#!/usr/bin/env python3
"""Download SuperMAG auroral-electrojet indices (SML and variants) into ``data/sml``.

The SuperMAG Web Service API returns the indices as JSON.  Running this
script interactively asks for the SuperMAG user id and the time range,
downloads the index series and stores it under ``data/sml``.

    $ ./scripts/pull_sml.py
    enter username (eg. JIJIN): jijin
    enter start date (eg. 20231201): 20231201
    enter duration (eg. 24:00): 24:00
    requesting SML indices for 2023-12-01T00:00 + 86400 s as user 'jijin'
    fetched 1440 samples -> .../data/sml/sml_20231201T0000Z_ext86400s.csv

The same query can be issued non-interactively with ``--user``, ``--start``
and ``--extent``, or as three positional values.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

BASE_URL = "https://supermag.jhuapl.edu/services/indices.php"
REGISTER_URL = "https://supermag.jhuapl.edu/index.php?page=register"

DEFAULT_START = "20231201"
DEFAULT_DURATION = "24:00"
DEFAULT_EXTENT = 86400
DEFAULT_FLAGS = "sml,smls,smld,mlat,mlt,glat,glon,stid,num"

# The service is most reliable for one-day windows; longer requests are
# split into daily chunks and concatenated.
MAX_CHUNK = 86400

REPO_ROOT = Path(__file__).resolve().parents[1]
SML_DIR = REPO_ROOT / "data" / "sml"


def ask(prompt: str, default: str = "") -> str:
    """Prompt the user and return ``default`` when the answer is empty."""
    try:
        answer = input(prompt).strip()
    except EOFError:
        answer = ""
    return answer or default


def ask_username() -> str:
    """Ask for the SuperMAG user id until a non-empty value is given."""
    while True:
        username = ask("enter username (eg. JIJIN): ")
        if username:
            return username
        print("A SuperMAG user id is required to query the indices service.")


def parse_start(text: str) -> datetime:
    """Parse a start timestamp into a UTC-aware ``datetime``.

    Accepted forms are ``YYYYMMDD``, ``YYYYMMDDHH``, ``YYYYMMDDHHMM``,
    ``YYYYMMDDHHMMSS`` and their dashed/colon separated equivalents.
    """
    cleaned = text.strip().replace(" ", "T")
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1]
    if "." in cleaned:
        cleaned = cleaned.split(".", 1)[0]
    compact = cleaned.replace("-", "").replace("T", "").replace(":", "")
    if compact.isdigit():
        formats = {
            8: "%Y%m%d",
            10: "%Y%m%d%H",
            12: "%Y%m%d%H%M",
            14: "%Y%m%d%H%M%S",
        }
        if len(compact) in formats:
            stamp = datetime.strptime(compact, formats[len(compact)])
            return stamp.replace(tzinfo=timezone.utc)
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            stamp = datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
        return stamp.replace(tzinfo=timezone.utc)
    raise ValueError(
        f"cannot interpret '{text}' as a start date; use YYYYMMDD (eg. 20231201)"
    )


def parse_extent(text: str) -> int:
    """Parse a duration given either in seconds or as ``HH:MM[:SS]``."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("duration must not be empty")
    if ":" in cleaned:
        parts = cleaned.split(":")
        if len(parts) > 3 or not all(part.strip() for part in parts):
            raise ValueError(
                f"cannot interpret '{text}' as a duration; use seconds or HH:MM[:SS]"
            )
        numbers = [int(part) for part in parts]
        while len(numbers) < 3:
            numbers.append(0)
        hours, minutes, seconds = numbers
        if minutes > 59 or seconds > 59:
            raise ValueError(f"'{text}' is not a valid HH:MM[:SS] duration")
        extent = hours * 3600 + minutes * 60 + seconds
    else:
        extent = int(float(cleaned))
    if extent <= 0:
        raise ValueError("duration must be positive")
    return extent


def build_url(
    logon: str, start: datetime, extent: int, flags: str = DEFAULT_FLAGS
) -> str:
    """Build the SuperMAG indices service URL for one request."""
    query = (
        "python&nohead"
        f"&start={quote(start.strftime('%Y-%m-%dT%H:%M'), safe=':')}"
        f"&logon={quote(logon, safe='')}"
        f"&extent={int(extent)}"
        f"&indices={quote(flags, safe=',')}"
    )
    return f"{BASE_URL}?{query}"


def fetch_indices(url: str, timeout: float = 180.0) -> list[dict[str, Any]]:
    """Fetch one JSON index record list from the SuperMAG service."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    text = response.text.strip()
    if not text:
        raise RuntimeError("SuperMAG returned an empty response")
    if text.upper().startswith("ERROR"):
        raise RuntimeError(f"SuperMAG rejected the request: {text}")
    if text.upper().startswith("OK"):
        return []
    start = text.find("[")
    if start < 0:
        raise RuntimeError(f"unexpected SuperMAG response: {text[:200]}")
    payload = json.loads(text[start:])
    if not isinstance(payload, list):
        raise RuntimeError("unexpected SuperMAG response: expected a JSON list")
    return payload


def iter_chunks(start: datetime, extent: int) -> list[tuple[datetime, int]]:
    """Split ``extent`` seconds from ``start`` into service-sized chunks."""
    chunks: list[tuple[datetime, int]] = []
    cursor = start
    remaining = extent
    while remaining > 0:
        span = min(remaining, MAX_CHUNK)
        chunks.append((cursor, span))
        cursor += timedelta(seconds=span)
        remaining -= span
    return chunks


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert the service records into a tidy time-indexed frame."""
    if not records:
        return pd.DataFrame()
    frame = pd.json_normalize(records)
    if "tval" in frame.columns:
        stamps = pd.to_datetime(frame["tval"], unit="s", utc=True)
        frame.insert(0, "time_utc", stamps.dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
        frame = frame.drop(columns=["tval"])
    return frame


def csv_safe(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace list-valued cells with JSON strings so the CSV stays flat."""
    safe = frame.copy()
    for column in safe.columns:
        if safe[column].map(lambda v: isinstance(v, (list, dict))).any():
            safe[column] = safe[column].map(
                lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v
            )
    return safe


def run(
    username: str,
    start: datetime,
    extent: int,
    flags: str = DEFAULT_FLAGS,
    out_dir: Path = SML_DIR,
) -> tuple[Path, Path]:
    """Download the index series and store it under ``out_dir``."""
    frames: list[pd.DataFrame] = []
    urls: list[str] = []
    for chunk_start, chunk_span in iter_chunks(start, extent):
        url = build_url(username, chunk_start, chunk_span, flags)
        urls.append(url)
        records = fetch_indices(url)
        if not records:
            print(f"  no data returned for {chunk_start:%Y-%m-%dT%H:%M} "
                  f"(+{chunk_span} s)")
            continue
        frames.append(records_to_frame(records))

    if not frames:
        raise RuntimeError("SuperMAG returned no index samples for this range")

    frame = pd.concat(frames, ignore_index=True)
    if "time_utc" in frame.columns:
        frame = frame.sort_values("time_utc").reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sml_{start.strftime('%Y%m%dT%H%M')}Z_ext{int(extent)}s"
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"

    csv_safe(frame).to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "extent_s": int(extent),
                "flags": flags,
                "requests": urls,
                "columns": list(frame.columns),
                "records": frame.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        )
    )
    return csv_path, json_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download SuperMAG SML indices into data/sml.",
        epilog=(
            "examples:\n"
            "  ./pull_sml.py\n"
            "  ./pull_sml.py --user jijin --start 2023-12-01T00:00 --extent 86400\n"
            "  ./pull_sml.py jijin 2023-12-01T00:00 86400\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "values",
        nargs="*",
        metavar="VALUE",
        help="optional positional user id, start time and duration",
    )
    parser.add_argument("--user", default=None, help="SuperMAG user id")
    parser.add_argument(
        "--start",
        default=None,
        help="start date, eg. YYYYMMDD, YYYYMMDDHHMM or YYYY-MM-DDTHH:MM",
    )
    parser.add_argument(
        "--extent",
        default=None,
        help="duration in seconds or HH:MM[:SS]",
    )
    parser.add_argument(
        "--flags", default=DEFAULT_FLAGS, help="comma separated index flags"
    )
    parser.add_argument(
        "--out-dir", default=None, help="output directory (default: data/sml)"
    )
    args = parser.parse_args(argv)

    positional = list(args.values) + [None, None, None]
    user_arg = args.user or positional[0]
    start_arg = args.start or positional[1]
    extent_arg = args.extent or positional[2]

    username = user_arg or ask_username()
    start_text = start_arg or ask(
        f"enter start date (eg. {DEFAULT_START}): ", DEFAULT_START
    )
    extent_text = extent_arg or ask(
        f"enter duration (eg. {DEFAULT_DURATION}): ", DEFAULT_DURATION
    )
    out_dir = Path(args.out_dir) if args.out_dir else SML_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        start = parse_start(start_text)
        extent = parse_extent(extent_text)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"requesting SML indices for {start:%Y-%m-%dT%H:%M} "
          f"+ {extent} s as user '{username}'")
    try:
        csv_path, json_path = run(username, start, extent, args.flags, out_dir)
    except Exception as exc:  # network, HTTP and service errors
        print(f"error: {exc}", file=sys.stderr)
        return 1

    frame = pd.read_csv(csv_path)
    print(f"fetched {len(frame)} samples -> {csv_path}")
    print(f"raw response -> {json_path}")
    if "SML" in frame.columns:
        column = pd.to_numeric(frame["SML"], errors="coerce")
        print(f"SML range: {column.min():.1f} to {column.max():.1f} nT")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        raise SystemExit(130)
