#!/usr/bin/env python3
"""Download SuperMAG all-station magnetometer data into ``data/all_stations``.

The SuperMAG Web Service API returns ground magnetometer observations for
all reporting stations in requested time windows. This script queries the
inventory, downloads high-fidelity multi-station time series in efficient
parallel chunks via SuperMAG's Web Service API, and compiles them into a
NetCDF file matching the official daily SuperMAG archive layout under
``../data/all_stations`` (or ``data/all_stations``).

Interactive usage:
    $ ./scripts/pull_supmag_meter.py
    enter username (eg. JIJIN): JIJIN
    enter start date (eg. 20231201): 20231201
    enter duration (eg. 24:00): 24:00

Non-interactive usage:
    $ ./scripts/pull_supmag_meter.py --user JIJIN --start 20231201 --extent 24:00
    $ ./scripts/pull_supmag_meter.py JIJIN 20231201 24:00
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import netCDF4 as nc
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from pull_sml import ask, parse_extent, parse_start  # noqa: E402
except ImportError:
    # Fallback in case pull_sml is not in path
    def ask(prompt: str, default: str = "") -> str:
        try:
            answer = input(prompt).strip()
        except EOFError:
            answer = ""
        return answer or default

    def parse_start(text: str) -> datetime:
        cleaned = text.strip().replace(" ", "T")
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1]
        compact = cleaned.replace("-", "").replace("T", "").replace(":", "")
        if compact.isdigit() and len(compact) == 8:
            stamp = datetime.strptime(compact, "%Y%m%d")
            return stamp.replace(tzinfo=timezone.utc)
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
            try:
                stamp = datetime.strptime(cleaned, fmt)
                return stamp.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        raise ValueError(f"cannot interpret '{text}' as a start date (eg. 20231201)")

    def parse_extent(text: str) -> int:
        cleaned = text.strip()
        if ":" in cleaned:
            parts = [int(p) for p in cleaned.split(":")]
            while len(parts) < 3:
                parts.append(0)
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        return int(float(cleaned))


INVENTORY_URL = "https://supermag.jhuapl.edu/services/inventory.php"
DATA_URL = "https://supermag.jhuapl.edu/services/data.php"

DEFAULT_USER = "JIJIN"
DEFAULT_START = "20231201"
DEFAULT_DURATION = "24:00"

STEP_SECONDS = 60
DEFAULT_CHUNK_SECONDS = 1800  # 30-minute chunks provide rapid, reliable downloads
FILL_VALUE = 999990.0

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "all_stations"

DESCRIPTION_HEADER = """ 
File contains N, E, Z from individual ground magnetometer stations.  
The coordinate system is defined as: N is positive in the direction of local
magnetic north; E is positive in the local magnetic east direction; and Z is positive
vertical down.
 
In case data are used in publications or presentations the following text
must be used for acknowledgements:
 

For the ground magnetometer data we gratefully acknowledge:
  INTERMAGNET; USGS Geomagnetism Program; CARISMA, PI Ian Mann; CANMOS, 
  Geomagnetism Unit of the Geological Survey of Canada; The S-RAMP
  Database, PI K. Yumoto and Dr. K. Shiokawa; The SPIDR database; 
  AARI, PI Oleg Troshichev; The MACCS program, PI M. Engebretson; 
  GIMA; MEASURE, UCLA IGPP and Florida Institute of Technology; 
  SAMBA, PI Eftyhia Zesta; 210 Chain, PI K. Yumoto; SAMNET, PI 
  Farideh Honary; IMAGE, PI Liisa Juusola; Finnish Meteorological 
  Institute, PI Liisa Juusola; Sodankylä Geophysical Observatory, 
  PI Tero Raita; UiT the Arctic University of Norway, Tromsø 
  Geophysical Observatory, PI Magnar G. Johnsen; GFZ German 
  Research Centre For Geosciences, PI Jürgen Matzka; Institute of 
  Geophysics, Polish Academy of Sciences, PI Anne Neska and Jan
  Reda; Polar Geophysical Institute, PI Alexander Yahnin and Yarolav
  Sakharov; Geological Survey of Sweden, PI Gerhard Schwarz; Swedish 
  Institute of Space Physics, PI Masatoshi Yamauchi; AUTUMN, PI 
  Martin Connors; DTU Space, Thom Edwards and PI Anna Willer; 
  South Pole and McMurdo Magnetometer, PI's Louis J. Lanzarotti
  and Alan T. Weatherwax; ICESTAR; RAPIDMAG; British Artarctic
  Survey; McMac, PI Dr. Peter Chi; BGS, PI Dr. Susan Macmillan; 
  Pushkov Institute of Terrestrial Magnetism, Ionosphere and Radio
  Wave Propagation (IZMIRAN);  MFGI, PI B. Heilig; University of 
  L’Aquila, PI M. Vellante; BCMT, V. Lesur and A. Chambodut; Data 
  obtained in cooperation with Geoscience Australia, PI Andrew Lewis; 
  PENGUIn, co-PIs Bob Clauer, Michael Hartinger, and Zhonghua Xu; 
  MagStar, PI Jennifer Gannon; LISN PI Cesar Valladares; Geophysical 
  and Astronomical Observatory of the University of Coimbra, PI Paulo 
  Ribeiro; Leibniz Institute of Atmospheric Physics, PI Jorge Chau; 
  SuperMAG, PI Rafael Mesquita; Data obtained in cooperation with 
  the Australian Bureau of Meteorology, PI Richard Marshall.
 
********************************************************
*              THESE ARE NOT FINAL DATA                *
* DESPITE EXTENSIVE CLEANING NUMEROUS BUGS STILL EXIST *
*        CHECK DATA QUALITY BEFORE PUBLICATION         *
********************************************************


Comments, questions or suggestions should be directed to:
    Dr. Jesper W. Gjerloev
    Johns Hopkins University
    Applied Physics Laboratory
    11100 Johns Hopkins Road
    Laurel, Maryland 20723
    (+1) 240-228-5410
    Jesper.Gjerloev@jhuapl.edu

"""

FLOAT_VARIABLES = {
    "dbe_geo": "Magnetic field east component, geographic coordinates [nT]",
    "dbe_nez": "Magnetic field east component, NEZ coordinates [nT]",
    "dbn_geo": "Magnetic field north component, geographic coordinates [nT]",
    "dbn_nez": "Magnetic field north component, NEZ coordinates [nT]",
    "dbz_geo": "Magnetic field vertical component, geographic coordinates [nT]",
    "dbz_nez": "Magnetic field vertical component, NEZ coordinates [nT]",
    "decl": "Magnetic Declination [degrees]",
    "glat": "Geographic Latitude [degrees]",
    "glon": "Geographic Longtiude [degrees]",
    "mcolat": "Magnetic Colatitude [degrees]",
    "mlt": "Magnetic Local Time [h]",
    "sza": "Solar Zenith Angle [degrees]",
}

BASELINE_LABELS = {"all": "All", "yearly": "Yearly", "none": "None"}


def _value(record: dict[str, Any], key: str) -> float:
    """Read one numeric field, mapping fill values onto NaN."""
    val = record.get(key)
    if val is None:
        return np.nan
    try:
        v = float(val)
        if not np.isfinite(v) or abs(v) >= FILL_VALUE:
            return np.nan
        return v
    except (ValueError, TypeError):
        return np.nan


def _component(record: dict[str, Any], name: str, part: str) -> float:
    """Read one N/E/Z component, mapping fill values onto NaN."""
    block = record.get(name)
    if not isinstance(block, dict):
        return np.nan
    val = block.get(part)
    if val is None:
        return np.nan
    try:
        v = float(val)
        if not np.isfinite(v) or abs(v) >= FILL_VALUE:
            return np.nan
        return v
    except (ValueError, TypeError):
        return np.nan


def fetch_inventory(
    logon: str, start: datetime, extent: int, timeout: float = 30.0
) -> list[str]:
    """Return the list of station IAGA codes reporting in the window."""
    query = (
        "python&nohead"
        f"&start={quote(start.strftime('%Y-%m-%dT%H:%M'), safe=':')}"
        f"&logon={quote(logon, safe='')}"
        f"&extent={int(extent)}"
    )
    url = f"{INVENTORY_URL}?{query}"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    text = response.text.strip()
    if text.upper().startswith("ERROR"):
        raise RuntimeError(f"SuperMAG rejected inventory query: {text}")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and lines[0].isdigit():
        lines = lines[1:]
    return lines


def fetch_chunk(
    logon: str,
    chunk_start: datetime,
    chunk_span: int,
    baseline: str = "all",
    retries: int = 3,
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    """Fetch one consolidated time chunk for all stations via Web Service API."""
    query = (
        "python&nohead"
        f"&start={quote(chunk_start.strftime('%Y-%m-%dT%H:%M'), safe=':')}"
        f"&logon={quote(logon, safe='')}"
        f"&extent={int(chunk_span)}"
        "&fmt=json"
        "&mlt&mag&geo&decl&sza"
    )
    if baseline != "all":
        query += f"&baseline={quote(baseline)}"

    url = f"{DATA_URL}?{query}"

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            text = resp.text.strip()
            if text.startswith("OK"):
                text = text[2:].strip()
            if not text:
                return []
            if text.upper().startswith("ERROR"):
                raise RuntimeError(f"SuperMAG error: {text}")

            idx = text.find("[")
            if idx < 0:
                return []
            return json.loads(text[idx:])
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * attempt)

    raise RuntimeError(
        f"Failed to fetch chunk {chunk_start:%Y-%m-%dT%H:%M} (+{chunk_span}s): {last_error}"
    )


def download_range(
    logon: str,
    start: datetime,
    extent: int,
    baseline: str = "all",
    workers: int = 4,
    chunk_seconds: int = DEFAULT_CHUNK_SECONDS,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Download data for all reporting stations across the requested extent."""
    print(f"querying inventory from {start:%Y-%m-%dT%H:%M} for {extent} s...")
    stations = fetch_inventory(logon, start, extent)
    if not stations:
        raise RuntimeError("No stations reported data for the requested window.")

    n_stations = len(stations)
    n_time = max(1, extent // STEP_SECONDS)
    station_to_idx = {st: i for i, st in enumerate(stations)}

    print(f"found {n_stations} reporting stations across {n_time} minutes")

    arrays = {
        name: np.full((n_time, n_stations), np.nan, dtype=np.float32)
        for name in FLOAT_VARIABLES
    }

    chunks: list[tuple[datetime, int, int]] = []
    cursor = start
    remaining = extent
    minute_offset = 0

    while remaining > 0:
        span = min(remaining, chunk_seconds)
        chunks.append((cursor, span, minute_offset))
        cursor += timedelta(seconds=span)
        remaining -= span
        minute_offset += span // STEP_SECONDS

    total_chunks = len(chunks)
    print(f"downloading {total_chunks} chunk(s) using {workers} parallel worker(s)...")

    lock = threading.Lock()
    completed = 0
    start_epoch = start.timestamp()

    def process_chunk(chunk_info: tuple[datetime, int, int]) -> int:
        c_start, c_span, _ = chunk_info
        records = fetch_chunk(logon, c_start, c_span, baseline)
        rec_count = len(records)

        with lock:
            for rec in records:
                tval = rec.get("tval")
                if tval is None:
                    continue
                min_idx = int(round((float(tval) - start_epoch) / STEP_SECONDS))
                if not (0 <= min_idx < n_time):
                    continue

                st_id = rec.get("iaga")
                if st_id not in station_to_idx:
                    continue
                s_idx = station_to_idx[st_id]

                arrays["dbn_nez"][min_idx, s_idx] = _component(rec, "N", "nez")
                arrays["dbn_geo"][min_idx, s_idx] = _component(rec, "N", "geo")
                arrays["dbe_nez"][min_idx, s_idx] = _component(rec, "E", "nez")
                arrays["dbe_geo"][min_idx, s_idx] = _component(rec, "E", "geo")
                arrays["dbz_nez"][min_idx, s_idx] = _component(rec, "Z", "nez")
                arrays["dbz_geo"][min_idx, s_idx] = _component(rec, "Z", "geo")
                arrays["decl"][min_idx, s_idx] = _value(rec, "decl")
                arrays["glat"][min_idx, s_idx] = _value(rec, "glat")
                arrays["glon"][min_idx, s_idx] = _value(rec, "glon")
                arrays["mcolat"][min_idx, s_idx] = _value(rec, "mcolat")
                arrays["mlt"][min_idx, s_idx] = _value(rec, "mlt")
                arrays["sza"][min_idx, s_idx] = _value(rec, "sza")

        return rec_count

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        future_map = {pool.submit(process_chunk, c): c for c in chunks}
        for future in as_completed(future_map):
            c_start, c_span, _ = future_map[future]
            try:
                rec_count = future.result()
            except Exception as exc:
                print(f"error on {c_start:%H:%M} (+{c_span}s): {exc}", file=sys.stderr)
            with lock:
                completed += 1
                sys.stdout.write(
                    f"\r[{completed}/{total_chunks}] fetched {completed*100//total_chunks}% "
                    f"({c_start:%Y-%m-%d %H:%M})"
                )
                sys.stdout.flush()

    print("\nall chunks downloaded successfully.")
    return stations, arrays


def build_description(
    stations: list[str],
    start: datetime,
    end: datetime,
    baseline: str,
    downloaded_at: str,
) -> str:
    """Format file description attribute matching official SuperMAG files."""
    stamp = "%Y-%m-%dT%H:%M:%S.000"
    tail = (
        f"This file downloaded from http://supermag.jhuapl.edu on {downloaded_at}\n"
        f"Data Revision:0006\n"
        f"Interval: {start.strftime(stamp)} - {end.strftime(stamp)} "
        f"Step: {STEP_SECONDS} seconds\n"
        f"Fill/Gap Value: NaN\n"
        f"Stations Selected: {','.join(stations)}\n"
        f"Baseline Subtraction: {BASELINE_LABELS.get(baseline, baseline.capitalize())}\n"
        "Parameters: IAGA Station ID | Geo. Lon. | Geo. Lat. | MLT | Mag Colat. "
        "| Mag. Declination | Solar Zenith Angle | Mag. Field NEZ | "
        "Mag. Field Geo.\n"
    )
    return DESCRIPTION_HEADER + tail


def write_netcdf(
    path: Path,
    start: datetime,
    end: datetime,
    stations: list[str],
    arrays: dict[str, np.ndarray],
    baseline: str,
    downloaded_at: str,
) -> None:
    """Write SuperMAG-layout NetCDF-4 dataset."""
    n_time = arrays["dbn_nez"].shape[0]
    n_station = len(stations)

    path.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("block", n_time)
        ds.createDimension("vector", n_station)
        ds.description = build_description(
            stations, start, end, baseline, downloaded_at
        )

        for name, desc in FLOAT_VARIABLES.items():
            var = ds.createVariable(name, "f4", ("block", "vector"), fill_value=np.nan)
            var.description = desc
            var[:] = arrays[name]

        station_ids = ds.createVariable("id", str, ("block", "vector"))
        station_ids.description = "Station Identifier [IAGA]"
        station_ids[:] = np.array(
            [list(stations) for _ in range(n_time)], dtype=object
        )

        stamps = [start + timedelta(minutes=m) for m in range(n_time)]
        for name, values in (
            ("time_yr", [st.year for st in stamps]),
            ("time_mo", [st.month for st in stamps]),
            ("time_dy", [st.day for st in stamps]),
            ("time_hr", [st.hour for st in stamps]),
            ("time_mt", [st.minute for st in stamps]),
            ("time_sc", [st.second for st in stamps]),
        ):
            dtype = "f8" if name == "time_sc" else "i2"
            var = ds.createVariable(name, dtype, ("block",))
            var.description = {
                "time_yr": "Start Year",
                "time_mo": "Start Month",
                "time_dy": "Start Day",
                "time_hr": "Start Hour",
                "time_mt": "Start Minute",
                "time_sc": "Start Second",
            }[name]
            var[:] = np.asarray(values, dtype=dtype)

        ext_var = ds.createVariable("extent", "f8", ("block",))
        ext_var.description = "Extent of Record [seconds]"
        ext_var[:] = np.full(n_time, float(STEP_SECONDS))

        npnt_var = ds.createVariable("npnt", "i4", ("block",))
        npnt_var.description = "Number of Points in Record"
        npnt_var[:] = np.full(n_time, n_station, dtype="i4")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download SuperMAG all-station magnetometer data into NetCDF format.",
        epilog=(
            "examples:\n"
            "  ./pull_supmag_meter.py\n"
            "  ./pull_supmag_meter.py --user JIJIN --start 20231201 --extent 24:00\n"
            "  ./pull_supmag_meter.py JIJIN 20231201 24:00\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "values",
        nargs="*",
        metavar="VALUE",
        help="optional positional user id, start date and duration",
    )
    parser.add_argument("--user", default=None, help="SuperMAG user id (eg. JIJIN)")
    parser.add_argument(
        "--start", default=None, help="start date, eg. YYYYMMDD or YYYYMMDDHHMM"
    )
    parser.add_argument(
        "--extent", default=None, help="duration in seconds or HH:MM[:SS]"
    )
    parser.add_argument(
        "--baseline",
        default="all",
        choices=sorted(BASELINE_LABELS),
        help="baseline subtraction (default: all)",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="parallel download workers (default: 4)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SECONDS,
        help=f"chunk size in seconds (default: {DEFAULT_CHUNK_SECONDS})",
    )
    parser.add_argument(
        "--output", default=None, help="explicit output NetCDF file path"
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="output directory (default: data/all_stations)",
    )
    args = parser.parse_args(argv)

    positional = list(args.values) + [None, None, None]
    user_arg = args.user or positional[0]
    start_arg = args.start or positional[1]
    extent_arg = args.extent or positional[2]

    username = user_arg or ask(f"enter username (eg. {DEFAULT_USER}): ", DEFAULT_USER)
    start_text = start_arg or ask(
        f"enter start date (eg. {DEFAULT_START}): ", DEFAULT_START
    )
    extent_text = extent_arg or ask(
        f"enter duration (eg. {DEFAULT_DURATION}): ", DEFAULT_DURATION
    )

    out_dir = Path(args.out_dir) if args.out_dir else DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        start = parse_start(start_text)
        extent = parse_extent(extent_text)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    end = start + timedelta(seconds=extent)
    downloaded_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if args.output:
        out_path = Path(args.output)
    else:
        # Match standard SuperMAG naming convention: SuperMAG_60s_YYYYMMDD_rev-0006.<timestamp>.netcdf
        ts = int(time.time())
        out_path = out_dir / f"SuperMAG_60s_{start:%Y%m%d}_rev-0006.{ts}.netcdf"

    print(
        f"requesting magnetometer data from {start:%Y-%m-%dT%H:%M} "
        f"+ {extent} s ({extent // 60} min) as user '{username}'"
    )

    try:
        stations, arrays = download_range(
            logon=username,
            start=start,
            extent=extent,
            baseline=args.baseline,
            workers=args.workers,
            chunk_seconds=args.chunk_size,
        )
        print(f"writing NetCDF dataset to {out_path} ...")
        write_netcdf(
            path=out_path,
            start=start,
            end=end,
            stations=stations,
            arrays=arrays,
            baseline=args.baseline,
            downloaded_at=downloaded_at,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"successfully wrote {out_path} ({file_size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        raise SystemExit(130)
