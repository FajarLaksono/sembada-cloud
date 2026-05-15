"""
Azure Public Dataset V2 — Downloader
=====================================
Downloads CSV.gz VM trace data + supplementary files (notebook, schema, TXT metadata).
Source: https://github.com/Azure/AzurePublicDataset

MODES (set via --mode flag or DOWNLOAD_MODE variable):
  core      → vmtable + subscriptions + deployments + supplementary (~421 MB)
  sample    → core + supplementary + first 3 CPU reading files (~2.9 GB)
  full      → core + supplementary + all 195 CPU reading files (~156 GB)

Usage:
  pip install requests tqdm
  python app/src/download_azure_v2.py --mode core
  python app/src/download_azure_v2.py --mode sample
  python app/src/download_azure_v2.py --mode full
  python app/src/download_azure_v2.py --mode full --workers 4   # parallel downloads
"""

import os
import sys
import time
import hashlib
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    from tqdm import tqdm
except ImportError:
    print("Missing dependencies. Run:  pip install requests tqdm")
    sys.exit(1)

# ── Base URL ──────────────────────────────────────────────────────────────────
BASE = "https://azurepublicdatasettraces.blob.core.windows.net/azurepublicdatasetv2"

# ── File definitions ──────────────────────────────────────────────────────────
CORE_FILES = [
    (f"{BASE}/trace_data/vmtable/vmtable.csv.gz",           "vmtable/vmtable.csv.gz"),
    (f"{BASE}/trace_data/subscriptions/subscriptions.csv.gz","subscriptions/subscriptions.csv.gz"),
    (f"{BASE}/trace_data/deployments/deployments.csv.gz",    "deployments/deployments.csv.gz"),
]

CPU_FILES = [
    (
        f"{BASE}/trace_data/vm_cpu_readings/vm_cpu_readings-file-{i}-of-195.csv.gz",
        f"vm_cpu_readings/vm_cpu_readings-file-{i}-of-195.csv.gz",
    )
    for i in range(1, 196)
]

SUPPLEMENTARY_FILES = [
    (
        f"{BASE}/Azure%202019%20Public%20Dataset%20V2%20-%20Trace%20Analysis.ipynb",
        "Azure 2019 Public Dataset V2 - Trace Analysis.ipynb",
    ),
    (f"{BASE}/schema.csv",                    "schema.csv"),
    (f"{BASE}/azure2019_data/category.txt",   "azure2019_data/category.txt"),
    (f"{BASE}/azure2019_data/cores.txt",      "azure2019_data/cores.txt"),
    (f"{BASE}/azure2019_data/cpu.txt",        "azure2019_data/cpu.txt"),
    (f"{BASE}/azure2019_data/deployment.txt", "azure2019_data/deployment.txt"),
    (f"{BASE}/azure2019_data/lifetime.txt",   "azure2019_data/lifetime.txt"),
    (f"{BASE}/azure2019_data/memory.txt",     "azure2019_data/memory.txt"),
]

MODES = {
    "core":   CORE_FILES + SUPPLEMENTARY_FILES,
    "sample": CORE_FILES + SUPPLEMENTARY_FILES + CPU_FILES[:3],
    "full":   CORE_FILES + SUPPLEMENTARY_FILES + CPU_FILES,
}

MODE_SIZES = {
    "core":   "~421 MB  (core tables + supplementary docs)",
    "sample": "~2.9 GB  (core + supplementary + first 3 CPU files)",
    "full":   "~156 GB  (complete dataset, 206 files)",
}

# ── Download logic ────────────────────────────────────────────────────────────
CHUNK = 4 * 1024 * 1024  # 4 MB chunks for large files
MAX_RETRIES = 5
RETRY_BACKOFF = [2, 4, 8, 16, 32]  # seconds between retries
print_lock = threading.Lock()


def create_session() -> requests.Session:
    """Create a requests session with retry and connection pooling."""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "azure-dataset-downloader/1.0"})
    return session


def sizeof_fmt(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num) < 1024.0:
            return f"{num:6.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def download_file(session: requests.Session, url: str, dest: Path, show_progress: bool = True) -> bool:
    """Download a single file with resume support, retries, and progress reporting."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(MAX_RETRIES):
        try:
            # HEAD request to get content length and determine resume behavior
            head = session.head(url, timeout=(10, 20))
            head.raise_for_status()
            total = int(head.headers.get("Content-Length", 0))
            total_label = sizeof_fmt(total) if total > 0 else "unknown size"

            # Skip if already fully downloaded
            if dest.exists() and total > 0 and dest.stat().st_size == total:
                with print_lock:
                    print(f"  ✓ Already complete: {dest.name} ({total_label})")
                return True

            existing = dest.stat().st_size if dest.exists() else 0
            headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}
            mode = "ab" if existing > 0 else "wb"

            with print_lock:
                if existing > 0:
                    print(f"  ↻ Resuming {dest.name} from {sizeof_fmt(existing)} / {total_label}")
                else:
                    print(f"  → Downloading {dest.name} ({total_label})")

            with session.get(url, headers=headers, stream=True, timeout=(10, 30)) as r:
                if existing > 0 and r.status_code == 200:
                    with print_lock:
                        print("    Server ignored Range header; restarting download from zero.")
                    existing = 0
                    mode = "wb"

                r.raise_for_status()
                bar_desc = dest.name[:40].ljust(40)
                progress = tqdm(
                    total=total,
                    initial=existing,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=bar_desc,
                    leave=False,
                    disable=not show_progress,
                    miniters=1,
                )

                with open(dest, mode) as f:
                    for chunk in r.iter_content(chunk_size=CHUNK):
                        if chunk:
                            f.write(chunk)
                            progress.update(len(chunk))

                progress.close()

            if total > 0 and dest.stat().st_size != total:
                raise IOError(
                    f"Size mismatch: expected {total}, got {dest.stat().st_size}"
                )

            with print_lock:
                print(f"  ✓ Finished {dest.name} ({sizeof_fmt(dest.stat().st_size)})")
            return True

        except (requests.RequestException, IOError) as e:
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            with print_lock:
                print(f"  ✗ Attempt {attempt+1}/{MAX_RETRIES} failed for {dest.name}: {e}")
                if attempt < MAX_RETRIES - 1:
                    print(f"    Retrying in {wait}s…")
            time.sleep(wait)

    with print_lock:
        print(f"  ✗✗ FAILED after {MAX_RETRIES} attempts: {dest.name}")
    return False


def download_all(
    file_list: list,
    output_dir: Path,
    workers: int = 1,
) -> None:
    total_files = len(file_list)
    failed = []
    session = create_session()

    print(f"\n{'─'*60}")
    print(f"  Output directory : {output_dir.resolve()}")
    print(f"  Files to download: {total_files}")
    print(f"  Parallel workers : {workers}")
    print(f"{'─'*60}\n")

    if workers == 1:
        # Sequential — cleaner progress bars
        for i, (url, rel_path) in enumerate(file_list, 1):
            dest = output_dir / rel_path
            print(f"[{i:3d}/{total_files}] {rel_path}")
            ok = download_file(session, url, dest, show_progress=True)
            if not ok:
                failed.append(rel_path)
    else:
        # Parallel — overall summary plus per-file status messages
        overall = tqdm(total=total_files, unit="file", desc="Overall", position=0)

        def _task(args):
            url, rel_path = args
            dest = output_dir / rel_path
            ok = download_file(session, url, dest, show_progress=False)
            with print_lock:
                overall.update(1)
                overall.set_postfix_str(rel_path[-35:])
            return (rel_path, ok)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_task, item): item for item in file_list}
            for future in as_completed(futures):
                rel_path, ok = future.result()
                if not ok:
                    failed.append(rel_path)

        overall.close()

    # Summary
    print(f"\n{'─'*60}")
    print(f"  ✓ Completed : {total_files - len(failed)}/{total_files} files")
    if failed:
        print(f"  ✗ Failed    : {len(failed)} files")
        for f in failed:
            print(f"      - {f}")
        print(f"\n  Re-run the script to retry failed files (resume is automatic).")
    else:
        print(f"  All files downloaded successfully.")
    print(f"{'─'*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download Azure Public Dataset V2 (2019 VM traces)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            [f"  {k:8s}  {v}" for k, v in MODE_SIZES.items()]
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["core", "sample", "full"],
        default="core",
        help="Download mode (default: core)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("azure_dataset_v2"),
        help="Destination folder (default: ./azure_dataset_v2)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel download threads (default: 1, max recommended: 4)",
    )
    parser.add_argument(
        "--cpu-files",
        type=str,
        default=None,
        help="Override: specific CPU file range, e.g. '1-10' downloads files 1 to 10 only",
    )
    args = parser.parse_args()

    file_list = MODES[args.mode].copy()

    # Custom CPU range override
    if args.cpu_files:
        try:
            start, end = map(int, args.cpu_files.split("-"))
            cpu_subset = CPU_FILES[start - 1 : end]
            # Replace cpu files in list with subset
            file_list = CORE_FILES + cpu_subset
            print(f"  CPU files override: {start}–{end} ({len(cpu_subset)} files)")
        except Exception:
            print("  --cpu-files must be in format START-END, e.g. '1-10'")
            sys.exit(1)

    print(f"\n  Azure Public Dataset V2 — Downloader")
    print(f"  Mode: {args.mode}  ({MODE_SIZES[args.mode]})")

    download_all(file_list, args.output_dir, workers=args.workers)


if __name__ == "__main__":
    main()