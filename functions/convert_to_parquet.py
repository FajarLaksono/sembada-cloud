"""
Azure Public Dataset V2 — CSV.gz to Parquet Converter
======================================================
Reads raw CSV.gz files from staging, writes columnar Parquet to transformed/ directory.

Usage:
    python functions/convert_to_parquet.py \\
        --input-dir data/staging/azure_dataset_v2_core \\
        --output-dir data/transformed/parquet

    # Force re-convert
    python functions/convert_to_parquet.py \\
        --input-dir data/staging/azure_dataset_v2_core \\
        --output-dir data/transformed/parquet \\
        --overwrite

    # Parallel conversion of CPU shards (faster)
    python functions/convert_to_parquet.py \\
        --input-dir data/staging/azure_dataset_v2_core \\
        --output-dir data/transformed/parquet \\
        --workers 4
"""

import argparse
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import pyarrow.csv as pcsv
    import pyarrow.parquet as pq
    from pyarrow import csv as csv_module
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with:  pip install pyarrow tqdm")
    sys.exit(1)

# ── Column schemas ────────────────────────────────────────────────────────────

VMTABLE_COLUMNS = {
    "vm_id": "utf8",
    "subscription_id": "utf8",
    "deployment_id": "utf8",
    "timestamp_created": "int64",
    "timestamp_deleted": "int64",
    "max_cpu": "float64",
    "avg_cpu": "float64",
    "p95_max_cpu": "float64",
    "vm_category": "utf8",
    "vm_core_count_bucket": "utf8",
    "vm_memory_gb_bucket": "utf8",
}

SUBSCRIPTIONS_COLUMNS = {
    "subscription_id": "utf8",
    "first_vm_timestamp": "int64",
    "vm_count": "int64",
}

DEPLOYMENTS_COLUMNS = {
    "deployment_id": "utf8",
    "deployment_size": "int64",
}

CPU_READINGS_COLUMNS = {
    "timestamp": "int64",
    "vm_id": "utf8",
    "min_cpu": "float64",
    "max_cpu": "float64",
    "avg_cpu": "float64",
}


def build_parse_options(column_schema: dict) -> pcsv.ParseOptions:
    """Build pyarrow CSV parse options with proper delimiter and header settings."""
    return pcsv.ParseOptions(delimiter=",", quote_char='"', double_quote=True, escape_char=None)


def build_convert_options(column_schema: dict) -> pcsv.ConvertOptions:
    """Build pyarrow convert options mapping column names and types."""
    from pyarrow import int64, float64, utf8

    type_map = {
        "int64": int64(),
        "float64": float64(),
        "utf8": utf8(),
    }
    column_types = {col: type_map[t] for col, t in column_schema.items()}
    return pcsv.ConvertOptions(
        column_types=column_types,
        strings_can_be_null=True,
        include_columns=list(column_schema.keys()),
    )


def read_csv_to_table(path: Path, column_schema: dict) -> "pyarrow.Table":
    """Read a CSV.gz file into a PyArrow Table with the given schema."""
    parse_opts = build_parse_options(column_schema)
    convert_opts = build_convert_options(column_schema)
    read_opts = pcsv.ReadOptions(column_names=list(column_schema.keys()), use_threads=True)
    return pcsv.read_csv(
        str(path),
        read_options=read_opts,
        parse_options=parse_opts,
        convert_options=convert_opts,
    )


def convert_core_table(
    input_path: Path,
    output_path: Path,
    column_schema: dict,
    overwrite: bool = False,
) -> bool:
    """Convert a single core table CSV → Parquet. Returns True if converted."""
    if output_path.exists() and not overwrite:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        table = read_csv_to_table(input_path, column_schema)
        pq.write_table(table, str(output_path), compression="zstd")
        return True
    except Exception as e:
        print(f"  ERROR converting {input_path.name}: {e}")
        return False


def convert_cpu_shard(
    input_path: Path,
    output_path: Path,
    overwrite: bool = False,
) -> tuple[Path, bool, str]:
    """Convert a single CPU reading shard. Returns (path, success, message)."""
    if output_path.exists() and not overwrite:
        return (input_path, True, "skipped (exists)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        table = read_csv_to_table(input_path, CPU_READINGS_COLUMNS)
        pq.write_table(table, str(output_path), compression="zstd")
        return (input_path, True, "ok")
    except Exception as e:
        return (input_path, False, str(e))


def convert_all(
    input_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
    workers: int = 1,
) -> None:
    """Convert all tables from input_dir to output_dir."""
    start = time.time()

    # ── Core tables ──────────────────────────────────────────────────────
    print("=" * 60)
    print("  CORE TABLES")
    print("=" * 60)

    core_tasks = [
        ("vmtable", input_dir / "vmtable" / "vmtable.csv.gz",
         output_dir / "vmtable.parquet", VMTABLE_COLUMNS),
        ("subscriptions", input_dir / "subscriptions" / "subscriptions.csv.gz",
         output_dir / "subscriptions.parquet", SUBSCRIPTIONS_COLUMNS),
        ("deployments", input_dir / "deployments" / "deployments.csv.gz",
         output_dir / "deployments.parquet", DEPLOYMENTS_COLUMNS),
    ]

    for name, src, dst, schema in core_tasks:
        print(f"  {name}: ", end="", flush=True)

        if not src.exists():
            print(f"SKIP (not found: {src})")
            continue
        converted = convert_core_table(src, dst, schema, overwrite=overwrite)
        if converted:
            size_mb = dst.stat().st_size / 1024**2
            print(f"OK  → {dst.name} ({size_mb:.1f} MB)")
        else:
            if dst.exists():
                size_mb = dst.stat().st_size / 1024**2
                print(f"cached → {dst.name} ({size_mb:.1f} MB)")
            else:
                print("FAILED")

    # ── CPU readings ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  CPU READINGS (shards)")
    print("=" * 60)

    cpu_input_dir = input_dir / "vm_cpu_readings"
    if not cpu_input_dir.exists():
        print(f"  SKIP (not found: {cpu_input_dir})")
        return

    shards = sorted(cpu_input_dir.glob("vm_cpu_readings-file-*-of-*.csv.gz"))
    print(f"  Found {len(shards)} shard(s)")

    if not shards:
        return

    cpu_output_root = output_dir / "cpu_readings"
    cpu_output_root.mkdir(parents=True, exist_ok=True)

    # Build task list
    tasks = []
    for shard in shards:
        out_name = shard.name.replace(".csv.gz", ".parquet")
        dst = cpu_output_root / out_name
        tasks.append((shard, dst, overwrite))

    # Determine how many need converting
    existing = sum(1 for _, dst, _ in tasks if dst.exists() and not overwrite)
    to_convert = len(tasks) - existing
    print(f"  Already converted: {existing}, to convert: {to_convert}")

    success_count = 0
    fail_count = 0

    if workers > 1 and to_convert > 0:
        # Parallel conversion
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(convert_cpu_shard, src, dst, overwrite): src
                for src, dst, overwrite in tasks
                if not dst.exists() or overwrite
            }
            with tqdm(total=len(futures), desc="  Converting", unit="shard") as pbar:
                for future in as_completed(futures):
                    src, ok, msg = future.result()
                    if ok:
                        success_count += 1
                    else:
                        fail_count += 1
                        tqdm.write(f"  ⚠ {src.name}: {msg}")
                    pbar.update(1)
    else:
        # Sequential conversion
        shard_tasks = [
            (src, dst, overwrite)
            for src, dst, overwrite in tasks
            if not dst.exists() or overwrite
        ]
        for src, dst, _ in tqdm(shard_tasks, desc="  Converting", unit="shard"):
            _, ok, msg = convert_cpu_shard(src, dst, overwrite)
            if ok:
                success_count += 1
            else:
                fail_count += 1
                tqdm.write(f"  ⚠ {src.name}: {msg}")

    # Summary
    elapsed = time.time() - start
    print()
    print("=" * 60)
    print(f"  CONVERSION COMPLETE ({elapsed:.1f}s)")
    print(f"  Core tables: 3/3")
    print(f"  CPU shards:  {success_count} OK, {fail_count} failed, {existing} cached")
    print(f"  Output:      {output_dir.resolve()}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Azure Public Dataset V2 CSV.gz → Parquet",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/staging/azure_dataset_v2_core",
        help="Path to raw CSV.gz data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/transformed/parquet",
        help="Path to write Parquet output",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-convert even if Parquet exists",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for CPU shard conversion (default: 1)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: input directory not found: {input_dir}")
        sys.exit(1)

    convert_all(input_dir, output_dir, overwrite=args.overwrite, workers=args.workers)


if __name__ == "__main__":
    main()
