"""Tests for CSV.gz → Parquet converter (app.src.convert_to_parquet)."""

import csv
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call


class TestColumnSchemas:
    """Module-level column schemas are correctly defined."""

    def test_vmtable_columns(self):
        from app.src.convert_to_parquet import VMTABLE_COLUMNS
        assert "vm_id" in VMTABLE_COLUMNS
        assert VMTABLE_COLUMNS["vm_id"] == "utf8"
        assert VMTABLE_COLUMNS["timestamp_created"] == "int64"
        assert VMTABLE_COLUMNS["avg_cpu"] == "float64"

    def test_subscriptions_columns(self):
        from app.src.convert_to_parquet import SUBSCRIPTIONS_COLUMNS
        assert list(SUBSCRIPTIONS_COLUMNS.keys()) == ["subscription_id", "first_vm_timestamp", "vm_count"]

    def test_deployments_columns(self):
        from app.src.convert_to_parquet import DEPLOYMENTS_COLUMNS
        assert list(DEPLOYMENTS_COLUMNS.keys()) == ["deployment_id", "deployment_size"]

    def test_cpu_readings_columns(self):
        from app.src.convert_to_parquet import CPU_READINGS_COLUMNS
        assert "timestamp" in CPU_READINGS_COLUMNS
        assert "vm_id" in CPU_READINGS_COLUMNS
        assert "avg_cpu" in CPU_READINGS_COLUMNS


class TestBuildOptions:
    """Build options functions return correct types."""

    def test_build_parse_options(self):
        from app.src.convert_to_parquet import build_parse_options
        from pyarrow.csv import ParseOptions

        opts = build_parse_options({"col": "utf8"})
        assert isinstance(opts, ParseOptions)

    def test_build_convert_options(self):
        from app.src.convert_to_parquet import build_convert_options
        from pyarrow.csv import ConvertOptions

        schema = {"id": "utf8", "val": "float64"}
        opts = build_convert_options(schema)
        assert isinstance(opts, ConvertOptions)
        assert "id" in opts.column_types
        assert "val" in opts.column_types

    def test_build_convert_options_columns_filtered(self):
        from app.src.convert_to_parquet import build_convert_options

        schema = {"a": "utf8", "b": "int64"}
        opts = build_convert_options(schema)
        assert opts.include_columns == ["a", "b"]


class TestReadCsvToTable:
    """read_csv_to_table with tiny temp CSV data."""

    def test_reads_csv_successfully(self, tmp_path):
        from app.src.convert_to_parquet import read_csv_to_table, VMTABLE_COLUMNS

        csv_path = tmp_path / "test.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["vm_a", "sub_1", "dep_1", "1000", "2000", "50.0", "10.0", "30.0", "Interactive", "4", "8"])
            writer.writerow(["vm_b", "sub_1", "dep_1", "2000", "3000", "80.0", "5.0", "60.0", "Batch", "8", "16"])

        table = read_csv_to_table(csv_path, VMTABLE_COLUMNS)
        assert table.num_rows == 2
        assert table.num_columns == len(VMTABLE_COLUMNS)
        assert table.column("vm_id")[0].as_py() == "vm_a"


class TestConvertCoreTable:
    """convert_core_table handles temp files correctly."""

    def test_converts_and_writes_parquet(self, tmp_path):
        from app.src.convert_to_parquet import convert_core_table, VMTABLE_COLUMNS
        import pyarrow as pa

        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.parquet"

        with open(input_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["vm_a", "s1", "d1", "0", "3600", "50", "10", "30", "Interactive", "4", "8"])

        result = convert_core_table(input_path, output_path, VMTABLE_COLUMNS, overwrite=False)
        assert result is True
        assert output_path.exists()

        table = pa.parquet.read_table(str(output_path))
        assert table.num_rows == 1

    def test_skips_if_exists_and_no_overwrite(self, tmp_path):
        from app.src.convert_to_parquet import convert_core_table, VMTABLE_COLUMNS

        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.parquet"

        output_path.write_text("fake parquet")
        with open(input_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["vm_a", "s1", "d1", "0", "3600", "50", "10", "30", "Interactive", "4", "8"])

        result = convert_core_table(input_path, output_path, VMTABLE_COLUMNS, overwrite=False)
        assert result is False

    def test_overwrite_when_requested(self, tmp_path):
        from app.src.convert_to_parquet import convert_core_table, VMTABLE_COLUMNS

        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.parquet"

        output_path.write_text("old")
        with open(input_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["vm_a", "s1", "d1", "0", "3600", "50", "10", "30", "Interactive", "4", "8"])

        result = convert_core_table(input_path, output_path, VMTABLE_COLUMNS, overwrite=True)
        assert result is True

    def test_returns_false_on_error(self, tmp_path):
        from app.src.convert_to_parquet import convert_core_table, VMTABLE_COLUMNS

        missing_input = tmp_path / "nonexistent.csv"
        output_path = tmp_path / "out.parquet"

        result = convert_core_table(missing_input, output_path, VMTABLE_COLUMNS)
        assert result is False


class TestConvertCpuShard:
    """convert_cpu_shard with temp files."""

    def test_converts_single_shard(self, tmp_path):
        from app.src.convert_to_parquet import convert_cpu_shard

        input_path = tmp_path / "shard.csv"
        output_path = tmp_path / "shard.parquet"

        with open(input_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["1000", "vm_a", "10", "50", "25"])
            writer.writerow(["2000", "vm_a", "5", "80", "40"])

        src, ok, msg = convert_cpu_shard(input_path, output_path)
        assert ok is True
        assert msg == "ok"
        assert output_path.exists()

    def test_skips_existing_shard(self, tmp_path):
        from app.src.convert_to_parquet import convert_cpu_shard

        input_path = tmp_path / "shard.csv"
        output_path = tmp_path / "shard.parquet"
        output_path.write_text("existing")

        with open(input_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["1000", "vm_a", "10", "50", "25"])

        src, ok, msg = convert_cpu_shard(input_path, output_path, overwrite=False)
        assert ok is True
        assert msg == "skipped (exists)"

    def test_reports_error(self, tmp_path):
        from app.src.convert_to_parquet import convert_cpu_shard

        missing = tmp_path / "nonexistent.csv"
        out = tmp_path / "out.parquet"

        src, ok, msg = convert_cpu_shard(missing, out)
        assert ok is False
        assert "ERROR" in msg or "does not exist" in msg or "No such file" in msg or "cannot find" in msg.lower() or "does not exist" in msg.lower()


class TestConvertAll:
    """convert_all orchestration with mocked sub-functions."""

    @patch("app.src.convert_to_parquet.convert_core_table")
    @patch("app.src.convert_to_parquet.convert_cpu_shard")
    def test_converts_core_tables_only(self, mock_cpu, mock_core, tmp_path):
        from app.src.convert_to_parquet import convert_all

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        (input_dir / "vmtable").mkdir(parents=True)
        (input_dir / "vmtable" / "vmtable.csv.gz").write_text("dummy")
        (input_dir / "subscriptions").mkdir(parents=True)
        (input_dir / "subscriptions" / "subscriptions.csv.gz").write_text("dummy")
        (input_dir / "deployments").mkdir(parents=True)
        (input_dir / "deployments" / "deployments.csv.gz").write_text("dummy")

        def _mock_convert(src, dst, schema, overwrite):
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text("fake")
            return True

        mock_core.side_effect = _mock_convert

        convert_all(input_dir, output_dir, overwrite=False, workers=1)

        assert mock_core.call_count == 3
        mock_cpu.assert_not_called()

    @patch("app.src.convert_to_parquet.convert_core_table")
    @patch("app.src.convert_to_parquet.convert_cpu_shard")
    def test_converts_cpu_shards_with_mock(self, mock_cpu, mock_core, tmp_path):
        from app.src.convert_to_parquet import convert_all

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        cpu_dir = input_dir / "vm_cpu_readings"
        cpu_dir.mkdir(parents=True)

        (input_dir / "vmtable").mkdir(parents=True)
        (input_dir / "vmtable" / "vmtable.csv.gz").write_text("dummy")
        (input_dir / "subscriptions").mkdir(parents=True)
        (input_dir / "subscriptions" / "subscriptions.csv.gz").write_text("dummy")
        (input_dir / "deployments").mkdir(parents=True)
        (input_dir / "deployments" / "deployments.csv.gz").write_text("dummy")

        for i in range(3):
            (cpu_dir / f"vm_cpu_readings-file-{i+1}-of-195.csv.gz").write_text("dummy")

        def _mock_convert(src, dst, schema, overwrite):
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text("fake")
            return True

        mock_core.side_effect = _mock_convert
        mock_cpu.return_value = (Path("f"), True, "ok")

        convert_all(input_dir, output_dir, overwrite=False, workers=2)

        assert mock_cpu.call_count == 3

    @patch("app.src.convert_to_parquet.convert_cpu_shard")
    @patch("app.src.convert_to_parquet.convert_core_table")
    def test_skips_if_no_cpu_dir(self, mock_core, mock_cpu, tmp_path):
        from app.src.convert_to_parquet import convert_all

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        (input_dir / "vmtable").mkdir(parents=True)
        (input_dir / "vmtable" / "vmtable.csv.gz").write_text("dummy")
        (input_dir / "subscriptions").mkdir(parents=True)
        (input_dir / "subscriptions" / "subscriptions.csv.gz").write_text("dummy")
        (input_dir / "deployments").mkdir(parents=True)
        (input_dir / "deployments" / "deployments.csv.gz").write_text("dummy")

        def _mock_convert(src, dst, schema, overwrite):
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text("fake")
            return True

        mock_core.side_effect = _mock_convert

        convert_all(input_dir, output_dir, overwrite=False, workers=1)

        assert mock_core.call_count == 3
        mock_cpu.assert_not_called()

    def test_handles_missing_core_inputs(self, tmp_path):
        from app.src.convert_to_parquet import convert_all

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        convert_all(input_dir, output_dir, overwrite=False, workers=1)


class TestMain:
    """CLI entry point via main()."""

    def test_main_runs_with_defaults(self):
        from app.src.convert_to_parquet import main
        from unittest.mock import patch as mock_patch

        with mock_patch("sys.argv", ["convert_to_parquet.py"]):
            with mock_patch("app.src.convert_to_parquet.Path.exists", return_value=True):
                with mock_patch("app.src.convert_to_parquet.convert_all") as mock_convert:
                    main()
                    mock_convert.assert_called_once()

    def test_main_exits_on_missing_input(self):
        from app.src.convert_to_parquet import main
        from unittest.mock import patch as mock_patch

        with mock_patch("sys.argv", ["convert_to_parquet.py"]):
            with mock_patch("app.src.convert_to_parquet.Path.exists", return_value=False):
                with pytest.raises(SystemExit):
                    main()
