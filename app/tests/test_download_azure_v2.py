"""Tests for Azure dataset downloader (app.src.download_azure_v2)."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, create_autospec


class TestModuleConstants:
    """Module-level constants are correctly defined."""

    def test_modes_keys(self):
        from app.src.download_azure_v2 import MODES
        assert set(MODES.keys()) == {"core", "sample", "full"}

    def test_core_includes_core_files_and_supplementary(self):
        from app.src.download_azure_v2 import MODES, CORE_FILES, SUPPLEMENTARY_FILES
        assert len(MODES["core"]) == len(CORE_FILES) + len(SUPPLEMENTARY_FILES)

    def test_sample_mode_size(self):
        from app.src.download_azure_v2 import MODES, CORE_FILES, SUPPLEMENTARY_FILES
        assert len(MODES["sample"]) == len(CORE_FILES) + len(SUPPLEMENTARY_FILES) + 3

    def test_full_mode_has_all_cpu_files(self):
        from app.src.download_azure_v2 import MODES, CPU_FILES
        assert len(MODES["full"]) >= len(CPU_FILES)
        assert len(CPU_FILES) == 195

    def test_mode_sizes_keys(self):
        from app.src.download_azure_v2 import MODE_SIZES
        assert set(MODE_SIZES.keys()) == {"core", "sample", "full"}

    def test_cpu_files_pattern(self):
        from app.src.download_azure_v2 import CPU_FILES
        url, rel_path = CPU_FILES[0]
        assert "file-1-of-195" in url
        assert url.startswith("https://")
        assert rel_path == "vm_cpu_readings/vm_cpu_readings-file-1-of-195.csv.gz"


class TestSizeofFmt:
    """sizeof_fmt formats byte sizes correctly."""

    def test_bytes(self):
        from app.src.download_azure_v2 import sizeof_fmt
        result = sizeof_fmt(512)
        assert "512.0" in result
        assert "B" in result

    def test_kilobytes(self):
        from app.src.download_azure_v2 import sizeof_fmt
        result = sizeof_fmt(1024 * 2)
        assert "2.0" in result
        assert "KB" in result

    def test_megabytes(self):
        from app.src.download_azure_v2 import sizeof_fmt
        result = sizeof_fmt(1024 * 1024 * 5)
        assert "5.0" in result
        assert "MB" in result

    def test_gigabytes(self):
        from app.src.download_azure_v2 import sizeof_fmt
        result = sizeof_fmt(1024 ** 3 * 3)
        assert "3.0" in result
        assert "GB" in result

    def test_terabytes(self):
        from app.src.download_azure_v2 import sizeof_fmt
        result = sizeof_fmt(1024 ** 4 * 2)
        assert "2.0" in result
        assert "TB" in result

    def test_zero(self):
        from app.src.download_azure_v2 import sizeof_fmt
        result = sizeof_fmt(0)
        assert "0.0" in result
        assert "B" in result


class TestCreateSession:
    """create_session returns a configured requests Session."""

    def test_returns_session(self):
        from app.src.download_azure_v2 import create_session
        import requests

        session = create_session()
        assert isinstance(session, requests.Session)
        assert "User-Agent" in session.headers
        session.close()


class TestDownloadFile:
    """download_file with mocked requests session."""

    def test_download_with_mocked_session(self, tmp_path):
        from app.src.download_azure_v2 import download_file

        dest = tmp_path / "test.csv.gz"
        mock_session = MagicMock()

        mock_head_resp = MagicMock()
        mock_head_resp.status_code = 200
        mock_head_resp.headers = {"Content-Length": "100"}
        mock_head_resp.raise_for_status = MagicMock()

        content = iter([b"x" * 100])
        mock_get_resp = MagicMock()
        mock_get_resp.__enter__.return_value = mock_get_resp
        mock_get_resp.__exit__.return_value = None
        mock_get_resp.status_code = 200
        mock_get_resp.headers = {"Content-Length": "100"}
        mock_get_resp.raise_for_status = MagicMock()
        mock_get_resp.iter_content.return_value = content

        mock_session.head.return_value = mock_head_resp
        mock_session.get.return_value = mock_get_resp

        result = download_file(mock_session, "https://ex.com/f", dest, show_progress=False)

        assert result is True
        assert dest.read_bytes() == b"x" * 100

    def test_skips_already_complete(self, tmp_path):
        from app.src.download_azure_v2 import download_file

        dest = tmp_path / "existing.csv.gz"
        dest.write_bytes(b"x" * 100)

        mock_session = MagicMock()
        mock_head_resp = MagicMock()
        mock_head_resp.status_code = 200
        mock_head_resp.headers = {"Content-Length": "100"}
        mock_head_resp.raise_for_status = MagicMock()
        mock_session.head.return_value = mock_head_resp

        result = download_file(mock_session, "https://ex.com/f", dest, show_progress=False)
        assert result is True
        assert mock_session.get.called is False

    def test_resumes_partial_download(self, tmp_path):
        from app.src.download_azure_v2 import download_file

        dest = tmp_path / "partial.csv.gz"
        dest.write_bytes(b"x" * 50)

        mock_session = MagicMock()
        mock_head_resp = MagicMock()
        mock_head_resp.status_code = 200
        mock_head_resp.headers = {"Content-Length": "100"}
        mock_head_resp.raise_for_status = MagicMock()

        content = iter([b"y" * 50])
        mock_get_resp = MagicMock()
        mock_get_resp.__enter__.return_value = mock_get_resp
        mock_get_resp.__exit__.return_value = None
        mock_get_resp.status_code = 206
        mock_get_resp.headers = {"Content-Length": "50"}
        mock_get_resp.raise_for_status = MagicMock()
        mock_get_resp.iter_content.return_value = content

        mock_session.head.return_value = mock_head_resp
        mock_session.get.return_value = mock_get_resp

        result = download_file(mock_session, "https://ex.com/f", dest, show_progress=False)

        assert result is True
        mock_session.get.assert_called_once()
        call_kwargs = mock_session.get.call_args[1]
        assert "Range" in call_kwargs.get("headers", {})

    def test_retry_on_failure(self, tmp_path):
        from app.src.download_azure_v2 import download_file
        import requests

        dest = tmp_path / "fail.csv.gz"
        mock_session = MagicMock()
        mock_session.head.side_effect = requests.RequestException("Connection error")

        with patch("app.src.download_azure_v2.time.sleep"):
            result = download_file(mock_session, "https://ex.com/f", dest, show_progress=False)
            assert result is False
            assert mock_session.head.call_count >= 2

    def test_size_mismatch_triggers_retry(self, tmp_path):
        from app.src.download_azure_v2 import download_file

        dest = tmp_path / "mismatch.csv.gz"

        mock_session = MagicMock()
        mock_head_resp = MagicMock()
        mock_head_resp.status_code = 200
        mock_head_resp.headers = {"Content-Length": "200"}
        mock_head_resp.raise_for_status = MagicMock()

        content = iter([b"x" * 100])
        mock_get_resp = MagicMock()
        mock_get_resp.__enter__.return_value = mock_get_resp
        mock_get_resp.__exit__.return_value = None
        mock_get_resp.status_code = 200
        mock_get_resp.headers = {"Content-Length": "200"}
        mock_get_resp.raise_for_status = MagicMock()
        mock_get_resp.iter_content.return_value = content

        mock_session.head.return_value = mock_head_resp
        mock_session.get.return_value = mock_get_resp

        with patch("app.src.download_azure_v2.time.sleep"):
            result = download_file(mock_session, "https://ex.com/f", dest, show_progress=False)
            assert result is False


class TestDownloadAll:
    """download_all orchestration with mocked download_file."""

    @patch("app.src.download_azure_v2.download_file")
    @patch("app.src.download_azure_v2.create_session")
    def test_downloads_sequentially(self, mock_create_session, mock_dl_file, tmp_path):
        from app.src.download_azure_v2 import download_all

        mock_dl_file.return_value = True
        mock_session = MagicMock()
        mock_create_session.return_value = mock_session

        file_list = [
            ("https://ex.com/a.csv.gz", "a.csv.gz"),
            ("https://ex.com/b.csv.gz", "b.csv.gz"),
        ]

        download_all(file_list, tmp_path, workers=1)

        assert mock_dl_file.call_count == 2

    @patch("app.src.download_azure_v2.download_file")
    @patch("app.src.download_azure_v2.create_session")
    def test_downloads_in_parallel(self, mock_create_session, mock_dl_file, tmp_path):
        from app.src.download_azure_v2 import download_all

        mock_dl_file.return_value = True
        mock_session = MagicMock()
        mock_create_session.return_value = mock_session

        file_list = [
            ("https://ex.com/a.csv.gz", "a.csv.gz"),
            ("https://ex.com/b.csv.gz", "b.csv.gz"),
            ("https://ex.com/c.csv.gz", "c.csv.gz"),
        ]

        download_all(file_list, tmp_path, workers=4)

        assert mock_dl_file.call_count == 3


class TestMain:
    """CLI entry point via main()."""

    def test_main_with_defaults(self):
        from app.src.download_azure_v2 import main
        from unittest.mock import patch as mock_patch

        with mock_patch("sys.argv", ["download_azure_v2.py"]):
            with mock_patch("app.src.download_azure_v2.download_all") as mock_download:
                main()
                mock_download.assert_called_once()

    def test_main_with_custom_mode(self):
        from app.src.download_azure_v2 import main
        from unittest.mock import patch as mock_patch

        with mock_patch("sys.argv", ["download_azure_v2.py", "--mode", "sample"]):
            with mock_patch("app.src.download_azure_v2.download_all") as mock_download:
                main()
                mock_download.assert_called_once()

    def test_main_with_cpu_files_override(self):
        from app.src.download_azure_v2 import main
        from unittest.mock import patch as mock_patch

        with mock_patch("sys.argv", ["download_azure_v2.py", "--cpu-files", "1-10"]):
            with mock_patch("app.src.download_azure_v2.download_all") as mock_download:
                main()
                mock_download.assert_called_once()

    def test_main_with_workers(self):
        from app.src.download_azure_v2 import main
        from unittest.mock import patch as mock_patch

        with mock_patch("sys.argv", ["download_azure_v2.py", "--workers", "4"]):
            with mock_patch("app.src.download_azure_v2.download_all") as mock_download:
                main()
                mock_download.assert_called_once()

    def test_main_invalid_cpu_range(self):
        from app.src.download_azure_v2 import main
        from unittest.mock import patch as mock_patch

        with mock_patch("sys.argv", ["download_azure_v2.py", "--cpu-files", "invalid"]):
            with pytest.raises(SystemExit):
                main()
