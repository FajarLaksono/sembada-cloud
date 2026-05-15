"""Tests for QA report utility."""

import pytest
import pandas as pd
from app.src.qa_report import load_run_log, check_model_compliance, generate_report, print_report


class TestLoadRunLog:
    def test_missing_file_returns_empty(self):
        result = load_run_log("nonexistent_file.csv")
        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestCheckModelCompliance:
    def test_regression_passing(self):
        row = pd.Series({"task": "regression", "mape": 10.0, "r2": 0.85})
        passes, failures = check_model_compliance(row)
        assert passes is True
        assert failures == []

    def test_regression_failing_mape(self):
        row = pd.Series({"task": "regression", "mape": 20.0, "r2": 0.85})
        passes, failures = check_model_compliance(row)
        assert passes is False
        assert any("MAPE" in f for f in failures)

    def test_regression_failing_r2(self):
        row = pd.Series({"task": "regression", "mape": 10.0, "r2": 0.5})
        passes, failures = check_model_compliance(row)
        assert passes is False
        assert any("R²" in f for f in failures)

    def test_classification_passing(self):
        row = pd.Series({"task": "classification", "f1_score": 0.9})
        passes, failures = check_model_compliance(row)
        assert passes is True

    def test_classification_failing_f1(self):
        row = pd.Series({"task": "classification", "f1_score": 0.7})
        passes, failures = check_model_compliance(row)
        assert passes is False
        assert any("F1" in f for f in failures)

    def test_unknown_task_returns_pass(self):
        row = pd.Series({"task": "clustering"})
        passes, failures = check_model_compliance(row)
        assert passes is True

    def test_classif_prefix_also_checked(self):
        row = pd.Series({"task": "classif", "f1_score": 0.7})
        passes, failures = check_model_compliance(row)
        assert passes is False


class TestGenerateReport:
    def test_empty_log(self):
        report = generate_report(pd.DataFrame())
        assert report["total_runs"] == 0
        assert report["passing"] == 0
        assert report["failing"] == 0
        assert report["failures"] == []

    def test_mixed_pass_fail(self):
        df = pd.DataFrame({
            "task": ["regression", "regression", "classification"],
            "mape": [10.0, 25.0, None],
            "r2": [0.85, 0.6, None],
            "f1_score": [None, None, 0.9],
            "run_id": ["r1", "r2", "r3"],
            "model_name": ["M1", "M2", "M3"],
        })
        report = generate_report(df)
        assert report["total_runs"] == 3
        assert report["passing"] == 2
        assert report["failing"] == 1
        assert len(report["failures"]) == 1
        assert report["failures"][0]["run_id"] == "r2"


class TestPrintReport:
    def test_prints_empty_report(self, capsys):
        report = {"total_runs": 0, "passing": 0, "failing": 0,
                  "failures": [], "models_by_task": {}}
        print_report(report)
        captured = capsys.readouterr()
        assert "QUALITY INSURANCE REPORT" in captured.out
        assert "STATUS:" in captured.out

    def test_prints_passing_report(self, capsys):
        report = {"total_runs": 2, "passing": 2, "failing": 0,
                  "failures": [], "models_by_task": {"regression": 2}}
        print_report(report)
        captured = capsys.readouterr()
        assert "ALL MODELS PASS" in captured.out

    def test_prints_failing_report(self, capsys):
        report = {"total_runs": 2, "passing": 1, "failing": 1,
                  "failures": [{"run_id": "r1", "model": "M1", "reasons": ["MAPE too high"]}],
                  "models_by_task": {"regression": 2}}
        print_report(report)
        captured = capsys.readouterr()
        assert "SOME MODELS FAIL" in captured.out
        assert "r1" in captured.out
        assert "M1" in captured.out
        assert "MAPE too high" in captured.out


class TestMain:
    def test_main_no_run_log(self):
        from app.src.qa_report import main
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("app.src.qa_report.load_run_log", lambda path=None: pd.DataFrame())
            main()

    def test_main_with_data(self):
        from app.src.qa_report import main
        mock_df = pd.DataFrame({
            "task": ["regression"], "mape": [10.0], "r2": [0.85],
            "run_id": ["r1"], "model_name": ["M1"],
        })
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("app.src.qa_report.load_run_log", lambda path=None: mock_df)
            main()
