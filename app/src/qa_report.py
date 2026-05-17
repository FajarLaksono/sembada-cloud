"""
Quality Assurance report generator.
Reads run_log.csv and prints compliance summary against success criteria.
"""

import pandas as pd
from pathlib import Path


SUCCESS_CRITERIA = {
    'regression': {'mape': 15.0, 'r2': 0.70},
    'classification': {'f1': 0.85},
    'clustering': {'silhouette_score': 0.3},
    'timeseries': {'mae': 5.0},
}


def load_run_log(path: str | Path = "models/run_log.csv") -> pd.DataFrame:
    """Load and validate the run log CSV."""
    log_path = Path(path)
    if not log_path.exists():
        print(f"Run log not found at {path}")
        return pd.DataFrame()
    return pd.read_csv(log_path)


def check_model_compliance(row: pd.Series) -> tuple[bool, list[str]]:
    """
    Check a single model run against success criteria.
    Returns (passes, list of failure reasons).
    """
    failures = []
    task = row.get('task', '')

    if 'regression' in task:
        if 'avg_cpu' in task:
            if row.get('r2', 0) < SUCCESS_CRITERIA['regression']['r2']:
                failures.append(f"R² {row['r2']:.3f} < {SUCCESS_CRITERIA['regression']['r2']}")
        else:
            if row.get('mape', 100) > SUCCESS_CRITERIA['regression']['mape']:
                failures.append(f"MAPE {row['mape']:.1f}% > {SUCCESS_CRITERIA['regression']['mape']}%")
            if row.get('r2', 0) < SUCCESS_CRITERIA['regression']['r2']:
                failures.append(f"R² {row['r2']:.3f} < {SUCCESS_CRITERIA['regression']['r2']}")

    elif 'classification' in task or 'classif' in task:
        if row.get('f1_score', 0) < SUCCESS_CRITERIA['classification']['f1']:
            failures.append(f"F1 {row['f1_score']:.3f} < {SUCCESS_CRITERIA['classification']['f1']}")

    return len(failures) == 0, failures


def generate_report(log_df: pd.DataFrame) -> dict:
    """Generate QA compliance report from run log."""
    report = {
        'total_runs': len(log_df),
        'passing': 0,
        'failing': 0,
        'failures': [],
        'models_by_task': {},
    }

    for _, row in log_df.iterrows():
        passes, failures = check_model_compliance(row)
        if passes:
            report['passing'] += 1
        else:
            report['failing'] += 1
            report['failures'].append({
                'run_id': row.get('run_id', '?'),
                'model': row.get('model_name', '?'),
                'reasons': failures,
            })

        task = row.get('task', 'unknown')
        if task not in report['models_by_task']:
            report['models_by_task'][task] = 0
        report['models_by_task'][task] += 1

    return report


def print_report(report: dict) -> None:
    """Print formatted QA report to stdout."""
    print("=" * 60)
    print("  QUALITY ASSURANCE REPORT — run_log.csv")
    print("=" * 60)
    print(f"\n  Total model runs: {report['total_runs']}")
    print(f"  Passing criteria: {report['passing']}")
    print(f"  Failing criteria: {report['failing']}")

    if report['total_runs'] > 0:
        pass_rate = report['passing'] / report['total_runs'] * 100
        print(f"  Pass rate:       {pass_rate:.0f}%")

    print(f"\n  Models by task:")
    for task, count in sorted(report['models_by_task'].items()):
        print(f"    {task}: {count}")

    if report['failures']:
        print(f"\n  Failures ({len(report['failures'])}):")
        for f in report['failures']:
            print(f"    [{f['run_id']}] {f['model']}:")
            for reason in f['reasons']:
                print(f"      - {reason}")

    print("=" * 60)

    if report['failing'] > 0:
        print("  STATUS: SOME MODELS FAIL QUALITY CHECKS")
    else:
        print("  STATUS: ALL MODELS PASS QUALITY CHECKS")

    print("=" * 60)


def main():
    log_df = load_run_log()
    if log_df.empty:
        print("No run_log.csv found. Run 03b first to generate model results.")
        return
    report = generate_report(log_df)
    print_report(report)


if __name__ == "__main__":
    main()
