# Implementation Plan: CI Notebook Execution with Real-Time Output & Artifact Reports

## 1. Workflow Files

Two separate workflow files — no ghost checks on PRs.

### `.github/workflows/ci.yml` (test — auto on push/PR)

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.13
        uses: actions/setup-python@v5
        with:
          python-version: "3.13"
          cache: "pip"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          if [ -f requirements.lock ]; then
            pip install -r requirements.lock
          else
            pip install -r requirements.txt
          fi
          pip install pytest coverage

      - name: Run unit tests with coverage
        run: |
          pytest app/tests/ -v --tb=short -x --cov=app.src --cov-report=term
        env:
          PYTHONPATH: ${{ github.workspace }}
```

### `.github/workflows/notebooks.yml` (notebooks — manual trigger)

```yaml
name: Notebooks

on:
  workflow_dispatch:

jobs:
  notebooks:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.13
        uses: actions/setup-python@v5
        with:
          python-version: "3.13"
          cache: "pip"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          if [ -f requirements.lock ]; then
            pip install -r requirements.lock
          else
            pip install -r requirements.txt
          fi
          pip install papermill jupyter nbformat

      - name: Execute notebooks (quality gates active)
        run: |
          papermill notebooks/03a_feature_engineering.ipynb notebooks/03a_output.ipynb \
            --log-output --progress-bar --execution-timeout 600
          papermill notebooks/03b_tabular_models.ipynb notebooks/03b_output.ipynb \
            --log-output --progress-bar --execution-timeout 600
          papermill notebooks/03c_timeseries_forecasting.ipynb notebooks/03c_output.ipynb \
            --log-output --progress-bar --execution-timeout 600
        env:
          PYTHONPATH: ${{ github.workspace }}

      - name: Convert to HTML reports
        run: |
          jupyter nbconvert --to html notebooks/03a_output.ipynb \
            --output 03a_report.html --TemplateExporter.exclude_input=True
          jupyter nbconvert --to html notebooks/03b_output.ipynb \
            --output 03b_report.html --TemplateExporter.exclude_input=True
          jupyter nbconvert --to html notebooks/03c_output.ipynb \
            --output 03c_report.html --TemplateExporter.exclude_input=True

      - name: Upload reports as artifacts
        uses: actions/upload-artifact@v4
        with:
          name: predictive-analysis-report
          path: notebooks/*_report.html
```

## 2. Files Updated

| File | Change |
|------|--------|
| `.github/workflows/ci.yml` | Rewritten — test only, auto on push/PR, no notebooks job |
| `.github/workflows/notebooks.yml` | **New** — notebooks + HTML + artifact upload, manual only |
| `requirements.txt` | Added `papermill>=2.6.0` |
| `AGENTS.md` | 3 papermill execute commands + CI/CD section |
| `docs/RUNNING.md` | CI/CD section split into two workflows |
| `docs/AZURE_PREDICTIVE_ANALYSIS_PLAN.md` | §7 YAMLs, architecture diagram, automation table |
| `docs/AUTOMATED_QA_TECH_SPEC.md` | §10 YAMLs, pipeline behavior table, CI checklist, AGENTS.md section |
| `docs/IMPLEMENTATION_PROGRESS.md` | CI checklist item |
| `.opencode/plans/ci-migration-plan.md` | This file |

## 3. How to Trigger Notebooks

1. GitHub → **Actions** tab → **Notebooks** workflow
2. Click **Run workflow** → select branch → **Run**
3. During execution: real-time cell output visible in the step log
4. When done: download `predictive-analysis-report.zip` from **Artifacts**
5. Open HTML files in browser — all graphs rendered inline

## 4. Expected Behavior

| Event | ci.yml | notebooks.yml |
|-------|--------|---------------|
| Push to main/develop | ✅ pytest runs | ❌ not triggered |
| PR to main | ✅ pytest runs | ❌ not triggered |
| Manual trigger (Actions → Notebooks) | ❌ not triggered | ✅ full notebook execution → HTML → artifacts |