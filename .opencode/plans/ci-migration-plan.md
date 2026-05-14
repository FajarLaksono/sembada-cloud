# Implementation Plan: CI Notebook Execution with Real-Time Output & Artifact Reports

## 1. CI Workflow (`.github/workflows/ci.yml`)

Replace single-job with two-job sequential workflow:

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python 3.14
        uses: actions/setup-python@v5
        with:
          python-version: "3.14"
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

  notebooks:
    if: github.event_name == 'workflow_dispatch'
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python 3.14
        uses: actions/setup-python@v5
        with:
          python-version: "3.14"
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
          papermill notebooks/03a_feature_engineering.ipynb notebooks/03a_output.ipynb --log-output --progress-bar --execution-timeout 600
          papermill notebooks/03b_tabular_models.ipynb notebooks/03b_output.ipynb --log-output --progress-bar --execution-timeout 600
          papermill notebooks/03c_timeseries_forecasting.ipynb notebooks/03c_output.ipynb --log-output --progress-bar --execution-timeout 600
        env:
          PYTHONPATH: ${{ github.workspace }}
      - name: Convert to HTML reports
        run: |
          jupyter nbconvert --to html notebooks/03a_output.ipynb --output 03a_report.html --TemplateExporter.exclude_input=True
          jupyter nbconvert --to html notebooks/03b_output.ipynb --output 03b_report.html --TemplateExporter.exclude_input=True
          jupyter nbconvert --to html notebooks/03c_output.ipynb --output 03c_report.html --TemplateExporter.exclude_input=True
      - name: Upload reports as artifacts
        uses: actions/upload-artifact@v4
        with:
          name: ml-reports
          path: notebooks/*_report.html
```

## 2. Files to Update

### `.github/workflows/ci.yml`
- Rewrite entirely with the two-job YAML above
- Remove `Test-Path` (PowerShell) → use `[ -f ... ]` (bash syntax for Ubuntu runner)
- `test` job: no papermill, no nbconvert, no jupyter — only pytest + coverage
- `notebooks` job: only runs on manual trigger, after tests pass
- Execute → HTML convert → artifact upload

### `docs/AZURE_PREDICTIVE_ANALYSIS_PLAN.md`
- **§7 (lines 1396–1445)**: Replace entire YAML block with new two-job version
- **Line 91**: `ci.yml ← pytest + nbconvert --execute on every PR` → `pytest on every push/PR, notebooks manual trigger`
- **Line 172**: `pytest + nbconvert --execute` → `pytest auto (push/PR), notebooks manual`
- **Line 1426**: `pip install pytest nbconvert jupyter` → `pip install pytest coverage`
- **Lines 1434–1444**: Replace nbconvert execute block with execute + HTML + upload

### `docs/AUTOMATED_QA_TECH_SPEC.md`
- **§10.3 (lines 498–543)**: Replace YAML block with two-job version
- **§10.4 (lines 546–553)**: Update Pipeline Behavior table:

  | Stage | Trigger | Purpose | On Failure |
  |-------|---------|---------|------------|
  | `pytest --cov` | push/PR/manual | Unit tests + coverage report | PR is blocked |
  | `papermill` | manual only (after tests pass) | End-to-end quality gates | `AssertionError` from failed gate |

- **Lines 920–922**: Update verification commands to show artifact flow
- **Line 954**: `Pipeline runs nbconvert --execute` → `Pipeline runs papermill (manual trigger, artifacts uploaded)`
- **Lines 1005–1016**: Update AGENTS.md CI section to match new YAML
- **Line 1043**: Update traceability matrix reference

### `docs/RUNNING.md`
- **Lines 113–125**: Update CI/CD Pipeline section:
  - Stage 3: unit tests (auto on push/PR)
  - Stage 4: notebooks (manual trigger via `workflow_dispatch`)
  - Add note about artifact download for HTML reports
  - Update the "Run full CI locally" command to match

### `AGENTS.md`
- **Line 73–74**: CI/CD section — keep local command for dev, add note about CI having two jobs

### `docs/IMPLEMENTATION_PROGRESS.md`
- **Line 118**: Update CI step description
- **Lines 517–522**: Update CI checklist items

## 3. How to Trigger Notebooks in GitHub UI

1. Go to repository on GitHub
2. Click **Actions** tab
3. Select **CI** workflow
4. Click **Run workflow** dropdown
5. Select branch, click green **Run workflow** button
6. `test` job runs first (~1-2 min)
7. If tests pass → `notebooks` job starts automatically
8. During execution: click the job to see real-time `--log-output` in the step logs
9. When done: download `ml-reports.zip` from **Artifacts** section
10. Open HTML files in any browser — all graphs, tables, and outputs rendered

## 4. Expected Behavior

| Scenario | Outcome |
|----------|---------|
| Push to main/develop | Tests run (~1-2 min). Notebooks do NOT run. |
| Open PR to main | Tests run. Notebooks do NOT run. |
| Manual trigger via UI | Tests run first. If pass → all 3 notebooks execute sequentially. Artifacts uploaded (HTML reports). |
| Quality gate fails | `papermill` exits non-zero → CI shows red ❌. Notebook output up to failure point is still included in HTML artifacts for debugging. |
| `--log-output` | Real-time stdout from each cell visible in Actions step log (training loss, metrics, assertions, print statements). |

## 5. Rationale

- **Save CI minutes**: ~30 min per full run only when manually triggered, never wasted on pushes
- **Real-time visibility**: `--log-output` streams ML training progress to the Actions log as it happens
- **Stakeholder reports**: HTML with embedded graphs, no Jupyter installation needed to view
- **Sequential safety**: Notebooks only run after tests pass, preventing wasted compute on broken code
- **Zero infra**: Artifacts stored by GitHub for 90 days, no S3/Pages setup needed
