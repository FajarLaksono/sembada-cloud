# Project: Sembada Cloud - Cloud Resource and cost prediction

## What this project does

This is a business analytics project to analyze and predict cloud resources and costs with machine learning and deep learning. complied to CRISP-ML(Q) and CAMS DevOps. 

## Datasets:

- Azure Public Dataset V2 (2019 VM traces): https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV2.md 

## Tech stack

- Python 3.13 

## Project Structure

- `notebooks/` - Analytics workplace with Jupyter.
- `app/src/` - app code
- `app/tests/` - Test files
- `docs/` - Documentation
- `deployment/` - Deployment files

## Code Standards

- Use Python 3.13.x features
- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Write docstrings for all public functions and classes
- Keep functions small and focused

## Conventions

- Module names: `snake_case`
- Class names: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private members: prefix with underscore `_private_method`

## Git Commit Messages

- Use conventional commits format: `type: description`
- Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`
- Keep commit messages under 72 characters

## When Asked to Make Changes

1. Understand the existing code first
2. Make minimal, focused changes
3. Ensure tests pass
4. Explain what you changed and why

## Development Commands

### Testing
- Run all unit tests: `pytest app/tests/ -v`
- Run specific test file: `pytest app/tests/test_features.py -v`
- Run with coverage: `pytest app/tests/ --cov=app.src`

### Linting & Formatting
- Check code style: `black --check app/src/ app/tests/ && flake8 app/src/`
- Format code: `black app/src/ app/tests/`

### Notebook Execution (quality gates active, real-time output)
- Execute 03a: `papermill notebooks/03a_feature_engineering.ipynb NUL --log-output --progress-bar --execution-timeout 600`
- Execute 03b: `papermill notebooks/03b_tabular_models.ipynb NUL --log-output --progress-bar --execution-timeout 600`
- Execute 03c: `papermill notebooks/03c_timeseries_forecasting.ipynb NUL --log-output --progress-bar --execution-timeout 600`

### Quality Insurance
- Generate QA compliance report: `python -m app.src.qa_report`

### Environment
- Freeze exact dependencies: `pip freeze --exclude-editable > requirements.lock`

### CI/CD (local)
- Run full CI locally: `pytest app/tests/ -v && papermill notebooks/03a_feature_engineering.ipynb NUL --log-output --progress-bar --execution-timeout 600`
- **On GitHub CI:** `ci.yml` runs `test` automatically on push/PR (synthetic fixtures). Notebooks run locally (100GB CPU readings cannot fit in CI).