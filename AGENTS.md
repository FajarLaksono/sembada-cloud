# Project: Sembada Cloud - Cloud Resource and cost prediction

## What this project does

This is a business analytics project to analyze and predict cloud resources and costs with machine learning and deep learning. complied to CRISP-ML(Q) and CAMS DevOps. 

## Datasets:

- Azure Public Dataset V2 (2019 VM traces): https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV2.md 

## Tech stack

- Python 3.14 

## Project Structure

- `notebooks/` - Analytics workplace with Jupyter.
- `app/src/` - app code
- `app/tests/` - Test files
- `docs/` - Documentation
- `deployment/` - Deployment files

## Code Standards

- Use Python 3.14.x features
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
- Check PEP 8 compliance: `flake8 app/src/ --max-line-length=120`
- Format code: `black app/src/ app/tests/`

### Notebook Execution
- Execute notebook end-to-end: `jupyter nbconvert --to notebook --execute notebooks/03_predictive_analysis.ipynb --output /dev/null --ExecutePreprocessor.timeout=600`

### CI/CD
- Run full CI locally: `pytest app/tests/ -v && jupyter nbconvert --to notebook --execute notebooks/03_predictive_analysis.ipynb`