# Project: Sembada Cloud - Cloud Resource and cost prediction

## What this project does

This is a business analytics project to analyze and predict cloud resources and costs with machine learning and deep learning. complied to CRISP-ML(Q) and CAMS DevOps. 

## Datasets:

- Google Borg cluster traces datasets: https://github.com/google/cluster-data 

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