# Contributing

Thank you for your interest in contributing!

## Getting Started

```bash
git clone https://github.com/SubhadipJana1409/day19-biomarker-shap
cd day19-biomarker-shap
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
pre-commit install
```

## Development Workflow

1. **Fork** the repo and create a feature branch: `git checkout -b feat/my-feature`
2. **Write tests** for any new functionality in `tests/`
3. **Run tests** locally: `pytest tests/ -v`
4. **Lint**: `ruff check src/ tests/` and `black src/ tests/`
5. **Commit** with a descriptive message: `git commit -m "feat: add new plot type"`
6. **Push** and open a Pull Request

## Code Style

- Black formatting (`line-length = 100`)
- Ruff for linting
- Type hints on all public functions
- Docstrings in NumPy style

## Reporting Bugs

Open an issue with:
- Python version
- OS
- Full error traceback
- Minimal reproducible example
