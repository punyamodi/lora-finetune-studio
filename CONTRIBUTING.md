# Contributing

## Development Setup

```bash
git clone https://github.com/punyamodi/lora-finetune-studio.git
cd lora-finetune-studio
pip install -e "."
pip install pytest ruff
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

This project follows PEP 8. Run `ruff check .` to lint and `ruff format .` to format.

## Pull Request Process

1. Fork the repository
2. Create a feature branch from `main`
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit a pull request with a clear description

## Reporting Issues

Open an issue on GitHub with a clear description and reproduction steps.
