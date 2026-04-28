# Contributing to Sentiment Analysis Dashboard

Thank you for your interest in contributing! This project welcomes contributions from the community.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report:

- Use a clear and descriptive title
- Describe the issue in detail
- Include steps to reproduce
- Include screenshots if applicable

### Suggesting Features

Feature suggestions are welcome! Please:

- Explain the use case
- Describe the desired functionality
- Include examples if applicable

### Pull Requests

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd sentiment-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

## Code Style

We follow [PEP 8](https://pep8.org/) style guidelines:

- 4 spaces for indentation
- Maximum line length: 79 characters
- Type hints for all functions
- Docstrings for all public functions

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=api --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

## Code of Conduct

Please note that this project is released with a [Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
