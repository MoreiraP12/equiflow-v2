# Contributing to equiflow

Thank you for your interest in contributing to equiflow! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/MoreiraP12/equiflow-v2.git
   cd equiflow-v2
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install system dependencies**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install graphviz
   
   # macOS
   brew install graphviz
   
   # Windows
   choco install graphviz
   ```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run only unit tests
pytest tests/test_unit.py -v

# Run only integration tests
pytest tests/test_eicu_integration.py -v

# Run with coverage
pytest tests/ --cov=equiflow --cov-report=html
```

## Code Style

We use [ruff](https://github.com/astral-sh/ruff) for linting. Please ensure your code passes linting:

```bash
ruff check equiflow/
```

### Style Guidelines

- **Line length**: 119 characters maximum
- **Docstrings**: NumPy style
- **Type hints**: Use type hints for all public functions
- **PEP 8**: Follow PEP 8 conventions

### Example Docstring

```python
def calculate_smd(
    group1: pd.Series,
    group2: pd.Series,
    variable_type: str = "continuous"
) -> float:
    """
    Calculate the Standardized Mean Difference between two groups.
    
    Parameters
    ----------
    group1 : pd.Series
        Values for the first group.
    group2 : pd.Series
        Values for the second group.
    variable_type : str, default="continuous"
        Type of variable: "continuous" or "categorical".
        
    Returns
    -------
    float
        The standardized mean difference.
        
    Examples
    --------
    >>> group1 = pd.Series([1, 2, 3, 4, 5])
    >>> group2 = pd.Series([2, 3, 4, 5, 6])
    >>> calculate_smd(group1, group2)
    0.632...
    """
```

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write tests for new functionality
   - Update documentation as needed
   - Follow the code style guidelines

3. **Run tests and linting**
   ```bash
   pytest tests/ -v
   ruff check equiflow/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

5. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a pull request on GitHub.

## Commit Message Convention

Use the following prefixes for commit messages:

- `Add:` New feature or functionality
- `Fix:` Bug fix
- `Update:` Update existing functionality
- `Remove:` Remove feature or file
- `Refactor:` Code refactoring without functional changes
- `Docs:` Documentation only changes
- `Test:` Test only changes
- `Style:` Code style changes (formatting, etc.)

## Reporting Issues

When reporting issues, please include:

1. **Description**: Clear description of the issue
2. **Steps to reproduce**: Minimal code example
3. **Expected behavior**: What you expected to happen
4. **Actual behavior**: What actually happened
5. **Environment**: Python version, OS, package version

## Feature Requests

We welcome feature requests! Please:

1. Check existing issues to avoid duplicates
2. Describe the use case clearly
3. Explain why this would be valuable to other users

## Questions?

Feel free to open an issue with the "question" label or reach out to the maintainers.

Thank you for contributing! 🎉
