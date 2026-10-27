# Contributing to Throttle

Thank you for considering contributing to Throttle! This document provides basic guidance for contributors.

## Development Setup

### Prerequisites
- Python 3.11 or higher (tested on 3.11, 3.12, 3.13, 3.14)
- pip

### Quick Start

```bash
# Clone the repository
git clone https://github.com/throttle-pro/throttle.git
cd throttle

# Create and activate a virtual environment
python3 -m venv .venv
. .venv/bin/activate

# Install in development mode
pip install -e .

# Install optional embeddings support (for semantic cache tier)
pip install -e .[embeddings]

# Verify installation
throttle --version
```

### Using Locked Dependencies (CI-grade)

For reproducible builds matching CI:

```bash
pip install --require-hashes \
  -r ci/requirements-build.lock \
  -r ci/requirements-runtime.lock

pip install --no-build-isolation --no-deps --editable .
pip check
```

## Running Tests

The test suite is offline-only and blocks non-loopback DNS/socket use to ensure tests don't make external network calls.

### Run All Tests

```bash
# Using unittest (simple, no extra dependencies)
PYTHONPATH=src python -m unittest discover -s tests -v

# Using pytest (matches CI behavior, requires pytest)
python -m pytest tests/ -v

# Using the CI test runner
python ci/run_tests.py --start-directory tests
```

### Run Specific Tests

```bash
# Run a single test file
python -m pytest tests/test_check.py -v

# Run a specific test
python -m pytest tests/test_cli.py::ParserAndPlanTests::test_parser_subcommands_match_expected -v

# Stop on first failure
python -m pytest tests/ -x
```

### Integration Tests

Some tests require a live backend (Ollama):

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve &

# Pull required models
ollama pull llama3.2:1b

# Run integration tests
python -m pytest tests/test_proxy_integration.py tests/test_proxy.py -v
```

All tests should pass before submitting a pull request. The suite covers:
- All CLI modes (plan, smoke, benchmark, golden, compare, check, autotune, gate, proxy)
- URL/proxy safety validation
- Response validation and streaming termination
- Cost separation and billing models
- Confidence interval and boundary logic
- Manifest tampering detection
- Saved run comparisons
- Six-run golden protocol validation
- Proxy cache behavior, scope isolation, and in-flight deduplication (unit tests)
- Proxy integration tests against live Ollama backend (skipped if Ollama unavailable)

## Building

```bash
# Build wheel and source distribution
pip install build
python -m build

# Output will be in dist/
```

## Code Quality

### Syntax Check

```bash
# Compile all Python files to check for syntax errors
python -m compileall -q -f src tests ci
```

### Code Style

Throttle prioritizes:
- **Correctness over cleverness**: Clear, auditable code
- **Explicit over implicit**: No hidden behavior or silent fallbacks
- **Safety by default**: Fail closed on invalid inputs
- **Evidence-based claims**: All performance numbers must be validated and traceable

There is no automated linter configuration (no ruff, black, mypy, etc.). Follow the existing code style in the file you're modifying.

## Pull Request Process

1. **Fork the repository** and create a feature branch from `main`
2. **Make your changes** with clear, focused commits
3. **Run the test suite** and ensure all tests pass:
   ```bash
   python -m pytest tests/ -v
   # or
   python ci/run_tests.py --start-directory tests
   ```
4. **Check syntax**:
   ```bash
   python -m compileall -q -f src tests ci
   ```
5. **Update documentation** if you're adding features or changing behavior
6. **Submit a pull request** with a clear description of:
   - What problem you're solving
   - How your changes address it
   - Any new test coverage you've added

## What to Contribute

Throttle welcomes contributions in several areas:

- **Bug fixes**: Incorrect behavior, edge cases, or safety issues
- **Test coverage**: Additional test cases for existing functionality
- **Documentation**: Clarifications, examples, or corrections
- **Performance improvements**: With validated benchmark evidence
- **New backends**: Support for additional inference engines
- **Protocol improvements**: Enhancements to the golden protocol or validation

## What NOT to Contribute (Without Discussion)

Please open an issue first before working on:

- Major architectural changes
- New CLI modes or subcommands
- Changes to the golden protocol validation logic
- Automatic configuration or "magic" behavior
- Breaking changes to saved report schema

## Reporting Issues

When reporting bugs, please include:

- Throttle version (`throttle --version`)
- Python version
- Operating system
- Minimal reproduction steps
- Expected vs actual behavior
- Sanitized command output (remove credentials/URLs)

## Project Structure

```
throttle/
├── src/throttle/          # Main package source
│   ├── cli.py            # CLI entry point and command definitions
│   ├── check.py          # throttle check: $/M tokens with CIs and verdicts
│   ├── benchmark.py      # plan / smoke / benchmark measurement engine
│   ├── compare.py        # Saved-report comparison and statistical analysis
│   ├── autotune.py       # throttle autotune (local, free)
│   ├── proxy.py          # Caching proxy server
│   ├── own/              # Own-your-intelligence loop commands
│   └── ...
├── tests/                # Test suite (unittest-based, run with pytest)
├── ci/                   # CI scripts and locked dependencies
├── data/                 # Reference data (negation pairs, etc.)
└── pyproject.toml       # Package metadata and configuration
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
