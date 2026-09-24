# Contributing to Executive Communication Coach

Thank you for your interest in contributing to **Executive Communication Coach**! We welcome contributions from the community.

---

## 1. Development Setup

### Prerequisites
- Python 3.10+ (macOS Apple Silicon or POSIX Linux)
- PortAudio development headers (e.g. `brew install portaudio` on macOS or `apt-get install libasound2-dev libportaudio2` on Ubuntu)
- Git

### Initializing the Workspace
```bash
# Fork and clone the repository
git clone https://github.com/ankitsingh99/executive-comm-coach.git
cd executive-comm-coach

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install editable package with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

---

## 2. Branching & Commit Conventions

- Create a feature branch with a descriptive name:
  ```bash
  git checkout -b feat/your-feature-name
  # or
  git checkout -b fix/issue-description
  ```

- We follow **Conventional Commits**:
  - `feat: ...` for new features
  - `fix: ...` for bug fixes
  - `docs: ...` for documentation updates
  - `refactor: ...` for code restructuring
  - `test: ...` for adding or updating tests
  - `ci: ...` for CI/CD workflow updates

---

## 3. Code Quality & Testing

Before submitting a Pull Request, ensure all linters, formatters, and test suites pass:

```bash
# Run formatting
black core/

# Run linting
flake8 core/

# Run unit & integration test suite
pytest -v core/tests/

# Run test coverage check
pytest --cov=core core/tests/
```

---

## 4. Submitting a Pull Request

1. Push your branch to your fork.
2. Open a Pull Request targeting `main`.
3. Fill out the provided Pull Request template completely.
4. Ensure all GitHub Actions CI checks pass.

---

## 5. Community & Conduct

Please review and adhere to our [Code of Conduct](CODE_OF_CONDUCT.md) in all project interactions.
