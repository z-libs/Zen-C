# Contributing to Zen C

First off, thank you for considering contributing to Zen C! It's people like you that make this project great.

We welcome all contributions, whether it's fixing bugs, adding documentation, proposing new features, or just reporting issues.

## How to Contribute

The general workflow for contributing is:

1.  **Fork the Repository**: Use the standard GitHub workflow to fork the repository to your own account.
2.  **Create a Feature Branch**: Create a new branch for your feature or bugfix. This keeps your changes organized and separate from the main branch.
    ```bash
    git checkout -b feature/NewThing
    ```
3.  **Make Changes**: Write your code or documentation changes.
4.  **Verify**: Ensure your changes work as expected and don't break existing functionality (see [Running Tests](#running-tests)).
5.  **Submit a Pull Request**: Push your branch to your fork and submit a Pull Request (PR) to the main Zen C repository.

## Issues and Pull Requests

We use GitHub Issues and Pull Requests to track bugs and features. To help us maintain quality:

-   **Use Templates**: When opening an Issue or PR, please use the provided templates.
    -   **Bug Report**: For reporting bugs.
    -   **Feature Request**: For suggesting new features.
    -   **Pull Request**: For submitting code changes.
-   **Be Descriptive**: Please provide as much detail as possible.
    -   **Automated Checks**: We have an automated workflow that checks the description length of new Issues and PRs. If the description is too short (< 50 characters), it will be automatically closed. This is to ensure we have enough information to help you.

## Development Guidelines

### Code Style
- Follow the existing C style found in the codebase. Consistency is key.
- Use `make format` to auto-format all source files with the provided `.clang-format`.
- Use `make lint` to verify formatting and check shell scripts.
- An `.editorconfig` file is provided for consistent editor settings (4-space indent, UTF-8, LF endings).
- **Pre-commit hooks**: Install with `pip install pre-commit && pre-commit install` to automatically check formatting, trailing whitespace, and shell scripts before each commit.
- Keep code clean and readable.

### Project Structure
If you are looking to extend the compiler, here is a quick map of the codebase:
*   **Parser**: `src/parser/` - Contains the recursive descent parser implementation.
*   **Codegen**: `src/codegen/` - Contains the transpiler logic that converts Zen C to GNU C/C11.
*   **Standard Library**: `std/` - The standard library modules, written in Zen C itself.

## Running Tests

The test suite is your best friend when developing. Please ensure all tests pass before submitting a PR.

### Run All Tests
```bash
make test
```

### Run Specific Test
```bash
./zc run tests/language/control_flow/test_match.zc
```

Or via the test runner:
```bash
make test only="tests/language/control_flow/test_match.zc"
```

### Test with Different Backends
```bash
./tests/scripts/run_tests.sh --cc clang    # Clang
./tests/scripts/run_tests.sh --cc zig      # Zig cc
./tests/scripts/run_tests.sh --cc tcc      # Tiny C Compiler
```

### Specialized Test Suites
```bash
make test-lsp      # LSP integration tests
make test-tcc      # Full suite with TCC backend
make test-misra    # MISRA C compliance checks
make test-asan     # AddressSanitizer / UBSan tests
```

### Warnings as Errors
To build with `-Werror` (recommended before submitting a PR):
```bash
make clean && make WERROR=1 -j$(nproc)
```

## Pull Request Process

1.  Ensure you have added tests for any new functionality.
2.  Ensure all existing tests pass.
3.  Update the documentation (Markdown files in `docs/` or `README.md`) if appropriate.
4.  Describe your changes clearly in the PR description. Link to any related issues.

Thank you for your contribution!
