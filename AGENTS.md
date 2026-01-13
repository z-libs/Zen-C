# Zen-C

A modern systems programming language that transpiles to human-readable GNU C/C11.

## Tech Stack

- **Language**: C (GCC/Clang/MSVC compatible)
- **Build System**: Make / CMake
- **Target Output**: GNU C / C11
- **Platforms**: Linux, macOS, Windows

## Directory Structure

```
Zen-C/
├── src/               # Compiler source code
│   ├── parser/        # Recursive descent parser
│   ├── codegen/       # Code generation (Zen C → C)
│   ├── lexer/         # Tokenizer
│   ├── ast/           # Abstract Syntax Tree definitions
│   ├── analysis/      # Type checking
│   ├── compat/        # Platform compatibility layer (Windows/POSIX)
│   ├── lsp/           # Language Server Protocol support
│   ├── repl/          # Interactive REPL
│   ├── plugins/       # Plugin manager
│   ├── utils/         # Utility functions
│   └── zen/           # Zen-specific helpers
├── plugins/           # Syntax extension plugins (regex, SQL, lisp, etc.)
├── std/               # Standard library (written in Zen C)
├── tests/             # Test suite
├── examples/          # Example programs
├── man/               # Man pages
└── plan/              # Design documents
```

## Common Commands

```bash
# Build (Make)
make

# Clean build
make clean && make

# Run tests
make test

# Install (system-wide)
sudo make install

# Build (CMake)
cmake -B build && cmake --build build

# CMake clean build
rm -rf build && cmake -B build && cmake --build build

# Run a Zen C file
./zc run <file>.zc

# Build executable
./zc build <file>.zc -o <output>

# Use different C compiler backend
./zc run <file>.zc --cc clang
```

## Code Style

- Follow existing C style in the codebase
- Use snake_case for functions and variables
- Prefix module functions (e.g., `parser_`, `codegen_`, `ast_`)

## Testing

- Add new tests in `tests/` directory
- Test files are `.zc` files
- Run `make test` to validate all tests pass
- Test with both GCC and Clang for compatibility

## Updating This File

Update this file when:
- Adding new modules or directories
- Changing build commands
- Modifying project structure
