# Coding Style

## Formatting
- Allman braces (`.clang-format` enforces this)
- 4-space indentation, no tabs
- 100-column limit
- Run `make format` to auto-fix formatting

## Naming
- `snake_case` for functions and variables
- `PascalCase` for types (structs, enums, typedefs)
- `SCREAMING_CASE` for macros and enum values
- `g_` prefix for global variables

## Comments
- `//` for implementation comments (space after `//`)
- `/** @brief */` for public API documentation
- `///<` for struct field documentation
- Never restate what the code does -- explain WHY
- Trailing `//` comments only for non-obvious behavior
- No em dashes, no decorative borders, no ASCII art
- Section dividers: `// ----` (hyphens only)
- New files: `// SPDX-License-Identifier: MIT` at top

## Includes
- Standard headers first (`<stdio.h>`, `<stdlib.h>`)
- Internal headers grouped by module
- No blank lines between same-group includes
- Use quotes for internal headers, angle brackets for system

## Error handling
- Return codes for recoverable errors
- `zpanic_at` for unrecoverable (parser errors, user mistakes)
- Check all allocation results (xmalloc never fails, but check anyway)
