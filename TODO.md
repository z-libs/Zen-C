# Zen C TODO

## Recently Completed

### Test Framework Modernization
- [x] Per-test failure isolation: `__zenc_assert` no longer calls `exit(1)` — uses counter instead
- [x] Named per-test output: each test prints name + OK/FAIL to stderr
- [x] `_z_run_tests()` returns failure count; `main()` uses it as exit code
- [x] Added `expect` keyword — non-fatal assertion, continues after failure
- [x] Multiple assertions in one test are all reported (not just the first)

### Emitter Refactoring
- [x] Fix `realloc` NULL-return bug, `emitter_vprintf`, `emitter_putc`, auto-indent, push/pop API
- [x] Migrate ~84 hardcoded `"    "` in codegen to auto-indent
- [x] Comptime temp file collision fix (`rand()` → `getpid()` + counter)

### Type Checker Tests
- [x] 13 new test files (from 47 lines to 450+ lines): type compat, operators, calls,
      returns, move, struct init, vardecl, traits, lifetime, match, cast, const-fold, misc

## Codegen
- [ ] Migrate remaining control instructions in codegen_stmt.c to jump tables (break, continue, return, asm, loops).
- [ ] Split codegen_stmt.c (2,548 lines) and codegen_decl.c (1,900+ lines) further by concern.

## Architecture & Modularity
- [ ] Divide the typechecker (src/analysis/typecheck.c) into several files.
- [ ] Split up parser_expr.c (8,016 lines) into a new folder (src/parser/expr/) with separate files for binary, unary, etc.
- [ ] Split parser_utils.c (6,752 lines) by concern: generics, imports, type resolution, comptime.
- [ ] Split parser_stmt.c (4,928 lines) by statement type: control flow, declarations, special (asm/defer/test).
- [ ] Split ParserContext god struct (parser.h:291-404) into focused sub-structs:
      CodegenState, ModuleState, GenericRegistry, LSPState, TypeValidationState.
      (The `cg` block at lines 376-391 is already a partial extraction: finish it.)
- [ ] Eliminate `g_parser_ctx` global — thread ctx through all call chains explicitly.
- [ ] Eliminate `g_config` macro (543+ refs) — pass config explicitly instead of via global `g_compiler`.
- [ ] Reduce AST dispatch duplication: 6+ parallel `switch(type)` chains across codegen, typechecker, LSP.
      Consider a formal visitor pattern or code-gen dispatchers.

## Features
- [ ] Improve traits.
- [ ] Improve comptime mechanics.

## Testing
- [x] Add comprehensive type checker tests (13 files covering type compatibility, operators,
      function calls, returns, move semantics, struct init, variable decl, traits, lifetime,
      match, cast, const-folding, lambdas, loops). Raised from 47 lines to 450+ lines.
- [ ] Add dedicated move checker tests (currently zero dedicated tests).
- [ ] Allow `expect` in comptime blocks for compile-time test assertions.
- [ ] Add test filtering via `ZC_TEST_FILTER` environment variable.
- [ ] Add `expect_eq`/`expect_ne`/`expect_approx_eq` helpers to standard library.

## Memory
- [ ] Fix LSP buffer leak: lsp_main.c arena-allocates JSON body then calls `zfree` (no-op on arena).
      For a long-running LSP server this leaks until process exit.
- [ ] Introduce sub-arenas or reset points for per-function allocations (move_check.c, etc.).

## Error Handling
- [ ] Unify 5+ error mechanisms: `zpanic_at`, `zpanic`, `zpanic_at_diag`, `fprintf(stderr, ...)`, return-NULL.
- [ ] Migrate `fprintf(stderr, ...)` in codegen runtime helpers to proper diagnostic pipeline
      (so LSP mode catches all errors).

## Cleanup
- [ ] Fix residual template and defer scope tracking bugs.
- [ ] Reduce function complexity hotspot warnings (for example, `parse_expr_prec_impl` at 2,873 lines).
- [ ] Clear temporary build files and logs.
