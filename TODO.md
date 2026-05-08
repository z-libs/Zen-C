# Zen C TODO

## Recently Completed

### LSP Crash Fixes, Memory Leaks & Test Coverage
- [x] 7 new LSP tests (21 total): partial code goto-def/references, unopened file
      request, empty source, didChange incremental update, codeAction
- [x] Fixed 3 NULL-pointer crashes: `callee` unchecked in goto-def (lsp_analysis.c:240),
      find-references (lsp_analysis.c:1522), and missing `ctx` check in completion (lsp_analysis.c:913)
- [x] Fixed real heap leaks: replaced `strdup()` with `xstrdup()` in json_rpc.c
      (system malloc vs arena mismatch — `zfree` is a no-op)
- [x] Fixed `tmpfile()` file descriptor leak on project reinit (lsp_project.c)
- [x] Clear stale AST on parse failure instead of retaining outdated index (lsp_project.c:280)

### Tuple Infrastructure Fix + 13 New Tests
- [x] Fixed nested tuple struct emission: field types now use `types[i]` array instead
      of naively splitting the mangled signature by `__` (which broke for nested tuples
      like `((int, int), (int, int))` — the inner `Tuple__int__int` contains `__`)
- [x] Added `TupleType.types[]` + `TupleType.count` fields to store individual
      field type names for correct codegen
- [x] Updated `register_tuple_with_types()` API and all callers (parser_type.c,
      parser_expr.c, parser_struct.c) to pass field type names individually
- [x] Fixed reverse-LIFO emission order so nested tuples are defined before parents
- [x] Two-pass model: forward declarations + struct bodies before enums
- [x] 13 new test files covering: 3/4/5/8/10-tuples, nested tuples, field mutation,
      comparison, enum variants, complex expressions, typed annotations, return types,
      edge cases, pointer type mangling, mixed arity destructuring

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
