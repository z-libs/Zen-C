# Zen C TODO

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
- [ ] Add comprehensive type checker tests (10-15 files: generic instantiation, trait resolution,
      type inference, comptime eval, error recovery). Currently 1 test file (47 lines) for 4,113 lines of logic.
- [ ] Add dedicated move checker tests (currently zero dedicated tests).
- [x] Fix parallel test runner: `rand()`-based temp filenames in `run_comptime_block` caused collisions
      when tests ran concurrently. Changed to `getpid()` + static counter.

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
