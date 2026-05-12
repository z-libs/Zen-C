// SPDX-License-Identifier: MIT
#ifndef CODEGEN_BACKEND_H
#define CODEGEN_BACKEND_H

#include "../ast/ast.h"

struct ParserContext;

/**
 * @brief A code generation backend.
 *
 * Each backend provides an entry point to emit the full program.
 * The default "c" backend wraps the existing C codegen.
 * A new backend (e.g. x64, wasm, LLVM IR) implements emit_program from scratch.
 */
typedef struct CodegenBackend
{
    /// Backend identifier (e.g. "c", "llvm", "x64").
    const char *name;

    /// Default output file extension (e.g. ".c", ".ll", ".s").
    const char *extension;

    /// Emit the entire program to the current emitter output.
    void (*emit_program)(struct ParserContext *ctx, struct ASTNode *root);

    /// Emit the language preamble (type aliases, runtime, includes).
    void (*emit_preamble)(struct ParserContext *ctx);
} CodegenBackend;

/// Register a backend implementation.
void codegen_register_backend(const CodegenBackend *backend);

/// Get a registered backend by name. Returns the default "c" backend if name is NULL.
const CodegenBackend *codegen_get_backend(const char *name);

/// Initialize the default backends (called once at startup).
void codegen_init_backends(void);

#endif // CODEGEN_BACKEND_H
