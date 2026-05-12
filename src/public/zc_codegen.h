#ifndef ZC_CODEGEN_H
#define ZC_CODEGEN_H

/**
 * @file zc_codegen.h
 * @brief Public API for libzc-codegen: C/C++/CUDA/ObjC code emission.
 */

struct ParserContext;
struct ASTNode;

/**
 * @brief Generate C/C++/CUDA/ObjC source from the AST.
 * The output is written to the file specified in the compiler config.
 *
 * @param ctx Parser context with parsed and type-checked AST.
 * @param root Root AST node to emit.
 */
void codegen_node(struct ParserContext *ctx, struct ASTNode *root);

/**
 * @brief Emit the C preamble (includes, type aliases, etc.).
 */
void emit_preamble(struct ParserContext *ctx);

/**
 * @brief Inline expression codegen for format string interpolation
 * (used internally by the parser, exposed for embedders).
 *
 * @return A C string representation of the expression. Caller must free.
 */
char *format_expression_as_c(struct ParserContext *ctx, struct ASTNode *node);

#endif // ZC_CODEGEN_H
