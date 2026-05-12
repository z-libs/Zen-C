#ifndef FORMAT_EXPR_H
#define FORMAT_EXPR_H

struct ParserContext;
struct ASTNode;

/**
 * @brief Serializes an expression AST node to its C representation
 * using the codegen in buffer mode. Used during parsing for f-string
 * and printf-style format string interpolation.
 *
 * @param ctx Parser context (must have valid cg state for codegen).
 * @param node The expression node to serialize.
 * @return A heap-allocated C string representation, caller must free.
 */
char *format_expression_as_c(struct ParserContext *ctx, struct ASTNode *node);

#endif
