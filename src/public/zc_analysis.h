#ifndef ZC_ANALYSIS_H
#define ZC_ANALYSIS_H

/**
 * @file zc_analysis.h
 * @brief Public API for libzc-analysis: type checking, move semantics.
 */

struct ParserContext;
struct ASTNode;
struct Token;

/**
 * @brief Full type-checking pass (MISRA, type errors, etc.).
 *
 * @param ctx Parser context with a fully parsed AST.
 * @param root Root AST node.
 * @return 0 on success, non-zero if errors found.
 */
int check_program(struct ParserContext *ctx, struct ASTNode *root);

/**
 * @brief Move-semantics-only checking pass.
 *
 * @param ctx Parser context with a fully parsed AST.
 * @param root Root AST node.
 * @return 0 on success, non-zero if move errors found.
 */
int check_moves_only(struct ParserContext *ctx, struct ASTNode *root);

/**
 * @brief Resolve a typedef alias to its base type.
 */
struct Type *resolve_alias(struct Type *t);

#endif // ZC_ANALYSIS_H
