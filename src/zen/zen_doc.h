// SPDX-License-Identifier: MIT
#ifndef ZEN_DOC_H
#ifndef ZC_ALLOW_INTERNAL
#error "zen/zen_doc.h is internal to Zen C. Include the appropriate public header instead."
#endif

#define ZEN_DOC_H

#include "../ast/ast.h"

/**
 * @brief Generates documentation for the given program AST.
 *
 * @param ctx The parser context (useful for type resolution).
 * @param root The root NODE_ROOT of the AST.
 */
void generate_docs(struct ParserContext *ctx, ASTNode *root);

#endif
