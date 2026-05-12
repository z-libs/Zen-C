#ifndef CONST_FOLD_H
#define CONST_FOLD_H

#include "ast/ast.h"

typedef struct ParserContext ParserContext;

// Evaluates a constant integer expression.
// Returns 1 if successful and sets *out_val.
// Returns 0 if the expression is not a compile-time constant.
int eval_const_int_expr(ASTNode *node, ParserContext *ctx, long long *out_val);

#endif
