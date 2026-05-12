// SPDX-License-Identifier: MIT
#ifndef TYPECHECK_EXPR_H
#define TYPECHECK_EXPR_H

#include "typecheck.h"

void check_expr_unary(TypeChecker *tc, ASTNode *node, int depth);
void check_expr_binary(TypeChecker *tc, ASTNode *node, int depth);
void check_expr_call(TypeChecker *tc, ASTNode *node, int depth);
void check_expr_var(TypeChecker *tc, ASTNode *node);
void check_expr_literal(TypeChecker *tc, ASTNode *node);
void check_struct_init(TypeChecker *tc, ASTNode *node, int depth);
int check_type_compatibility(TypeChecker *tc, Type *target, Type *value, Token t,
                             ASTNode *value_node, int is_call_arg);
void apply_implicit_struct_pointer_conversions(TypeChecker *tc, ASTNode **expr_ptr,
                                               Type *expected_type);
void check_move_for_rvalue(TypeChecker *tc, ASTNode *rvalue);

#endif
