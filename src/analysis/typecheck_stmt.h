#ifndef TYPECHECK_STMT_H
#define TYPECHECK_STMT_H

#include "typecheck.h"

void check_block(TypeChecker *tc, ASTNode *block, int depth);
void check_var_decl(TypeChecker *tc, ASTNode *node, int depth);
void check_function(TypeChecker *tc, ASTNode *node, int depth);
void check_loop_passes(TypeChecker *tc, ASTNode *node, int depth);
void tc_check_trait(TypeChecker *tc, ASTNode *node, int depth);
void tc_check_impl(TypeChecker *tc, ASTNode *node, int depth);
void tc_check_impl_trait(TypeChecker *tc, ASTNode *node, int depth);
int block_always_returns(ASTNode *block);
int stmt_always_returns(ASTNode *stmt);

#endif
