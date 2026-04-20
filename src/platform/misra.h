#ifndef PLATFORM_MISRA_H
#define PLATFORM_MISRA_H

#include <stdio.h>
#include <stdbool.h>
#include "zprep.h" // For Token

typedef struct Type Type;
typedef struct ASTNode ASTNode;
typedef struct TypeChecker TypeChecker;

/**
 * @brief Emits a strictly stripped-down C preamble devoid of dynamic
 *        memory and I/O headers, ensuring 100% MISRA Compliance of the Core artifact.
 */
void emit_misra_preamble(FILE *out);

// --- MISRA C:2012 Compliance Modules ---

// Section 10/11/12: Essential Type Model & Conversions
void misra_check_ess_type_categories(struct TypeChecker *tc, struct Type *left, struct Type *right,
                                     Token token);
void misra_check_ess_type_composite(struct TypeChecker *tc, struct Type *target,
                                    struct Type *source, Token token);
void misra_check_implicit_conversion(struct TypeChecker *tc, struct Type *target,
                                     struct Type *source, Token token);
void misra_check_char_arithmetic(struct TypeChecker *tc, struct Type *left, struct Type *right,
                                 const char *op, Token token);
void misra_check_bitwise_operand(struct TypeChecker *tc, struct Type *t, Token token);
void misra_check_shift_amount(struct TypeChecker *tc, long long amount, int width, Token token);
void misra_check_pointer_conversion(struct TypeChecker *tc, struct Type *target,
                                    struct Type *source, Token token);
void misra_check_void_ptr_cast(struct TypeChecker *tc, struct Type *target, struct Type *source,
                               Token token);
void misra_check_cast(struct TypeChecker *tc, struct Type *target, struct Type *source, Token token,
                      bool is_composite);
void misra_check_null_pointer_constant(struct TypeChecker *tc, struct ASTNode *node, Token token);
void misra_check_binary_op_essential_types(struct TypeChecker *tc, struct ASTNode *left,
                                           struct ASTNode *right, Token token);
void misra_check_unsigned_wrap(struct TypeChecker *tc, const char *op, long long left,
                               long long right, long long res, struct Type *type, Token token);

// Section 13/14/15: Expressions & Control Flow
void misra_check_side_effects_sizeof(struct TypeChecker *tc, struct ASTNode *expr);
void misra_check_assignment_result_used(struct TypeChecker *tc, Token token);
void misra_check_inc_dec_result_used(struct TypeChecker *tc, Token token);
void misra_check_condition_boolean(struct TypeChecker *tc, struct Type *t, Token token);
void misra_check_invariant_condition(struct TypeChecker *tc, Token token);
void misra_check_loop_counter_float(struct TypeChecker *tc, struct Type *t, Token token);
void misra_check_initializer_side_effects(struct TypeChecker *tc, struct ASTNode *node);

// Section 16: Match/Switch
void misra_check_match_stmt(struct TypeChecker *tc, struct ASTNode *node);
void misra_check_strict_match(struct TypeChecker *tc, struct ASTNode *node);
void misra_check_shadowing(struct TypeChecker *tc, const char *name, Token token);
void misra_check_double_initialization(struct TypeChecker *tc, const char *field_name, Token token);

// Section 17: Functions
void misra_check_recursion(struct TypeChecker *tc, Token token);
void misra_check_function_return_usage(struct TypeChecker *tc, struct ASTNode *node);
void misra_check_array_param_size(TypeChecker *tc, int expected, int actual, Token token);
void misra_check_null_pointer_constant(TypeChecker *tc, struct ASTNode *node, Token token);
void misra_check_external_array_size(TypeChecker *tc, Type *t, Token token, int is_static,
                                     int is_local);
void misra_check_param_modified(TypeChecker *tc, struct ASTNode *left, Token token);
void misra_check_pointer_arithmetic(TypeChecker *tc, Type *t, Token token);
void misra_check_pointer_nesting(TypeChecker *tc, Type *t, Token token);
void misra_check_struct_decl(TypeChecker *tc, struct ASTNode *node);
void misra_check_compound_body(TypeChecker *tc, struct ASTNode *body, const char *stmt_name);
void misra_check_terminal_else(TypeChecker *tc, struct ASTNode *if_node);
void misra_check_param_nesting(TypeChecker *tc, struct ASTNode *func_node);
void misra_check_unused_param(struct TypeChecker *tc, const char *name, Token token);
void misra_check_const_ptr_param(struct TypeChecker *tc, const char *name, Token token);
void misra_check_goto(TypeChecker *tc, Token token);
void misra_check_goto_constraint(TypeChecker *tc, Token goto_tok, Token label_tok);
void misra_check_union(TypeChecker *tc, Token token);
void misra_check_iteration_termination(TypeChecker *tc, Token token);
void misra_check_stdarg(TypeChecker *tc, Token token);
void misra_check_vla(TypeChecker *tc, Type *t, Token token);
void misra_check_flexible_array(struct ASTNode *strct, struct ASTNode *field);
void misra_check_identifier_collision(Token tok, const char *name1, const char *name2, int limit);

// Section 2: Unused Code
void misra_audit_unused_symbols(struct TypeChecker *tc);

// Section 5: Identifiers
void misra_audit_identifier_uniqueness(struct TypeChecker *tc);
void misra_audit_block_scope(struct TypeChecker *tc);

// Section 19: Overlapping Storage
void misra_check_assignment_overlap(struct TypeChecker *tc, struct ASTNode *left,
                                    struct ASTNode *right, Token token);

// Zen C Extensions
void misra_check_raw_block(struct TypeChecker *tc, Token token);
void misra_check_preprocessor_directive(struct TypeChecker *tc, Token token);
void misra_check_preprocessor_expression(struct TypeChecker *tc, Token tok, const char *expression);
void misra_check_preprocessor_expression_parser(struct ParserContext *ctx, Token tok,
                                                const char *expression);
void misra_check_plugin_block(struct TypeChecker *tc, Token token);
void misra_check_reserved_identifier(struct TypeChecker *tc, const char *name, Token token);

#endif // PLATFORM_MISRA_H
