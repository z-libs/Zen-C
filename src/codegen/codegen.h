
#ifndef CODEGEN_H
#define CODEGEN_H

#include "../ast/ast.h"
#include "../parser/parser.h"
#include "../zprep.h"
#include <stdio.h>

// Main codegen entry points.
void codegen_node(ParserContext *ctx, ASTNode *node, FILE *out);
void codegen_node_single(ParserContext *ctx, ASTNode *node, FILE *out);
void codegen_walker(ParserContext *ctx, ASTNode *node, FILE *out);
void codegen_expression(ParserContext *ctx, ASTNode *node, FILE *out);

// Utility functions (codegen_utils.c).
char *infer_type(ParserContext *ctx, ASTNode *node);
ASTNode *find_struct_def_codegen(ParserContext *ctx, const char *name);
char *get_field_type_str(ParserContext *ctx, const char *struct_name, const char *field_name);
char *extract_call_args(const char *args);
void emit_var_decl_type(ParserContext *ctx, FILE *out, const char *type_str, const char *var_name);
char *replace_string_type(const char *args);
const char *parse_original_method_name(const char *mangled);
void emit_auto_type(ParserContext *ctx, ASTNode *init_expr, Token t, FILE *out);
char *codegen_type_to_string(Type *t);
void emit_func_signature(FILE *out, ASTNode *func, const char *name_override);

// Declaration emission  (codegen_decl.c).
void emit_preamble(ParserContext *ctx, FILE *out);
void emit_includes_and_aliases(ASTNode *node, FILE *out);
void emit_type_aliases(ASTNode *node, FILE *out);
void emit_global_aliases(ParserContext *ctx, FILE *out);
void emit_struct_defs(ParserContext *ctx, ASTNode *node, FILE *out);
void emit_trait_defs(ASTNode *node, FILE *out);
void emit_enum_protos(ASTNode *node, FILE *out);
void emit_globals(ParserContext *ctx, ASTNode *node, FILE *out);
void emit_lambda_defs(ParserContext *ctx, FILE *out);
void emit_protos(ASTNode *node, FILE *out);
void emit_impl_vtables(ParserContext *ctx, FILE *out);
void emit_tests_and_runner(ParserContext *ctx, ASTNode *node, FILE *out);
void print_type_defs(ParserContext *ctx, FILE *out, ASTNode *nodes);

// Global state (shared across modules).
extern ASTNode *global_user_structs;
extern char *g_current_impl_type;
extern int tmp_counter;
extern int defer_count;
extern ASTNode *defer_stack[];
extern ASTNode *g_current_lambda;

#define MAX_DEFER 1024

#endif
