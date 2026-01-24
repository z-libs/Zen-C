
#include "codegen.h"
#include "zprep.h"
#include "../constants.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../plugins/plugin_manager.h"
#include "ast.h"
#include "zprep_plugin.h"

// Emit literal expression (int, float, string, char)
static void codegen_literal_expr(ASTNode *node, FILE *out)
{
    if (node->literal.type_kind == TOK_STRING)
    {
        fprintf(out, "\"%s\"", node->literal.string_val);
    }
    else if (node->literal.type_kind == TOK_CHAR)
    {
        fprintf(out, "%s", node->literal.string_val);
    }
    else if (node->literal.type_kind == 1) // float
    {
        fprintf(out, "%f", node->literal.float_val);
    }
    else // int
    {
        if (node->literal.int_val > 9223372036854775807ULL)
        {
            fprintf(out, "%lluULL", (unsigned long long)node->literal.int_val);
        }
        else
        {
            fprintf(out, "%llu", (unsigned long long)node->literal.int_val);
        }
    }
}

// Emit variable reference expression
static void codegen_var_expr(ParserContext *ctx, ASTNode *node, FILE *out)
{
    if (g_current_lambda)
    {
        for (int i = 0; i < g_current_lambda->lambda.num_captures; i++)
        {
            if (strcmp(node->var_ref.name, g_current_lambda->lambda.captured_vars[i]) == 0)
            {
                fprintf(out, "ctx->%s", node->var_ref.name);
                return;
            }
        }
    }

    if (node->resolved_type && strcmp(node->resolved_type, "unknown") == 0)
    {
        if (node->var_ref.suggestion)
        {
            char msg[256];
            sprintf(msg, "Undefined variable '%s'", node->var_ref.name);
            char help[256];
            sprintf(help, "Did you mean '%s'?", node->var_ref.suggestion);
            zwarn_at(node->token, "%s\n   = help: %s", msg, help);
        }
    }
    fprintf(out, "%s", node->var_ref.name);
}

// Emit lambda expression
static void codegen_lambda_expr(ASTNode *node, FILE *out)
{
    if (node->lambda.num_captures > 0)
    {
        fprintf(out,
                "({ struct Lambda_%d_Ctx *ctx = malloc(sizeof(struct "
                "Lambda_%d_Ctx));\n",
                node->lambda.lambda_id, node->lambda.lambda_id);
        for (int i = 0; i < node->lambda.num_captures; i++)
        {
            fprintf(out, "ctx->%s = ", node->lambda.captured_vars[i]);
            int found = 0;
            if (g_current_lambda)
            {
                for (int k = 0; k < g_current_lambda->lambda.num_captures; k++)
                {
                    if (strcmp(node->lambda.captured_vars[i],
                               g_current_lambda->lambda.captured_vars[k]) == 0)
                    {
                        fprintf(out, "ctx->%s", node->lambda.captured_vars[i]);
                        found = 1;
                        break;
                    }
                }
            }
            if (!found)
            {
                fprintf(out, "%s", node->lambda.captured_vars[i]);
            }
            fprintf(out, ";\n");
        }
        fprintf(out, "(z_closure_T){.func = _lambda_%d, .ctx = ctx}; })", node->lambda.lambda_id);
    }
    else
    {
        fprintf(out, "((z_closure_T){.func = (void*)_lambda_%d, .ctx = NULL})",
                node->lambda.lambda_id);
    }
}

void codegen_expression(ParserContext *ctx, ASTNode *node, FILE *out)
{
    if (!node)
    {
        return;
    }
    switch (node->type)
    {
    case NODE_MATCH:
        codegen_match_internal(ctx, node, out, 1);
        break;
    case NODE_EXPR_BINARY:
        if (strncmp(node->binary.op, "??", 2) == 0 && strlen(node->binary.op) == 2)
        {
            fprintf(out, "({ ");
            emit_auto_type(ctx, node->binary.left, node->token, out);
            fprintf(out, " _l = (");
            codegen_expression(ctx, node->binary.left, out);
            fprintf(out, "); _l ? _l : (");
            codegen_expression(ctx, node->binary.right, out);
            fprintf(out, "); })");
        }
        else if (strcmp(node->binary.op, "?\?=") == 0)
        {
            fprintf(out, "({ if (!(");
            codegen_expression(ctx, node->binary.left, out);
            fprintf(out, ")) ");
            codegen_expression(ctx, node->binary.left, out);
            fprintf(out, " = (");
            codegen_expression(ctx, node->binary.right, out);
            fprintf(out, "); ");
            codegen_expression(ctx, node->binary.left, out);
            fprintf(out, "; })");
        }
        else if ((strcmp(node->binary.op, "==") == 0 || strcmp(node->binary.op, "!=") == 0))
        {
            char *t1 = infer_type(ctx, node->binary.left);

            int is_ptr = 0;
            if (t1)
            {
                char *check = t1;
                int depth = 0;
                while (depth++ < 10)
                {
                    if (strchr(check, '*'))
                    {
                        is_ptr = 1;
                        break;
                    }
                    int resolved = 0;
                    ASTNode *alias = global_user_structs;
                    if (alias)
                    {
                        while (alias)
                        {
                            if (alias->type == NODE_TYPE_ALIAS &&
                                strcmp(check, alias->type_alias.alias) == 0)
                            {
                                check = alias->type_alias.original_type;
                                resolved = 1;
                                break;
                            }
                            alias = alias->next;
                        }
                    }
                    if (!resolved)
                    {
                        break;
                    }
                }
            }

            int is_basic = IS_BASIC_TYPE(t1);

            ASTNode *def = t1 ? find_struct_def(ctx, t1) : NULL;
            if (t1 && def && (def->type == NODE_STRUCT || def->type == NODE_ENUM) && !is_basic &&
                !is_ptr)
            {
                char *base = t1;
                if (strncmp(base, "struct ", 7) == 0)
                {
                    base += 7;
                }

                if (strcmp(node->binary.op, "!=") == 0)
                {
                    fprintf(out, "(!");
                }
                fprintf(out, "%s__eq(", base);

                if (node->binary.left->type == NODE_EXPR_VAR ||
                    node->binary.left->type == NODE_EXPR_INDEX ||
                    node->binary.left->type == NODE_EXPR_MEMBER)
                {
                    fprintf(out, "&");
                    codegen_expression(ctx, node->binary.left, out);
                }
                else
                {
                    fprintf(out, "({ ZC_AUTO _t = ");
                    codegen_expression(ctx, node->binary.left, out);
                    fprintf(out, "; &_t; })");
                }

                fprintf(out, ", ");

                if (node->binary.right->type == NODE_EXPR_VAR ||
                    node->binary.right->type == NODE_EXPR_INDEX ||
                    node->binary.right->type == NODE_EXPR_MEMBER)
                {
                    fprintf(out, "&");
                    codegen_expression(ctx, node->binary.right, out);
                }
                else
                {
                    fprintf(out, "({ ZC_AUTO _t = ");
                    codegen_expression(ctx, node->binary.right, out);
                    fprintf(out, "; &_t; })");
                }

                fprintf(out, ")");
                if (strcmp(node->binary.op, "!=") == 0)
                {
                    fprintf(out, ")");
                }
            }
            else if (t1 && (strcmp(t1, "string") == 0 || strcmp(t1, "char*") == 0 ||
                            strcmp(t1, "const char*") == 0))
            {
                // Check if comparing to NULL - don't use strcmp for NULL comparisons
                int is_null_compare = 0;
                if (node->binary.right->type == NODE_EXPR_VAR &&
                    strcmp(node->binary.right->var_ref.name, "NULL") == 0)
                {
                    is_null_compare = 1;
                }
                else if (node->binary.left->type == NODE_EXPR_VAR &&
                         strcmp(node->binary.left->var_ref.name, "NULL") == 0)
                {
                    is_null_compare = 1;
                }

                if (is_null_compare)
                {
                    // Direct pointer comparison for NULL
                    fprintf(out, "(");
                    codegen_expression(ctx, node->binary.left, out);
                    fprintf(out, " %s ", node->binary.op);
                    codegen_expression(ctx, node->binary.right, out);
                    fprintf(out, ")");
                }
                else
                {
                    fprintf(out, "(strcmp(");
                    codegen_expression(ctx, node->binary.left, out);
                    fprintf(out, ", ");
                    codegen_expression(ctx, node->binary.right, out);
                    if (strcmp(node->binary.op, "==") == 0)
                    {
                        fprintf(out, ") == 0)");
                    }
                    else
                    {
                        fprintf(out, ") != 0)");
                    }
                }
            }
            else
            {
                fprintf(out, "(");
                codegen_expression(ctx, node->binary.left, out);
                fprintf(out, " %s ", node->binary.op);
                codegen_expression(ctx, node->binary.right, out);
                fprintf(out, ")");
            }
        }
        else
        {
            fprintf(out, "(");
            codegen_expression(ctx, node->binary.left, out);
            fprintf(out, " %s ", node->binary.op);
            codegen_expression(ctx, node->binary.right, out);
            fprintf(out, ")");
        }
        break;
    case NODE_EXPR_VAR:
        codegen_var_expr(ctx, node, out);
        break;
    case NODE_LAMBDA:
        codegen_lambda_expr(node, out);
        break;
    case NODE_EXPR_LITERAL:
        codegen_literal_expr(node, out);
        break;
    case NODE_EXPR_CALL:
    {
        if (node->call.callee->type == NODE_EXPR_MEMBER)
        {
            ASTNode *target = node->call.callee->member.target;
            char *method = node->call.callee->member.field;

            if (strcmp(method, "len") == 0)
            {
                if (target->type_info && target->type_info->kind == TYPE_ARRAY)
                {
                    if (target->type_info->array_size > 0)
                    {
                        fprintf(out, "%d", target->type_info->array_size);
                    }
                    else
                    {
                        codegen_expression(ctx, target, out);
                        fprintf(out, ".len");
                    }
                    return;
                }
            }

            char *type = infer_type(ctx, target);
            if (type)
            {
                char *clean = xstrdup(type);
                char *ptr = strchr(clean, '*');
                if (ptr)
                {
                    *ptr = 0;
                }

                char *base = clean;
                if (strncmp(base, "struct ", 7) == 0)
                {
                    base += 7;
                }

                if (!strchr(type, '*') && target->type == NODE_EXPR_CALL)
                {
                    fprintf(out, "({ %s _t = ", type);
                    codegen_expression(ctx, target, out);
                    fprintf(out, "; %s__%s(&_t", base, method);
                    ASTNode *arg = node->call.args;
                    while (arg)
                    {
                        fprintf(out, ", ");
                        codegen_expression(ctx, arg, out);
                        arg = arg->next;
                    }
                    fprintf(out, "); })");
                }
                else
                {
                    // Mixin Lookup Logic
                    char *call_base = base;
                    int need_cast = 0;
                    char mixin_func_name[128];
                    sprintf(mixin_func_name, "%s__%s", base, method);

                    char *resolved_method_suffix = NULL;

                    if (!find_func(ctx, mixin_func_name))
                    {
                        // Try resolving as a trait method: Struct__Trait_Method
                        StructRef *ref = ctx->parsed_impls_list;
                        while (ref)
                        {
                            if (ref->node && ref->node->type == NODE_IMPL_TRAIT)
                            {
                                if (strcmp(ref->node->impl_trait.target_type, base) == 0)
                                {
                                    char trait_mangled[256];
                                    sprintf(trait_mangled, "%s__%s_%s", base,
                                            ref->node->impl_trait.trait_name, method);
                                    if (find_func(ctx, trait_mangled))
                                    {
                                        char *suffix =
                                            xmalloc(strlen(ref->node->impl_trait.trait_name) +
                                                    strlen(method) + 2);
                                        sprintf(suffix, "%s_%s", ref->node->impl_trait.trait_name,
                                                method);
                                        resolved_method_suffix = suffix;
                                        break;
                                    }
                                }
                            }
                            ref = ref->next;
                        }

                        if (resolved_method_suffix)
                        {
                            method = resolved_method_suffix;
                        }
                        else
                        {
                            // Method not found on primary struct, check mixins
                            ASTNode *def = find_struct_def(ctx, base);
                            if (def && def->type == NODE_STRUCT && def->strct.used_structs)
                            {
                                for (int k = 0; k < def->strct.used_struct_count; k++)
                                {
                                    char mixin_check[128];
                                    sprintf(mixin_check, "%s__%s", def->strct.used_structs[k],
                                            method);
                                    if (find_func(ctx, mixin_check))
                                    {
                                        call_base = def->strct.used_structs[k];
                                        need_cast = 1;
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    fprintf(out, "%s__%s(", call_base, method);
                    if (need_cast)
                    {
                        fprintf(out, "(%s*)%s", call_base, strchr(type, '*') ? "" : "&");
                    }
                    else if (!strchr(type, '*'))
                    {
                        fprintf(out, "&");
                    }
                    codegen_expression(ctx, target, out);
                    ASTNode *arg = node->call.args;
                    while (arg)
                    {
                        fprintf(out, ", ");
                        codegen_expression(ctx, arg, out);
                        arg = arg->next;
                    }
                    fprintf(out, ")");
                }
                free(clean);
                return;
            }
        }
        if (node->call.callee->type == NODE_EXPR_VAR)
        {
            ASTNode *def = find_struct_def(ctx, node->call.callee->var_ref.name);
            if (def && def->type == NODE_STRUCT)
            {
                fprintf(out, "(struct %s){0}", node->call.callee->var_ref.name);
                return;
            }
        }

        if (node->call.callee->type_info && node->call.callee->type_info->kind == TYPE_FUNCTION)
        {
            fprintf(out, "({ z_closure_T _c = ");
            codegen_expression(ctx, node->call.callee, out);
            fprintf(out, "; ");

            Type *ft = node->call.callee->type_info;
            char *ret = codegen_type_to_string(ft->inner);
            if (strcmp(ret, "string") == 0)
            {
                free(ret);
                ret = xstrdup("char*");
            }

            fprintf(out, "((%s (*)(void*", ret);
            for (int i = 0; i < ft->arg_count; i++)
            {
                char *as = codegen_type_to_string(ft->args[i]);
                fprintf(out, ", %s", as);
                free(as);
            }
            if (ft->is_varargs)
            {
                fprintf(out, ", ...");
            }
            fprintf(out, "))_c.func)(_c.ctx");

            ASTNode *arg = node->call.args;
            while (arg)
            {
                fprintf(out, ", ");
                codegen_expression(ctx, arg, out);
                arg = arg->next;
            }
            fprintf(out, "); })");
            free(ret);
            break;
        }

        codegen_expression(ctx, node->call.callee, out);
        fprintf(out, "(");

        if (node->call.arg_names && node->call.callee->type == NODE_EXPR_VAR)
        {
            char *fn_name = node->call.callee->var_ref.name;
            FuncSig *sig = find_func(ctx, fn_name);

            if (sig && sig->arg_types)
            {
                for (int p = 0; p < sig->total_args; p++)
                {
                    ASTNode *arg = node->call.args;

                    for (int i = 0; i < node->call.arg_count && arg; i++, arg = arg->next)
                    {
                        if (node->call.arg_names[i] && p < node->call.arg_count)
                        {

                            // For now, emit in order provided...
                        }
                    }
                }
            }

            ASTNode *arg = node->call.args;
            int first = 1;
            while (arg)
            {
                if (!first)
                {
                    fprintf(out, ", ");
                }
                first = 0;
                codegen_expression(ctx, arg, out);
                arg = arg->next;
            }
        }
        else
        {
            FuncSig *sig = NULL;
            if (node->call.callee->type == NODE_EXPR_VAR)
            {
                sig = find_func(ctx, node->call.callee->var_ref.name);
            }

            ASTNode *arg = node->call.args;
            int arg_idx = 0;
            while (arg)
            {
                int handled = 0;
                if (sig && arg_idx < sig->total_args)
                {
                    Type *param_t = sig->arg_types[arg_idx];
                    Type *arg_t = arg->type_info;

                    if (param_t && param_t->kind == TYPE_ARRAY && param_t->array_size == 0 &&
                        arg_t && arg_t->kind == TYPE_ARRAY && arg_t->array_size > 0)
                    {
                        char *inner = type_to_string(param_t->inner);
                        fprintf(out, "(Slice_%s){.data = ", inner);
                        codegen_expression(ctx, arg, out);
                        fprintf(out, ", .len = %d, .cap = %d}", arg_t->array_size,
                                arg_t->array_size);
                        free(inner);
                        handled = 1;
                    }
                }

                if (!handled)
                {
                    codegen_expression(ctx, arg, out);
                }

                if (arg->next)
                {
                    fprintf(out, ", ");
                }
                arg = arg->next;
                arg_idx++;
            }
        }
        fprintf(out, ")");
        break;
    }
    case NODE_EXPR_MEMBER:
        if (strcmp(node->member.field, "len") == 0)
        {
            if (node->member.target->type_info)
            {
                if (node->member.target->type_info->kind == TYPE_ARRAY)
                {
                    if (node->member.target->type_info->array_size > 0)
                    {
                        fprintf(out, "%d", node->member.target->type_info->array_size);
                        break;
                    }
                }
            }
        }

        if (node->member.is_pointer_access == 2)
        {
            fprintf(out, "({ ");
            emit_auto_type(ctx, node->member.target, node->token, out);
            fprintf(out, " _t = (");
            codegen_expression(ctx, node->member.target, out);
            char *field = node->member.field;
            if (field && field[0] >= '0' && field[0] <= '9')
            {
                fprintf(out, "); _t ? _t->v%s : 0; })", field);
            }
            else
            {
                fprintf(out, "); _t ? _t->%s : 0; })", field);
            }
        }
        else
        {
            codegen_expression(ctx, node->member.target, out);
            // Verify actual type instead of trusting is_pointer_access flag
            char *lt = infer_type(ctx, node->member.target);
            int actually_ptr = 0;
            if (lt && (lt[strlen(lt) - 1] == '*' || strstr(lt, "*")))
            {
                actually_ptr = 1;
            }
            if (lt)
            {
                free(lt);
            }
            char *field = node->member.field;
            if (field && field[0] >= '0' && field[0] <= '9')
            {
                fprintf(out, "%sv%s", actually_ptr ? "->" : ".", field);
            }
            else
            {
                fprintf(out, "%s%s", actually_ptr ? "->" : ".", field);
            }
        }
        break;
    case NODE_EXPR_INDEX:
    {
        int is_slice_struct = 0;
        if (node->index.array->type_info)
        {
            if (node->index.array->type_info->kind == TYPE_ARRAY &&
                node->index.array->type_info->array_size == 0)
            {
                is_slice_struct = 1;
            }
        }
        if (node->index.array->resolved_type)
        {
            if (strncmp(node->index.array->resolved_type, "Slice_", 6) == 0)
            {
                is_slice_struct = 1;
            }
        }

        if (!is_slice_struct && !node->index.array->type_info && !node->index.array->resolved_type)
        {
            char *inferred = infer_type(ctx, node->index.array);
            if (inferred && strncmp(inferred, "Slice_", 6) == 0)
            {
                is_slice_struct = 1;
            }
            if (inferred)
            {
                free(inferred);
            }
        }

        if (is_slice_struct)
        {
            if (node->index.array->type == NODE_EXPR_VAR)
            {
                codegen_expression(ctx, node->index.array, out);
                fprintf(out, ".data[_z_check_bounds(");
                codegen_expression(ctx, node->index.index, out);
                fprintf(out, ", ");
                codegen_expression(ctx, node->index.array, out);
                fprintf(out, ".len)]");
            }
            else
            {
                codegen_expression(ctx, node->index.array, out);
                fprintf(out, ".data[");
                codegen_expression(ctx, node->index.index, out);
                fprintf(out, "]");
            }
        }
        else
        {
            int fixed_size = -1;
            if (node->index.array->type_info && node->index.array->type_info->kind == TYPE_ARRAY)
            {
                fixed_size = node->index.array->type_info->array_size;
            }

            codegen_expression(ctx, node->index.array, out);
            fprintf(out, "[");
            if (fixed_size > 0)
            {
                fprintf(out, "_z_check_bounds(");
            }
            codegen_expression(ctx, node->index.index, out);
            if (fixed_size > 0)
            {
                fprintf(out, ", %d)", fixed_size);
            }
            fprintf(out, "]");
        }
    }
    break;
    case NODE_EXPR_SLICE:
    {
        int known_size = -1;
        int is_slice_struct = 0;
        if (node->slice.array->type_info)
        {
            if (node->slice.array->type_info->kind == TYPE_ARRAY)
            {
                known_size = node->slice.array->type_info->array_size;
                if (known_size == 0)
                {
                    is_slice_struct = 1;
                }
            }
        }

        char *tname = "unknown";
        if (node->type_info && node->type_info->inner)
        {
            tname = codegen_type_to_string(node->type_info->inner);
        }

        fprintf(out, "({ ");
        emit_auto_type(ctx, node->slice.array, node->token, out);
        fprintf(out, " _arr = ");
        codegen_expression(ctx, node->slice.array, out);
        fprintf(out, "; int _start = ");
        if (node->slice.start)
        {
            codegen_expression(ctx, node->slice.start, out);
        }
        else
        {
            fprintf(out, "0");
        }
        fprintf(out, "; int _len = ");

        if (node->slice.end)
        {
            codegen_expression(ctx, node->slice.end, out);
            fprintf(out, " - _start; ");
        }
        else
        {
            if (known_size > 0)
            {
                fprintf(out, "%d - _start; ", known_size);
            }
            else if (is_slice_struct)
            {
                fprintf(out, "_arr.len - _start; ");
            }
            else
            {
                fprintf(out, "/* UNSAFE: Full Slice on unknown size */ 0; ");
            }
        }

        if (is_slice_struct)
        {
            fprintf(out,
                    "(Slice_%s){ .data = _arr.data + _start, .len = _len, .cap = "
                    "_len }; })",
                    tname);
        }
        else
        {
            fprintf(out, "(Slice_%s){ .data = _arr + _start, .len = _len, .cap = _len }; })",
                    tname);
        }
        break;
    }
    case NODE_BLOCK:
    {
        int saved = defer_count;
        fprintf(out, "({ ");
        codegen_walker(ctx, node->block.statements, out);
        for (int i = defer_count - 1; i >= saved; i--)
        {
            codegen_node_single(ctx, defer_stack[i], out);
        }
        defer_count = saved;
        fprintf(out, " })");
        break;
    }
    case NODE_TRY:
    {
        char *type_name = "Result";
        if (g_current_func_ret_type)
        {
            type_name = g_current_func_ret_type;
        }
        else if (node->try_stmt.expr->type_info && node->try_stmt.expr->type_info->name)
        {
            type_name = node->try_stmt.expr->type_info->name;
        }

        if (strcmp(type_name, "__auto_type") == 0 || strcmp(type_name, "unknown") == 0)
        {
            type_name = "Result";
        }

        char *search_name = type_name;
        if (strncmp(search_name, "struct ", 7) == 0)
        {
            search_name += 7;
        }

        int is_enum = 0;
        StructRef *er = ctx->parsed_enums_list;
        while (er)
        {
            if (er->node && er->node->type == NODE_ENUM &&
                strcmp(er->node->enm.name, search_name) == 0)
            {
                is_enum = 1;
                break;
            }
            er = er->next;
        }
        if (!is_enum)
        {
            ASTNode *ins = ctx->instantiated_structs;
            while (ins)
            {
                if (ins->type == NODE_ENUM && strcmp(ins->enm.name, search_name) == 0)
                {
                    is_enum = 1;
                    break;
                }
                ins = ins->next;
            }
        }

        fprintf(out, "({ ");
        emit_auto_type(ctx, node->try_stmt.expr, node->token, out);
        fprintf(out, " _try = ");
        codegen_expression(ctx, node->try_stmt.expr, out);

        if (is_enum)
        {
            fprintf(out,
                    "; if (_try.tag == %s_Err_Tag) return (%s_Err(_try.data.Err)); "
                    "_try.data.Ok; })",
                    search_name, search_name);
        }
        else
        {
            fprintf(out,
                    "; if (!_try.is_ok) return %s__Err(_try.err); "
                    "_try.val; })",
                    search_name);
        }
        break;
    }
    case NODE_RAW_STMT:
        fprintf(out, "%s", node->raw_stmt.content);
        break;
    case NODE_PLUGIN:
    {
        // Plugin registry - declare external plugins
        ZPlugin *found = zptr_find_plugin(node->plugin_stmt.plugin_name);

        if (found)
        {
            ZApi api = {.filename = g_current_filename ? g_current_filename : "input.zc",
                        .current_line = node->line,
                        .out = out,
                        .hoist_out = ctx->hoist_out};
            found->fn(node->plugin_stmt.body, &api);
        }
        else
        {
            fprintf(out, "/* Unknown plugin: %s */\n", node->plugin_stmt.plugin_name);
        }
        break;
    }
    case NODE_EXPR_UNARY:
        if (node->unary.op && strcmp(node->unary.op, "&_rval") == 0)
        {
            fprintf(out, "({ ");
            emit_auto_type(ctx, node->unary.operand, node->token, out);
            fprintf(out, " _t = (");
            codegen_expression(ctx, node->unary.operand, out);
            fprintf(out, "); &_t; })");
        }
        else if (node->unary.op && strcmp(node->unary.op, "?") == 0)
        {
            fprintf(out, "({ ");
            emit_auto_type(ctx, node->unary.operand, node->token, out);
            fprintf(out, " _t = (");
            codegen_expression(ctx, node->unary.operand, out);
            fprintf(out, "); if (_t.tag != 0) return _t; _t.data.Ok; })");
        }
        else if (node->unary.op && strcmp(node->unary.op, "_post++") == 0)
        {
            fprintf(out, "(");
            codegen_expression(ctx, node->unary.operand, out);
            fprintf(out, "++)");
        }
        else if (node->unary.op && strcmp(node->unary.op, "_post--") == 0)
        {
            fprintf(out, "(");
            codegen_expression(ctx, node->unary.operand, out);
            fprintf(out, "--)");
        }
        else
        {
            fprintf(out, "(%s", node->unary.op);
            codegen_expression(ctx, node->unary.operand, out);
            fprintf(out, ")");
        }
        break;
    case NODE_VA_START:
        fprintf(out, "va_start(");
        codegen_expression(ctx, node->va_start.ap, out);
        fprintf(out, ", ");
        codegen_expression(ctx, node->va_start.last_arg, out);
        fprintf(out, ")");
        break;
    case NODE_VA_END:
        fprintf(out, "va_end(");
        codegen_expression(ctx, node->va_end.ap, out);
        fprintf(out, ")");
        break;
    case NODE_VA_COPY:
        fprintf(out, "va_copy(");
        codegen_expression(ctx, node->va_copy.dest, out);
        fprintf(out, ", ");
        codegen_expression(ctx, node->va_copy.src, out);
        fprintf(out, ")");
        break;
    case NODE_VA_ARG:
    {
        char *type_str = codegen_type_to_string(node->va_arg.type_info);
        fprintf(out, "va_arg(");
        codegen_expression(ctx, node->va_arg.ap, out);
        fprintf(out, ", %s)", type_str);
        free(type_str);
        break;
    }
    case NODE_EXPR_CAST:
        fprintf(out, "(%s)(", node->cast.target_type);
        codegen_expression(ctx, node->cast.expr, out);
        fprintf(out, ")");
        break;
    case NODE_EXPR_SIZEOF:
        if (node->size_of.target_type)
        {
            fprintf(out, "sizeof(%s)", node->size_of.target_type);
        }
        else
        {
            fprintf(out, "sizeof(");
            codegen_expression(ctx, node->size_of.expr, out);
            fprintf(out, ")");
        }
        break;
    case NODE_TYPEOF:
        if (node->size_of.target_type)
        {
            fprintf(out, "typeof(%s)", node->size_of.target_type);
        }
        else
        {
            fprintf(out, "typeof(");
            codegen_expression(ctx, node->size_of.expr, out);
            fprintf(out, ")");
        }
        break;

    case NODE_REFLECTION:
    {
        Type *t = node->reflection.target_type;
        if (node->reflection.kind == 0)
        { // @type_name
            char *s = codegen_type_to_string(t);
            fprintf(out, "\"%s\"", s);
            free(s);
        }
        else
        { // @fields
            if (t->kind != TYPE_STRUCT || !t->name)
            {
                fprintf(out, "((void*)0)");
                break;
            }
            char *sname = t->name;
            // Find definition
            ASTNode *def = find_struct_def(ctx, sname);
            if (!def)
            {
                fprintf(out, "((void*)0)");
                break;
            }

            fprintf(out,
                    "({ static struct { char *name; char *type; unsigned long offset; } "
                    "_fields_%s[] = {",
                    sname);
            ASTNode *f = def->strct.fields;
            while (f)
            {
                if (f->type == NODE_FIELD)
                {
                    fprintf(out, "{ \"%s\", \"%s\", __builtin_offsetof(struct %s, %s) }, ",
                            f->field.name, f->field.type, sname, f->field.name);
                }
                f = f->next;
            }
            fprintf(out, "{ 0 } }; (void*)_fields_%s; })", sname);
        }
        break;
    }
    case NODE_EXPR_STRUCT_INIT:
    {
        const char *struct_name = node->struct_init.struct_name;
        if (strcmp(struct_name, "Self") == 0 && g_current_impl_type)
        {
            struct_name = g_current_impl_type;
        }

        int is_zen_struct = 0;
        StructRef *sr = ctx->parsed_structs_list;
        while (sr)
        {
            if (sr->node && sr->node->type == NODE_STRUCT &&
                strcmp(sr->node->strct.name, struct_name) == 0)
            {
                is_zen_struct = 1;
                break;
            }
            sr = sr->next;
        }

        if (is_zen_struct)
        {
            fprintf(out, "(struct %s){", struct_name);
        }
        else
        {
            fprintf(out, "(%s){", struct_name);
        }
        ASTNode *f = node->struct_init.fields;
        while (f)
        {
            fprintf(out, ".%s = ", f->var_decl.name);
            codegen_expression(ctx, f->var_decl.init_expr, out);
            if (f->next)
            {
                fprintf(out, ", ");
            }
            f = f->next;
        }
        fprintf(out, "}");
        break;
    }
    case NODE_EXPR_ARRAY_LITERAL:
        fprintf(out, "{");
        ASTNode *elem = node->array_literal.elements;
        int first = 1;
        while (elem)
        {
            if (!first)
            {
                fprintf(out, ", ");
            }
            codegen_expression(ctx, elem, out);
            elem = elem->next;
            first = 0;
        }
        fprintf(out, "}");
        break;
    case NODE_TERNARY:
        fprintf(out, "((");
        codegen_expression(ctx, node->ternary.cond, out);
        fprintf(out, ") ? (");
        codegen_expression(ctx, node->ternary.true_expr, out);
        fprintf(out, ") : (");
        codegen_expression(ctx, node->ternary.false_expr, out);
        fprintf(out, "))");
        break;
    case NODE_AWAIT:
    {
        char *ret_type = "void*";
        int free_ret = 0;
        if (node->type_info)
        {
            char *t = codegen_type_to_string(node->type_info);
            if (t)
            {
                ret_type = t;
                free_ret = 1;
            }
        }
        else if (node->resolved_type)
        {
            ret_type = node->resolved_type;
        }

        if (strcmp(ret_type, "Async") == 0 || strcmp(ret_type, "void*") == 0)
        {
            char *inf = infer_type(ctx, node);
            if (inf && strcmp(inf, "Async") != 0 && strcmp(inf, "void*") != 0)
            {
                if (free_ret)
                {
                    free(ret_type);
                }
                ret_type = inf;
                free_ret = 0;
            }
        }

        int needs_long_cast = 0;
        int returns_struct = 0;
        if (strstr(ret_type, "*") == NULL && strcmp(ret_type, "string") != 0 &&
            strcmp(ret_type, "void") != 0 && strcmp(ret_type, "Async") != 0)
        {
            if (strcmp(ret_type, "int") != 0 && strcmp(ret_type, "bool") != 0 &&
                strcmp(ret_type, "char") != 0 && strcmp(ret_type, "float") != 0 &&
                strcmp(ret_type, "double") != 0 && strcmp(ret_type, "long") != 0 &&
                strcmp(ret_type, "usize") != 0 && strcmp(ret_type, "isize") != 0 &&
                strncmp(ret_type, "uint", 4) != 0 && strncmp(ret_type, "int", 3) != 0)
            {
                returns_struct = 1;
            }
            else
            {
                needs_long_cast = 1;
            }
            if (strncmp(ret_type, "struct", 6) == 0)
            {
                returns_struct = 1;
            }
        }

        fprintf(out, "({ Async _a = ");
        codegen_expression(ctx, node->unary.operand, out);
        fprintf(out, "; void* _r; pthread_join(_a.thread, &_r); ");
        if (strcmp(ret_type, "void") == 0)
        {
            fprintf(out, "})");
        }
        else
        {
            if (returns_struct)
            {
                fprintf(out, "%s _val = *(%s*)_r; free(_r); _val; })", ret_type, ret_type);
            }
            else
            {
                if (needs_long_cast)
                {
                    fprintf(out, "(%s)(long)_r; })", ret_type);
                }
                else
                {
                    fprintf(out, "(%s)_r; })", ret_type);
                }
            }
        }
        if (free_ret)
        {
            free(ret_type);
        }
        break;
    }
    default:
        break;
    }
}
