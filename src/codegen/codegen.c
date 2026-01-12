
#include "codegen.h"
#include "zprep.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../plugins/plugin_manager.h"
#include "ast.h"
#include "zprep_plugin.h"

// static function for internal use.
static char *g_current_func_ret_type = NULL;
static void codegen_match_internal(ParserContext *ctx, ASTNode *node, FILE *out, int use_result)
{
    int id = tmp_counter++;
    int is_self = (node->match_stmt.expr->type == NODE_EXPR_VAR &&
                   strcmp(node->match_stmt.expr->var_ref.name, "self") == 0);

    char *ret_type = infer_type(ctx, node);
    int is_expr = (use_result && ret_type && strcmp(ret_type, "void") != 0);

    fprintf(out, "({ ");
    emit_auto_type(ctx, node->match_stmt.expr, node->token, out);
    fprintf(out, " _m_%d = ", id);
    if (is_self)
    {
        fprintf(out, "*(");
    }
    codegen_expression(ctx, node->match_stmt.expr, out);
    if (is_self)
    {
        fprintf(out, ")");
    }
    fprintf(out, "; ");

    if (is_expr)
    {
        fprintf(out, "%s _r_%d; ", ret_type, id);
    }

    char *expr_type = infer_type(ctx, node->match_stmt.expr);
    int is_option = (expr_type && strncmp(expr_type, "Option_", 7) == 0);
    int is_result = (expr_type && strncmp(expr_type, "Result_", 7) == 0);

    char *enum_name = NULL;
    ASTNode *chk = node->match_stmt.cases;
    int has_wildcard = 0;
    while (chk)
    {
        if (strcmp(chk->match_case.pattern, "_") == 0)
        {
            has_wildcard = 1;
        }
        else if (!enum_name)
        {
            EnumVariantReg *reg = find_enum_variant(ctx, chk->match_case.pattern);
            if (reg)
            {
                enum_name = reg->enum_name;
            }
        }
        chk = chk->next;
    }

    if (enum_name && !has_wildcard)
    {
        // Iterate through all registered variants for this enum
        EnumVariantReg *v = ctx->enum_variants;
        while (v)
        {
            if (strcmp(v->enum_name, enum_name) == 0)
            {
                int covered = 0;
                ASTNode *c2 = node->match_stmt.cases;
                while (c2)
                {
                    if (strcmp(c2->match_case.pattern, v->variant_name) == 0)
                    {
                        covered = 1;
                        break;
                    }
                    c2 = c2->next;
                }
                if (!covered)
                {
                    zwarn_at(node->token, "Non-exhaustive match: Missing variant '%s'",
                             v->variant_name);
                }
            }
            v = v->next;
        }
    }

    ASTNode *c = node->match_stmt.cases;
    int first = 1;
    while (c)
    {
        if (!first)
        {
            fprintf(out, " else ");
        }
        fprintf(out, "if (");
        if (strcmp(c->match_case.pattern, "_") == 0)
        {
            fprintf(out, "1");
        }
        else if (is_option)
        {
            if (strcmp(c->match_case.pattern, "Some") == 0)
            {
                fprintf(out, "_m_%d.is_some", id);
            }
            else if (strcmp(c->match_case.pattern, "None") == 0)
            {
                fprintf(out, "!_m_%d.is_some", id);
            }
            else
            {
                fprintf(out, "1");
            }
        }
        else if (is_result)
        {
            if (strcmp(c->match_case.pattern, "Ok") == 0)
            {
                fprintf(out, "_m_%d.is_ok", id);
            }
            else if (strcmp(c->match_case.pattern, "Err") == 0)
            {
                fprintf(out, "!_m_%d.is_ok", id);
            }
            else
            {
                fprintf(out, "1");
            }
        }
        else
        {
            EnumVariantReg *reg = find_enum_variant(ctx, c->match_case.pattern);
            if (reg)
            {
                fprintf(out, "_m_%d.tag == %d", id, reg->tag_id);
            }
            else if (c->match_case.pattern[0] == '"')
            {
                fprintf(out, "strcmp(_m_%d, %s) == 0", id, c->match_case.pattern);
            }
            else if (isdigit(c->match_case.pattern[0]) || c->match_case.pattern[0] == '-')
            {
                // Numeric pattern
                fprintf(out, "_m_%d == %s", id, c->match_case.pattern);
            }
            else if (c->match_case.pattern[0] == '\'')
            {
                // Char literal pattern
                fprintf(out, "_m_%d == %s", id, c->match_case.pattern);
            }
            else
            {
                fprintf(out, "1");
            }
        }
        fprintf(out, ") { ");
        if (c->match_case.binding_name)
        {
            if (is_option)
            {
                if (strstr(g_config.cc, "tcc"))
                {
                    fprintf(out, "__typeof__(_m_%d.val) %s = _m_%d.val; ", id,
                            c->match_case.binding_name, id);
                }
                else
                {
                    fprintf(out, "__auto_type %s = _m_%d.val; ", c->match_case.binding_name, id);
                }
            }
            if (is_result)
            {
                if (strcmp(c->match_case.pattern, "Ok") == 0)
                {
                    if (strstr(g_config.cc, "tcc"))
                    {
                        fprintf(out, "__typeof__(_m_%d.val) %s = _m_%d.val; ", id,
                                c->match_case.binding_name, id);
                    }
                    else
                    {
                        fprintf(out, "__auto_type %s = _m_%d.val; ", c->match_case.binding_name,
                                id);
                    }
                }
                else
                {
                    if (strstr(g_config.cc, "tcc"))
                    {
                        fprintf(out, "__typeof__(_m_%d.err) %s = _m_%d.err; ", id,
                                c->match_case.binding_name, id);
                    }
                    else
                    {
                        fprintf(out, "__auto_type %s = _m_%d.err; ", c->match_case.binding_name,
                                id);
                    }
                }
            }
            else
            {
                char *f = strrchr(c->match_case.pattern, '_');
                if (f)
                {
                    f++;
                }
                else
                {
                    f = c->match_case.pattern;
                }
                fprintf(out, "__auto_type %s = _m_%d.data.%s; ", c->match_case.binding_name, id, f);
            }
        }

        // Check if body is a string literal (should auto-print).
        ASTNode *body = c->match_case.body;
        int is_string_literal = (body->type == NODE_EXPR_LITERAL && body->literal.type_kind == 2);

        if (is_expr)
        {
            fprintf(out, "_r_%d = ", id);
            if (is_string_literal)
            {
                codegen_node_single(ctx, body, out);
            }
            else
            {
                if (body->type == NODE_BLOCK)
                {
                    int saved = defer_count;
                    fprintf(out, "({ ");
                    ASTNode *stmt = body->block.statements;
                    while (stmt)
                    {
                        codegen_node_single(ctx, stmt, out);
                        stmt = stmt->next;
                    }
                    for (int i = defer_count - 1; i >= saved; i--)
                    {
                        codegen_node_single(ctx, defer_stack[i], out);
                    }
                    defer_count = saved;
                    fprintf(out, " })");
                }
                else
                {
                    codegen_node_single(ctx, body, out);
                }
            }
            fprintf(out, ";");
        }
        else
        {
            if (is_string_literal)
            {
                fprintf(out, "({ printf(\"%%s\", ");
                codegen_expression(ctx, body, out);
                fprintf(out, "); printf(\"\\n\"); 0; })");
            }
            else
            {
                codegen_node_single(ctx, body, out);
            }
        }

        fprintf(out, " }");
        first = 0;
        c = c->next;
    }

    if (is_expr)
    {
        fprintf(out, " _r_%d; })", id);
    }
    else
    {
        fprintf(out, " })");
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

            int is_basic = 0;
            if (t1)
            {
                is_basic = (strcmp(t1, "int") == 0 || strcmp(t1, "bool") == 0 ||
                            strcmp(t1, "char") == 0 || strcmp(t1, "void") == 0 ||
                            strcmp(t1, "float") == 0 || strcmp(t1, "double") == 0 ||
                            strcmp(t1, "usize") == 0 || strcmp(t1, "size_t") == 0 ||
                            strcmp(t1, "ssize_t") == 0 || strcmp(t1, "__auto_type") == 0);
            }

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
                fprintf(out, "%s_eq(&", base);
                codegen_expression(ctx, node->binary.left, out);
                fprintf(out, ", ");
                codegen_expression(ctx, node->binary.right, out);
                fprintf(out, ")");
                if (strcmp(node->binary.op, "!=") == 0)
                {
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
        break;
    case NODE_LAMBDA:
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
            fprintf(out, "(z_closure_T){.func = _lambda_%d, .ctx = ctx}; })",
                    node->lambda.lambda_id);
        }
        else
        {
            fprintf(out, "((z_closure_T){.func = (void*)_lambda_%d, .ctx = NULL})",
                    node->lambda.lambda_id);
        }
        break;
    case NODE_EXPR_LITERAL:
        if (node->literal.type_kind == TOK_STRING)
        {
            fprintf(out, "\"%s\"", node->literal.string_val);
        }
        else if (node->literal.type_kind == TOK_CHAR)
        {
            fprintf(out, "%s", node->literal.string_val);
        }
        else if (node->literal.type_kind == 1)
        {
            fprintf(out, "%f", node->literal.float_val);
        }

        else
        {
            fprintf(out, "%d", node->literal.int_val);
        }
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
                    fprintf(out, "; %s_%s(&_t", base, method);
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
                    fprintf(out, "%s_%s(", base, method);
                    if (!strchr(type, '*'))
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
            char *ret = type_to_string(ft->inner);
            if (strcmp(ret, "string") == 0)
            {
                free(ret);
                ret = xstrdup("char*");
            }

            fprintf(out, "((%s (*)(void*", ret);
            for (int i = 0; i < ft->arg_count; i++)
            {
                char *as = type_to_string(ft->args[i]);
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
            ASTNode *arg = node->call.args;
            while (arg)
            {
                codegen_expression(ctx, arg, out);
                if (arg->next)
                {
                    fprintf(out, ", ");
                }
                arg = arg->next;
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
            fprintf(out, "); _t ? _t->%s : 0; })", node->member.field);
        }
        else
        {
            codegen_expression(ctx, node->member.target, out);
            fprintf(out, "%s%s", node->member.is_pointer_access ? "->" : ".", node->member.field);
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
            tname = type_to_string(node->type_info->inner);
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
                    "; if (!_try.is_ok) return %s_Err(_try.err); "
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
            char *s = type_to_string(t);
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
        fprintf(out, "(struct %s){", struct_name);
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
            char *t = type_to_string(node->type_info);
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
        fprintf(out, "; void* _r; z_thread_join(_a.thread, &_r); ");
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

void codegen_node_single(ParserContext *ctx, ASTNode *node, FILE *out)
{
    if (!node)
    {
        return;
    }
    switch (node->type)
    {
    case NODE_MATCH:
        codegen_match_internal(ctx, node, out, 0); // 0 = statement context
        fprintf(out, ";\n");
        break;
    case NODE_FUNCTION:
        if (!node->func.body)
        {
            break;
        }

        if (node->func.is_async)
        {
            fprintf(out, "struct %s_Args {\n", node->func.name);
            char *args_copy = xstrdup(node->func.args);
            char *token = strtok(args_copy, ",");
            int arg_count = 0;
            char **arg_names = xmalloc(32 * sizeof(char *));

            while (token)
            {
                while (*token == ' ')
                {
                    token++; // trim leading
                }
                char *last_space = strrchr(token, ' ');
                if (last_space)
                {
                    *last_space = 0;
                    char *type = token;
                    char *name = last_space + 1;
                    fprintf(out, "%s %s;\n", type, name);

                    arg_names[arg_count++] = xstrdup(name);
                }
                token = strtok(NULL, ",");
            }
            free(args_copy);
            fprintf(out, "};\n");

            fprintf(out, "void* _runner_%s(void* _args)\n", node->func.name);
            fprintf(out, "{\n");
            fprintf(out, "    struct %s_Args* args = (struct %s_Args*)_args;\n", node->func.name,
                    node->func.name);

            // Determine mechanism: struct/large-type? -> malloc; primitive -> cast
            int returns_struct = 0;
            char *rt = node->func.ret_type;
            if (strcmp(rt, "void") != 0 && strcmp(rt, "Async") != 0)
            {
                if (strstr(rt, "*") == NULL && strcmp(rt, "string") != 0 &&
                    strcmp(rt, "int") != 0 && strcmp(rt, "bool") != 0 && strcmp(rt, "char") != 0 &&
                    strcmp(rt, "float") != 0 && strcmp(rt, "double") != 0 &&
                    strcmp(rt, "long") != 0 && strcmp(rt, "usize") != 0 &&
                    strcmp(rt, "isize") != 0 && strncmp(rt, "uint", 4) != 0 &&
                    strncmp(rt, "int", 3) != 0)
                {
                    returns_struct = 1;
                }
            }

            // Call Impl
            if (returns_struct)
            {
                fprintf(out, "    %s *res_ptr = malloc(sizeof(%s));\n", rt, rt);
                fprintf(out, "    *res_ptr = ");
            }
            else if (strcmp(rt, "void") != 0 && strcmp(rt, "Async") != 0)
            {
                fprintf(out, "    %s res = ", rt);
            }
            else
            {
                fprintf(out, "    ");
            }

            fprintf(out, "_impl_%s(", node->func.name);
            for (int i = 0; i < arg_count; i++)
            {
                fprintf(out, "%sargs->%s", i > 0 ? ", " : "", arg_names[i]);
            }
            fprintf(out, ");\n");
            fprintf(out, "    free(args);\n");

            if (returns_struct)
            {
                fprintf(out, "    return (void*)res_ptr;\n");
            }
            else if (strcmp(rt, "void") != 0)
            {
                fprintf(out, "    return (void*)(long)res;\n");
            }
            else
            {
                fprintf(out, "    return NULL;\n");
            }
            fprintf(out, "}\n");

            fprintf(out, "%s _impl_%s(%s)\n", node->func.ret_type, node->func.name,
                    node->func.args);
            fprintf(out, "{\n");
            defer_count = 0;
            codegen_walker(ctx, node->func.body, out);
            for (int i = defer_count - 1; i >= 0; i--)
            {
                codegen_node_single(ctx, defer_stack[i], out);
            }
            fprintf(out, "}\n");

            // 4. Define Public Wrapper (Spawns Thread)
            fprintf(out, "Async %s(%s)\n", node->func.name, node->func.args);
            fprintf(out, "{\n");
            fprintf(out, "    struct %s_Args* args = malloc(sizeof(struct %s_Args));\n",
                    node->func.name, node->func.name);
            for (int i = 0; i < arg_count; i++)
            {
                fprintf(out, "    args->%s = %s;\n", arg_names[i], arg_names[i]);
            }

            fprintf(out, "    z_thread_t th;\n");
            fprintf(out, "    z_thread_create(&th, _runner_%s, args);\n", node->func.name);
            fprintf(out, "    return (Async){.thread=th, .result=NULL};\n");
            fprintf(out, "}\n");

            break;
        }

        defer_count = 0;
        fprintf(out, "\n");

        // Emit GCC attributes before function
        {
            int has_attrs = node->func.constructor || node->func.destructor ||
                            node->func.noinline || node->func.unused || node->func.weak ||
                            node->func.cold || node->func.hot || node->func.noreturn ||
                            node->func.pure || node->func.section;
            if (has_attrs)
            {
                fprintf(out, "__attribute__((");
                int first = 1;
#define EMIT_ATTR(cond, name)                                                                      \
    if (cond)                                                                                      \
    {                                                                                              \
        if (!first)                                                                                \
            fprintf(out, ", ");                                                                    \
        fprintf(out, name);                                                                        \
        first = 0;                                                                                 \
    }
                EMIT_ATTR(node->func.constructor, "constructor");
                EMIT_ATTR(node->func.destructor, "destructor");
                EMIT_ATTR(node->func.noinline, "noinline");
                EMIT_ATTR(node->func.unused, "unused");
                EMIT_ATTR(node->func.weak, "weak");
                EMIT_ATTR(node->func.cold, "cold");
                EMIT_ATTR(node->func.hot, "hot");
                EMIT_ATTR(node->func.noreturn, "noreturn");
                EMIT_ATTR(node->func.pure, "pure");
                if (node->func.section)
                {
                    if (!first)
                    {
                        fprintf(out, ", ");
                    }
                    fprintf(out, "section(\"%s\")", node->func.section);
                }
#undef EMIT_ATTR
                fprintf(out, ")) ");
            }
        }

        if (node->func.is_inline)
        {
            fprintf(out, "inline ");
        }
        fprintf(out, "%s %s(%s)\n", node->func.ret_type, node->func.name, node->func.args);
        fprintf(out, "{\n");
        char *prev_ret = g_current_func_ret_type;
        g_current_func_ret_type = node->func.ret_type;
        codegen_walker(ctx, node->func.body, out);
        for (int i = defer_count - 1; i >= 0; i--)
        {
            codegen_node_single(ctx, defer_stack[i], out);
        }
        g_current_func_ret_type = prev_ret;
        fprintf(out, "}\n");
        break;

    case NODE_DEFER:
        if (defer_count < MAX_DEFER)
        {
            defer_stack[defer_count++] = node->defer_stmt.stmt;
        }
        break;
    case NODE_IMPL:
        g_current_impl_type = node->impl.struct_name;
        codegen_walker(ctx, node->impl.methods, out);
        g_current_impl_type = NULL;
        break;
    case NODE_IMPL_TRAIT:
        g_current_impl_type = node->impl_trait.target_type;
        codegen_walker(ctx, node->impl_trait.methods, out);

        if (strcmp(node->impl_trait.trait_name, "Drop") == 0)
        {
            char *tname = node->impl_trait.target_type;
            fprintf(out, "\n// RAII Glue\n");
            fprintf(out, "void %s_Drop_glue(%s *self) {\n", tname, tname);
            fprintf(out, "    %s_Drop_drop(self);\n", tname);
            fprintf(out, "}\n");
        }
        g_current_impl_type = NULL;
        break;
    case NODE_DESTRUCT_VAR:
    {
        int id = tmp_counter++;
        fprintf(out, "    ");
        emit_auto_type(ctx, node->destruct.init_expr, node->token, out);
        fprintf(out, " _tmp_%d = ", id);
        codegen_expression(ctx, node->destruct.init_expr, out);
        fprintf(out, ";\n");

        if (node->destruct.is_guard)
        {
            // var Some(val) = opt else ...
            char *variant = node->destruct.guard_variant;
            char *check = "val"; // field to access

            if (strcmp(variant, "Some") == 0)
            {
                fprintf(out, "    if (!_tmp_%d.is_some) {\n", id);
            }
            else if (strcmp(variant, "Ok") == 0)
            {
                fprintf(out, "    if (!_tmp_%d.is_ok) {\n", id);
            }
            else if (strcmp(variant, "Err") == 0)
            {
                fprintf(out, "    if (_tmp_%d.is_ok) {\n", id); // Err if NOT ok
                check = "err";
            }
            else
            {
                // Generic guard? Assume .is_variant present?
                fprintf(out, "    if (!_tmp_%d.is_%s) {\n", id, variant);
            }

            // Else block
            codegen_walker(ctx, node->destruct.else_block->block.statements, out);
            fprintf(out, "    }\n");

            // Bind value
            if (strstr(g_config.cc, "tcc"))
            {
                fprintf(out, "    __typeof__(_tmp_%d.%s) %s = _tmp_%d.%s;\n", id, check,
                        node->destruct.names[0], id, check);
            }
            else
            {
                fprintf(out, "    __auto_type %s = _tmp_%d.%s;\n", node->destruct.names[0], id,
                        check);
            }
        }
        else
        {
            for (int i = 0; i < node->destruct.count; i++)
            {
                if (node->destruct.is_struct_destruct)
                {
                    char *field = node->destruct.field_names ? node->destruct.field_names[i]
                                                             : node->destruct.names[i];
                    if (strstr(g_config.cc, "tcc"))
                    {
                        fprintf(out, "    __typeof__(_tmp_%d.%s) %s = _tmp_%d.%s;\n", id, field,
                                node->destruct.names[i], id, field);
                    }
                    else
                    {
                        fprintf(out, "    __auto_type %s = _tmp_%d.%s;\n", node->destruct.names[i],
                                id, field);
                    }
                }
                else
                {
                    if (strstr(g_config.cc, "tcc"))
                    {
                        fprintf(out, "    __typeof__(_tmp_%d.v%d) %s = _tmp_%d.v%d;\n", id, i,
                                node->destruct.names[i], id, i);
                    }
                    else
                    {
                        fprintf(out, "    __auto_type %s = _tmp_%d.v%d;\n", node->destruct.names[i],
                                id, i);
                    }
                }
            }
        }
        break;
    }
    case NODE_BLOCK:
    {
        int saved = defer_count;
        fprintf(out, "    {\n");
        codegen_walker(ctx, node->block.statements, out);
        for (int i = defer_count - 1; i >= saved; i--)
        {
            codegen_node_single(ctx, defer_stack[i], out);
        }
        defer_count = saved;
        fprintf(out, "    }\n");
        break;
    }
    case NODE_VAR_DECL:
        fprintf(out, "    ");
        if (node->var_decl.is_static)
        {
            fprintf(out, "static ");
        }
        if (node->var_decl.is_autofree)
        {
            fprintf(out, "__attribute__((cleanup(_z_autofree_impl))) ");
        }
        {
            char *tname = NULL;
            Type *tinfo = node->var_decl.type_info;
            if (tinfo && tinfo->name)
            {
                tname = tinfo->name;
            }
            else if (node->var_decl.type_str && strcmp(node->var_decl.type_str, "__auto_type") != 0)
            {
                tname = node->var_decl.type_str;
            }

            if (tname)
            {
                ASTNode *def = find_struct_def(ctx, tname);
                if (def && def->type_info && def->type_info->has_drop)
                {
                    fprintf(out, "__attribute__((cleanup(%s_Drop_glue))) ", tname);
                }
            }
        }
        if (node->var_decl.type_str && strcmp(node->var_decl.type_str, "__auto_type") != 0)
        {
            emit_var_decl_type(ctx, out, node->var_decl.type_str, node->var_decl.name);
            add_symbol(ctx, node->var_decl.name, node->var_decl.type_str, node->var_decl.type_info);
            if (node->var_decl.init_expr)
            {
                fprintf(out, " = ");
                codegen_expression(ctx, node->var_decl.init_expr, out);
            }
            fprintf(out, ";\n");
        }
        else
        {
            char *inferred = NULL;
            if (node->var_decl.init_expr)
            {
                inferred = infer_type(ctx, node->var_decl.init_expr);
            }

            if (inferred && strcmp(inferred, "__auto_type") != 0)
            {
                emit_var_decl_type(ctx, out, inferred, node->var_decl.name);
                add_symbol(ctx, node->var_decl.name, inferred, NULL);
            }
            else
            {
                emit_auto_type(ctx, node->var_decl.init_expr, node->token, out);
                fprintf(out, " %s", node->var_decl.name);

                if (inferred)
                {
                    add_symbol(ctx, node->var_decl.name, inferred, NULL);
                }
                else
                {
                    // Here we are cooked.
                }
            }

            fprintf(out, " = ");
            codegen_expression(ctx, node->var_decl.init_expr, out);
            fprintf(out, ";\n");
        }
        break;
    case NODE_CONST:
        fprintf(out, "    const ");
        if (node->var_decl.type_str)
        {
            fprintf(out, "%s %s", node->var_decl.type_str, node->var_decl.name);
        }
        else
        {
            emit_auto_type(ctx, node->var_decl.init_expr, node->token, out);
            fprintf(out, " %s", node->var_decl.name);
        }
        fprintf(out, " = ");
        codegen_expression(ctx, node->var_decl.init_expr, out);
        fprintf(out, ";\n");
        break;
    case NODE_FIELD:
        if (node->field.bit_width > 0)
        {
            fprintf(out, "    %s %s : %d;\n", node->field.type, node->field.name,
                    node->field.bit_width);
        }
        else
        {
            fprintf(out, "    ");
            emit_var_decl_type(ctx, out, node->field.type, node->field.name);
            fprintf(out, ";\n");
        }
        break;
    case NODE_IF:
        fprintf(out, "if (");
        codegen_expression(ctx, node->if_stmt.condition, out);
        fprintf(out, ") ");
        codegen_node_single(ctx, node->if_stmt.then_body, out);
        if (node->if_stmt.else_body)
        {
            fprintf(out, " else ");
            codegen_node_single(ctx, node->if_stmt.else_body, out);
        }
        break;
    case NODE_UNLESS:
        fprintf(out, "if (!(");
        codegen_expression(ctx, node->unless_stmt.condition, out);
        fprintf(out, ")) ");
        codegen_node_single(ctx, node->unless_stmt.body, out);
        break;
    case NODE_GUARD:
        fprintf(out, "if (!(");
        codegen_expression(ctx, node->guard_stmt.condition, out);
        fprintf(out, ")) ");
        codegen_node_single(ctx, node->guard_stmt.body, out);
        break;
    case NODE_WHILE:
        fprintf(out, "while (");
        codegen_expression(ctx, node->while_stmt.condition, out);
        fprintf(out, ") ");
        codegen_node_single(ctx, node->while_stmt.body, out);
        break;
    case NODE_FOR:
        fprintf(out, "for (");
        if (node->for_stmt.init)
        {
            if (node->for_stmt.init->type == NODE_VAR_DECL)
            {
                ASTNode *v = node->for_stmt.init;
                if (v->var_decl.type_str && strcmp(v->var_decl.type_str, "__auto_type") != 0)
                {
                    fprintf(out, "%s %s = (%s)(", v->var_decl.type_str, v->var_decl.name,
                            v->var_decl.type_str);
                    codegen_expression(ctx, v->var_decl.init_expr, out);
                    fprintf(out, ")");
                }
                else
                {
                    emit_auto_type(ctx, v->var_decl.init_expr, v->token, out);
                    fprintf(out, " %s = ", v->var_decl.name);
                    codegen_expression(ctx, v->var_decl.init_expr, out);
                }
            }
            else
            {
                codegen_expression(ctx, node->for_stmt.init, out);
            }
        }
        fprintf(out, "; ");
        if (node->for_stmt.condition)
        {
            codegen_expression(ctx, node->for_stmt.condition, out);
        }
        fprintf(out, "; ");
        if (node->for_stmt.step)
        {
            codegen_expression(ctx, node->for_stmt.step, out);
        }
        fprintf(out, ") ");
        codegen_node_single(ctx, node->for_stmt.body, out);
        break;
    case NODE_BREAK:
        if (node->break_stmt.target_label)
        {
            fprintf(out, "goto __break_%s;\n", node->break_stmt.target_label);
        }
        else
        {
            fprintf(out, "break;\n");
        }
        break;
    case NODE_CONTINUE:
        if (node->continue_stmt.target_label)
        {
            fprintf(out, "goto __continue_%s;\n", node->continue_stmt.target_label);
        }
        else
        {
            fprintf(out, "continue;\n");
        }
        break;
    case NODE_GOTO:
        if (node->goto_stmt.goto_expr)
        {
            // Computed goto: goto *expr;
            fprintf(out, "goto *(");
            codegen_expression(ctx, node->goto_stmt.goto_expr, out);
            fprintf(out, ");\n");
        }
        else
        {
            fprintf(out, "goto %s;\n", node->goto_stmt.label_name);
        }
        break;
    case NODE_LABEL:
        fprintf(out, "%s:;\n", node->label_stmt.label_name);
        break;
    case NODE_DO_WHILE:
        fprintf(out, "do ");
        codegen_node_single(ctx, node->do_while_stmt.body, out);
        fprintf(out, " while (");
        codegen_expression(ctx, node->do_while_stmt.condition, out);
        fprintf(out, ");\n");
        break;
    // Loop constructs: loop, repeat, for-in
    case NODE_LOOP:
        // loop { ... } => while (1) { ... }
        fprintf(out, "while (1) ");
        codegen_node_single(ctx, node->loop_stmt.body, out);
        break;
    case NODE_REPEAT:
        fprintf(out, "for (int _rpt_i = 0; _rpt_i < (%s); _rpt_i++) ", node->repeat_stmt.count);
        codegen_node_single(ctx, node->repeat_stmt.body, out);
        break;
    case NODE_FOR_RANGE:
        fprintf(out, "for (");
        if (strstr(g_config.cc, "tcc"))
        {
            fprintf(out, "__typeof__((");
            codegen_expression(ctx, node->for_range.start, out);
            fprintf(out, ")) %s = ", node->for_range.var_name);
        }
        else
        {
            fprintf(out, "__auto_type %s = ", node->for_range.var_name);
        }
        codegen_expression(ctx, node->for_range.start, out);
        fprintf(out, "; %s < ", node->for_range.var_name);
        codegen_expression(ctx, node->for_range.end, out);
        fprintf(out, "; %s", node->for_range.var_name);
        if (node->for_range.step)
        {
            fprintf(out, " += %s) ", node->for_range.step);
        }
        else
        {
            fprintf(out, "++) ");
        }
        codegen_node_single(ctx, node->for_range.body, out);
        break;
    case NODE_ASM:
    {
        int is_extended = (node->asm_stmt.num_outputs > 0 || node->asm_stmt.num_inputs > 0 ||
                           node->asm_stmt.num_clobbers > 0);

        if (node->asm_stmt.is_volatile)
        {
            fprintf(out, "    __asm__ __volatile__(");
        }
        else
        {
            fprintf(out, "    __asm__(");
        }

        char *code = node->asm_stmt.code;
        char *transformed = xmalloc(strlen(code) * 3); // Generous buffer
        char *dst = transformed;

        for (char *p = code; *p; p++)
        {
            if (*p == '{')
            {
                // Find matching }
                char *end = strchr(p + 1, '}');
                if (end)
                {
                    // Extract variable name
                    int var_len = end - p - 1;
                    char var_name[64];
                    strncpy(var_name, p + 1, var_len);
                    var_name[var_len] = 0;

                    // Find variable index
                    int idx = -1;

                    // Check outputs first
                    for (int i = 0; i < node->asm_stmt.num_outputs; i++)
                    {
                        if (strcmp(node->asm_stmt.outputs[i], var_name) == 0)
                        {
                            idx = i;
                            break;
                        }
                    }

                    // Then check inputs
                    if (idx == -1)
                    {
                        for (int i = 0; i < node->asm_stmt.num_inputs; i++)
                        {
                            if (strcmp(node->asm_stmt.inputs[i], var_name) == 0)
                            {
                                idx = node->asm_stmt.num_outputs + i;
                                break;
                            }
                        }
                    }

                    if (idx >= 0)
                    {
                        // Replace with %N
                        dst += sprintf(dst, "%%%d", idx);
                    }
                    else
                    {
                        // Variable not found - error or keep as-is?
                        dst += sprintf(dst, "{%s}", var_name);
                    }

                    p = end; // Skip past }
                }
                else
                {
                    *dst++ = *p;
                }
            }
            else if (*p == '%')
            {
                if (is_extended)
                {
                    *dst++ = '%';
                    *dst++ = '%';
                }
                else
                {
                    *dst++ = '%';
                }
            }
            else
            {
                *dst++ = *p;
            }
        }
        *dst = 0;

        fprintf(out, "\"");
        for (char *p = transformed; *p; p++)
        {
            if (*p == '\n')
            {
                fprintf(out, "\\n\"\n        \"");
            }
            else if (*p == '"')
            {
                fprintf(out, "\\\"");
            }
            else if (*p == '\\')
            {
                fprintf(out, "\\\\");
            }
            else
            {
                fputc(*p, out);
            }
        }
        fprintf(out, "\\n\"");

        if (node->asm_stmt.num_outputs > 0)
        {
            fprintf(out, "\n        : ");
            for (int i = 0; i < node->asm_stmt.num_outputs; i++)
            {
                if (i > 0)
                {
                    fprintf(out, ", ");
                }

                // Determine constraint
                char *mode = node->asm_stmt.output_modes[i];
                if (strcmp(mode, "out") == 0)
                {
                    fprintf(out, "\"=r\"(%s)", node->asm_stmt.outputs[i]);
                }
                else if (strcmp(mode, "inout") == 0)
                {
                    fprintf(out, "\"+r\"(%s)", node->asm_stmt.outputs[i]);
                }
                else
                {
                    fprintf(out, "\"=r\"(%s)", node->asm_stmt.outputs[i]);
                }
            }
        }

        if (node->asm_stmt.num_inputs > 0)
        {
            fprintf(out, "\n        : ");
            for (int i = 0; i < node->asm_stmt.num_inputs; i++)
            {
                if (i > 0)
                {
                    fprintf(out, ", ");
                }
                fprintf(out, "\"r\"(%s)", node->asm_stmt.inputs[i]);
            }
        }
        else if (node->asm_stmt.num_outputs > 0)
        {
            fprintf(out, "\n        : ");
        }

        if (node->asm_stmt.num_clobbers > 0)
        {
            fprintf(out, "\n        : ");
            for (int i = 0; i < node->asm_stmt.num_clobbers; i++)
            {
                if (i > 0)
                {
                    fprintf(out, ", ");
                }
                fprintf(out, "\"%s\"", node->asm_stmt.clobbers[i]);
            }
        }

        fprintf(out, ");\n");
        break;
    }
    case NODE_RETURN:
        fprintf(out, "    return ");
        codegen_expression(ctx, node->ret.value, out);
        fprintf(out, ";\n");
        break;
    case NODE_EXPR_MEMBER:
    {
        codegen_expression(ctx, node->member.target, out);
        char *lt = infer_type(ctx, node->member.target);
        if (lt && (lt[strlen(lt) - 1] == '*' || strstr(lt, "*")))
        {
            fprintf(out, "->%s", node->member.field);
        }
        else
        {
            fprintf(out, ".%s", node->member.field);
        }
        if (lt)
        {
            free(lt);
        }
        break;
    }
    case NODE_REPL_PRINT:
    {
        fprintf(out, "{ ");
        emit_auto_type(ctx, node->repl_print.expr, node->token, out);
        fprintf(out, " _zval = (");
        codegen_expression(ctx, node->repl_print.expr, out);
        fprintf(out, "); fprintf(stdout, _z_str(_zval), _zval); fprintf(stdout, "
                     "\"\\n\"); }\n");
        break;
    }
    case NODE_AWAIT:
    {
        char *ret_type = "void*";
        int free_ret = 0;
        if (node->type_info)
        {
            char *t = type_to_string(node->type_info);
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

        // Fallback: If type is still Async/void* (likely from Future type, not
        // Result type), try to infer
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
                free_ret = 0; // infer_type ownership ambiguous, avoid double free
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
        fprintf(out, "; void* _r; z_thread_join(_a.thread, &_r); ");
        if (strcmp(ret_type, "void") == 0)
        {
            fprintf(out, "})"); // result unused
        }
        else
        {
            if (returns_struct)
            {
                // Dereference and free
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
        fprintf(out, ";\n"); // Statement terminator
        break;
    }
    default:
        codegen_expression(ctx, node, out);
        fprintf(out, ";\n");
        break;
    }
}

// Walks AST nodes and generates code.
void codegen_walker(ParserContext *ctx, ASTNode *node, FILE *out)
{
    while (node)
    {
        codegen_node_single(ctx, node, out);
        node = node->next;
    }
}
