
#include "../ast/ast.h"
#include "../parser/parser.h"
#include "../zprep.h"
#include "codegen.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Global state
ASTNode *global_user_structs = NULL;
char *g_current_impl_type = NULL;
int tmp_counter = 0;
ASTNode *defer_stack[MAX_DEFER];
int defer_count = 0;
ASTNode *g_current_lambda = NULL;

// Helper to emit variable declarations with array types.
void emit_var_decl_type(ParserContext *ctx, FILE *out, const char *type_str, const char *var_name)
{
    (void)ctx;

    char *bracket = strchr(type_str, '[');

    if (bracket)
    {
        int base_len = bracket - type_str;
        fprintf(out, "%.*s %s%s", base_len, type_str, var_name, bracket);
    }
    else
    {
        fprintf(out, "%s %s", type_str, var_name);
    }
}

// Find struct definition
ASTNode *find_struct_def_codegen(ParserContext *ctx, const char *name)
{
    if (!name)
    {
        return NULL;
    }
    ASTNode *s = global_user_structs;
    while (s)
    {
        if (s->type == NODE_STRUCT && strcmp(s->strct.name, name) == 0 && !s->strct.is_incomplete)
        {
            return s;
        }
        s = s->next;
    }

    // Check parsed structs list (imports)-
    StructRef *sr = ctx->parsed_structs_list;
    while (sr)
    {
        if (sr->node && sr->node->type == NODE_STRUCT && strcmp(sr->node->strct.name, name) == 0 &&
            !sr->node->strct.is_incomplete)
        {
            return sr->node;
        }
        sr = sr->next;
    }
    s = ctx->instantiated_structs;
    while (s)
    {
        if (s->type == NODE_STRUCT && strcmp(s->strct.name, name) == 0 && !s->strct.is_incomplete)
        {
            return s;
        }
        s = s->next;
    }
    return NULL;
}

// Get field type from struct.
char *get_field_type_str(ParserContext *ctx, const char *struct_name, const char *field_name)
{
    char clean_name[256];
    strncpy(clean_name, struct_name, sizeof(clean_name) - 1);
    clean_name[sizeof(clean_name) - 1] = 0;

    char *ptr = strchr(clean_name, '<');
    if (ptr)
    {
        *ptr = 0;
    }

    ASTNode *def = find_struct_def_codegen(ctx, clean_name);
    if (!def)
    {
        return NULL;
    }

    ASTNode *f = def->strct.fields;
    while (f)
    {
        if (strcmp(f->field.name, field_name) == 0)
        {
            return f->field.type;
        }
        f = f->next;
    }
    return NULL;
}

// Type inference.
char *infer_type(ParserContext *ctx, ASTNode *node)
{
    if (!node)
    {
        return NULL;
    }
    if (node->resolved_type && strcmp(node->resolved_type, "unknown") != 0)
    {
        return node->resolved_type;
    }

    if (node->type == NODE_EXPR_LITERAL)
    {
        if (node->type_info)
        {
            return type_to_string(node->type_info);
        }
        return NULL;
    }

    if (node->type == NODE_EXPR_VAR)
    {
        Symbol *sym = find_symbol_entry(ctx, node->var_ref.name);
        if (sym)
        {
            if (sym->type_name)
            {
                return sym->type_name;
            }
            if (sym->type_info)
            {
                return codegen_type_to_string(sym->type_info);
            }
        }
    }

    if (node->type == NODE_EXPR_CALL)
    {
        if (node->call.callee->type == NODE_EXPR_VAR)
        {
            FuncSig *sig = find_func(ctx, node->call.callee->var_ref.name);
            if (sig)
            {
                if (sig->is_async)
                {
                    return "Async";
                }
                if (sig->ret_type)
                {
                    return codegen_type_to_string(sig->ret_type);
                }
            }

            // Fallback for known stdlib memory functions.
            if (strcmp(node->call.callee->var_ref.name, "malloc") == 0 ||
                strcmp(node->call.callee->var_ref.name, "calloc") == 0 ||
                strcmp(node->call.callee->var_ref.name, "realloc") == 0)
            {
                return "void*";
            }
            ASTNode *sdef = find_struct_def_codegen(ctx, node->call.callee->var_ref.name);
            if (sdef)
            {
                return node->call.callee->var_ref.name;
            }
        }
        // Method call: target.method() - look up Type_method signature.
        if (node->call.callee->type == NODE_EXPR_MEMBER)
        {
            char *target_type = infer_type(ctx, node->call.callee->member.target);
            if (target_type)
            {
                char clean_type[256];
                strcpy(clean_type, target_type);
                char *ptr = strchr(clean_type, '*');
                if (ptr)
                {
                    *ptr = 0;
                }

                char *base = clean_type;
                if (strncmp(base, "struct ", 7) == 0)
                {
                    base += 7;
                }

                char func_name[512];
                sprintf(func_name, "%s__%s", base, node->call.callee->member.field);

                FuncSig *sig = find_func(ctx, func_name);
                if (sig && sig->ret_type)
                {
                    return codegen_type_to_string(sig->ret_type);
                }
            }
        }

        if (node->call.callee->type == NODE_EXPR_VAR)
        {
            Symbol *sym = find_symbol_entry(ctx, node->call.callee->var_ref.name);
            if (sym && sym->type_info && sym->type_info->kind == TYPE_FUNCTION &&
                sym->type_info->inner)
            {
                return type_to_string(sym->type_info->inner);
            }
        }
    }

    if (node->type == NODE_TRY)
    {
        char *inner_type = infer_type(ctx, node->try_stmt.expr);
        if (inner_type)
        {
            // Extract T from Result<T> or Option<T>
            char *start = strchr(inner_type, '<');
            if (start)
            {
                start++; // Skip <
                char *end = strrchr(inner_type, '>');
                if (end && end > start)
                {
                    int len = end - start;
                    char *extracted = xmalloc(len + 1);
                    strncpy(extracted, start, len);
                    extracted[len] = 0;
                    return extracted;
                }
            }
        }
    }

    if (node->type == NODE_EXPR_MEMBER)
    {
        char *parent_type = infer_type(ctx, node->member.target);
        if (!parent_type)
        {
            return NULL;
        }

        char clean_name[256];
        strcpy(clean_name, parent_type);
        char *ptr = strchr(clean_name, '*');
        if (ptr)
        {
            *ptr = 0;
        }

        return get_field_type_str(ctx, clean_name, node->member.field);
    }

    if (node->type == NODE_EXPR_BINARY)
    {
        if (strcmp(node->binary.op, "??") == 0)
        {
            return infer_type(ctx, node->binary.left);
        }

        const char *op = node->binary.op;
        char *left_type = infer_type(ctx, node->binary.left);
        char *right_type = infer_type(ctx, node->binary.right);

        int is_logical = (strcmp(op, "&&") == 0 || strcmp(op, "||") == 0 || strcmp(op, "==") == 0 ||
                          strcmp(op, "!=") == 0 || strcmp(op, "<") == 0 || strcmp(op, ">") == 0 ||
                          strcmp(op, "<=") == 0 || strcmp(op, ">=") == 0);

        if (is_logical)
        {
            return xstrdup("int");
        }

        if (left_type && strcmp(left_type, "usize") == 0)
        {
            return "usize";
        }
        if (right_type && strcmp(right_type, "usize") == 0)
        {
            return "usize";
        }
        if (left_type && strcmp(left_type, "double") == 0)
        {
            return "double";
        }

        return left_type ? left_type : right_type;
    }

    if (node->type == NODE_MATCH)
    {
        ASTNode *case_node = node->match_stmt.cases;
        while (case_node)
        {
            char *type = infer_type(ctx, case_node->match_case.body);
            if (type && strcmp(type, "void") != 0 && strcmp(type, "unknown") != 0)
            {
                return type;
            }
            case_node = case_node->next;
        }
        return NULL;
    }

    if (node->type == NODE_EXPR_INDEX)
    {
        char *array_type = infer_type(ctx, node->index.array);
        if (array_type)
        {
            // If T*, returns T. If T[], returns T.
            char *ptr = strrchr(array_type, '*');
            if (ptr)
            {
                int len = ptr - array_type;
                char *buf = xmalloc(len + 1);
                strncpy(buf, array_type, len);
                buf[len] = 0;
                return buf;
            }
        }
        return "int";
    }

    if (node->type == NODE_EXPR_UNARY)
    {
        if (strcmp(node->unary.op, "&") == 0)
        {
            char *inner = infer_type(ctx, node->unary.operand);
            if (inner)
            {
                char *buf = xmalloc(strlen(inner) + 2);
                sprintf(buf, "%s*", inner);
                return buf;
            }
        }
        if (strcmp(node->unary.op, "*") == 0)
        {
            char *inner = infer_type(ctx, node->unary.operand);
            if (inner)
            {
                char *ptr = strchr(inner, '*');
                if (ptr)
                {
                    // Return base type (naive)
                    int len = ptr - inner;
                    char *dup = xmalloc(len + 1);
                    strncpy(dup, inner, len);
                    dup[len] = 0;
                    return dup;
                }
            }
        }
        return infer_type(ctx, node->unary.operand);
    }

    if (node->type == NODE_AWAIT)
    {
        // Infer underlying type T from await Async<T>
        // If it's a direct call await foo(), we know T from foo's signature.
        if (node->unary.operand->type == NODE_EXPR_CALL &&
            node->unary.operand->call.callee->type == NODE_EXPR_VAR)
        {
            FuncSig *sig = find_func(ctx, node->unary.operand->call.callee->var_ref.name);
            if (sig && sig->ret_type)
            {
                return codegen_type_to_string(sig->ret_type);
            }
        }

        return "void*";
    }

    if (node->type == NODE_EXPR_CAST)
    {
        return node->cast.target_type;
    }

    if (node->type == NODE_EXPR_STRUCT_INIT)
    {
        return node->struct_init.struct_name;
    }

    if (node->type == NODE_EXPR_LITERAL)
    {
        if (node->literal.type_kind == TOK_STRING)
        {
            return "string";
        }
        if (node->literal.type_kind == TOK_CHAR)
        {
            return "char";
        }
        if (node->literal.type_kind == 1)
        {
            return "double";
        }
        return "int";
    }

    return NULL;
}

// Extract variable names from argument string.
char *extract_call_args(const char *args)
{
    if (!args || strlen(args) == 0)
    {
        return xstrdup("");
    }
    char *out = xmalloc(strlen(args) + 1);
    out[0] = 0;

    char *dup = xstrdup(args);
    char *p = strtok(dup, ",");
    while (p)
    {
        while (*p == ' ')
        {
            p++;
        }
        char *last_space = strrchr(p, ' ');
        char *ptr_star = strrchr(p, '*');

        char *name = p;
        if (last_space)
        {
            name = last_space + 1;
        }
        if (ptr_star && ptr_star > last_space)
        {
            name = ptr_star + 1;
        }

        if (strlen(out) > 0)
        {
            strcat(out, ", ");
        }
        strcat(out, name);

        p = strtok(NULL, ",");
    }
    free(dup);
    return out;
}

// Parse original method name from mangled name.
const char *parse_original_method_name(const char *mangled)
{
    const char *last = strrchr(mangled, '_');
    return last ? last + 1 : mangled;
}

// Replace string type in arguments.
char *replace_string_type(const char *args)
{
    if (!args)
    {
        return NULL;
    }
    char *res = xmalloc(strlen(args) * 2 + 1);
    res[0] = 0;
    const char *p = args;
    while (*p)
    {
        const char *match = strstr(p, "string ");
        if (match)
        {
            if (match > args && (isalnum(*(match - 1)) || *(match - 1) == '_'))
            {
                strncat(res, p, match - p + 6);
                p = match + 6;
            }
            else
            {
                strncat(res, p, match - p);
                strcat(res, "const char* ");
                p = match + 7;
            }
        }
        else
        {
            strcat(res, p);
            break;
        }
    }
    return res;
}

// Helper to emit auto type or fallback.
void emit_auto_type(ParserContext *ctx, ASTNode *init_expr, Token t, FILE *out)
{
    char *inferred = NULL;
    if (init_expr)
    {
        inferred = infer_type(ctx, init_expr);
    }

    if (inferred && strcmp(inferred, "__auto_type") != 0 && strcmp(inferred, "unknown") != 0)
    {
        fprintf(out, "%s", inferred);
    }
    else
    {
        if (strstr(g_config.cc, "tcc"))
        {
            zpanic_with_suggestion(t,
                                   "Type inference failed for variable initialization and TCC does "
                                   "not support __auto_type",
                                   "Please specify the type explicitly: 'var x: Type = ...'");
        }
        else
        {
            fprintf(out, "ZC_AUTO");
        }
    }
}
// C-compatible type stringifier for codegen.
// Identical to type_to_string but strictly uses 'struct T' for structs to support
// external/non-typedef'd types.
char *codegen_type_to_string(Type *t)
{
    return type_to_c_string(t);
}

// Emit function signature using Type info for correct C codegen
void emit_func_signature(FILE *out, ASTNode *func, const char *name_override)
{
    if (!func || func->type != NODE_FUNCTION)
    {
        return;
    }

    // Emit CUDA qualifiers (for both forward declarations and definitions)
    if (g_config.use_cuda)
    {
        if (func->func.cuda_global)
        {
            fprintf(out, "__global__ ");
        }
        if (func->func.cuda_device)
        {
            fprintf(out, "__device__ ");
        }
        if (func->func.cuda_host)
        {
            fprintf(out, "__host__ ");
        }
    }

    // Return type
    char *ret_str;
    if (func->func.ret_type_info)
    {
        ret_str = codegen_type_to_string(func->func.ret_type_info);
    }
    else if (func->func.ret_type)
    {
        ret_str = xstrdup(func->func.ret_type);
    }
    else
    {
        ret_str = xstrdup("void");
    }

    fprintf(out, "%s %s(", ret_str, name_override ? name_override : func->func.name);
    free(ret_str);

    // Args
    if (func->func.arg_count == 0 && !func->func.is_varargs)
    {
        fprintf(out, "void");
    }
    else
    {
        for (int i = 0; i < func->func.arg_count; i++)
        {
            if (i > 0)
            {
                fprintf(out, ", ");
            }

            char *type_str = NULL;
            if (func->func.arg_types && func->func.arg_types[i])
            {
                type_str = codegen_type_to_string(func->func.arg_types[i]);
            }
            else
            {
                type_str = xstrdup("void*"); // Fallback
            }

            const char *name = "";
            if (func->func.param_names && func->func.param_names[i])
            {
                name = func->func.param_names[i];
            }

            // check if array type
            char *bracket = strchr(type_str, '[');
            if (bracket)
            {
                int base_len = bracket - type_str;
                fprintf(out, "%.*s %s%s", base_len, type_str, name, bracket);
            }
            else
            {
                fprintf(out, "%s %s", type_str, name);
            }
            free(type_str);
        }
        if (func->func.is_varargs)
        {
            if (func->func.arg_count > 0)
            {
                fprintf(out, ", ");
            }
            fprintf(out, "...");
        }
    }
    fprintf(out, ")");
}
