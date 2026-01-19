
#include "../zen/zen_facts.h"
#include "parser.h"
#include <ctype.h>
#include <stdio.h>

#include <stdlib.h>
#include <string.h>

Type *get_field_type(ParserContext *ctx, Type *struct_type, const char *field_name);

static int type_is_unsigned(Type *t)
{
    if (!t)
    {
        return 0;
    }

    return (t->kind == TYPE_U8 || t->kind == TYPE_U16 || t->kind == TYPE_U32 ||
            t->kind == TYPE_U64 || t->kind == TYPE_USIZE || t->kind == TYPE_BYTE ||
            t->kind == TYPE_U128 || t->kind == TYPE_UINT ||
            (t->kind == TYPE_STRUCT && t->name &&
             (0 == strcmp(t->name, "uint8_t") || 0 == strcmp(t->name, "uint16_t") ||
              0 == strcmp(t->name, "uint32_t") || 0 == strcmp(t->name, "uint64_t") ||
              0 == strcmp(t->name, "size_t"))));
}

static void check_format_string(ASTNode *call, Token t)
{
    if (call->type != NODE_EXPR_CALL)
    {
        return;
    }
    ASTNode *callee = call->call.callee;
    if (callee->type != NODE_EXPR_VAR)
    {
        return;
    }

    char *fname = callee->var_ref.name;
    if (!fname)
    {
        return;
    }

    if (strcmp(fname, "printf") != 0 && strcmp(fname, "sprintf") != 0 &&
        strcmp(fname, "fprintf") != 0 && strcmp(fname, "dprintf") != 0)
    {
        return;
    }

    int fmt_idx = 0;
    if (strcmp(fname, "fprintf") == 0 || strcmp(fname, "sprintf") == 0 ||
        strcmp(fname, "dprintf") == 0)
    {
        fmt_idx = 1;
    }

    ASTNode *args = call->call.args;
    ASTNode *fmt_arg = args;
    for (int i = 0; i < fmt_idx; i++)
    {
        if (!fmt_arg)
        {
            return;
        }
        fmt_arg = fmt_arg->next;
    }
    if (!fmt_arg)
    {
        return;
    }

    if (fmt_arg->type != NODE_EXPR_LITERAL || fmt_arg->literal.type_kind != TOK_STRING)
    {
        return;
    }

    const char *fmt = fmt_arg->literal.string_val;

    ASTNode *curr_arg = fmt_arg->next;
    int arg_num = fmt_idx + 2;

    for (int i = 0; fmt[i]; i++)
    {
        if (fmt[i] == '%')
        {
            i++;
            if (fmt[i] == 0)
            {
                break;
            }
            if (fmt[i] == '%')
            {
                continue;
            }

            // Flags.
            while (fmt[i] == '-' || fmt[i] == '+' || fmt[i] == ' ' || fmt[i] == '#' ||
                   fmt[i] == '0')
            {
                i++;
            }

            // Width.
            while (isdigit(fmt[i]))
            {
                i++;
            }

            if (fmt[i] == '*')
            {
                i++;
                if (!curr_arg)
                {
                    warn_format_string(t, arg_num, "width(int)", "missing");
                }
                else
                {
                    /* check int */
                    curr_arg = curr_arg->next;
                    arg_num++;
                }
            }

            // Precision.
            if (fmt[i] == '.')
            {
                i++;
                while (isdigit(fmt[i]))
                {
                    i++;
                }

                if (fmt[i] == '*')
                {
                    i++;
                    if (!curr_arg)
                    {
                        warn_format_string(t, arg_num, "precision(int)", "missing");
                    }
                    else
                    {
                        /* check int */
                        curr_arg = curr_arg->next;
                        arg_num++;
                    }
                }
            }

            // Length.
            if (fmt[i] == 'h' || fmt[i] == 'l' || fmt[i] == 'L' || fmt[i] == 'z' || fmt[i] == 'j' ||
                fmt[i] == 't')
            {
                if (fmt[i] == 'h' && fmt[i + 1] == 'h')
                {
                    i++;
                }
                else if (fmt[i] == 'l' && fmt[i + 1] == 'l')
                {
                    i++;
                }
                i++;
            }

            char spec = fmt[i];

            if (!curr_arg)
            {
                warn_format_string(t, arg_num, "argument", "missing");
                continue;
            }

            Type *vt = curr_arg->type_info;
            char *got_type = vt ? type_to_string(vt) : "?";

            if (spec == 'd' || spec == 'i' || spec == 'u' || spec == 'x' || spec == 'X' ||
                spec == 'o')
            {
                if (vt && vt->kind != TYPE_INT && vt->kind != TYPE_I64 && !type_is_unsigned(vt) &&
                    vt->kind != TYPE_CHAR)
                {
                    warn_format_string(t, arg_num, "integer", got_type);
                }
            }
            else if (spec == 's')
            {
                if (vt && vt->kind != TYPE_STRING && vt->kind != TYPE_POINTER &&
                    vt->kind != TYPE_ARRAY)
                {
                    warn_format_string(t, arg_num, "string", got_type);
                }
            }
            else if (spec == 'f' || spec == 'F' || spec == 'e' || spec == 'E' || spec == 'g' ||
                     spec == 'G')
            {
                if (vt && vt->kind != TYPE_FLOAT && vt->kind != TYPE_F64)
                {
                    warn_format_string(t, arg_num, "float", got_type);
                }
            }
            else if (spec == 'p')
            {
                if (vt && vt->kind != TYPE_POINTER && vt->kind != TYPE_ARRAY)
                {
                    warn_format_string(t, arg_num, "pointer", got_type);
                }
            }

            curr_arg = curr_arg->next;
            arg_num++;
        }
    }
}

ASTNode *parse_expression(ParserContext *ctx, Lexer *l)
{
    return parse_expr_prec(ctx, l, PREC_NONE);
}

Precedence get_token_precedence(Token t)
{
    if (t.type == TOK_INT || t.type == TOK_FLOAT || t.type == TOK_STRING || t.type == TOK_IDENT ||
        t.type == TOK_FSTRING)
    {
        return PREC_NONE;
    }

    if (t.type == TOK_QUESTION)
    {
        return PREC_CALL;
    }

    if (t.type == TOK_ARROW && t.start[0] == '-')
    {
        return PREC_CALL;
    }

    if (t.type == TOK_Q_DOT)
    {
        return PREC_CALL;
    }

    if (t.type == TOK_QQ)
    {
        return PREC_OR;
    }

    if (t.type == TOK_OR)
    {
        return PREC_OR;
    }

    if (t.type == TOK_AND)
    {
        return PREC_AND;
    }

    if (t.type == TOK_QQ_EQ)
    {
        return PREC_ASSIGNMENT;
    }

    if (t.type == TOK_PIPE)
    {
        return PREC_TERM;
    }

    if (t.type == TOK_LANGLE || t.type == TOK_RANGLE)
    {
        return PREC_COMPARISON;
    }

    if (t.type == TOK_OP)
    {
        if (is_token(t, "=") || is_token(t, "+=") || is_token(t, "-=") || is_token(t, "*=") ||
            is_token(t, "/=") || is_token(t, "%=") || is_token(t, "|=") || is_token(t, "&=") ||
            is_token(t, "^=") || is_token(t, "<<=") || is_token(t, ">>="))
        {
            return PREC_ASSIGNMENT;
        }

        if (is_token(t, "||") || is_token(t, "or"))
        {
            return PREC_OR;
        }

        if (is_token(t, "&&") || is_token(t, "and"))
        {
            return PREC_AND;
        }

        if (is_token(t, "|"))
        {
            return PREC_TERM;
        }

        if (is_token(t, "^"))
        {
            return PREC_TERM;
        }

        if (is_token(t, "&"))
        {
            return PREC_TERM;
        }

        if (is_token(t, "<<") || is_token(t, ">>"))
        {
            return PREC_TERM;
        }

        if (is_token(t, "==") || is_token(t, "!="))
        {
            return PREC_EQUALITY;
        }

        if (is_token(t, "<") || is_token(t, ">") || is_token(t, "<=") || is_token(t, ">="))
        {
            return PREC_COMPARISON;
        }

        if (is_token(t, "+") || is_token(t, "-"))
        {
            return PREC_TERM;
        }

        if (is_token(t, "*") || is_token(t, "/") || is_token(t, "%"))
        {
            return PREC_FACTOR;
        }

        if (is_token(t, "."))
        {
            return PREC_CALL;
        }

        if (is_token(t, "|>"))
        {
            return PREC_TERM;
        }
    }

    if (t.type == TOK_LBRACKET || t.type == TOK_LPAREN)
    {
        return PREC_CALL;
    }

    if (is_token(t, "??"))
    {
        return PREC_OR;
    }

    if (is_token(t, "\?\?="))
    {
        return PREC_ASSIGNMENT;
    }

    return PREC_NONE;
}

// Helper to check if a variable name is in a list.
static int is_in_list(const char *name, char **list, int count)
{
    for (int i = 0; i < count; i++)
    {
        if (0 == strcmp(name, list[i]))
        {
            return 1;
        }
    }
    return 0;
}

// Recursively find all variable references in an expression/statement.
static void find_var_refs(ASTNode *node, char ***refs, int *ref_count)
{
    if (!node)
    {
        return;
    }

    if (node->type == NODE_EXPR_VAR)
    {
        *refs = xrealloc(*refs, sizeof(char *) * (*ref_count + 1));
        (*refs)[*ref_count] = xstrdup(node->var_ref.name);
        (*ref_count)++;
    }

    switch (node->type)
    {
    case NODE_EXPR_BINARY:
        find_var_refs(node->binary.left, refs, ref_count);
        find_var_refs(node->binary.right, refs, ref_count);
        break;
    case NODE_EXPR_UNARY:
        find_var_refs(node->unary.operand, refs, ref_count);
        break;
    case NODE_EXPR_CALL:
        find_var_refs(node->call.callee, refs, ref_count);
        for (ASTNode *arg = node->call.args; arg; arg = arg->next)
        {
            find_var_refs(arg, refs, ref_count);
        }
        break;
    case NODE_EXPR_MEMBER:
        find_var_refs(node->member.target, refs, ref_count);
        break;
    case NODE_EXPR_INDEX:
        find_var_refs(node->index.array, refs, ref_count);
        find_var_refs(node->index.index, refs, ref_count);
        break;
    case NODE_EXPR_SLICE:
        find_var_refs(node->slice.array, refs, ref_count);
        find_var_refs(node->slice.start, refs, ref_count);
        find_var_refs(node->slice.end, refs, ref_count);
        break;
    case NODE_BLOCK:
        for (ASTNode *stmt = node->block.statements; stmt; stmt = stmt->next)
        {
            find_var_refs(stmt, refs, ref_count);
        }
        break;
    case NODE_RETURN:
        find_var_refs(node->ret.value, refs, ref_count);
        break;
    case NODE_VAR_DECL:
    case NODE_CONST:
        find_var_refs(node->var_decl.init_expr, refs, ref_count);
        break;
    case NODE_IF:
        find_var_refs(node->if_stmt.condition, refs, ref_count);
        find_var_refs(node->if_stmt.then_body, refs, ref_count);
        if (node->if_stmt.else_body)
        {
            find_var_refs(node->if_stmt.else_body, refs, ref_count);
        }
        break;
    case NODE_WHILE:
        find_var_refs(node->while_stmt.condition, refs, ref_count);
        find_var_refs(node->while_stmt.body, refs, ref_count);
        break;
    case NODE_FOR:
        find_var_refs(node->for_stmt.init, refs, ref_count);
        find_var_refs(node->for_stmt.condition, refs, ref_count);
        find_var_refs(node->for_stmt.step, refs, ref_count);
        find_var_refs(node->for_stmt.body, refs, ref_count);
        break;
    case NODE_MATCH:
        find_var_refs(node->match_stmt.expr, refs, ref_count);
        for (ASTNode *c = node->match_stmt.cases; c; c = c->next)
        {
            find_var_refs(c->match_case.body, refs, ref_count);
        }
        break;
    default:
        break;
    }
}

// Helper to find variable declarations in a subtree
static void find_declared_vars(ASTNode *node, char ***decls, int *count)
{
    if (!node)
    {
        return;
    }

    if (node->type == NODE_VAR_DECL)
    {
        *decls = xrealloc(*decls, sizeof(char *) * (*count + 1));
        (*decls)[*count] = xstrdup(node->var_decl.name);
        (*count)++;
    }

    if (node->type == NODE_MATCH_CASE && node->match_case.binding_name)
    {
        *decls = xrealloc(*decls, sizeof(char *) * (*count + 1));
        (*decls)[*count] = xstrdup(node->match_case.binding_name);
        (*count)++;
    }

    switch (node->type)
    {
    case NODE_BLOCK:
        for (ASTNode *stmt = node->block.statements; stmt; stmt = stmt->next)
        {
            find_declared_vars(stmt, decls, count);
        }
        break;
    case NODE_IF:
        find_declared_vars(node->if_stmt.then_body, decls, count);
        find_declared_vars(node->if_stmt.else_body, decls, count);
        break;
    case NODE_WHILE:
        find_declared_vars(node->while_stmt.body, decls, count);
        break;
    case NODE_FOR:
        find_declared_vars(node->for_stmt.init, decls, count);
        find_declared_vars(node->for_stmt.body, decls, count);
        break;
    case NODE_MATCH:
        for (ASTNode *c = node->match_stmt.cases; c; c = c->next)
        {
            find_declared_vars(c, decls, count);
            find_declared_vars(c->match_case.body, decls, count);
        }
        break;
    default:
        break;
    }
}

// Analyze lambda body to find captured variables.
void analyze_lambda_captures(ParserContext *ctx, ASTNode *lambda)
{
    if (!lambda || lambda->type != NODE_LAMBDA)
    {
        return;
    }

    char **all_refs = NULL;
    int num_refs = 0;
    find_var_refs(lambda->lambda.body, &all_refs, &num_refs);

    char **local_decls = NULL;
    int num_local_decls = 0;
    find_declared_vars(lambda->lambda.body, &local_decls, &num_local_decls);

    char **captures = xmalloc(sizeof(char *) * 16);
    char **capture_types = xmalloc(sizeof(char *) * 16);
    int num_captures = 0;

    for (int i = 0; i < num_refs; i++)
    {
        const char *var_name = all_refs[i];

        if (is_in_list(var_name, lambda->lambda.param_names, lambda->lambda.num_params))
        {
            continue;
        }

        if (is_in_list(var_name, local_decls, num_local_decls))
        {
            continue;
        }

        if (is_in_list(var_name, captures, num_captures))
        {
            continue;
        }

        if (strcmp(var_name, "printf") == 0 || strcmp(var_name, "malloc") == 0 ||
            strcmp(var_name, "strcmp") == 0 || strcmp(var_name, "free") == 0 ||
            strcmp(var_name, "Vec_new") == 0 || strcmp(var_name, "Vec_push") == 0)
        {
            continue;
        }

        FuncSig *fs = ctx->func_registry;
        int is_func = 0;
        while (fs)
        {
            if (0 == strcmp(fs->name, var_name))
            {
                is_func = 1;
                break;
            }
            fs = fs->next;
        }
        if (is_func)
        {
            continue;
        }

        Scope *s = ctx->current_scope;
        int is_local = 0;
        int is_found = 0;
        while (s)
        {
            Symbol *cur = s->symbols;
            while (cur)
            {
                if (0 == strcmp(cur->name, var_name))
                {
                    is_found = 1;
                    if (s->parent != NULL)
                    {
                        is_local = 1;
                    }
                    break;
                }
                cur = cur->next;
            }
            if (is_found)
            {
                break;
            }
            s = s->parent;
        }

        if (is_found && !is_local)
        {
            continue;
        }

        captures[num_captures] = xstrdup(var_name);

        Type *t = find_symbol_type_info(ctx, var_name);
        if (t)
        {
            capture_types[num_captures] = type_to_string(t);
        }
        else
        {
            // Fallback for global/unknown
            // If looks like a function, use "void*" (for closure ctx)
            // else default to "int" or "void*"
            capture_types[num_captures] = xstrdup("int");
        }
        num_captures++;
    }

    lambda->lambda.captured_vars = captures;
    lambda->lambda.captured_types = capture_types;
    lambda->lambda.num_captures = num_captures;

    if (local_decls)
    {
        for (int i = 0; i < num_local_decls; i++)
        {
            free(local_decls[i]);
        }
        free(local_decls);
    }
    for (int i = 0; i < num_refs; i++)
    {
        free(all_refs[i]);
    }
    if (all_refs)
    {
        free(all_refs);
    }
}

ASTNode *parse_lambda(ParserContext *ctx, Lexer *l)
{
    lexer_next(l);

    if (lexer_peek(l).type != TOK_LPAREN)
    {
        zpanic_at(lexer_peek(l), "Expected '(' after 'fn' in lambda");
    }

    lexer_next(l);

    char **param_names = xmalloc(sizeof(char *) * 16);
    char **param_types = xmalloc(sizeof(char *) * 16);
    int num_params = 0;

    while (lexer_peek(l).type != TOK_RPAREN)
    {
        if (num_params > 0)
        {
            if (lexer_peek(l).type != TOK_COMMA)
            {
                zpanic_at(lexer_peek(l), "Expected ',' between parameters");
            }

            lexer_next(l);
        }

        Token name_tok = lexer_next(l);
        if (name_tok.type != TOK_IDENT)
        {
            zpanic_at(name_tok, "Expected parameter name");
        }

        param_names[num_params] = token_strdup(name_tok);

        if (lexer_peek(l).type != TOK_COLON)
        {
            zpanic_at(lexer_peek(l), "Expected ':' after parameter name");
        }

        lexer_next(l);

        char *param_type_str = parse_type(ctx, l);
        param_types[num_params] = param_type_str;
        num_params++;
    }
    lexer_next(l);

    char *return_type = xstrdup("void");
    if (lexer_peek(l).type == TOK_ARROW)
    {
        lexer_next(l);
        return_type = parse_type(ctx, l);
    }

    ASTNode *body = NULL;
    if (lexer_peek(l).type == TOK_LBRACE)
    {
        body = parse_block(ctx, l);
    }
    else
    {
        zpanic_at(lexer_peek(l), "Expected '{' for lambda body");
    }

    ASTNode *lambda = ast_create(NODE_LAMBDA);
    lambda->lambda.param_names = param_names;
    lambda->lambda.param_types = param_types;
    lambda->lambda.return_type = return_type;
    lambda->lambda.body = body;
    lambda->lambda.num_params = num_params;
    lambda->lambda.lambda_id = ctx->lambda_counter++;
    lambda->lambda.is_expression = 0;
    register_lambda(ctx, lambda);
    analyze_lambda_captures(ctx, lambda);

    return lambda;
}

// Helper to create AST for f-string content.
static ASTNode *create_fstring_block(ParserContext *ctx, const char *content)
{
    ASTNode *block = ast_create(NODE_BLOCK);
    block->type_info = type_new(TYPE_STRING);
    block->resolved_type = xstrdup("string");

    ASTNode *head = NULL, *tail = NULL;

    ASTNode *decl_b = ast_create(NODE_RAW_STMT);
    decl_b->raw_stmt.content = xstrdup("static char _b[4096]; _b[0]=0;");
    if (!head)
    {
        head = decl_b;
    }
    else
    {
        tail->next = decl_b;
    }
    tail = decl_b;

    ASTNode *decl_t = ast_create(NODE_RAW_STMT);
    decl_t->raw_stmt.content = xstrdup("char _t[128];");
    tail->next = decl_t;
    tail = decl_t;

    const char *cur = content;
    while (*cur)
    {
        char *brace = strchr(cur, '{');
        if (!brace)
        {
            if (strlen(cur) > 0)
            {
                ASTNode *cat = ast_create(NODE_RAW_STMT);
                cat->raw_stmt.content = xmalloc(strlen(cur) + 20);
                sprintf(cat->raw_stmt.content, "strcat(_b, \"%s\");", cur);
                tail->next = cat;
                tail = cat;
            }
            break;
        }

        if (brace > cur)
        {
            int len = brace - cur;
            char *txt = xmalloc(len + 1);
            strncpy(txt, cur, len);
            txt[len] = 0;
            ASTNode *cat = ast_create(NODE_RAW_STMT);
            cat->raw_stmt.content = xmalloc(len + 20);
            sprintf(cat->raw_stmt.content, "strcat(_b, \"%s\");", txt);
            tail->next = cat;
            tail = cat;
            free(txt);
        }

        char *end_brace = strchr(brace, '}');
        if (!end_brace)
        {
            break;
        }

        char *colon = NULL;
        char *p = brace + 1;
        int depth = 1;
        while (p < end_brace)
        {
            if (*p == '{')
            {
                depth++;
            }
            if (*p == '}')
            {
                depth--;
            }
            if (depth == 1 && *p == ':' && !colon)
            {
                if ((p + 1) < end_brace && *(p + 1) == ':')
                {
                    p++;
                }
                else
                {
                    colon = p;
                }
            }
            p++;
        }

        char *expr_str;
        char *fmt = NULL;

        if (colon && colon < end_brace)
        {
            int expr_len = colon - (brace + 1);
            expr_str = xmalloc(expr_len + 1);
            strncpy(expr_str, brace + 1, expr_len);
            expr_str[expr_len] = 0;

            int fmt_len = end_brace - (colon + 1);
            fmt = xmalloc(fmt_len + 1);
            strncpy(fmt, colon + 1, fmt_len);
            fmt[fmt_len] = 0;
        }
        else
        {
            int expr_len = end_brace - (brace + 1);
            expr_str = xmalloc(expr_len + 1);
            strncpy(expr_str, brace + 1, expr_len);
            expr_str[expr_len] = 0;
        }

        Lexer sub_l;
        lexer_init(&sub_l, expr_str);
        ASTNode *expr_node = parse_expression(ctx, &sub_l);

        if (expr_node && expr_node->type == NODE_EXPR_VAR)
        {
            Symbol *sym = find_symbol_entry(ctx, expr_node->var_ref.name);
            if (sym)
            {
                sym->is_used = 1;
            }
        }

        ASTNode *call_sprintf = ast_create(NODE_EXPR_CALL);
        ASTNode *callee = ast_create(NODE_EXPR_VAR);
        callee->var_ref.name = xstrdup("sprintf");
        call_sprintf->call.callee = callee;

        ASTNode *arg_t = ast_create(NODE_EXPR_VAR);
        arg_t->var_ref.name = xstrdup("_t");

        ASTNode *arg_fmt = NULL;
        if (fmt)
        {
            char *fmt_str = xmalloc(strlen(fmt) + 3);
            sprintf(fmt_str, "%%%s", fmt);
            arg_fmt = ast_create(NODE_EXPR_LITERAL);
            arg_fmt->literal.type_kind = 2;
            arg_fmt->literal.string_val = fmt_str;
            arg_fmt->type_info = type_new(TYPE_STRING);
        }
        else
        {
            // _z_str(expr)
            ASTNode *call_macro = ast_create(NODE_EXPR_CALL);
            ASTNode *macro_callee = ast_create(NODE_EXPR_VAR);
            macro_callee->var_ref.name = xstrdup("_z_str");
            call_macro->call.callee = macro_callee;
            Lexer l2;
            lexer_init(&l2, expr_str);
            ASTNode *expr_copy = parse_expression(ctx, &l2);

            call_macro->call.args = expr_copy;
            arg_fmt = call_macro;
        }

        call_sprintf->call.args = arg_t;
        arg_t->next = arg_fmt;
        arg_fmt->next = expr_node;

        tail->next = call_sprintf;
        tail = call_sprintf;

        // strcat(_b, _t)
        ASTNode *cat_t = ast_create(NODE_RAW_STMT);
        cat_t->raw_stmt.content = xstrdup("strcat(_b, _t);");
        tail->next = cat_t;
        tail = cat_t;

        cur = end_brace + 1;
        free(expr_str);
        if (fmt)
        {
            free(fmt);
        }
    }

    // Return _b
    ASTNode *ret_b = ast_create(NODE_RAW_STMT);
    ret_b->raw_stmt.content = xstrdup("_b;");
    tail->next = ret_b;
    tail = ret_b;

    block->block.statements = head;
    return block;
}

ASTNode *parse_primary(ParserContext *ctx, Lexer *l)
{
    ASTNode *node = NULL;
    Token t = lexer_next(l);

    // ** Prefixes **

    // Literals
    if (t.type == TOK_INT)
    {
        node = ast_create(NODE_EXPR_LITERAL);
        node->literal.type_kind = 0;
        node->type_info = type_new(TYPE_INT);
        char *s = token_strdup(t);
        unsigned long long val;
        if (t.len > 2 && s[0] == '0' && s[1] == 'b')
        {
            val = strtoull(s + 2, NULL, 2);
        }
        else
        {
            val = strtoull(s, NULL, 0);
        }
        node->literal.int_val = (unsigned long long)val;
        free(s);
    }
    else if (t.type == TOK_FLOAT)
    {
        node = ast_create(NODE_EXPR_LITERAL);
        node->literal.type_kind = 1;
        node->literal.float_val = atof(t.start);
        node->type_info = type_new(TYPE_F64);
    }
    else if (t.type == TOK_STRING)
    {
        node = ast_create(NODE_EXPR_LITERAL);
        node->literal.type_kind = TOK_STRING;
        node->literal.string_val = xmalloc(t.len);
        strncpy(node->literal.string_val, t.start + 1, t.len - 2);
        node->literal.string_val[t.len - 2] = 0;
        node->type_info = type_new(TYPE_STRING);
    }
    else if (t.type == TOK_FSTRING)
    {
        char *inner = xmalloc(t.len);
        strncpy(inner, t.start + 2, t.len - 3);
        inner[t.len - 3] = 0;
        node = create_fstring_block(ctx, inner);
        free(inner);
    }
    else if (t.type == TOK_CHAR)
    {
        node = ast_create(NODE_EXPR_LITERAL);
        node->literal.type_kind = TOK_CHAR;
        node->literal.string_val = token_strdup(t);
        node->type_info = type_new(TYPE_I8);
    }

    else if (t.type == TOK_SIZEOF)
    {
        if (lexer_peek(l).type != TOK_LPAREN)
        {
            zpanic_at(lexer_peek(l), "Expected ( after sizeof");
        }
        lexer_next(l);

        int pos = l->pos;
        int col = l->col;
        int line = l->line;
        Type *ty = parse_type_formal(ctx, l);

        if (ty->kind != TYPE_UNKNOWN && lexer_peek(l).type == TOK_RPAREN)
        {
            lexer_next(l);
            char *ts = type_to_string(ty);
            node = ast_create(NODE_EXPR_SIZEOF);
            node->size_of.target_type = ts;
            node->size_of.expr = NULL;
            node->type_info = type_new(TYPE_USIZE);
        }
        else
        {
            l->pos = pos;
            l->col = col;
            l->line = line;
            ASTNode *ex = parse_expression(ctx, l);
            if (lexer_next(l).type != TOK_RPAREN)
            {
                zpanic_at(lexer_peek(l), "Expected ) after sizeof identifier");
            }
            node = ast_create(NODE_EXPR_SIZEOF);
            node->size_of.target_type = NULL;
            node->size_of.expr = ex;
            node->type_info = type_new(TYPE_USIZE);
        }
    }

    else if (t.type == TOK_IDENT && strncmp(t.start, "typeof", 6) == 0 && t.len == 6)
    {
        if (lexer_peek(l).type != TOK_LPAREN)
        {
            zpanic_at(lexer_peek(l), "Expected ( after typeof");
        }
        lexer_next(l);

        int pos = l->pos;
        int col = l->col;
        int line = l->line;
        Type *ty = parse_type_formal(ctx, l);

        if (ty->kind != TYPE_UNKNOWN && lexer_peek(l).type == TOK_RPAREN)
        {
            lexer_next(l);
            char *ts = type_to_string(ty);
            node = ast_create(NODE_TYPEOF);
            node->size_of.target_type = ts;
            node->size_of.expr = NULL;
        }
        else
        {
            l->pos = pos;
            l->col = col;
            l->line = line;
            ASTNode *ex = parse_expression(ctx, l);
            if (lexer_next(l).type != TOK_RPAREN)
            {
                zpanic_at(lexer_peek(l), "Expected ) after typeof expression");
            }
            node = ast_create(NODE_TYPEOF);
            node->size_of.target_type = NULL;
            node->size_of.expr = ex;
        }
    }

    else if (t.type == TOK_AT)
    {
        Token ident = lexer_next(l);
        if (ident.type != TOK_IDENT)
        {
            zpanic_at(ident, "Expected intrinsic name after @");
        }

        int kind = -1;
        if (strncmp(ident.start, "type_name", 9) == 0 && ident.len == 9)
        {
            kind = 0;
        }
        else if (strncmp(ident.start, "fields", 6) == 0 && ident.len == 6)
        {
            kind = 1;
        }
        else
        {
            zpanic_at(ident, "Unknown intrinsic @%.*s", ident.len, ident.start);
        }

        {
            Token t = lexer_next(l);
            if (t.type != TOK_LPAREN)
            {
                zpanic_at(t, "Expected ( after intrinsic");
            }
        }

        Type *target = parse_type_formal(ctx, l);

        {
            Token t = lexer_next(l);
            if (t.type != TOK_RPAREN)
            {
                zpanic_at(t, "Expected ) after intrinsic type");
            }
        }

        node = ast_create(NODE_REFLECTION);
        node->reflection.kind = kind;
        node->reflection.target_type = target;
        node->type_info = (kind == 0) ? type_new(TYPE_STRING) : type_new_ptr(type_new(TYPE_VOID));
    }

    else if (t.type == TOK_IDENT && strncmp(t.start, "match", 5) == 0 && t.len == 5)
    {
        ASTNode *expr = parse_expression(ctx, l);
        skip_comments(l);
        {
            Token t = lexer_next(l);
            if (t.type != TOK_LBRACE)
            {
                zpanic_at(t, "Expected { after match expression");
            }
        }

        ASTNode *h = 0, *tl = 0;
        while (1)
        {
            skip_comments(l);
            if (lexer_peek(l).type == TOK_RBRACE)
            {
                break;
            }
            if (lexer_peek(l).type == TOK_COMMA)
            {
                lexer_next(l);
            }

            skip_comments(l);
            if (lexer_peek(l).type == TOK_RBRACE)
            {
                break;
            }

            Token p = lexer_next(l);
            char *pattern = token_strdup(p);
            int is_default = (strcmp(pattern, "_") == 0);

            char *binding = NULL;
            int is_destructure = 0;
            skip_comments(l);
            if (!is_default && lexer_peek(l).type == TOK_LPAREN)
            {
                lexer_next(l);
                Token b = lexer_next(l);
                if (b.type != TOK_IDENT)
                {
                    zpanic_at(b, "Expected binding name");
                }
                binding = token_strdup(b);
                if (lexer_next(l).type != TOK_RPAREN)
                {
                    zpanic_at(lexer_peek(l), "Expected )");
                }
                is_destructure = 1;
            }

            ASTNode *guard = NULL;
            skip_comments(l);
            if (lexer_peek(l).type == TOK_IDENT && strncmp(lexer_peek(l).start, "if", 2) == 0)
            {
                lexer_next(l);
                guard = parse_expression(ctx, l);
            }

            skip_comments(l);
            if (lexer_next(l).type != TOK_ARROW)
            {
                zpanic_at(lexer_peek(l), "Expected '=>'");
            }

            ASTNode *body;
            skip_comments(l);
            Token pk = lexer_peek(l);
            if (pk.type == TOK_LBRACE)
            {
                body = parse_block(ctx, l);
            }
            else if (pk.type == TOK_ASSERT ||
                     (pk.type == TOK_IDENT && strncmp(pk.start, "assert", 6) == 0))
            {
                body = parse_assert(ctx, l);
            }
            else if (pk.type == TOK_IDENT && strncmp(pk.start, "return", 6) == 0)
            {
                body = parse_return(ctx, l);
            }
            else
            {
                body = parse_expression(ctx, l);
            }

            ASTNode *c = ast_create(NODE_MATCH_CASE);
            c->match_case.pattern = pattern;
            c->match_case.binding_name = binding;
            c->match_case.is_destructuring = is_destructure;
            c->match_case.guard = guard;
            c->match_case.body = body;
            c->match_case.is_default = is_default;
            if (!h)
            {
                h = c;
            }
            else
            {
                tl->next = c;
            }
            tl = c;
        }
        lexer_next(l);
        node = ast_create(NODE_MATCH);
        node->match_stmt.expr = expr;
        node->match_stmt.cases = h;
    }

    else if (t.type == TOK_IDENT)
    {
        if (t.len == 2 && strncmp(t.start, "fn", 2) == 0 && lexer_peek(l).type == TOK_LPAREN)
        {
            l->pos -= t.len;
            l->col -= t.len;
            return parse_lambda(ctx, l);
        }

        char *ident = token_strdup(t);

        if (lexer_peek(l).type == TOK_OP && lexer_peek(l).start[0] == '!' && lexer_peek(l).len == 1)
        {
            node = parse_macro_call(ctx, l, ident);
            if (node)
            {
                free(ident);
                return node;
            }
        }

        if (lexer_peek(l).type == TOK_ARROW)
        {
            lexer_next(l);
            return parse_arrow_lambda_single(ctx, l, ident);
        }

        char *acc = ident;
        while (1)
        {
            int changed = 0;
            if (lexer_peek(l).type == TOK_DCOLON)
            {
                lexer_next(l);
                Token suffix = lexer_next(l);
                if (suffix.type != TOK_IDENT)
                {
                    zpanic_at(suffix, "Expected identifier after ::");
                }

                SelectiveImport *si =
                    (!ctx->current_module_prefix) ? find_selective_import(ctx, acc) : NULL;
                if (si)
                {
                    char *tmp =
                        xmalloc(strlen(si->source_module) + strlen(si->symbol) + suffix.len + 4);

                    char base[256];
                    sprintf(base, "%s_%s", si->source_module, si->symbol);
                    ASTNode *def = find_struct_def(ctx, base);
                    int is_type = (def != NULL);

                    if (is_type)
                    {
                        // Check Enum Variant
                        int is_variant = 0;
                        if (def->type == NODE_ENUM)
                        {
                            ASTNode *v = def->enm.variants;
                            char sbuf[128];
                            strncpy(sbuf, suffix.start, suffix.len);
                            sbuf[suffix.len] = 0;
                            while (v)
                            {
                                if (strcmp(v->variant.name, sbuf) == 0)
                                {
                                    is_variant = 1;
                                    break;
                                }
                                v = v->next;
                            }
                        }
                        if (is_variant)
                        {
                            sprintf(tmp, "%s_%s_%.*s", si->source_module, si->symbol, suffix.len,
                                    suffix.start);
                        }
                        else
                        {
                            sprintf(tmp, "%s_%s__%.*s", si->source_module, si->symbol, suffix.len,
                                    suffix.start);
                        }
                    }
                    else
                    {
                        sprintf(tmp, "%s_%s_%.*s", si->source_module, si->symbol, suffix.len,
                                suffix.start);
                    }

                    free(acc);
                    acc = tmp;
                }
                else
                {
                    Module *mod = find_module(ctx, acc);
                    if (mod)
                    {
                        if (mod->is_c_header)
                        {
                            char *tmp = xmalloc(suffix.len + 1);
                            strncpy(tmp, suffix.start, suffix.len);
                            tmp[suffix.len] = 0;
                            free(acc);
                            acc = tmp;
                        }

                        else
                        {
                            char *tmp = xmalloc(strlen(mod->base_name) + suffix.len + 2);
                            sprintf(tmp, "%s_", mod->base_name);
                            strncat(tmp, suffix.start, suffix.len);
                            free(acc);
                            acc = tmp;
                        }
                    }
                    else
                    {
                        char *tmp = xmalloc(strlen(acc) + suffix.len + 3);

                        ASTNode *def = find_struct_def(ctx, acc);
                        if (def)
                        {
                            int is_variant = 0;
                            if (def->type == NODE_ENUM)
                            {
                                ASTNode *v = def->enm.variants;
                                char sbuf[128];
                                strncpy(sbuf, suffix.start, suffix.len);
                                sbuf[suffix.len] = 0;
                                while (v)
                                {
                                    if (strcmp(v->variant.name, sbuf) == 0)
                                    {
                                        is_variant = 1;
                                        break;
                                    }
                                    v = v->next;
                                }
                            }
                            if (is_variant)
                            {
                                sprintf(tmp, "%s_%.*s", acc, suffix.len, suffix.start);
                            }
                            else
                            {
                                sprintf(tmp, "%s__%.*s", acc, suffix.len, suffix.start);
                            }
                        }
                        else
                        {
                            int handled_as_generic = 0;
                            for (int i = 0; i < ctx->known_generics_count; i++)
                            {
                                char *gname = ctx->known_generics[i];
                                int glen = strlen(gname);
                                if (strncmp(acc, gname, glen) == 0 && acc[glen] == '_')
                                {
                                    ASTNode *tpl_def = find_struct_def(ctx, gname);
                                    if (tpl_def)
                                    {
                                        int is_variant = 0;
                                        if (tpl_def->type == NODE_ENUM)
                                        {
                                            ASTNode *v = tpl_def->enm.variants;
                                            char sbuf[128];
                                            strncpy(sbuf, suffix.start, suffix.len);
                                            sbuf[suffix.len] = 0;
                                            while (v)
                                            {
                                                if (strcmp(v->variant.name, sbuf) == 0)
                                                {
                                                    is_variant = 1;
                                                    break;
                                                }
                                                v = v->next;
                                            }
                                        }
                                        if (is_variant)
                                        {
                                            sprintf(tmp, "%s_%.*s", acc, suffix.len, suffix.start);
                                        }
                                        else
                                        {
                                            sprintf(tmp, "%s__%.*s", acc, suffix.len, suffix.start);
                                        }
                                        handled_as_generic = 1;
                                        break; // Found and handled
                                    }
                                }
                            }
                            // Explicit check for Vec to ensure it works
                            if (!handled_as_generic && strncmp(acc, "Vec_", 4) == 0)
                            {
                                sprintf(tmp, "%s__%.*s", acc, suffix.len, suffix.start);
                                handled_as_generic = 1;
                            }

                            // Also check registered templates list
                            if (!handled_as_generic)
                            {
                                GenericTemplate *gt = ctx->templates;
                                while (gt)
                                {
                                    char *gname = gt->name;
                                    int glen = strlen(gname);
                                    if ((strncmp(acc, gname, glen) == 0 && acc[glen] == '_') ||
                                        strcmp(acc, gname) == 0)
                                    {
                                        ASTNode *tpl_def = gt->struct_node;
                                        if (tpl_def)
                                        {
                                            int is_variant = 0;
                                            if (tpl_def->type == NODE_ENUM)
                                            {
                                                ASTNode *v = tpl_def->enm.variants;
                                                char sbuf[128];
                                                strncpy(sbuf, suffix.start, suffix.len);
                                                sbuf[suffix.len] = 0;
                                                while (v)
                                                {
                                                    if (strcmp(v->variant.name, sbuf) == 0)
                                                    {
                                                        is_variant = 1;
                                                        break;
                                                    }
                                                    v = v->next;
                                                }
                                            }
                                            if (is_variant)
                                            {
                                                sprintf(tmp, "%s_%.*s", acc, suffix.len,
                                                        suffix.start);
                                            }
                                            else
                                            {
                                                sprintf(tmp, "%s__%.*s", acc, suffix.len,
                                                        suffix.start);
                                            }
                                            handled_as_generic = 1;
                                            break;
                                        }
                                    }
                                    gt = gt->next;
                                }
                            }

                            if (!handled_as_generic)
                            {
                                sprintf(tmp, "%s_%.*s", acc, suffix.len, suffix.start);
                            }
                        }

                        free(acc);
                        acc = tmp;
                    }
                }
                changed = 1;
            }

            if (lexer_peek(l).type == TOK_LANGLE)
            {
                Lexer lookahead = *l;
                lexer_next(&lookahead);
                parse_type(ctx, &lookahead);
                if (lexer_peek(&lookahead).type == TOK_RANGLE)
                {
                    lexer_next(l);
                    Type *formal_type = parse_type_formal(ctx, l);
                    char *concrete_type = type_to_string(formal_type); // mangled for naming
                    char *unmangled_type =
                        type_to_c_string(formal_type); // C-compatible for substitution
                    lexer_next(l);

                    int is_struct = 0;
                    GenericTemplate *st = ctx->templates;
                    while (st)
                    {
                        if (strcmp(st->name, acc) == 0)
                        {
                            is_struct = 1;
                            break;
                        }
                        st = st->next;
                    }
                    if (!is_struct && (strcmp(acc, "Result") == 0 || strcmp(acc, "Option") == 0))
                    {
                        is_struct = 1;
                    }

                    if (is_struct)
                    {
                        instantiate_generic(ctx, acc, concrete_type, unmangled_type, t);

                        char *clean_type = sanitize_mangled_name(concrete_type);

                        char *m = xmalloc(strlen(acc) + strlen(clean_type) + 2);
                        sprintf(m, "%s_%s", acc, clean_type);
                        free(clean_type);

                        free(acc);
                        acc = m;
                    }
                    else
                    {
                        char *m =
                            instantiate_function_template(ctx, acc, concrete_type, unmangled_type);
                        if (m)
                        {
                            free(acc);
                            acc = m;
                        }
                        else
                        {
                            zpanic_at(t, "Unknown generic %s", acc);
                        }
                    }
                    changed = 1;
                }
            }
            if (!changed)
            {
                break;
            }
        }

        if (lexer_peek(l).type == TOK_LBRACE)
        {
            int is_struct_init = 0;
            Lexer pl = *l;
            lexer_next(&pl);
            Token fi = lexer_peek(&pl);

            if (fi.type == TOK_RBRACE)
            {
                is_struct_init = 1;
            }
            else if (fi.type == TOK_IDENT)
            {
                lexer_next(&pl);
                if (lexer_peek(&pl).type == TOK_COLON)
                {
                    is_struct_init = 1;
                }
            }
            if (is_struct_init)
            {
                char *struct_name = acc;
                if (!ctx->current_module_prefix)
                {
                    SelectiveImport *si = find_selective_import(ctx, acc);
                    if (si)
                    {
                        struct_name = xmalloc(strlen(si->source_module) + strlen(si->symbol) + 2);
                        sprintf(struct_name, "%s_%s", si->source_module, si->symbol);
                    }
                }
                if (struct_name == acc && ctx->current_module_prefix && !is_known_generic(ctx, acc))
                {
                    char *prefixed = xmalloc(strlen(ctx->current_module_prefix) + strlen(acc) + 2);
                    sprintf(prefixed, "%s_%s", ctx->current_module_prefix, acc);
                    struct_name = prefixed;
                }
                lexer_next(l);
                node = ast_create(NODE_EXPR_STRUCT_INIT);
                node->struct_init.struct_name = struct_name;
                Type *init_type = type_new(TYPE_STRUCT);
                init_type->name = xstrdup(struct_name);
                node->type_info = init_type;

                ASTNode *head = NULL, *tail = NULL;
                int first = 1;
                while (lexer_peek(l).type != TOK_RBRACE)
                {
                    if (!first && lexer_peek(l).type == TOK_COMMA)
                    {
                        lexer_next(l);
                    }
                    if (lexer_peek(l).type == TOK_RBRACE)
                    {
                        break;
                    }
                    Token fn = lexer_next(l);
                    if (lexer_next(l).type != TOK_COLON)
                    {
                        zpanic_at(lexer_peek(l), "Expected :");
                    }
                    ASTNode *val = parse_expression(ctx, l);
                    ASTNode *assign = ast_create(NODE_VAR_DECL);
                    assign->var_decl.name = token_strdup(fn);
                    assign->var_decl.init_expr = val;
                    if (!head)
                    {
                        head = assign;
                    }
                    else
                    {
                        tail->next = assign;
                    }
                    tail = assign;
                    first = 0;
                }
                lexer_next(l);
                node->struct_init.fields = head;
                Type *st = type_new(TYPE_STRUCT);
                st->name = xstrdup(struct_name);
                node->type_info = st;
                return node; // Struct init cannot be called/indexed usually, return
                             // early
            }
        }

        // E. readln(args...) Magic
        FuncSig *sig = find_func(ctx, acc);
        if (strcmp(acc, "readln") == 0 && lexer_peek(l).type == TOK_LPAREN)
        {
            lexer_next(l); // eat (

            // Parse args
            ASTNode *args[16];
            int ac = 0;
            if (lexer_peek(l).type != TOK_RPAREN)
            {
                while (1)
                {
                    args[ac++] = parse_expression(ctx, l);
                    if (lexer_peek(l).type == TOK_COMMA)
                    {
                        lexer_next(l);
                    }
                    else
                    {
                        break;
                    }
                }
            }
            if (lexer_next(l).type != TOK_RPAREN)
            {
                zpanic_at(lexer_peek(l), "Expected )");
            }

            if (ac == 0)
            {
                // readln() -> _z_readln_raw()
                node = ast_create(NODE_EXPR_CALL);
                ASTNode *callee = ast_create(NODE_EXPR_VAR);
                callee->var_ref.name = xstrdup("_z_readln_raw");
                node->call.callee = callee;
                node->type_info = type_new(TYPE_STRING);
            }
            else
            {
                // readln(vars...) -> _z_scan_helper("fmt", &vars...)
                char fmt[256];
                fmt[0] = 0;
                for (int i = 0; i < ac; i++)
                {
                    Type *t = args[i]->type_info;
                    if (!t && args[i]->type == NODE_EXPR_VAR)
                    {
                        t = find_symbol_type_info(ctx, args[i]->var_ref.name);
                    }

                    if (!t)
                    {
                        strcat(fmt, "%d"); // Fallback
                    }
                    else
                    {
                        if (t->kind == TYPE_INT || t->kind == TYPE_I32 || t->kind == TYPE_BOOL)
                        {
                            strcat(fmt, "%d");
                        }
                        else if (t->kind == TYPE_F64)
                        {
                            strcat(fmt, "%lf");
                        }
                        else if (t->kind == TYPE_F32 || t->kind == TYPE_FLOAT)
                        {
                            strcat(fmt, "%f");
                        }
                        else if (t->kind == TYPE_STRING)
                        {
                            strcat(fmt, "%s");
                        }
                        else if (t->kind == TYPE_CHAR || t->kind == TYPE_I8 || t->kind == TYPE_U8 ||
                                 t->kind == TYPE_BYTE)
                        {
                            strcat(fmt, " %c"); // Space skip whitespace
                        }
                        else
                        {
                            strcat(fmt, "%d");
                        }
                    }
                    if (i < ac - 1)
                    {
                        strcat(fmt, " ");
                    }
                }

                node = ast_create(NODE_EXPR_CALL);
                ASTNode *callee = ast_create(NODE_EXPR_VAR);
                callee->var_ref.name = xstrdup("_z_scan_helper");
                node->call.callee = callee;
                node->type_info = type_new(TYPE_INT); // Returns count

                ASTNode *fmt_node = ast_create(NODE_EXPR_LITERAL);
                fmt_node->literal.type_kind = 2; // string
                fmt_node->literal.string_val = xstrdup(fmt);

                ASTNode *head = fmt_node, *tail = fmt_node;

                for (int i = 0; i < ac; i++)
                {
                    // Create Unary & (AddressOf) node wrapping the arg
                    ASTNode *addr = ast_create(NODE_EXPR_UNARY);
                    addr->unary.op = xstrdup("&");
                    addr->unary.operand = args[i];
                    // Link
                    tail->next = addr;
                    tail = addr;
                }
                node->call.args = head;
            }
            free(acc);
        }
        else if (sig && lexer_peek(l).type == TOK_LPAREN)
        {
            lexer_next(l);
            ASTNode *head = NULL, *tail = NULL;
            int args_provided = 0;
            char **arg_names = NULL;
            int has_named = 0;

            if (lexer_peek(l).type != TOK_RPAREN)
            {
                while (1)
                {
                    char *arg_name = NULL;

                    Token t1 = lexer_peek(l);
                    if (t1.type == TOK_IDENT)
                    {
                        Token t2 = lexer_peek2(l);
                        if (t2.type == TOK_COLON)
                        {
                            arg_name = token_strdup(t1);
                            has_named = 1;
                            lexer_next(l);
                            lexer_next(l);
                        }
                    }

                    ASTNode *arg = parse_expression(ctx, l);
                    if (!head)
                    {
                        head = arg;
                    }
                    else
                    {
                        tail->next = arg;
                    }
                    tail = arg;
                    args_provided++;

                    arg_names = xrealloc(arg_names, args_provided * sizeof(char *));
                    arg_names[args_provided - 1] = arg_name;

                    arg_names = xrealloc(arg_names, args_provided * sizeof(char *));
                    arg_names[args_provided - 1] = arg_name;

                    if (lexer_peek(l).type == TOK_COMMA)
                    {
                        lexer_next(l);
                    }
                    else
                    {
                        break;
                    }
                }
            }
            if (lexer_next(l).type != TOK_RPAREN)
            {
                zpanic_at(lexer_peek(l), "Expected )");
            }
            for (int i = args_provided; i < sig->total_args; i++)
            {
                if (sig->defaults[i])
                {
                    ASTNode *def = ast_create(NODE_RAW_STMT);
                    def->raw_stmt.content = xstrdup(sig->defaults[i]);
                    if (!head)
                    {
                        head = def;
                    }
                    else
                    {
                        tail->next = def;
                    }
                    tail = def;
                }
            }
            node = ast_create(NODE_EXPR_CALL);
            node->token = t; // Set source token
            ASTNode *callee = ast_create(NODE_EXPR_VAR);
            callee->var_ref.name = acc;
            node->call.callee = callee;
            node->call.args = head;
            node->call.arg_names = has_named ? arg_names : NULL;
            node->call.arg_count = args_provided;
            if (sig)
            {
                node->definition_token = sig->decl_token;
            }
            if (sig->is_async)
            {
                Type *async_type = type_new(TYPE_STRUCT);
                async_type->name = xstrdup("Async");
                node->type_info = async_type;
                node->resolved_type = xstrdup("Async");
            }
            else if (sig->ret_type)
            {
                node->type_info = sig->ret_type;
                node->resolved_type = type_to_string(sig->ret_type);
            }
            else
            {
                node->resolved_type = xstrdup("void");
            }
        }
        else if (!sig && !find_symbol_entry(ctx, acc) && lexer_peek(l).type == TOK_LPAREN)
        {
            lexer_next(l); // eat (
            ASTNode *head = NULL, *tail = NULL;
            char **arg_names = NULL;
            int args_provided = 0;
            int has_named = 0;

            if (lexer_peek(l).type != TOK_RPAREN)
            {
                while (1)
                {
                    char *arg_name = NULL;

                    // Check for named argument: name: value
                    Token t1 = lexer_peek(l);
                    if (t1.type == TOK_IDENT)
                    {
                        Token t2 = lexer_peek2(l);
                        if (t2.type == TOK_COLON)
                        {
                            arg_name = token_strdup(t1);
                            has_named = 1;
                            lexer_next(l);
                            lexer_next(l);
                        }
                    }

                    ASTNode *arg = parse_expression(ctx, l);
                    if (!head)
                    {
                        head = arg;
                    }
                    else
                    {
                        tail->next = arg;
                    }
                    tail = arg;
                    args_provided++;

                    arg_names = xrealloc(arg_names, args_provided * sizeof(char *));
                    arg_names[args_provided - 1] = arg_name;

                    if (lexer_peek(l).type == TOK_COMMA)
                    {
                        lexer_next(l);
                    }
                    else
                    {
                        break;
                    }
                }
            }
            if (lexer_next(l).type != TOK_RPAREN)
            {
                zpanic_at(lexer_peek(l), "Expected )");
            }

            node = ast_create(NODE_EXPR_CALL);
            node->token = t;
            ASTNode *callee = ast_create(NODE_EXPR_VAR);
            callee->var_ref.name = acc;
            node->call.callee = callee;
            node->call.args = head;
            node->call.arg_names = has_named ? arg_names : NULL;
            node->call.arg_count = args_provided;
            // Unknown return type - let codegen infer it
            node->resolved_type = xstrdup("unknown");
            // Fall through to Postfix
        }
        else
        {
            node = ast_create(NODE_EXPR_VAR);
            node->token = t; // Set source token
            node->var_ref.name = acc;
            node->type_info = find_symbol_type_info(ctx, acc);

            Symbol *sym = find_symbol_entry(ctx, acc);
            if (sym)
            {
                sym->is_used = 1;
                node->definition_token = sym->decl_token;
            }

            char *type_str = find_symbol_type(ctx, acc);

            if (type_str)
            {
                node->resolved_type = type_str;
                node->var_ref.suggestion = NULL;
            }
            else
            {
                node->resolved_type = xstrdup("unknown");
                if (should_suppress_undef_warning(ctx, acc))
                {
                    node->var_ref.suggestion = NULL;
                }
                else
                {
                    node->var_ref.suggestion = find_similar_symbol(ctx, acc);
                }
            }
        }
    }

    else if (t.type == TOK_LPAREN)
    {

        Lexer lookahead = *l;
        int is_lambda = 0;
        char **params = xmalloc(sizeof(char *) * 16);
        int nparams = 0;

        while (1)
        {
            if (lexer_peek(&lookahead).type != TOK_IDENT)
            {
                break;
            }
            params[nparams++] = token_strdup(lexer_next(&lookahead));
            Token sep = lexer_peek(&lookahead);
            if (sep.type == TOK_COMMA)
            {
                lexer_next(&lookahead);
                continue;
            }
            else if (sep.type == TOK_RPAREN)
            {
                lexer_next(&lookahead);
                if (lexer_peek(&lookahead).type == TOK_ARROW)
                {
                    lexer_next(&lookahead);
                    is_lambda = 1;
                }
                break;
            }
            else
            {
                break;
            }
        }

        if (is_lambda && nparams > 0)
        {
            *l = lookahead; // Commit
            return parse_arrow_lambda_multi(ctx, l, params, nparams);
        }

        int saved = l->pos;
        if (lexer_peek(l).type == TOK_IDENT)
        {
            Lexer cast_look = *l;
            lexer_next(&cast_look); // eat ident
            while (lexer_peek(&cast_look).type == TOK_DCOLON)
            { // handle A::B
                lexer_next(&cast_look);
                if (lexer_peek(&cast_look).type == TOK_IDENT)
                {
                    lexer_next(&cast_look);
                }
                else
                {
                    break;
                }
            }
            while (lexer_peek(&cast_look).type == TOK_OP && is_token(lexer_peek(&cast_look), "*"))
            {
                lexer_next(&cast_look);
            }

            if (lexer_peek(&cast_look).type == TOK_RPAREN)
            {
                lexer_next(&cast_look); // eat )
                Token next = lexer_peek(&cast_look);
                // Heuristic: It's a cast if followed by literal, ident, paren, or &/*
                if (next.type == TOK_STRING || next.type == TOK_INT || next.type == TOK_FLOAT ||
                    (next.type == TOK_OP &&
                     (is_token(next, "&") || is_token(next, "*") || is_token(next, "!"))) ||
                    next.type == TOK_IDENT || next.type == TOK_LPAREN)
                {

                    Type *cast_type_obj = parse_type_formal(ctx, l);
                    char *cast_type = type_to_string(cast_type_obj);
                    {
                        Token t = lexer_next(l);
                        if (t.type != TOK_RPAREN)
                        {
                            zpanic_at(t, "Expected ) after cast");
                        }
                    }
                    ASTNode *target = parse_expr_prec(ctx, l, PREC_UNARY);

                    node = ast_create(NODE_EXPR_CAST);
                    node->cast.target_type = cast_type;
                    node->cast.expr = target;
                    node->type_info = cast_type_obj;
                    return node; // Casts are usually unary, handled here.
                }
            }
        }
        l->pos = saved; // Reset if not a cast

        ASTNode *expr = parse_expression(ctx, l);
        if (lexer_next(l).type != TOK_RPAREN)
        {
            zpanic_at(lexer_peek(l), "Expected )");
        }
        node = expr;
    }

    else if (t.type == TOK_LBRACKET)
    {
        ASTNode *head = NULL, *tail = NULL;
        int count = 0;
        while (lexer_peek(l).type != TOK_RBRACKET)
        {
            ASTNode *elem = parse_expression(ctx, l);
            count++;
            if (!head)
            {
                head = elem;
                tail = elem;
            }
            else
            {
                tail->next = elem;
                tail = elem;
            }
            if (lexer_peek(l).type == TOK_COMMA)
            {
                lexer_next(l);
            }
            else
            {
                break;
            }
        }
        if (lexer_next(l).type != TOK_RBRACKET)
        {
            zpanic_at(lexer_peek(l), "Expected ] after array literal");
        }
        node = ast_create(NODE_EXPR_ARRAY_LITERAL);
        node->array_literal.elements = head;
        node->array_literal.count = count;
        if (head && head->type_info)
        {
            Type *elem_type = head->type_info;
            Type *arr_type = type_new(TYPE_ARRAY);
            arr_type->inner = elem_type;
            arr_type->array_size = count;
            node->type_info = arr_type;
        }
    }
    else
    {
        zpanic_at(t, "Unexpected token in parse_primary: %.*s", t.len, t.start);
    }

    while (1)
    {
        if (lexer_peek(l).type == TOK_LPAREN)
        {
            Token op = lexer_next(l); // consume '('
            ASTNode *head = NULL, *tail = NULL;
            char **arg_names = NULL;
            int arg_count = 0;
            int has_named = 0;

            if (lexer_peek(l).type != TOK_RPAREN)
            {
                while (1)
                {
                    char *arg_name = NULL;

                    // Check for named argument: IDENT : expr
                    Token t1 = lexer_peek(l);
                    if (t1.type == TOK_IDENT)
                    {
                        Token t2 = lexer_peek2(l);
                        if (t2.type == TOK_COLON)
                        {
                            arg_name = token_strdup(t1);
                            has_named = 1;
                            lexer_next(l); // eat IDENT
                            lexer_next(l); // eat :
                        }
                    }

                    ASTNode *arg = parse_expression(ctx, l);
                    if (!head)
                    {
                        head = arg;
                    }
                    else
                    {
                        tail->next = arg;
                    }
                    tail = arg;

                    arg_names = xrealloc(arg_names, (arg_count + 1) * sizeof(char *));
                    arg_names[arg_count] = arg_name;
                    arg_count++;

                    if (lexer_peek(l).type == TOK_COMMA)
                    {
                        lexer_next(l);
                    }
                    else
                    {
                        break;
                    }
                }
            }
            {
                Token t = lexer_next(l);
                if (t.type != TOK_RPAREN)
                {
                    zpanic_at(t, "Expected ) after call arguments");
                }
            }

            ASTNode *call = ast_create(NODE_EXPR_CALL);
            call->call.callee = node;
            call->call.args = head;
            call->call.arg_names = has_named ? arg_names : NULL;
            call->call.arg_count = arg_count;
            check_format_string(call, op);

            // Try to infer type if callee has function type info
            call->resolved_type = xstrdup("unknown"); // Default (was int)
            if (node->type_info && node->type_info->kind == TYPE_FUNCTION && node->type_info->inner)
            {
                call->type_info = node->type_info->inner;

                // Update resolved_type based on real return
                // (Optional: type_to_string(call->type_info))
            }
            node = call;
        }

        else if (lexer_peek(l).type == TOK_LBRACKET)
        {
            Token bracket = lexer_next(l); // consume '['
            ASTNode *index = parse_expression(ctx, l);
            {
                Token t = lexer_next(l);
                if (t.type != TOK_RBRACKET)
                {
                    zpanic_at(t, "Expected ] after index");
                }
            }

            // Static Array Bounds Check
            if (node->type_info && node->type_info->kind == TYPE_ARRAY &&
                node->type_info->array_size > 0)
            {
                if (index->type == NODE_EXPR_LITERAL && index->literal.type_kind == 0)
                {
                    int idx = index->literal.int_val;
                    if (idx < 0 || idx >= node->type_info->array_size)
                    {
                        warn_array_bounds(bracket, idx, node->type_info->array_size);
                    }
                }
            }

            int overloaded_get = 0;
            if (node->type_info && node->type_info->kind != TYPE_ARRAY &&
                (node->type_info->kind == TYPE_STRUCT ||
                 (node->type_info->kind == TYPE_POINTER && node->type_info->inner &&
                  node->type_info->inner->kind == TYPE_STRUCT)))
            {
                Type *st = node->type_info;
                char *struct_name = (st->kind == TYPE_STRUCT) ? st->name : st->inner->name;
                int is_ptr = (st->kind == TYPE_POINTER);

                char mangled[256];
                sprintf(mangled, "%s__get", struct_name);
                FuncSig *sig = find_func(ctx, mangled);
                if (sig)
                {
                    // Rewrite to Call: node.get(index)
                    ASTNode *call = ast_create(NODE_EXPR_CALL);
                    ASTNode *callee = ast_create(NODE_EXPR_VAR);
                    callee->var_ref.name = xstrdup(mangled);
                    call->call.callee = callee;

                    // Arg 1: Self
                    ASTNode *arg1 = node;
                    if (sig->total_args > 0 && sig->arg_types[0]->kind == TYPE_POINTER && !is_ptr)
                    {
                        // Needs ptr, have value -> &node
                        ASTNode *addr = ast_create(NODE_EXPR_UNARY);
                        addr->unary.op = xstrdup("&");
                        addr->unary.operand = node;
                        addr->type_info = type_new_ptr(st);
                        arg1 = addr;
                    }
                    else if (is_ptr && sig->arg_types[0]->kind != TYPE_POINTER)
                    {
                        // Needs value, have ptr -> *node
                        ASTNode *deref = ast_create(NODE_EXPR_UNARY);
                        deref->unary.op = xstrdup("*");
                        deref->unary.operand = node;
                        arg1 = deref;
                    }

                    // Arg 2: Index
                    arg1->next = index;
                    index->next = NULL;
                    call->call.args = arg1;

                    call->type_info = sig->ret_type;
                    call->resolved_type = type_to_string(sig->ret_type);

                    node = call;
                    overloaded_get = 1;
                }
            }

            if (!overloaded_get)
            {
                ASTNode *idx_node = ast_create(NODE_EXPR_INDEX);
                idx_node->index.array = node;
                idx_node->index.index = index;
                idx_node->type_info = (node->type_info && node->type_info->inner)
                                          ? node->type_info->inner
                                          : type_new(TYPE_INT);
                node = idx_node;
            }
        }

        else
        {
            break;
        }
    }

    return node;
}

int is_comparison_op(const char *op)
{
    return (strcmp(op, "==") == 0 || strcmp(op, "!=") == 0 || strcmp(op, "<") == 0 ||
            strcmp(op, ">") == 0 || strcmp(op, "<=") == 0 || strcmp(op, ">=") == 0);
}

Type *get_field_type(ParserContext *ctx, Type *struct_type, const char *field_name)
{
    if (!struct_type)
    {
        return NULL;
    }

    // Built-in Fields for Arrays/Slices
    if (struct_type->kind == TYPE_ARRAY)
    {
        if (strcmp(field_name, "len") == 0)
        {
            return type_new(TYPE_INT);
        }
        if (struct_type->array_size == 0)
        { // Slice
            if (strcmp(field_name, "cap") == 0)
            {
                return type_new(TYPE_INT);
            }
            if (strcmp(field_name, "data") == 0)
            {
                return type_new_ptr(struct_type->inner);
            }
        }
    }

    char *sname = struct_type->name;
    // Handle Pointers (User* -> User)
    if (struct_type->kind == TYPE_POINTER && struct_type->inner)
    {
        sname = struct_type->inner->name;
    }
    if (!sname)
    {
        return NULL;
    }

    ASTNode *def = find_struct_def(ctx, sname);
    if (!def)
    {
        return NULL;
    }

    ASTNode *f = def->strct.fields;
    while (f)
    {
        if (strcmp(f->field.name, field_name) == 0)
        {
            return f->type_info;
        }
        f = f->next;
    }
    return NULL;
}

const char *get_operator_method(const char *op)
{
    // Arithmetic
    if (strcmp(op, "+") == 0)
    {
        return "add";
    }
    if (strcmp(op, "-") == 0)
    {
        return "sub";
    }
    if (strcmp(op, "*") == 0)
    {
        return "mul";
    }
    if (strcmp(op, "/") == 0)
    {
        return "div";
    }
    if (strcmp(op, "%") == 0)
    {
        return "rem";
    }

    // Comparison
    if (strcmp(op, "==") == 0)
    {
        return "eq";
    }
    if (strcmp(op, "!=") == 0)
    {
        return "neq";
    }
    if (strcmp(op, "<") == 0)
    {
        return "lt";
    }
    if (strcmp(op, ">") == 0)
    {
        return "gt";
    }
    if (strcmp(op, "<=") == 0)
    {
        return "le";
    }
    if (strcmp(op, ">=") == 0)
    {
        return "ge";
    }

    if (strcmp(op, "&") == 0)
    {
        return "bitand";
    }
    if (strcmp(op, "|") == 0)
    {
        return "bitor";
    }
    if (strcmp(op, "^") == 0)
    {
        return "bitxor";
    }
    if (strcmp(op, "<<") == 0)
    {
        return "shl";
    }
    if (strcmp(op, ">>") == 0)
    {
        return "shr";
    }

    return NULL;
}

char *resolve_struct_name_from_type(ParserContext *ctx, Type *t, int *is_ptr_out,
                                    char **allocated_out)
{
    if (!t)
    {
        return NULL;
    }
    char *struct_name = NULL;
    *allocated_out = NULL;
    *is_ptr_out = 0;

    const char *alias_target = NULL;
    if (t->kind == TYPE_STRUCT)
    {
        alias_target = find_type_alias(ctx, t->name);
    }

    if (alias_target)
    {
        char *tpl = xstrdup(alias_target);
        char *args_ptr = strchr(tpl, '<');
        if (args_ptr)
        {
            *args_ptr = 0;
            args_ptr++;
            char *end = strrchr(args_ptr, '>');
            if (end)
            {
                *end = 0;
            }

            const char *c_type = args_ptr;
            if (strcmp(args_ptr, "f32") == 0)
            {
                c_type = "float";
            }
            else if (strcmp(args_ptr, "f64") == 0)
            {
                c_type = "double";
            }
            else if (strcmp(args_ptr, "i32") == 0)
            {
                c_type = "int";
            }
            else if (strcmp(args_ptr, "u32") == 0)
            {
                c_type = "uint";
            }
            else if (strcmp(args_ptr, "bool") == 0)
            {
                c_type = "bool";
            }

            char *clean = sanitize_mangled_name(c_type);
            char *mangled = xmalloc(strlen(tpl) + strlen(clean) + 2);
            sprintf(mangled, "%s_%s", tpl, clean);
            struct_name = mangled;
            *allocated_out = mangled;
            free(clean);
            *is_ptr_out = 0;
        }
        else if (strchr(alias_target, '*'))
        {
            *is_ptr_out = 1;
            char *tmp = xstrdup(alias_target);
            char *p = strchr(tmp, '*');
            if (p)
            {
                *p = 0;
            }
            struct_name = xstrdup(tmp);
            *allocated_out = struct_name;
            free(tmp);
        }
        else
        {
            struct_name = xstrdup(alias_target);
            *allocated_out = struct_name;
            *is_ptr_out = 0;
        }
        free(tpl);
    }
    else
    {
        Type *struct_type = NULL;
        if (t->kind == TYPE_STRUCT)
        {
            struct_type = t;
            struct_name = t->name;
            *is_ptr_out = 0;
        }
        else if (t->kind == TYPE_POINTER && t->inner->kind == TYPE_STRUCT)
        {
            struct_type = t->inner;
            *is_ptr_out = 1;
        }

        if (struct_type)
        {
            if (struct_type->args && struct_type->arg_count > 0 && struct_type->name)
            {
                // It's a generic type instance (e.g. Foo<T>).
                // We must construct Foo_T, ensuring we measure SANITIZED length.
                int len = strlen(struct_type->name) + 1;

                // Pass 1: Calculate Length
                for (int i = 0; i < struct_type->arg_count; i++)
                {
                    Type *arg = struct_type->args[i];
                    if (arg)
                    {
                        char *s = type_to_string(arg);
                        if (s)
                        {
                            char *clean = sanitize_mangled_name(s);
                            if (clean)
                            {
                                len += strlen(clean) + 1; // +1 for '_'
                                free(clean);
                            }
                            free(s);
                        }
                    }
                }

                char *mangled = xmalloc(len + 1);
                strcpy(mangled, struct_type->name);

                // Pass 2: Build String
                for (int i = 0; i < struct_type->arg_count; i++)
                {
                    Type *arg = struct_type->args[i];
                    if (arg)
                    {
                        char *arg_str = type_to_string(arg);
                        if (arg_str)
                        {
                            char *clean = sanitize_mangled_name(arg_str);
                            if (clean)
                            {
                                strcat(mangled, "_");
                                strcat(mangled, clean);
                                free(clean);
                            }
                            free(arg_str);
                        }
                    }
                }
                struct_name = mangled;
                *allocated_out = mangled;
            }
            else if (struct_type->name && strchr(struct_type->name, '<'))
            {
                // Fallback: It's a generic type string. We need to mangle it.
                char *tpl = xstrdup(struct_type->name);
                char *args_ptr = strchr(tpl, '<');
                if (args_ptr)
                {
                    *args_ptr = 0;
                    args_ptr++;
                    char *end = strrchr(args_ptr, '>');
                    if (end)
                    {
                        *end = 0;
                    }

                    char *clean = sanitize_mangled_name(args_ptr);
                    char *mangled = xmalloc(strlen(tpl) + strlen(clean) + 2);
                    sprintf(mangled, "%s_%s", tpl, clean);
                    struct_name = mangled;
                    *allocated_out = mangled;
                    free(clean);
                }
                free(tpl);
            }
            else
            {
                struct_name = struct_type->name;
            }
        }
    }
    return struct_name;
}

ASTNode *parse_expr_prec(ParserContext *ctx, Lexer *l, Precedence min_prec)
{
    Token t = lexer_peek(l);
    ASTNode *lhs = NULL;

    if (t.type == TOK_QUESTION)
    {
        Lexer lookahead = *l;
        lexer_next(&lookahead);
        Token next = lexer_peek(&lookahead);

        if (next.type == TOK_STRING || next.type == TOK_FSTRING)
        {
            lexer_next(l); // consume '?'
            Token t_str = lexer_next(l);

            char *inner = xmalloc(t_str.len);
            if (t_str.type == TOK_FSTRING)
            {
                strncpy(inner, t_str.start + 2, t_str.len - 3);
                inner[t_str.len - 3] = 0;
            }
            else
            {
                strncpy(inner, t_str.start + 1, t_str.len - 2);
                inner[t_str.len - 2] = 0;
            }

            // Reuse printf sugar to generate the prompt print
            char *print_code = process_printf_sugar(ctx, inner, 0, "stdout", NULL, NULL);
            free(inner);

            // Checks for (args...) suffix for SCAN mode
            if (lexer_peek(l).type == TOK_LPAREN)
            {
                lexer_next(l); // consume (

                // Parse args
                ASTNode *args[16];
                int ac = 0;
                if (lexer_peek(l).type != TOK_RPAREN)
                {
                    while (1)
                    {
                        args[ac++] = parse_expression(ctx, l);
                        if (lexer_peek(l).type == TOK_COMMA)
                        {
                            lexer_next(l);
                        }
                        else
                        {
                            break;
                        }
                    }
                }
                if (lexer_next(l).type != TOK_RPAREN)
                {
                    zpanic_at(lexer_peek(l), "Expected )");
                }

                char fmt[256];
                fmt[0] = 0;
                for (int i = 0; i < ac; i++)
                {
                    Type *t = args[i]->type_info;
                    if (!t && args[i]->type == NODE_EXPR_VAR)
                    {
                        t = find_symbol_type_info(ctx, args[i]->var_ref.name);
                    }

                    if (!t)
                    {
                        strcat(fmt, "%d");
                    }
                    else
                    {
                        if (t->kind == TYPE_INT || t->kind == TYPE_I32 || t->kind == TYPE_BOOL)
                        {
                            strcat(fmt, "%d");
                        }
                        else if (t->kind == TYPE_F64)
                        {
                            strcat(fmt, "%lf");
                        }
                        else if (t->kind == TYPE_F32 || t->kind == TYPE_FLOAT)
                        {
                            strcat(fmt, "%f");
                        }
                        else if (t->kind == TYPE_STRING)
                        {
                            strcat(fmt, "%s");
                        }
                        else if (t->kind == TYPE_CHAR || t->kind == TYPE_I8 || t->kind == TYPE_U8 ||
                                 t->kind == TYPE_BYTE)
                        {
                            strcat(fmt, " %c");
                        }
                        else
                        {
                            strcat(fmt, "%d");
                        }
                    }
                    if (i < ac - 1)
                    {
                        strcat(fmt, " ");
                    }
                }

                ASTNode *block = ast_create(NODE_BLOCK);

                ASTNode *s1 = ast_create(NODE_RAW_STMT);
                // Append semicolon to ensure it's a valid statement
                char *s1_code = xmalloc(strlen(print_code) + 2);
                sprintf(s1_code, "%s;", print_code);
                s1->raw_stmt.content = s1_code;
                free(print_code);

                ASTNode *call = ast_create(NODE_EXPR_CALL);
                ASTNode *callee = ast_create(NODE_EXPR_VAR);
                callee->var_ref.name = xstrdup("_z_scan_helper");
                call->call.callee = callee;
                call->type_info = type_new(TYPE_INT);

                ASTNode *fmt_node = ast_create(NODE_EXPR_LITERAL);
                fmt_node->literal.type_kind = TOK_STRING;
                fmt_node->literal.string_val = xstrdup(fmt);
                ASTNode *head = fmt_node, *tail = fmt_node;

                for (int i = 0; i < ac; i++)
                {
                    ASTNode *addr = ast_create(NODE_EXPR_UNARY);
                    addr->unary.op = xstrdup("&");
                    addr->unary.operand = args[i];
                    tail->next = addr;
                    tail = addr;
                }
                call->call.args = head;

                // Link Statements
                s1->next = call;
                block->block.statements = s1;

                return block;
            }
            else
            {
                // String Mode (Original)
                size_t len = strlen(print_code);
                if (len > 5)
                {
                    print_code[len - 5] = 0; // Strip "0; })"
                }

                char *final_code = xmalloc(strlen(print_code) + 64);
                sprintf(final_code, "%s readln(); })", print_code);
                free(print_code);

                ASTNode *n = ast_create(NODE_RAW_STMT);
                n->raw_stmt.content = final_code;
                return n;
            }
        }
    }
    if (t.type == TOK_OP && is_token(t, "!"))
    {
        Lexer lookahead = *l;
        lexer_next(&lookahead);
        Token next = lexer_peek(&lookahead);

        if (next.type == TOK_STRING || next.type == TOK_FSTRING)
        {
            lexer_next(l); // consume '!'
            Token t_str = lexer_next(l);

            char *inner = xmalloc(t_str.len);
            if (t_str.type == TOK_FSTRING)
            {
                strncpy(inner, t_str.start + 2, t_str.len - 3);
                inner[t_str.len - 3] = 0;
            }
            else
            {
                strncpy(inner, t_str.start + 1, t_str.len - 2);
                inner[t_str.len - 2] = 0;
            }

            // Check for .. suffix (.. suppresses newline)
            int newline = 1;
            if (lexer_peek(l).type == TOK_DOTDOT)
            {
                lexer_next(l); // consume ..
                newline = 0;
            }

            char *code = process_printf_sugar(ctx, inner, newline, "stderr", NULL, NULL);
            free(inner);

            ASTNode *n = ast_create(NODE_RAW_STMT);
            n->raw_stmt.content = code;
            return n;
        }
    }

    if (t.type == TOK_AWAIT)
    {
        lexer_next(l); // consume await
        ASTNode *operand = parse_expr_prec(ctx, l, PREC_UNARY);

        lhs = ast_create(NODE_AWAIT);
        lhs->unary.operand = operand;
        // Type inference: await Async<T> yields T
        // If operand is a call to an async function, look up its ret_type (not
        // Async)
        if (operand->type == NODE_EXPR_CALL && operand->call.callee->type == NODE_EXPR_VAR)
        {
            FuncSig *sig = find_func(ctx, operand->call.callee->var_ref.name);
            if (sig && sig->is_async && sig->ret_type)
            {
                lhs->type_info = sig->ret_type;
                lhs->resolved_type = type_to_string(sig->ret_type);
            }
            else if (sig && !sig->is_async)
            {
                // Not an async function - shouldn't await it
                lhs->type_info = type_new(TYPE_VOID);
                lhs->resolved_type = xstrdup("void");
            }
            else
            {
                lhs->type_info = type_new_ptr(type_new(TYPE_VOID));
                lhs->resolved_type = xstrdup("void*");
            }
        }
        else
        {
            // Awaiting a variable - harder to determine underlying type
            // Fallback to void* for now (could be improved with metadata)
            lhs->type_info = type_new_ptr(type_new(TYPE_VOID));
            lhs->resolved_type = xstrdup("void*");
        }

        goto after_unary;
    }

    if (t.type == TOK_OP &&
        (is_token(t, "-") || is_token(t, "!") || is_token(t, "*") || is_token(t, "&") ||
         is_token(t, "~") || is_token(t, "&&") || is_token(t, "++") || is_token(t, "--")))
    {
        lexer_next(l); // consume op
        ASTNode *operand = parse_expr_prec(ctx, l, PREC_UNARY);

        char *method = NULL;
        if (is_token(t, "-"))
        {
            method = "neg";
        }
        if (is_token(t, "!"))
        {
            method = "not";
        }
        if (is_token(t, "~"))
        {
            method = "bitnot";
        }

        if (method && operand->type_info)
        {
            Type *ot = operand->type_info;
            int is_ptr = 0;
            char *allocated_name = NULL;
            char *struct_name = resolve_struct_name_from_type(ctx, ot, &is_ptr, &allocated_name);

            if (struct_name)
            {
                char mangled[256];
                sprintf(mangled, "%s__%s", struct_name, method);

                if (find_func(ctx, mangled))
                {
                    // Rewrite: ~x -> Struct_bitnot(x)
                    ASTNode *call = ast_create(NODE_EXPR_CALL);
                    ASTNode *callee = ast_create(NODE_EXPR_VAR);
                    callee->var_ref.name = xstrdup(mangled);
                    call->call.callee = callee;

                    // Handle 'self' argument adjustment (Pointer vs Value)
                    ASTNode *arg = operand;
                    FuncSig *sig = find_func(ctx, mangled);

                    if (sig->total_args > 0 && sig->arg_types[0]->kind == TYPE_POINTER && !is_ptr)
                    {
                        int is_rvalue =
                            (operand->type == NODE_EXPR_CALL || operand->type == NODE_EXPR_BINARY ||
                             operand->type == NODE_MATCH);
                        ASTNode *addr = ast_create(NODE_EXPR_UNARY);
                        addr->unary.op = is_rvalue ? xstrdup("&_rval") : xstrdup("&");
                        addr->unary.operand = operand;
                        addr->type_info = type_new_ptr(ot);
                        arg = addr;
                    }
                    else if (is_ptr && sig->arg_types[0]->kind != TYPE_POINTER)
                    {
                        // Function wants Value, we have Pointer -> Dereference (*)
                        ASTNode *deref = ast_create(NODE_EXPR_UNARY);
                        deref->unary.op = xstrdup("*");
                        deref->unary.operand = operand;
                        deref->type_info = ot->inner;
                        arg = deref;
                    }

                    call->call.args = arg;
                    call->type_info = sig->ret_type;
                    call->resolved_type = type_to_string(sig->ret_type);
                    lhs = call;

                    if (allocated_name)
                    {
                        free(allocated_name);
                    }
                    // Skip standard unary node creation
                    goto after_unary;
                }
                if (allocated_name)
                {
                    free(allocated_name);
                }
            }
        }

        // Standard Unary Node (for primitives or if no overload found)
        lhs = ast_create(NODE_EXPR_UNARY);
        lhs->unary.op = token_strdup(t);
        lhs->unary.operand = operand;

        if (operand->type_info)
        {
            if (is_token(t, "&"))
            {
                lhs->type_info = type_new_ptr(operand->type_info);
            }
            else if (is_token(t, "*"))
            {
                if (operand->type_info->kind == TYPE_POINTER)
                {
                    lhs->type_info = operand->type_info->inner;
                }
            }
            else
            {
                lhs->type_info = operand->type_info;
            }
        }

    after_unary:; // Label to skip standard creation if overloaded
    }

    else if (is_token(t, "sizeof"))
    {
        lexer_next(l);
        if (lexer_peek(l).type == TOK_LPAREN)
        {
            const char *start = l->src + l->pos;
            int depth = 0;
            while (1)
            {
                Token tk = lexer_peek(l);
                if (tk.type == TOK_EOF)
                {
                    zpanic_at(tk, "Unterminated sizeof");
                }
                if (tk.type == TOK_LPAREN)
                {
                    depth++;
                }
                if (tk.type == TOK_RPAREN)
                {
                    depth--;
                    if (depth == 0)
                    {
                        lexer_next(l);
                        break;
                    }
                }
                lexer_next(l);
            }
            int len = (l->src + l->pos) - start;
            char *content = xmalloc(len + 8);
            sprintf(content, "sizeof%.*s", len, start);
            lhs = ast_create(NODE_RAW_STMT);
            lhs->raw_stmt.content = content;
            lhs->type_info = type_new(TYPE_INT);
        }
        else
        {
            zpanic_at(lexer_peek(l), "sizeof must be followed by (");
        }
    }
    else
    {
        lhs = parse_primary(ctx, l);
    }

    while (1)
    {
        Token op = lexer_peek(l);

        Precedence prec = get_token_precedence(op);

        // Handle postfix ++ and -- (highest postfix precedence)
        if (op.type == TOK_OP && op.len == 2 &&
            ((op.start[0] == '+' && op.start[1] == '+') ||
             (op.start[0] == '-' && op.start[1] == '-')))
        {
            lexer_next(l); // consume ++ or --
            ASTNode *node = ast_create(NODE_EXPR_UNARY);
            node->unary.op = (op.start[0] == '+') ? xstrdup("_post++") : xstrdup("_post--");
            node->unary.operand = lhs;
            node->type_info = lhs->type_info;
            lhs = node;
            continue;
        }

        if (prec == PREC_NONE || prec < min_prec)
        {
            break;
        }

        // Pointer access: ->
        if (op.type == TOK_ARROW && op.start[0] == '-')
        {
            lexer_next(l);
            Token field = lexer_next(l);
            if (field.type != TOK_IDENT)
            {
                zpanic_at(field, "Expected field name after ->");
                break;
            }
            ASTNode *node = ast_create(NODE_EXPR_MEMBER);
            node->member.target = lhs;
            node->member.field = token_strdup(field);
            node->member.is_pointer_access = 1;

            node->type_info = get_field_type(ctx, lhs->type_info, node->member.field);
            if (node->type_info)
            {
                node->resolved_type = type_to_string(node->type_info);
            }
            else
            {
                node->resolved_type = xstrdup("unknown");
            }

            lhs = node;
            continue;
        }

        // Null-safe access: ?.
        if (op.type == TOK_Q_DOT)
        {
            lexer_next(l);
            Token field = lexer_next(l);
            if (field.type != TOK_IDENT)
            {
                zpanic_at(field, "Expected field name after ?.");
                break;
            }
            ASTNode *node = ast_create(NODE_EXPR_MEMBER);
            node->member.target = lhs;
            node->member.field = token_strdup(field);
            node->member.is_pointer_access = 2;

            node->type_info = get_field_type(ctx, lhs->type_info, node->member.field);
            if (node->type_info)
            {
                node->resolved_type = type_to_string(node->type_info);
            }

            lhs = node;
            continue;
        }

        // Postfix ? (Result Unwrap OR Ternary)
        if (op.type == TOK_QUESTION)
        {
            // Disambiguate
            Lexer lookahead = *l;
            lexer_next(&lookahead); // skip ?
            Token next = lexer_peek(&lookahead);

            // Heuristic: If next token starts an expression => Ternary
            // (Ident, Number, String, (, {, -, !, *, etc)
            int is_ternary = 0;
            if (next.type == TOK_INT || next.type == TOK_FLOAT || next.type == TOK_STRING ||
                next.type == TOK_IDENT || next.type == TOK_LPAREN || next.type == TOK_LBRACE ||
                next.type == TOK_SIZEOF || next.type == TOK_DEFER || next.type == TOK_AUTOFREE ||
                next.type == TOK_FSTRING || next.type == TOK_CHAR)
            {
                is_ternary = 1;
            }
            // Check unary ops
            if (next.type == TOK_OP)
            {
                if (is_token(next, "-") || is_token(next, "!") || is_token(next, "*") ||
                    is_token(next, "&") || is_token(next, "~"))
                {
                    is_ternary = 1;
                }
            }

            if (is_ternary)
            {
                if (PREC_TERNARY < min_prec)
                {
                    break; // Return to caller to handle precedence
                }

                lexer_next(l); // consume ?
                ASTNode *true_expr = parse_expression(ctx, l);
                expect(l, TOK_COLON, "Expected : in ternary");
                ASTNode *false_expr = parse_expr_prec(ctx, l, PREC_TERNARY); // Right associative

                ASTNode *tern = ast_create(NODE_TERNARY);
                zen_trigger_at(TRIGGER_TERNARY, lhs->token);

                tern->ternary.cond = lhs;
                tern->ternary.true_expr = true_expr;
                tern->ternary.false_expr = false_expr;

                // Type inference hint: Both branches should match?
                // Logic later in codegen/semant.
                lhs = tern;
                continue;
            }

            // Otherwise: Unwrap (High Precedence)
            if (PREC_CALL < min_prec)
            {
                break;
            }

            lexer_next(l);
            ASTNode *n = ast_create(NODE_TRY);
            n->try_stmt.expr = lhs;
            lhs = n;
            continue;
        }

        // Pipe: |>
        if (op.type == TOK_PIPE || (op.type == TOK_OP && is_token(op, "|>")))
        {
            lexer_next(l);
            ASTNode *rhs = parse_expr_prec(ctx, l, prec + 1);
            if (rhs->type == NODE_EXPR_CALL)
            {
                ASTNode *old_args = rhs->call.args;
                lhs->next = old_args;
                rhs->call.args = lhs;
                lhs = rhs;
            }
            else
            {
                ASTNode *call = ast_create(NODE_EXPR_CALL);
                call->call.callee = rhs;
                call->call.args = lhs;
                lhs->next = NULL;
                lhs = call;
            }
            continue;
        }

        lexer_next(l); // Consume operator/paren/bracket

        // Call: (...)
        if (op.type == TOK_LPAREN)
        {
            ASTNode *call = ast_create(NODE_EXPR_CALL);
            call->call.callee = lhs;
            ASTNode *head = NULL, *tail = NULL;
            char **arg_names = NULL;
            int arg_count = 0;
            int has_named = 0;

            if (lexer_peek(l).type != TOK_RPAREN)
            {
                while (1)
                {
                    char *arg_name = NULL;

                    // Check for named argument: IDENT : expr
                    Token t1 = lexer_peek(l);
                    if (t1.type == TOK_IDENT)
                    {
                        // Lookahead for colon
                        Token t2 = lexer_peek2(l);
                        if (t2.type == TOK_COLON)
                        {
                            arg_name = token_strdup(t1);
                            has_named = 1;
                            lexer_next(l); // eat IDENT
                            lexer_next(l); // eat :
                        }
                    }

                    ASTNode *arg = parse_expression(ctx, l);
                    if (!head)
                    {
                        head = arg;
                    }
                    else
                    {
                        tail->next = arg;
                    }
                    tail = arg;

                    // Store arg name
                    arg_names = xrealloc(arg_names, (arg_count + 1) * sizeof(char *));
                    arg_names[arg_count] = arg_name;
                    arg_count++;

                    if (lexer_peek(l).type == TOK_COMMA)
                    {
                        lexer_next(l);
                    }
                    else
                    {
                        break;
                    }
                }
            }
            if (lexer_next(l).type != TOK_RPAREN)
            {
                zpanic_at(lexer_peek(l), "Expected )");
            }
            call->call.args = head;
            call->call.arg_names = has_named ? arg_names : NULL;
            call->call.arg_count = arg_count;

            call->resolved_type = xstrdup("unknown");
            if (lhs->type_info && lhs->type_info->kind == TYPE_FUNCTION && lhs->type_info->inner)
            {
                call->type_info = lhs->type_info->inner;
            }

            lhs = call;
            continue;
        }

        // Index: [...] or Slice: [start..end]
        if (op.type == TOK_LBRACKET || (op.type == TOK_OP && is_token(op, "[")))
        {
            ASTNode *start = NULL;
            ASTNode *end = NULL;
            int is_slice = 0;

            // Fallback: If LHS is a variable but missing type info, look it up now
            if (!lhs->type_info && lhs->type == NODE_EXPR_VAR)
            {
                Type *sym_type = find_symbol_type_info(ctx, lhs->var_ref.name);
                if (sym_type)
                {
                    lhs->type_info = sym_type;
                    lhs->resolved_type = type_to_string(sym_type);
                }
            }

            // Case: [..] or [..end]
            if (lexer_peek(l).type == TOK_DOTDOT || lexer_peek(l).type == TOK_DOTDOT_LT)
            {
                is_slice = 1;
                lexer_next(l); // consume .. or ..<
                if (lexer_peek(l).type != TOK_RBRACKET)
                {
                    end = parse_expression(ctx, l);
                }
            }
            else
            {
                // Case: [start] or [start..] or [start..end]
                start = parse_expression(ctx, l);
                if (lexer_peek(l).type == TOK_DOTDOT || lexer_peek(l).type == TOK_DOTDOT_LT)
                {
                    is_slice = 1;
                    lexer_next(l); // consume ..
                    if (lexer_peek(l).type != TOK_RBRACKET)
                    {
                        end = parse_expression(ctx, l);
                    }
                }
            }

            if (lexer_next(l).type != TOK_RBRACKET)
            {
                zpanic_at(lexer_peek(l), "Expected ]");
            }

            if (is_slice)
            {
                ASTNode *node = ast_create(NODE_EXPR_SLICE);
                node->slice.array = lhs;
                node->slice.start = start;
                node->slice.end = end;

                // Type Inference & Registration
                if (lhs->type_info)
                {
                    Type *inner = NULL;
                    if (lhs->type_info->kind == TYPE_ARRAY)
                    {
                        inner = lhs->type_info->inner;
                    }
                    else if (lhs->type_info->kind == TYPE_POINTER)
                    {
                        inner = lhs->type_info->inner;
                    }

                    if (inner)
                    {
                        node->type_info = type_new(TYPE_ARRAY);
                        node->type_info->inner = inner;
                        node->type_info->array_size = 0; // Slice

                        // Clean up string for registration (e.g. "int" from "int*")
                        char *inner_str = type_to_string(inner);

                        // Strip * if it somehow managed to keep one, though
                        // parse_type_formal should handle it For now assume type_to_string
                        // gives base type
                        register_slice(ctx, inner_str);
                    }
                }

                lhs = node;
            }
            else
            {
                ASTNode *node = ast_create(NODE_EXPR_INDEX);
                node->index.array = lhs;
                node->index.index = start;

                char *struct_name = NULL;
                Type *t = lhs->type_info;
                int is_ptr = 0;

                if (t)
                {
                    if (t->kind == TYPE_STRUCT)
                    {
                        struct_name = t->name;
                    }
                    else if (t->kind == TYPE_POINTER && t->inner && t->inner->kind == TYPE_STRUCT)
                    {
                        struct_name = t->inner->name;
                        is_ptr = 1;
                    }
                }
                if (!struct_name && lhs->resolved_type)
                {
                    char *s = lhs->resolved_type;
                    if (strncmp(s, "struct ", 7) == 0)
                    {
                        s += 7;
                    }
                    ASTNode *def = find_struct_def(ctx, s);
                    if (def && def->type == NODE_STRUCT)
                    {
                        struct_name = s;
                        if (strchr(lhs->resolved_type, '*'))
                        {
                            is_ptr = 1;
                        }
                    }
                    if (!struct_name)
                    {
                        def = find_struct_def(ctx, lhs->resolved_type);
                        if (def && def->type == NODE_STRUCT)
                        {
                            struct_name = lhs->resolved_type;
                            // Just assume val type
                        }
                    }
                }

                if (struct_name)
                {
                    char mangled[256];
                    sprintf(mangled, "%s__get", struct_name);

                    if (find_func(ctx, mangled))
                    {
                        // Rewrite to Call
                        ASTNode *call = ast_create(NODE_EXPR_CALL);
                        ASTNode *callee = ast_create(NODE_EXPR_VAR);
                        callee->var_ref.name = xstrdup(mangled);
                        call->call.callee = callee;

                        ASTNode *self_arg = lhs;
                        FuncSig *sig = find_func(ctx, mangled);

                        // Pointer adjustment logic
                        if (sig->total_args > 0 && sig->arg_types[0]->kind == TYPE_POINTER &&
                            !is_ptr)
                        {
                            ASTNode *addr = ast_create(NODE_EXPR_UNARY);
                            addr->unary.op = xstrdup("&");
                            addr->unary.operand = lhs;
                            if (t)
                            {
                                addr->type_info = type_new_ptr(t);
                            }
                            self_arg = addr;
                        }
                        else if (is_ptr && sig->arg_types[0]->kind != TYPE_POINTER)
                        {
                            ASTNode *deref = ast_create(NODE_EXPR_UNARY);
                            deref->unary.op = xstrdup("*");
                            deref->unary.operand = lhs;
                            self_arg = deref;
                        }

                        self_arg->next = start;
                        call->call.args = self_arg;
                        call->type_info = sig->ret_type;
                        call->resolved_type = type_to_string(sig->ret_type);

                        lhs = call;
                        continue;
                    }
                }

                // Static Array Bounds Check
                if (lhs->type_info && lhs->type_info->kind == TYPE_ARRAY &&
                    lhs->type_info->array_size > 0)
                {
                    if (start->type == NODE_EXPR_LITERAL && start->literal.type_kind == 0)
                    {
                        int idx = start->literal.int_val;
                        if (idx < 0 || idx >= lhs->type_info->array_size)
                        {
                            warn_array_bounds(op, idx, lhs->type_info->array_size);
                        }
                    }
                }

                lhs = node;
            }
            continue;
        }

        // Member: .
        if (op.type == TOK_OP && is_token(op, "."))
        {
            Token field = lexer_next(l);
            if (field.type != TOK_IDENT)
            {
                zpanic_at(field, "Expected field name after .");
                break;
            }
            ASTNode *node = ast_create(NODE_EXPR_MEMBER);
            node->member.target = lhs;
            node->member.field = token_strdup(field);
            node->member.is_pointer_access = 0;

            if (lhs->type_info && lhs->type_info->kind == TYPE_POINTER)
            {
                node->member.is_pointer_access = 1;

                // Special case: .val() on pointer = dereference
                if (strcmp(node->member.field, "val") == 0 && lexer_peek(l).type == TOK_LPAREN)
                {
                    lexer_next(l); // consume (
                    if (lexer_peek(l).type == TOK_RPAREN)
                    {
                        lexer_next(l); // consume )
                        // Rewrite to dereference: *ptr
                        ASTNode *deref = ast_create(NODE_EXPR_UNARY);
                        deref->unary.op = xstrdup("*");
                        deref->unary.operand = lhs;
                        deref->type_info = lhs->type_info->inner;
                        lhs = deref;
                        continue;
                    }
                }
            }
            else if (lhs->type == NODE_EXPR_VAR)
            {
                char *type = find_symbol_type(ctx, lhs->var_ref.name);
                if (type && strchr(type, '*'))
                {
                    node->member.is_pointer_access = 1;

                    // Special case: .val() on pointer = dereference
                    if (strcmp(node->member.field, "val") == 0 && lexer_peek(l).type == TOK_LPAREN)
                    {
                        lexer_next(l); // consume (
                        if (lexer_peek(l).type == TOK_RPAREN)
                        {
                            lexer_next(l); // consume )
                            // Rewrite to dereference: *ptr
                            ASTNode *deref = ast_create(NODE_EXPR_UNARY);
                            deref->unary.op = xstrdup("*");
                            deref->unary.operand = lhs;
                            // Try to get inner type
                            if (lhs->type_info && lhs->type_info->kind == TYPE_POINTER)
                            {
                                deref->type_info = lhs->type_info->inner;
                            }
                            lhs = deref;
                            continue;
                        }
                    }
                }
                if (strcmp(lhs->var_ref.name, "self") == 0 && !node->member.is_pointer_access)
                {
                    node->member.is_pointer_access = 1;
                }
            }

            node->type_info = get_field_type(ctx, lhs->type_info, node->member.field);

            if (!node->type_info && lhs->type_info)
            {
                char *struct_name = NULL;
                Type *st = lhs->type_info;
                if (st->kind == TYPE_STRUCT)
                {
                    struct_name = st->name;
                }
                else if (st->kind == TYPE_POINTER && st->inner && st->inner->kind == TYPE_STRUCT)
                {
                    struct_name = st->inner->name;
                }

                if (struct_name)
                {
                    char mangled[256];
                    sprintf(mangled, "%s__%s", struct_name, node->member.field);

                    FuncSig *sig = find_func(ctx, mangled);
                    if (sig)
                    {
                        // It is a method! Create a Function Type Info to carry the return
                        // type
                        Type *ft = type_new(TYPE_FUNCTION);
                        ft->name = xstrdup(mangled);
                        ft->inner = sig->ret_type; // Return type
                        node->type_info = ft;
                    }
                }
            }

            if (node->type_info)
            {
                node->resolved_type = type_to_string(node->type_info);
            }
            else
            {
                node->resolved_type = xstrdup("unknown");
            }

            lhs = node;
            continue;
        }

        ASTNode *rhs = parse_expr_prec(ctx, l, prec + 1);
        ASTNode *bin = ast_create(NODE_EXPR_BINARY);
        bin->token = op;
        if (op.type == TOK_OP)
        {
            if (is_token(op, "&") || is_token(op, "|") || is_token(op, "^"))
            {
                zen_trigger_at(TRIGGER_BITWISE, op);
            }
            else if (is_token(op, "<<") || is_token(op, ">>"))
            {
                zen_trigger_at(TRIGGER_BITWISE, op);
            }
        }
        bin->binary.left = lhs;
        bin->binary.right = rhs;

        if (op.type == TOK_LANGLE)
        {
            bin->binary.op = xstrdup("<");
        }
        else if (op.type == TOK_RANGLE)
        {
            bin->binary.op = xstrdup(">");
        }
        else if (op.type == TOK_AND)
        {
            bin->binary.op = xstrdup("&&");
        }
        else if (op.type == TOK_OR)
        {
            bin->binary.op = xstrdup("||");
        }
        else
        {
            bin->binary.op = token_strdup(op);
        }

        if (strcmp(bin->binary.op, "/") == 0 || strcmp(bin->binary.op, "%") == 0)
        {
            if (rhs->type == NODE_EXPR_LITERAL && rhs->literal.type_kind == 0 &&
                rhs->literal.int_val == 0)
            {
                warn_division_by_zero(op);
            }
        }

        if (is_comparison_op(bin->binary.op))
        {
            // Check for identical operands (x == x)
            if (lhs->type == NODE_EXPR_VAR && rhs->type == NODE_EXPR_VAR)
            {
                if (strcmp(lhs->var_ref.name, rhs->var_ref.name) == 0)
                {
                    if (strcmp(bin->binary.op, "==") == 0 || strcmp(bin->binary.op, ">=") == 0 ||
                        strcmp(bin->binary.op, "<=") == 0)
                    {
                        warn_comparison_always_true(op, "Comparing a variable to itself");
                    }
                    else if (strcmp(bin->binary.op, "!=") == 0 ||
                             strcmp(bin->binary.op, ">") == 0 || strcmp(bin->binary.op, "<") == 0)
                    {
                        warn_comparison_always_false(op, "Comparing a variable to itself");
                    }
                }
            }
            else if (lhs->type == NODE_EXPR_LITERAL && lhs->literal.type_kind == 0 &&
                     rhs->type == NODE_EXPR_LITERAL && rhs->literal.type_kind == 0)
            {
                // Check if literals make sense (e.g. 5 > 5)
                if (lhs->literal.int_val == rhs->literal.int_val)
                {
                    if (strcmp(bin->binary.op, "==") == 0 || strcmp(bin->binary.op, ">=") == 0 ||
                        strcmp(bin->binary.op, "<=") == 0)
                    {
                        warn_comparison_always_true(op, "Comparing identical literals");
                    }
                    else
                    {
                        warn_comparison_always_false(op, "Comparing identical literals");
                    }
                }
            }

            if (lhs->type_info && type_is_unsigned(lhs->type_info))
            {
                if (rhs->type == NODE_EXPR_LITERAL && rhs->literal.type_kind == 0 &&
                    rhs->literal.int_val == 0)
                {
                    if (strcmp(bin->binary.op, ">=") == 0)
                    {
                        warn_comparison_always_true(op, "Unsigned value is always >= 0");
                    }
                    else if (strcmp(bin->binary.op, "<") == 0)
                    {
                        warn_comparison_always_false(op, "Unsigned value is never < 0");
                    }
                }
            }
        }

        if (strcmp(bin->binary.op, "=") == 0 || strcmp(bin->binary.op, "+=") == 0 ||
            strcmp(bin->binary.op, "-=") == 0 || strcmp(bin->binary.op, "*=") == 0 ||
            strcmp(bin->binary.op, "/=") == 0)
        {

            if (lhs->type == NODE_EXPR_VAR)
            {
                // Check if the variable is const
                Type *t = find_symbol_type_info(ctx, lhs->var_ref.name);
                if (t && t->is_const)
                {
                    zpanic_at(op, "Cannot assign to const variable '%s'", lhs->var_ref.name);
                }
            }
        }

        int is_compound = 0;
        size_t op_len = strlen(bin->binary.op);

        // Check if operator ends with '=' but is not ==, !=, <=, >=
        if (op_len > 1 && bin->binary.op[op_len - 1] == '=')
        {
            char c = bin->binary.op[0];
            if (c != '=' && c != '!' && c != '<' && c != '>')
            {
                is_compound = 1;
            }
            // Special handle for <<= and >>=
            if (strcmp(bin->binary.op, "<<=") == 0 || strcmp(bin->binary.op, ">>=") == 0)
            {
                is_compound = 1;
            }
        }

        if (is_compound)
        {
            ASTNode *op_node = ast_create(NODE_EXPR_BINARY);
            op_node->binary.left = lhs;
            op_node->binary.right = rhs;

            // Extract the base operator (remove last char '=')
            char *inner_op = xmalloc(op_len);
            strncpy(inner_op, bin->binary.op, op_len - 1);
            inner_op[op_len - 1] = '\0';
            op_node->binary.op = inner_op;

            // Inherit type info temporarily
            if (lhs->type_info && rhs->type_info && type_eq(lhs->type_info, rhs->type_info))
            {
                op_node->type_info = lhs->type_info;
            }

            const char *inner_method = get_operator_method(inner_op);
            if (inner_method)
            {
                Type *lt = lhs->type_info;
                int is_lhs_ptr = 0;
                char *allocated_name = NULL;
                char *struct_name =
                    resolve_struct_name_from_type(ctx, lt, &is_lhs_ptr, &allocated_name);

                if (struct_name)
                {
                    char mangled[256];
                    sprintf(mangled, "%s__%s", struct_name, inner_method);
                    FuncSig *sig = find_func(ctx, mangled);
                    if (sig)
                    {
                        // Rewrite op_node from BINARY -> CALL
                        ASTNode *call = ast_create(NODE_EXPR_CALL);
                        ASTNode *callee = ast_create(NODE_EXPR_VAR);
                        callee->var_ref.name = xstrdup(mangled);
                        call->call.callee = callee;

                        // Handle 'self' argument
                        ASTNode *arg1 = lhs;
                        if (sig->total_args > 0 && sig->arg_types[0]->kind == TYPE_POINTER &&
                            !is_lhs_ptr)
                        {
                            ASTNode *addr = ast_create(NODE_EXPR_UNARY);
                            addr->unary.op = xstrdup("&");
                            addr->unary.operand = lhs;
                            addr->type_info = type_new_ptr(lt);
                            arg1 = addr;
                        }
                        else if (is_lhs_ptr && sig->arg_types[0]->kind != TYPE_POINTER)
                        {
                            ASTNode *deref = ast_create(NODE_EXPR_UNARY);
                            deref->unary.op = xstrdup("*");
                            deref->unary.operand = lhs;
                            arg1 = deref;
                        }

                        call->call.args = arg1;
                        arg1->next = rhs;
                        rhs->next = NULL;
                        call->type_info = sig->ret_type;

                        // Replace op_node with the call
                        op_node = call;
                    }
                }
                if (allocated_name)
                {
                    free(allocated_name);
                }
            }

            free(bin->binary.op);
            bin->binary.op = xstrdup("=");
            bin->binary.right = op_node;
        }

        // Index Set Overload: Call(get, idx) = val  -->  Call(set, idx, val)
        if (strcmp(bin->binary.op, "=") == 0 && lhs->type == NODE_EXPR_CALL)
        {
            if (lhs->call.callee->type == NODE_EXPR_VAR)
            {
                char *name = lhs->call.callee->var_ref.name;
                // Check if it ends in "_get"
                size_t len = strlen(name);
                if (len > 4 && strcmp(name + len - 4, "_get") == 0)
                {
                    char *set_name = xstrdup(name);
                    set_name[len - 3] = 's'; // Replace 'g' with 's' -> _set
                    set_name[len - 2] = 'e';
                    set_name[len - 1] = 't';

                    if (find_func(ctx, set_name))
                    {
                        // Create NEW Call Node for Set
                        ASTNode *set_call = ast_create(NODE_EXPR_CALL);
                        ASTNode *set_callee = ast_create(NODE_EXPR_VAR);
                        set_callee->var_ref.name = set_name;
                        set_call->call.callee = set_callee;

                        // Clone argument list (Shallow copy of arg nodes to preserve chain
                        // for get)
                        ASTNode *lhs_args = lhs->call.args;
                        ASTNode *new_head = NULL;
                        ASTNode *new_tail = NULL;

                        while (lhs_args)
                        {
                            ASTNode *arg_copy = xmalloc(sizeof(ASTNode));
                            memcpy(arg_copy, lhs_args, sizeof(ASTNode));
                            arg_copy->next = NULL;

                            if (!new_head)
                            {
                                new_head = arg_copy;
                            }
                            else
                            {
                                new_tail->next = arg_copy;
                            }
                            new_tail = arg_copy;

                            lhs_args = lhs_args->next;
                        }

                        // Append RHS to new args
                        ASTNode *val_expr = bin->binary.right;
                        if (new_tail)
                        {
                            new_tail->next = val_expr;
                        }
                        else
                        {
                            new_head = val_expr;
                        }

                        set_call->call.args = new_head;
                        set_call->type_info = type_new(TYPE_VOID);

                        lhs = set_call; // Use the new Set call as the result
                        continue;
                    }
                    else
                    {
                        free(set_name);
                    }
                }
            }
        }

        const char *method = get_operator_method(bin->binary.op);

        if (method)
        {
            Type *lt = lhs->type_info;
            int is_lhs_ptr = 0;
            char *allocated_name = NULL;
            char *struct_name =
                resolve_struct_name_from_type(ctx, lt, &is_lhs_ptr, &allocated_name);

            if (struct_name)
            {
                char mangled[256];
                sprintf(mangled, "%s__%s", struct_name, method);

                FuncSig *sig = find_func(ctx, mangled);

                if (sig)
                {
                    ASTNode *call = ast_create(NODE_EXPR_CALL);
                    ASTNode *callee = ast_create(NODE_EXPR_VAR);
                    callee->var_ref.name = xstrdup(mangled);
                    call->call.callee = callee;

                    ASTNode *arg1 = lhs;

                    // Check if function expects a pointer for 'self'
                    if (sig->total_args > 0 && sig->arg_types[0] &&
                        sig->arg_types[0]->kind == TYPE_POINTER)
                    {
                        if (!is_lhs_ptr)
                        {
                            // Value -> Pointer.
                            int is_rvalue =
                                (lhs->type == NODE_EXPR_CALL || lhs->type == NODE_EXPR_BINARY ||
                                 lhs->type == NODE_EXPR_STRUCT_INIT ||
                                 lhs->type == NODE_EXPR_CAST || lhs->type == NODE_MATCH);

                            ASTNode *addr = ast_create(NODE_EXPR_UNARY);
                            addr->unary.op = is_rvalue ? xstrdup("&_rval") : xstrdup("&");
                            addr->unary.operand = lhs;
                            addr->type_info = type_new_ptr(lt);
                            arg1 = addr;
                        }
                    }
                    else
                    {
                        // Function expects value
                        if (is_lhs_ptr)
                        {
                            // Have pointer, need value -> *lhs
                            ASTNode *deref = ast_create(NODE_EXPR_UNARY);
                            deref->unary.op = xstrdup("*");
                            deref->unary.operand = lhs;
                            if (lt && lt->kind == TYPE_POINTER)
                            {
                                deref->type_info = lt->inner;
                            }
                            arg1 = deref;
                        }
                    }

                    call->call.args = arg1;
                    arg1->next = rhs;
                    rhs->next = NULL;

                    call->type_info = sig->ret_type;
                    call->resolved_type = type_to_string(sig->ret_type);

                    lhs = call;
                    if (allocated_name)
                    {
                        free(allocated_name);
                    }
                    continue; // Loop again with result as new lhs
                }
                if (allocated_name)
                {
                    free(allocated_name);
                }
            }
        }

        // Standard Type Checking (if no overload found)
        if (lhs->type_info && rhs->type_info)
        {
            if (is_comparison_op(bin->binary.op))
            {
                bin->type_info = type_new(TYPE_INT); // bool
                char *t1 = type_to_string(lhs->type_info);
                char *t2 = type_to_string(rhs->type_info);
                // Skip type check if either operand is void* (escape hatch type)
                int skip_check = (strcmp(t1, "void*") == 0 || strcmp(t2, "void*") == 0);

                // Allow comparing pointers/strings with integer literal 0 (NULL)
                if (!skip_check)
                {
                    int lhs_is_ptr =
                        (lhs->type_info->kind == TYPE_POINTER ||
                         lhs->type_info->kind == TYPE_STRING || (t1 && strstr(t1, "*")));
                    int rhs_is_ptr =
                        (rhs->type_info->kind == TYPE_POINTER ||
                         rhs->type_info->kind == TYPE_STRING || (t2 && strstr(t2, "*")));

                    if (lhs_is_ptr && rhs->type == NODE_EXPR_LITERAL && rhs->literal.int_val == 0)
                    {
                        skip_check = 1;
                    }
                    if (rhs_is_ptr && lhs->type == NODE_EXPR_LITERAL && lhs->literal.int_val == 0)
                    {
                        skip_check = 1;
                    }
                }

                if (!skip_check && !type_eq(lhs->type_info, rhs->type_info) &&
                    !(is_integer_type(lhs->type_info) && is_integer_type(rhs->type_info)))
                {
                    char msg[256];
                    sprintf(msg, "Type mismatch in comparison: cannot compare '%s' and '%s'", t1,
                            t2);

                    char suggestion[256];
                    sprintf(suggestion, "Both operands must have compatible types for comparison");

                    zpanic_with_suggestion(op, msg, suggestion);
                }
            }
            else
            {
                if (type_eq(lhs->type_info, rhs->type_info))
                {
                    bin->type_info = lhs->type_info;
                }
                else
                {
                    // Check aliases
                    char *al = NULL, *ar = NULL;
                    int pl = 0, pr = 0;
                    char *sl = resolve_struct_name_from_type(ctx, lhs->type_info, &pl, &al);
                    char *sr = resolve_struct_name_from_type(ctx, rhs->type_info, &pr, &ar);

                    int alias_match = 0;
                    if (sl && sr && strcmp(sl, sr) == 0 && pl == pr)
                    {
                        alias_match = 1;
                        bin->type_info = lhs->type_info;
                    }
                    if (al)
                    {
                        free(al);
                    }
                    if (ar)
                    {
                        free(ar);
                    }

                    char *t1 = type_to_string(lhs->type_info);
                    char *t2 = type_to_string(rhs->type_info);

                    // Allow pointer arithmetic: ptr + int, ptr - int, int + ptr
                    int is_ptr_arith = 0;
                    if (!alias_match)
                    {
                        if (strcmp(bin->binary.op, "+") == 0 || strcmp(bin->binary.op, "-") == 0)
                        {
                            int lhs_is_ptr = (lhs->type_info->kind == TYPE_POINTER ||
                                              lhs->type_info->kind == TYPE_STRING ||
                                              (t1 && strstr(t1, "*") != NULL));
                            int rhs_is_ptr = (rhs->type_info->kind == TYPE_POINTER ||
                                              rhs->type_info->kind == TYPE_STRING ||
                                              (t2 && strstr(t2, "*") != NULL));
                            int lhs_is_int =
                                (lhs->type_info->kind == TYPE_INT ||
                                 lhs->type_info->kind == TYPE_I32 ||
                                 lhs->type_info->kind == TYPE_I64 ||
                                 lhs->type_info->kind == TYPE_ISIZE ||
                                 lhs->type_info->kind == TYPE_USIZE ||
                                 (t1 && (strcmp(t1, "int") == 0 || strcmp(t1, "isize") == 0 ||
                                         strcmp(t1, "usize") == 0 || strcmp(t1, "size_t") == 0 ||
                                         strcmp(t1, "ptrdiff_t") == 0)));
                            int rhs_is_int =
                                (rhs->type_info->kind == TYPE_INT ||
                                 rhs->type_info->kind == TYPE_I32 ||
                                 rhs->type_info->kind == TYPE_I64 ||
                                 rhs->type_info->kind == TYPE_ISIZE ||
                                 rhs->type_info->kind == TYPE_USIZE ||
                                 (t2 && (strcmp(t2, "int") == 0 || strcmp(t2, "isize") == 0 ||
                                         strcmp(t2, "usize") == 0 || strcmp(t2, "size_t") == 0 ||
                                         strcmp(t2, "ptrdiff_t") == 0)));

                            if ((lhs_is_ptr && rhs_is_int) || (lhs_is_int && rhs_is_ptr))
                            {
                                is_ptr_arith = 1;
                                bin->type_info = lhs_is_ptr ? lhs->type_info : rhs->type_info;
                            }
                        }
                    }

                    if (!is_ptr_arith && !alias_match)
                    {
                        char msg[256];
                        sprintf(msg, "Type mismatch in binary operation '%s'", bin->binary.op);

                        char suggestion[512];
                        sprintf(suggestion,
                                "Left operand has type '%s', right operand has type '%s'\n   = "
                                "note: Consider casting one operand to match the other",
                                t1, t2);

                        zpanic_with_suggestion(op, msg, suggestion);
                    }
                }
            }
        }

        lhs = bin;
    }
    return lhs;
}

ASTNode *parse_arrow_lambda_single(ParserContext *ctx, Lexer *l, char *param_name)
{
    ASTNode *lambda = ast_create(NODE_LAMBDA);
    lambda->lambda.param_names = xmalloc(sizeof(char *));
    lambda->lambda.param_names[0] = param_name;
    lambda->lambda.num_params = 1;

    // Default param type: int
    lambda->lambda.param_types = xmalloc(sizeof(char *));
    lambda->lambda.param_types[0] = xstrdup("int");

    // Create Type Info: int -> int
    Type *t = type_new(TYPE_FUNCTION);
    t->inner = type_new(TYPE_INT); // Return
    t->args = xmalloc(sizeof(Type *));
    t->args[0] = type_new(TYPE_INT); // Arg
    t->arg_count = 1;
    lambda->type_info = t;

    // Body parsing...
    ASTNode *body_block = NULL;
    if (lexer_peek(l).type == TOK_LBRACE)
    {
        body_block = parse_block(ctx, l);
    }
    else
    {
        ASTNode *expr = parse_expression(ctx, l);
        ASTNode *ret = ast_create(NODE_RETURN);
        ret->ret.value = expr;
        body_block = ast_create(NODE_BLOCK);
        body_block->block.statements = ret;
    }
    lambda->lambda.body = body_block;
    lambda->lambda.return_type = xstrdup("int");
    lambda->lambda.lambda_id = ctx->lambda_counter++;
    lambda->lambda.is_expression = 1;
    register_lambda(ctx, lambda);
    analyze_lambda_captures(ctx, lambda);
    return lambda;
}

ASTNode *parse_arrow_lambda_multi(ParserContext *ctx, Lexer *l, char **param_names, int num_params)
{
    ASTNode *lambda = ast_create(NODE_LAMBDA);
    lambda->lambda.param_names = param_names;
    lambda->lambda.num_params = num_params;

    // Type Info construction
    Type *t = type_new(TYPE_FUNCTION);
    t->inner = type_new(TYPE_INT);
    t->args = xmalloc(sizeof(Type *) * num_params);
    t->arg_count = num_params;

    lambda->lambda.param_types = xmalloc(sizeof(char *) * num_params);
    for (int i = 0; i < num_params; i++)
    {
        lambda->lambda.param_types[i] = xstrdup("int");
        t->args[i] = type_new(TYPE_INT);
    }
    lambda->type_info = t;

    // Body parsing...
    ASTNode *body_block = NULL;
    if (lexer_peek(l).type == TOK_LBRACE)
    {
        body_block = parse_block(ctx, l);
    }
    else
    {
        ASTNode *expr = parse_expression(ctx, l);
        ASTNode *ret = ast_create(NODE_RETURN);
        ret->ret.value = expr;
        body_block = ast_create(NODE_BLOCK);
        body_block->block.statements = ret;
    }
    lambda->lambda.body = body_block;
    lambda->lambda.return_type = xstrdup("int");
    lambda->lambda.lambda_id = ctx->lambda_counter++;
    lambda->lambda.is_expression = 1;
    register_lambda(ctx, lambda);
    analyze_lambda_captures(ctx, lambda);
    return lambda;
}
