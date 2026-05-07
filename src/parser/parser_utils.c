#include "../codegen/codegen.h"
#include "../plugins/plugin_manager.h"
#include "parser.h"
#include "platform/misra.h"
#include "../constants.h"
#include "../ast/primitives.h"
#include <ctype.h>
#include "analysis/const_fold.h"

// Forward declaration
static void audit_section_5(ParserContext *ctx, Scope *scope, const char *name,
                            const char *link_name, Token tok);

static int is_unmangle_primitive(const char *base);

void parser_audit_preprocessor(ParserContext *ctx, Token tok)
{
    CompilerConfig *cfg = &ctx->compiler->config;
    const char *p = tok.start;
    while (isspace(*p) || *p == '#')
    {
        p++;
    }

    if (strncmp(p, "define", 6) == 0)
    {
        if (cfg->misra_mode)
        {
            zerror_at(tok,
                      "MISRA Violation: '#' directives are prohibited (MISRA Rule Zen 1.4). Use "
                      "'def' instead.");

            // Rule 21.1: #define of standard identifiers
            const char *id_start = p + 6;
            while (isspace(*id_start))
            {
                id_start++;
            }
            const char *id_end = id_start;
            while (isalnum(*id_end) || *id_end == '_')
            {
                id_end++;
            }
            int id_len = id_end - id_start;
            if (id_len > 0)
            {
                char id[128];
                if (id_len >= 128)
                {
                    id_len = 127;
                }
                strncpy(id, id_start, id_len);
                id[id_len] = 0;

                if (strcmp(id, "errno") == 0 || strcmp(id, "assert") == 0 ||
                    strcmp(id, "NULL") == 0 || strcmp(id, "static_assert") == 0 ||
                    strcmp(id, "bool") == 0 || strcmp(id, "restrict") == 0 ||
                    strcmp(id, "inline") == 0)
                {
                    zerror_at(
                        tok, "MISRA Rule 21.1: #define shall not be used on a reserved macro name");
                }
            }
        }
        else
        {
            zwarn_at_diag(0, tok,
                          "Preprocessor directive '#define' is deprecated. Use 'def' instead.");
        }
        // still try to parse it for constant extraction (backward compat)
        char *content = xmalloc(tok.len + 1);
        strncpy(content, tok.start, tok.len);
        content[tok.len] = 0;
        try_parse_macro_const(ctx, content);
        free(content);
    }
    else if (strncmp(p, "include", 7) == 0)
    {
        if (cfg->misra_mode)
        {
            zerror_at(
                tok,
                "MISRA Violation: '#include' is prohibited (Rule Zen 1.4). Use 'import' instead.");
        }
        else
        {
            zwarn_at_diag(0, tok,
                          "Preprocessor directive '#include' is deprecated. Use 'import' instead.");
        }
    }
    else if (strncmp(p, "if", 2) == 0 || strncmp(p, "elif", 4) == 0 ||
             strncmp(p, "ifdef", 5) == 0 || strncmp(p, "ifndef", 6) == 0)
    {
        int is_elif = (strncmp(p, "elif", 4) == 0);
        int is_ifdef = (strncmp(p, "ifdef", 5) == 0);
        int is_ifndef = (strncmp(p, "ifndef", 6) == 0);

        if (cfg->misra_mode)
        {
            zerror_at(tok, "MISRA Violation: '#' preprocessor conditions are prohibited (MISRA "
                           "Rule Zen 1.4). Use "
                           "'@cfg(...)' instead.");

            // Perform specific expression audits (Rule 20.8, 20.9)
            if (is_ifdef || is_ifndef)
            {
                // Just an identifier
                const char *expr_start = p + (is_ifdef ? 5 : 6);
                while (isspace(*expr_start))
                {
                    expr_start++;
                }
                // 20.9 technically only applies to #if/#elif, but we check ifdefs too if they are
                // legacy
            }
            else
            {
                const char *expr_start = p + (is_elif ? 4 : 2);
                while (isspace(*expr_start))
                {
                    expr_start++;
                }

                int expr_len = (tok.start + tok.len) - expr_start;
                if (expr_len > 0)
                {
                    char *expr_buf = xmalloc(expr_len + 1);
                    strncpy(expr_buf, expr_start, expr_len);
                    expr_buf[expr_len] = 0;

                    // Truncate at comment or newline to avoid Rule 20.9 false positives
                    char *comment = strstr(expr_buf, "//");
                    if (comment)
                    {
                        *comment = 0;
                    }
                    char *nl = strchr(expr_buf, '\n');
                    if (nl)
                    {
                        *nl = 0;
                    }
                    char *cr = strchr(expr_buf, '\r');
                    if (cr)
                    {
                        *cr = 0;
                    }

                    misra_check_preprocessor_expression_parser(ctx, tok, expr_buf);
                    free(expr_buf);
                }
            }
        }
        else
        {
            zwarn_at_diag(0, tok,
                          "Preprocessor directive '#' conditions are deprecated. Use "
                          "'@cfg(...)' instead.");
        }
    }
    else if (strncmp(p, "undef", 5) == 0 || strncmp(p, "error", 5) == 0 ||
             strncmp(p, "warning", 7) == 0 || strncmp(p, "pragma", 6) == 0)
    {
        int is_undef = (strncmp(p, "undef", 5) == 0);
        if (cfg->misra_mode)
        {
            zerror_at(tok, "MISRA Violation: '#' directives are prohibited (MISRA Rule Zen 1.4).");

            if (is_undef)
            {
                // Rule 21.1: #undef of standard identifiers
                const char *id_start = p + 5;
                while (isspace(*id_start))
                {
                    id_start++;
                }
                const char *id_end = id_start;
                while (isalnum(*id_end) || *id_end == '_')
                {
                    id_end++;
                }
                int id_len = id_end - id_start;
                if (id_len > 0)
                {
                    char id[128];
                    if (id_len >= 128)
                    {
                        id_len = 127;
                    }
                    strncpy(id, id_start, id_len);
                    id[id_len] = 0;

                    // Common standard macro names that shouldn't be undefined
                    if (strcmp(id, "errno") == 0 || strcmp(id, "assert") == 0 ||
                        strcmp(id, "NULL") == 0 || strcmp(id, "static_assert") == 0)
                    {
                        zerror_at(
                            tok,
                            "MISRA Rule 21.1: #undef shall not be used on a reserved macro name");
                    }
                }
            }
        }
        else
        {
            zwarn_at_diag(0, tok, "Preprocessor directive '#' is deprecated.");
        }
    }
}

void try_parse_macro_const(ParserContext *ctx, const char *content)
{
    CompilerConfig *cfg = &ctx->compiler->config;
    Lexer l;
    lexer_init(&l, content);
    l.emit_comments = 0;

    lexer_next(&l); // Skip start

    // Manual skip of #
    const char *p = content;
    while (isspace(*p))
    {
        p++;
    }
    if (*p == '#')
    {
        p++;
    }

    // Now lex the rest
    lexer_init(&l, p);

    // Expect 'define'
    Token def = lexer_next(&l);
    if (def.type != TOK_IDENT || strncmp(def.start, "define", 6) != 0)
    {
        return;
    }

    // Expect NAME
    Token name = lexer_next(&l);
    if (name.type != TOK_IDENT)
    {
        return;
    }

    const char *p_scan = name.start + name.len;
    while (*p_scan && *p_scan != '\n')
    {
        // Simple scan for # and ## without full lexing to catch them in all defines
        if (*p_scan == '#')
        {
            if (*(p_scan + 1) == '#')
            {
                if (cfg->misra_mode)
                {
                    zerror_at(name, "MISRA Rule 20.10: '##' operator used in macro");
                }
                p_scan++;
            }
            else
            {
                if (cfg->misra_mode)
                {
                    zerror_at(name, "MISRA Rule 20.10: '#' operator used in macro");
                }
            }
        }
        p_scan++;
    }

    if (*(name.start + name.len) == '(')
    {
        if (cfg->misra_mode)
        {
            // Advanced audit for function-like macros (20.11, 20.12)
            const char *pm = name.start + name.len;
            while (isspace(*pm))
            {
                pm++;
            }
            if (*pm == '(')
            {
                pm++; // skip '('
                char *params[32];
                int param_count = 0;
                while (*pm && *pm != ')' && param_count < 32)
                {
                    while (isspace(*pm) || *pm == ',')
                    {
                        pm++;
                    }
                    if (!isalpha(*pm) && *pm != '_')
                    {
                        break;
                    }
                    const char *p_start = pm;
                    while (is_ident_char(*pm))
                    {
                        pm++;
                    }
                    int len = pm - p_start;
                    params[param_count] = xmalloc(len + 1);
                    strncpy(params[param_count], p_start, len);
                    params[param_count][len] = 0;
                    param_count++;
                }
                while (*pm && *pm != ')')
                {
                    pm++;
                }
                if (*pm == ')')
                {
                    pm++;
                }

                // Body usage tracking
                unsigned int used_norm = 0;
                unsigned int used_op = 0;

                const char *pb = pm;
                while (*pb && *pb != '\n')
                {
                    while (isspace(*pb))
                    {
                        pb++;
                    }
                    if (!*pb || *pb == '\n')
                    {
                        break;
                    }

                    if (*pb == '#')
                    {
                        int is_concat = (pb[1] == '#');
                        pb += (is_concat ? 2 : 1);
                        while (isspace(*pb))
                        {
                            pb++;
                        }
                        if (isalpha(*pb) || *pb == '_')
                        {
                            const char *id_start = pb;
                            while (is_ident_char(*pb))
                            {
                                pb++;
                            }
                            int id_len = pb - id_start;
                            for (int i = 0; i < param_count; i++)
                            {
                                if (id_len == (int)strlen(params[i]) &&
                                    strncmp(id_start, params[i], id_len) == 0)
                                {
                                    used_op |= (1 << i);
                                    if (!is_concat)
                                    {
                                        // Rule 20.11 check: #param followed by ##
                                        const char *pafter = pb;
                                        while (isspace(*pafter))
                                        {
                                            pafter++;
                                        }
                                        if (pafter[0] == '#' && pafter[1] == '#')
                                        {
                                            zerror_at(
                                                name,
                                                "MISRA Rule 20.11: # parameter followed by ##");
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else if (isalpha(*pb) || *pb == '_')
                    {
                        const char *id_start = pb;
                        while (is_ident_char(*pb))
                        {
                            pb++;
                        }
                        int id_len = pb - id_start;
                        // Check if it's followed or preceded by ## (handled above for follow)
                        // Actually, we can just check for ## around it
                        const char *pafter = pb;
                        while (isspace(*pafter))
                        {
                            pafter++;
                        }
                        int follows_concat = (pafter[0] == '#' && pafter[1] == '#');

                        for (int i = 0; i < param_count; i++)
                        {
                            if (id_len == (int)strlen(params[i]) &&
                                strncmp(id_start, params[i], id_len) == 0)
                            {
                                if (follows_concat)
                                {
                                    used_op |= (1 << i);
                                }
                                else
                                {
                                    used_norm |= (1 << i);
                                }
                            }
                        }
                    }
                    else
                    {
                        pb++;
                    }
                }

                // Rule 20.12 check: parameter used as op AND normally
                for (int i = 0; i < param_count; i++)
                {
                    if ((used_op & (1 << i)) && (used_norm & (1 << i)))
                    {
                        char msg[128];
                        snprintf(msg, sizeof(msg),
                                 "MISRA Rule 20.12: parameter '%s' used as both operand to #/## "
                                 "and normal token",
                                 params[i]);
                        zerror_at(name, "%s", msg);
                    }
                    free(params[i]);
                }
            }
        }
        return; // Fixed-size scanner already did 20.10 above
    }

    // Check remaining tokens for SAFETY
    Lexer check_l = l;
    int balance = 0;
    while (1)
    {
        Token ct = lexer_next(&check_l);
        if (ct.type == TOK_EOF)
        {
            break;
        }
        if (ct.type == TOK_LPAREN)
        {
            balance++;
        }
        else if (ct.type == TOK_RPAREN)
        {
            balance--;
        }
        else if (ct.type == TOK_LBRACE || ct.type == TOK_RBRACE || ct.type == TOK_SEMICOLON)
        {
            return; // Unsafe or complex
        }

        // MISRA Rule 20.10: The # and ## preprocessor operators should not be used
        if (cfg->misra_mode && ct.type == TOK_OP)
        {
            if (ct.len == 1 && *ct.start == '#')
            {
                zerror_at(ct, "MISRA Rule 20.10: '#' operator used in macro");
            }
            else if (ct.len == 2 && strncmp(ct.start, "##", 2) == 0)
            {
                zerror_at(ct, "MISRA Rule 20.10: '##' operator used in macro");
            }
        }

        if (ct.type == TOK_IDENT)
        {
            char *tok_str = token_strdup(ct);
            int is_prim = is_primitive_type_name(tok_str);

            // Check other keywords not covered by is_primitive_type_name
            if (!is_prim)
            {
                if (is_token(ct, "signed") || is_token(ct, "unsigned") || is_token(ct, "struct") ||
                    is_token(ct, "union") || is_token(ct, "enum") || is_token(ct, "const") ||
                    is_token(ct, "volatile") || is_token(ct, "extern") || is_token(ct, "static") ||
                    is_token(ct, "register") || is_token(ct, "auto") || is_token(ct, "typedef"))
                {
                    is_prim = 1;
                }
            }

            free(tok_str);

            if (is_prim)
            {
                return;
            }
        }
    }
    if (balance != 0)
    {
        return; // Unbalanced
    }

    // Ensure we have something to parse
    if (lexer_peek(&l).type == TOK_EOF)
    {
        return;
    }

    // Try parse expression
    // We need to handle potential parsing errors gracefully.
    // If parse_expression errors, zpanic unwinds.
    // But we filtered hopefully unsafe tokens.

    ASTNode *expr = parse_expression(ctx, &l);
    if (!expr)
    {
        return;
    }

    long long val;
    if (eval_const_int_expr(expr, ctx, &val))
    {
        // Success! Register as constant.
        char *n = token_strdup(name);

        // Check if already defined?
        ZenSymbol *existing = find_symbol_entry(ctx, n);
        if (!existing)
        {
            // Add to symbol table
            add_symbol_with_token(ctx, n, "int", type_new(TYPE_INT), name, 0); // Placeholder type
            // find_symbol_entry to set properties
            ZenSymbol *sym = find_symbol_entry(ctx, n);
            if (sym)
            {
                sym->is_const_value = 1;
                sym->const_int_val = (int)val;
                sym->is_def = 1;
            }
        }
        else
        {
            free(n);
        }
    }
}
#include <stdlib.h>
#include <string.h>

void instantiate_methods(ParserContext *ctx, GenericImplTemplate *it,
                         const char *mangled_struct_name, const char *arg,
                         const char *unmangled_arg);

Token z_parse_expect(Lexer *l, ZenTokenType type, const char *msg)
{
    Token t = lexer_next(l);
    if (t.type != type)
    {
        zpanic_at(t, "Expected %s, but got '%.*s'", msg, t.len, t.start);
        return (Token){type, t.start, 0, t.line, t.col, t.file};
    }
    return t;
}

// Helper to check if a type name is a primitive type
int is_primitive_type_name(const char *name)
{
    return find_primitive_kind(name) != TYPE_UNKNOWN;
}

TypeKind get_primitive_type_kind(const char *name)
{
    return find_primitive_kind(name);
}

// Forward declaration
char *ast_to_string_recursive(ASTNode *node, int depth);

char *ast_to_string(ASTNode *node)
{
    return ast_to_string_recursive(node, 0);
}

// Temporary lightweight AST printer for default args
// Comprehensive AST printer for default args and other code generation needs
char *ast_to_string_recursive(ASTNode *node, int depth)
{
    const int MAX_DEPTH = 32;
    if (!node || depth > MAX_DEPTH)
    {
        return xstrdup(depth > MAX_DEPTH ? "..." : "");
    }

    size_t buf_size = MAX_PATH_LEN;
    char *buf = xmalloc(buf_size);
    buf[0] = 0;

    switch (node->type)
    {
    case NODE_EXPR_LITERAL:
        if (node->literal.type_kind == LITERAL_INT)
        {
            snprintf(buf, buf_size, "%llu", node->literal.int_val);
        }
        else if (node->literal.type_kind == LITERAL_FLOAT)
        {
            snprintf(buf, buf_size, "%f", node->literal.float_val);
        }
        else if (node->literal.type_kind == LITERAL_STRING)
        {
            size_t s_len = strlen(node->literal.string_val);
            size_t required = s_len + 16;
            if (required > buf_size)
            {
                char *new_buf = xrealloc(buf, required);
                buf = new_buf;
                buf_size = required;
            }
            snprintf(buf, buf_size, "\"%s\"", node->literal.string_val);
        }
        else if (node->literal.type_kind == LITERAL_CHAR)
        {
            if (node->literal.int_val == '\'')
            {
                snprintf(buf, buf_size, "'\\''");
            }
            else if (node->literal.int_val == '\n')
            {
                snprintf(buf, buf_size, "'\\n'");
            }
            else if (node->literal.int_val == '\\')
            {
                snprintf(buf, buf_size, "'\\\\'");
            }
            else if (node->literal.int_val == '\0')
            {
                snprintf(buf, buf_size, "'\\0'");
            }
            else
            {
                snprintf(buf, buf_size, "'%c'", (char)node->literal.int_val);
            }
        }
        break;
    case NODE_EXPR_VAR:
        snprintf(buf, buf_size, "%s", node->var_ref.name ? node->var_ref.name : "");
        break;
    case NODE_EXPR_BINARY:
    {
        char *l = ast_to_string_recursive(node->binary.left, depth + 1);
        char *r = ast_to_string_recursive(node->binary.right, depth + 1);
        // Add parens to be safe
        snprintf(buf, buf_size, "(%s %s %s)", l, node->binary.op ? node->binary.op : "?", r);
        free(l);
        free(r);
        break;
    }
    case NODE_EXPR_UNARY:
    {
        char *o = ast_to_string_recursive(node->unary.operand, depth + 1);
        snprintf(buf, buf_size, "(%s%s)", node->unary.op ? node->unary.op : "?", o);
        free(o);
        break;
    }
    case NODE_EXPR_CAST:
    {
        char *e = ast_to_string_recursive(node->cast.expr, depth + 1);
        snprintf(buf, buf_size, "((%s)%s)", node->cast.target_type ? node->cast.target_type : "?",
                 e);
        free(e);
        break;
    }
    case NODE_EXPR_CALL:
    {
        char *callee = ast_to_string_recursive(node->call.callee, depth + 1);
        snprintf(buf, buf_size, "%s(", callee);
        free(callee);

        ASTNode *arg = node->call.args;
        int first = 1;
        while (arg)
        {
            if (!first)
            {
                if (strlen(buf) + 4 < buf_size)
                {
                    strcat(buf, ", ");
                }
            }
            char *a = ast_to_string_recursive(arg, depth + 1);
            if (strlen(buf) + strlen(a) + 4 < buf_size)
            {
                strcat(buf, a);
            }
            free(a);
            first = 0;
            arg = arg->next;
        }
        if (strlen(buf) + 2 < buf_size)
        {
            strcat(buf, ")");
        }
        break;
    }
    case NODE_EXPR_STRUCT_INIT:
    {
        char *name = node->struct_init.struct_name;
        snprintf(buf, buf_size, "%s{", name ? name : "?");

        ASTNode *field = node->struct_init.fields;
        int first = 1;
        while (field)
        {
            if (!first)
            {
                if (strlen(buf) + 4 < buf_size)
                {
                    strcat(buf, ", ");
                }
            }
            if (field->type == NODE_VAR_DECL)
            {
                if (strlen(buf) + (field->var_decl.name ? strlen(field->var_decl.name) : 0) + 4 <
                    buf_size)
                {
                    strcat(buf, field->var_decl.name ? field->var_decl.name : "?");
                    strcat(buf, ": ");
                }
                char *val = ast_to_string_recursive(field->var_decl.init_expr, depth + 1);
                if (strlen(buf) + strlen(val) + 2 < buf_size)
                {
                    strcat(buf, val);
                }
                free(val);
            }
            first = 0;
            field = field->next;
        }
        if (strlen(buf) + 2 < buf_size)
        {
            strcat(buf, "}");
        }
        break;
    }
    case NODE_EXPR_MEMBER:
    {
        char *t = ast_to_string_recursive(node->member.target, depth + 1);
        snprintf(buf, buf_size, "%s.%s", t, node->member.field ? node->member.field : "?");
        free(t);
        break;
    }
    case NODE_EXPR_INDEX:
    {
        char *arr = ast_to_string_recursive(node->index.array, depth + 1);
        char *idx = ast_to_string_recursive(node->index.index, depth + 1);
        snprintf(buf, buf_size, "%s[%s]", arr, idx);
        free(arr);
        free(idx);
        break;
    }
    default:
        snprintf(buf, buf_size, "<expr>");
        break;
    }
    return buf;
}

int is_token(Token t, const char *s)
{
    int len = strlen(s);
    return (t.len == len && strncmp(t.start, s, len) == 0);
}

char *token_strdup(Token t)
{
    char *s = xmalloc(t.len + 1);
    strncpy(s, t.start, t.len);
    s[t.len] = 0;
    return s;
}

char *token_get_string_content(Token t)
{
    int is_multi = 0;
    int is_fstring = (t.type == TOK_FSTRING);
    int is_raw = (t.type == TOK_RAW_STRING);
    int start_offset = 1;
    int end_offset = 1;

    if (is_fstring)
    {
        is_multi = (t.len >= 7 && t.start[1] == '"' && t.start[2] == '"' && t.start[3] == '"');
        start_offset = is_multi ? 4 : 2;
        end_offset = is_multi ? 3 : 1;
    }
    else if (is_raw)
    {
        is_multi = (t.len >= 7 && t.start[1] == '"' && t.start[2] == '"' && t.start[3] == '"');
        start_offset = is_multi ? 4 : 2;
        end_offset = is_multi ? 3 : 1;
    }
    else // TOK_STRING
    {
        is_multi = (t.len >= 6 && t.start[0] == '"' && t.start[1] == '"' && t.start[2] == '"');
        start_offset = is_multi ? 3 : 1;
        end_offset = is_multi ? 3 : 1;
    }

    int content_len = t.len - start_offset - end_offset;
    if (content_len < 0)
    {
        content_len = 0;
    }

    char *content = xmalloc(content_len + 1);
    strncpy(content, t.start + start_offset, content_len);
    content[content_len] = '\0';
    return content;
}

void skip_comments(Lexer *l)
{
    int prev_emit = l->emit_comments;
    l->emit_comments = 1;
    while (lexer_peek(l).type == TOK_COMMENT)
    {
        Token tk = lexer_next(l);
        if (g_config.keep_comments && g_parser_ctx)
        {
            if (g_parser_ctx->last_doc_comment)
            {
                // Concatenate if multiple comments
                size_t old_len = strlen(g_parser_ctx->last_doc_comment);
                char *new_c = xmalloc(old_len + tk.len + 2);
                sprintf(new_c, "%s\n%.*s", g_parser_ctx->last_doc_comment, tk.len, tk.start);
                free(g_parser_ctx->last_doc_comment);
                g_parser_ctx->last_doc_comment = new_c;
            }
            else
            {
                g_parser_ctx->last_doc_comment = xmalloc(tk.len + 1);
                strncpy(g_parser_ctx->last_doc_comment, tk.start, tk.len);
                g_parser_ctx->last_doc_comment[tk.len] = 0;
            }
        }
    }
    l->emit_comments = prev_emit;
}

// C reserved words that conflict with C when used as identifiers.

static const char *C_RESERVED_WORDS[] = {
    // C types that could be used as names
    "double", "float", "signed", "unsigned", "short", "long", "auto", "register",
    // C keywords
    "switch", "case", "default", "do", "goto", "typedef", "static", "extern", "volatile", "inline",
    "restrict", "sizeof", "const",
    // C11+ keywords
    "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex", "_Generic", "_Imaginary", "_Noreturn",
    "_Static_assert", "_Thread_local", NULL};

int is_c_reserved_word(const char *name)
{
    for (int i = 0; C_RESERVED_WORDS[i] != NULL; i++)
    {
        if (strcmp(name, C_RESERVED_WORDS[i]) == 0)
        {
            return 1;
        }
    }
    return 0;
}

void warn_c_reserved_word(Token t, const char *name)
{
    zwarn_at(t, "Identifier '%s' conflicts with C reserved word", name);
    fprintf(stderr, COLOR_CYAN "   = note: " COLOR_RESET
                               "This will cause compilation errors in the generated C code\n");
}

char *consume_until_semicolon(Lexer *l)
{
    const char *s = l->src + l->pos;
    int d = 0;
    while (1)
    {
        Token t = lexer_peek(l);
        if (t.type == TOK_EOF)
        {
            break;
        }
        if (t.type == TOK_LBRACE || t.type == TOK_LPAREN || t.type == TOK_LBRACKET)
        {
            d++;
        }
        if (t.type == TOK_RBRACE || t.type == TOK_RPAREN || t.type == TOK_RBRACKET)
        {
            d--;
        }

        if (d == 0 && t.type == TOK_SEMICOLON)
        {
            int len = t.start - s;
            char *r = xmalloc(len + 1);
            strncpy(r, s, len);
            r[len] = 0;
            lexer_next(l);
            return r;
        }
        lexer_next(l);
    }
    return xstrdup("");
}

void enter_scope(ParserContext *ctx)
{
    Scope *s = symbol_scope_create(ctx->current_scope, NULL);
    ctx->current_scope = s;
}

void exit_scope(ParserContext *ctx)
{
    if (!ctx->current_scope || ctx->current_scope == ctx->global_scope)
    {
        return;
    }

    // Check for unused variables (legacy logic)
    ZenSymbol *sym = ctx->current_scope->symbols;
    while (sym)
    {
        if (!sym->is_used && strcmp(sym->name, "self") != 0 && sym->name[0] != '_')
        {
            // Could emit warning here
        }
        sym = sym->next;
    }

    ctx->current_scope = ctx->current_scope->parent;
}

// Helper to register a symbol for LSP persistent queries
void register_symbol_to_lsp(ParserContext *ctx, ZenSymbol *s)
{
    if (!ctx || !s)
    {
        return;
    }

    // Deduplicate: Don't add if same name, kind, and location already exists
    ZenSymbol *curr = ctx->all_symbols;
    while (curr)
    {
        if (curr->kind == s->kind && curr->decl_token.line == s->decl_token.line &&
            curr->decl_token.col == s->decl_token.col && curr->name && s->name &&
            strcmp(curr->name, s->name) == 0)
        {
            return;
        }
        curr = curr->next;
    }

    ZenSymbol *lsp_copy = xmalloc(sizeof(ZenSymbol));
    memcpy(lsp_copy, s, sizeof(ZenSymbol));
    lsp_copy->original = s; // Link clone back to the original symbol for global auditing
    lsp_copy->next = ctx->all_symbols;
    ctx->all_symbols = lsp_copy;
    if (s->name)
    {
        lsp_copy->name = xstrdup(s->name);
    }
    if (s->cfg_condition)
    {
        lsp_copy->cfg_condition = xstrdup(s->cfg_condition);
    }

    lsp_copy->is_local = s->is_local;
}

void add_symbol(ParserContext *ctx, const char *n, const char *t, Type *type_info, int is_export)
{
    add_symbol_with_token(ctx, n, t, type_info, (Token){0}, is_export);
}

void add_symbol_with_token(ParserContext *ctx, const char *n, const char *t, Type *type_info,
                           Token tok, int is_export)
{
    if (!ctx->current_scope)
    {
        if (!ctx->global_scope)
        {
            ctx->global_scope = symbol_scope_create(NULL, "Global");
        }
        ctx->current_scope = ctx->global_scope;
    }

    if (strcmp(n, "it") != 0 && strcmp(n, "self") != 0)
    {
        audit_section_5(ctx, ctx->current_scope, n, NULL, tok);
    }

    // In LSP/MISRA mode, check for existing symbol in the current scope to avoid duplicates
    if (g_config.mode_lsp || g_config.misra_mode)
    {
        ZenSymbol *existing = symbol_lookup_local(ctx->current_scope, n);
        if (existing)
        {
            existing->type_name = t ? xstrdup(t) : NULL;
            existing->type_info = type_info;
            existing->decl_token = tok;
            return;
        }
    }

    ZenSymbol *s = symbol_add(ctx->current_scope, n, SYM_VARIABLE);
    s->is_local = (ctx->current_scope != ctx->global_scope);
    s->type_name = t ? xstrdup(t) : NULL;
    s->type_info = type_info;
    s->decl_token = tok;
    s->is_export = is_export;

    register_symbol_to_lsp(ctx, s);
}

Type *find_symbol_type_info(ParserContext *ctx, const char *n)
{
    ZenSymbol *sym = symbol_lookup(ctx->current_scope, n);
    if (sym)
    {
        return sym->type_info;
    }

    // Fallback: check for enum variants (MISRA Rule Zen 1.3)
    EnumVariantReg *ev = find_enum_variant(ctx, n);
    if (ev)
    {
        Type *t = type_new(TYPE_ENUM);
        t->name = xstrdup(ev->enum_name);
        return t;
    }

    return NULL;
}

char *find_symbol_type(ParserContext *ctx, const char *n)
{
    ZenSymbol *sym = symbol_lookup(ctx->current_scope, n);
    if (sym)
    {
        return sym->type_name ? xstrdup(sym->type_name) : NULL;
    }

    // Fallback: check for enum variants (MISRA Rule Zen 1.3)
    EnumVariantReg *ev = find_enum_variant(ctx, n);
    if (ev)
    {
        return xstrdup(ev->enum_name);
    }

    return NULL;
}

ZenSymbol *find_symbol_entry(ParserContext *ctx, const char *n)
{
    return symbol_lookup(ctx->current_scope, n);
}

// LSP: Search flat symbol list (works after scopes are destroyed).
ZenSymbol *find_symbol_in_all(ParserContext *ctx, const char *n)
{
    ZenSymbol *sym = ctx->all_symbols;
    while (sym)
    {
        if (strcmp(sym->name, n) == 0)
        {
            return sym;
        }
        sym = sym->next;
    }
    return NULL;
}

void init_builtins()
{
    static int init = 0;
    if (init)
    {
        return;
    }
    init = 1;
}

void register_func(ParserContext *ctx, Scope *scope, const char *name, int count, char **defaults,
                   Type **arg_types, Type *ret_type, int is_varargs, int is_async, int is_pure,
                   const char *link_name, Token decl_token, int is_export)
{
    // In LSP/MISRA mode, check for existing function in the registry to avoid duplicates
    if (g_config.mode_lsp || g_config.misra_mode)
    {
        FuncSig *existing = find_func(ctx, name);
        if (existing)
        {
            existing->decl_token = decl_token;
            existing->total_args = count;
            existing->defaults = defaults;
            existing->arg_types = arg_types;
            existing->ret_type = ret_type;
            existing->is_varargs = is_varargs;
            existing->is_async = is_async;
            existing->is_pure = is_pure;
            // Note: symbol update happens below
        }
    }

    FuncSig *f = NULL;
    if (g_config.mode_lsp || g_config.misra_mode)
    {
        f = find_func(ctx, name);
    }

    if (!f)
    {
        f = xmalloc(sizeof(FuncSig));
        f->name = xstrdup(name);
        f->next = ctx->func_registry;
        ctx->func_registry = f;
    }

    f->decl_token = decl_token;
    f->total_args = count;
    f->defaults = defaults;
    f->arg_types = arg_types;
    f->ret_type = ret_type;
    f->is_varargs = is_varargs;
    f->is_async = is_async;
    f->is_pure = is_pure;
    f->required = 0;

    // Unified logic: check for existing symbol to avoid duplicates
    Scope *target_scope = scope ? scope : ctx->current_scope;
    audit_section_5(ctx, target_scope ? target_scope : ctx->global_scope, name, link_name,
                    decl_token);

    ZenSymbol *sym = symbol_lookup_local(target_scope, name);
    if (!sym)
    {
        sym = symbol_add(scope ? scope : ctx->current_scope, name, SYM_FUNCTION);
    }
    else
    {
        sym->kind = SYM_FUNCTION; // Ensure kind is correct if it was a placeholder
    }
    sym->data.sig = f;
    sym->decl_token = decl_token;
    sym->is_export = is_export;
    if (link_name)
    {
        f->link_name = xstrdup(link_name);
        sym->link_name = f->link_name;
    }

    register_symbol_to_lsp(ctx, sym);

    // Create formal type for the function pointer
    Type *ft = type_new(TYPE_FUNCTION);
    ft->arg_count = count;
    ft->args = arg_types;
    ft->inner = ret_type;
    ft->is_raw = 1;          // Static functions are raw pointers, not closures
    ft->traits.has_drop = 0; // Static functions don't need drop
    sym->type_info = ft;
}

void register_func_template(ParserContext *ctx, const char *name, const char *param, ASTNode *node)
{
    GenericFuncTemplate *t = xcalloc(1, sizeof(GenericFuncTemplate));
    t->name = xstrdup(name);
    t->generic_param = xstrdup(param);
    t->func_node = node;
    t->next = ctx->func_templates;
    ctx->func_templates = t;
}

void register_deprecated_func(ParserContext *ctx, const char *name, const char *reason)
{
    DeprecatedFunc *d = xmalloc(sizeof(DeprecatedFunc));
    d->name = xstrdup(name);
    d->reason = reason ? xstrdup(reason) : NULL;
    d->next = ctx->deprecated_funcs;
    ctx->deprecated_funcs = d;
}

DeprecatedFunc *find_deprecated_func(ParserContext *ctx, const char *name)
{
    DeprecatedFunc *d = ctx->deprecated_funcs;
    while (d)
    {
        if (strcmp(d->name, name) == 0)
        {
            return d;
        }
        d = d->next;
    }
    return NULL;
}

GenericFuncTemplate *find_func_template(ParserContext *ctx, const char *name)
{
    GenericFuncTemplate *t = ctx->func_templates;
    while (t)
    {
        if (strcmp(t->name, name) == 0)
        {
            return t;
        }
        t = t->next;
    }
    return NULL;
}

void register_generic(ParserContext *ctx, char *name)
{
    for (int i = 0; i < ctx->known_generics_count; i++)
    {
        if (strcmp(ctx->known_generics[i], name) == 0)
        {
            return;
        }
    }
    ctx->known_generics[ctx->known_generics_count++] = xstrdup(name);
}

int is_known_generic(ParserContext *ctx, char *name)
{
    for (int i = 0; i < ctx->known_generics_count; i++)
    {
        if (strcmp(ctx->known_generics[i], name) == 0)
        {
            return 1;
        }
    }
    return 0;
}

int is_generic_dependent_str(ParserContext *ctx, const char *type_str)
{
    if (!type_str || !ctx)
    {
        return 0;
    }
    for (int i = 0; i < ctx->known_generics_count; i++)
    {
        const char *g = ctx->known_generics[i];
        const char *p = strstr(type_str, g);
        while (p)
        {
            // Boundaries: Must not be preceded or followed by identifier chars (except Ptr or _)
            int valid = 1;
            if (p > type_str && is_ident_char(*(p - 1)) && *(p - 1) != '_')
            {
                valid = 0;
            }
            if (valid)
            {
                const char *next = p + strlen(g);
                if (*next != '\0' && is_ident_char(*next) && *next != '_')
                {
                    // Allow Ptr suffix (mangled)
                    if (strncmp(next, "Ptr", 3) != 0)
                    {
                        valid = 0;
                    }
                }
            }
            if (valid)
            {
                return 1;
            }
            p = strstr(p + 1, g);
        }
    }
    return 0;
}

void register_impl_template(ParserContext *ctx, const char *sname, const char *param, ASTNode *node)
{
    GenericImplTemplate *t = xmalloc(sizeof(GenericImplTemplate));
    t->struct_name = xstrdup(sname);
    t->generic_param = xstrdup(param);
    t->impl_node = node;
    t->next = ctx->impl_templates;
    ctx->impl_templates = t;

    // Late binding: Check if any existing instantiations match this new impl
    // template
    Instantiation *inst = ctx->instantiations;
    while (inst)
    {
        if (inst->template_name && strcmp(inst->template_name, sname) == 0)
        {
            instantiate_methods(ctx, t, inst->name, inst->concrete_arg, inst->unmangled_arg);
        }
        inst = inst->next;
    }
}

void add_to_struct_list(ParserContext *ctx, ASTNode *node)
{
    StructRef *r = xmalloc(sizeof(StructRef));
    r->node = node;
    r->next = ctx->parsed_structs_list;
    ctx->parsed_structs_list = r;
}

void register_type_alias(ParserContext *ctx, const char *alias, const char *original,
                         Type *type_info, int is_opaque, const char *defined_in_file, Token tok,
                         int is_export)
{
    // In LSP mode, check for existing type alias to avoid duplicates
    if (g_config.mode_lsp)
    {
        TypeAlias *existing = find_type_alias_node(ctx, alias);
        if (existing)
        {
            existing->original_type = xstrdup(original);
            existing->type_info = type_info;
            existing->is_opaque = is_opaque;
            existing->defined_in_file = defined_in_file ? xstrdup(defined_in_file) : NULL;
            // Symbol update will happen in add_symbol (called below)
        }
    }

    TypeAlias *ta = NULL;
    if (g_config.mode_lsp)
    {
        ta = find_type_alias_node(ctx, alias);
    }

    if (!ta)
    {
        ta = xmalloc(sizeof(TypeAlias));
        ta->alias = xstrdup(alias);
        ta->next = ctx->type_aliases;
        ctx->type_aliases = ta;
    }

    ta->original_type = xstrdup(original);
    ta->type_info = type_info;
    ta->is_opaque = is_opaque;
    ta->defined_in_file = defined_in_file ? xstrdup(defined_in_file) : NULL;

    // Unified logic: check for existing symbol to avoid duplicates
    audit_section_5(ctx, ctx->current_scope, alias, NULL, tok);
    ZenSymbol *sym = symbol_lookup_local(ctx->current_scope, alias);
    if (!sym)
    {
        sym = symbol_add(ctx->current_scope, alias, SYM_ALIAS);
    }
    else
    {
        sym->kind = SYM_ALIAS;
    }
    sym->decl_token = tok;
    sym->is_export = is_export;
    sym->data.alias.original_type = xstrdup(original);
    sym->type_info = type_info;
    register_symbol_to_lsp(ctx, sym);
}

const char *find_type_alias(ParserContext *ctx, const char *alias)
{
    ZenSymbol *sym = symbol_lookup_kind(ctx->current_scope, alias, SYM_ALIAS);
    if (sym)
    {
        return sym->data.alias.original_type;
    }

    TypeAlias *ta = find_type_alias_node(ctx, alias);
    return ta ? ta->original_type : NULL;
}

TypeAlias *find_type_alias_node(ParserContext *ctx, const char *alias)
{
    TypeAlias *ta = ctx->type_aliases;
    while (ta)
    {
        if (strcmp(ta->alias, alias) == 0)
        {

            return ta;
        }
        ta = ta->next;
    }
    return NULL;
}

void add_to_enum_list(ParserContext *ctx, ASTNode *node)
{
    StructRef *r = xmalloc(sizeof(StructRef));
    r->node = node;
    r->next = ctx->parsed_enums_list;
    ctx->parsed_enums_list = r;
}

void add_to_func_list(ParserContext *ctx, ASTNode *node)
{
    StructRef *curr = ctx->parsed_funcs_list;
    while (curr)
    {
        if (curr->node == node)
        {
            return;
        }
        curr = curr->next;
    }
    StructRef *r = xmalloc(sizeof(StructRef));
    r->node = node;
    r->next = ctx->parsed_funcs_list;
    ctx->parsed_funcs_list = r;
}

void add_to_impl_list(ParserContext *ctx, ASTNode *node)
{
    StructRef *curr = ctx->parsed_impls_list;
    while (curr)
    {
        if (curr->node == node)
        {
            return;
        }
        curr = curr->next;
    }
    StructRef *r = xmalloc(sizeof(StructRef));
    r->node = node;
    r->next = ctx->parsed_impls_list;
    ctx->parsed_impls_list = r;
}

void add_to_global_list(ParserContext *ctx, ASTNode *node)
{
    StructRef *curr = ctx->parsed_globals_list;
    while (curr)
    {
        if (curr->node == node)
        {
            return;
        }
        curr = curr->next;
    }
    StructRef *r = xmalloc(sizeof(StructRef));
    r->node = node;
    r->next = ctx->parsed_globals_list;
    ctx->parsed_globals_list = r;
}

void register_builtins(ParserContext *ctx)
{
    Type *t = type_new(TYPE_BOOL);
    t->is_const = 1;
    add_symbol(ctx, "true", "bool", t, 0);

    t = type_new(TYPE_BOOL);
    t->is_const = 1;
    add_symbol(ctx, "false", "bool", t, 0);

    // Register 'free'
    Type *void_t = type_new(TYPE_VOID);
    add_symbol(ctx, "free", "void", void_t, 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;

    // Register common libc functions to avoid warnings
    add_symbol(ctx, "strdup", "string", type_new(TYPE_STRING), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "malloc", "void*", type_new_ptr(void_t), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "realloc", "void*", type_new_ptr(void_t), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "calloc", "void*", type_new_ptr(void_t), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "puts", "int", type_new(TYPE_INT), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "printf", "int", type_new(TYPE_INT), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "strcmp", "int", type_new(TYPE_INT), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "strlen", "int", type_new(TYPE_INT), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "strcpy", "string", type_new(TYPE_STRING), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "strcat", "string", type_new(TYPE_STRING), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "memset", "void*", type_new_ptr(void_t), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "memcpy", "void*", type_new_ptr(void_t), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "exit", "void", void_t, 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;

    // Stdio Globals
    add_symbol(ctx, "stdin", "void*", type_new_ptr(void_t), 0);
    add_symbol(ctx, "stdout", "void*", type_new_ptr(void_t), 0);
    add_symbol(ctx, "stderr", "void*", type_new_ptr(void_t), 0);

    // File I/O
    add_symbol(ctx, "fopen", "void*", type_new_ptr(void_t), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "fclose", "int", type_new(TYPE_INT), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "fread", "usize", type_new(TYPE_USIZE), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "fwrite", "usize", type_new(TYPE_USIZE), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "fseek", "int", type_new(TYPE_INT), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "ftell", "long", type_new(TYPE_I64), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "rewind", "void", void_t, 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "fprintf", "int", type_new(TYPE_INT), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "vprintf", "int", type_new(TYPE_INT), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "vfprintf", "int", type_new(TYPE_INT), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "sprintf", "int", type_new(TYPE_INT), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "vsnprintf", "int", type_new(TYPE_INT), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "snprintf", "int", type_new(TYPE_INT), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "feof", "int", type_new(TYPE_INT), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "ferror", "int", type_new(TYPE_INT), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "mkdir", "int", type_new(TYPE_INT), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "rmdir", "int", type_new(TYPE_INT), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "chdir", "int", type_new(TYPE_INT), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "getcwd", "string", type_new(TYPE_STRING), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "system", "int", type_new(TYPE_INT), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "getenv", "string", type_new(TYPE_STRING), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "fgets", "string", type_new(TYPE_STRING), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;
    add_symbol(ctx, "usleep", "int", type_new(TYPE_INT), 0);
    ctx->current_scope->symbols->kind = SYM_FUNCTION;

    ASTNode *va_def = ast_create(NODE_STRUCT);
    va_def->strct.name = xstrdup("va_list");
    register_struct_def(ctx, "va_list", va_def);
    register_impl(ctx, "Copy", "va_list");
}

void register_comptime_builtins(ParserContext *ctx)
{
    Type *void_t = type_new(TYPE_VOID);
    add_symbol(ctx, "yield", "void", void_t, 0);
    add_symbol(ctx, "code", "void", void_t, 0); // Alias for yield
    add_symbol(ctx, "compile_warn", "void", void_t, 0);
    add_symbol(ctx, "compile_error", "void", void_t, 0);

    register_extern_symbol(ctx, "yield");
    register_extern_symbol(ctx, "code");
    register_extern_symbol(ctx, "compile_warn");
    register_extern_symbol(ctx, "compile_error");
}

void add_instantiated_func(ParserContext *ctx, ASTNode *fn)
{
    fn->next = ctx->instantiated_funcs;
    ctx->instantiated_funcs = fn;
}

void register_enum_variant(ParserContext *ctx, const char *vname, const char *ename, int tag)
{
    // In LSP mode, check for existing variant to avoid duplicates
    if (g_config.mode_lsp)
    {
        EnumVariantReg *existing = find_enum_variant(ctx, vname);
        if (existing)
        {
            existing->tag_id = tag;
            return;
        }
    }

    // Record entry in the enum variant registry for global lookup
    audit_section_5(ctx, ctx->global_scope, vname, NULL, (Token){0});

    EnumVariantReg *r = xcalloc(1, sizeof(EnumVariantReg));
    r->enum_name = ename ? xstrdup(ename) : NULL;
    r->variant_name = vname ? xstrdup(vname) : NULL;
    r->tag_id = tag;
    r->next = ctx->enum_variants;
    ctx->enum_variants = r;
}

EnumVariantReg *find_enum_variant(ParserContext *ctx, const char *name)
{
    char *ename = NULL;
    const char *vname = name;
    const char *sep = strstr(name, "::");
    if (!sep)
    {
        sep = strstr(name, "__");
    }

    if (sep)
    {
        int elen = (int)(sep - name);
        ename = xmalloc(elen + 1);
        strncpy(ename, name, elen);
        ename[elen] = 0;
        vname = sep + 2;
    }

    EnumVariantReg *r = ctx->enum_variants;
    while (r)
    {
        if (strcmp(r->variant_name, vname) == 0)
        {
            if (!ename || strcmp(r->enum_name, ename) == 0)
            {
                if (ename)
                {
                    free(ename);
                }
                return r;
            }
        }
        r = r->next;
    }
    if (ename)
    {
        free(ename);
    }
    return NULL;
}

void register_lambda(ParserContext *ctx, ASTNode *node)
{
    LambdaRef *ref = xmalloc(sizeof(LambdaRef));
    ref->node = node;
    ref->next = ctx->global_lambdas;
    ctx->global_lambdas = ref;
}

void register_extern_symbol(ParserContext *ctx, const char *name)
{
    // Check for duplicates
    for (int i = 0; i < ctx->extern_symbol_count; i++)
    {
        if (strcmp(ctx->extern_symbols[i], name) == 0)
        {
            return;
        }
    }

    // Grow array if needed
    if (ctx->extern_symbol_count == 0)
    {
        ctx->extern_symbols = xmalloc(sizeof(char *) * 64);
    }
    else if (ctx->extern_symbol_count % 64 == 0)
    {
        ctx->extern_symbols =
            xrealloc(ctx->extern_symbols, sizeof(char *) * (ctx->extern_symbol_count + 64));
    }

    ctx->extern_symbols[ctx->extern_symbol_count++] = xstrdup(name);
}

int is_extern_symbol(ParserContext *ctx, const char *name)
{
    for (int i = 0; i < ctx->extern_symbol_count; i++)
    {
        if (strcmp(ctx->extern_symbols[i], name) == 0)
        {
            return 1;
        }
    }
    return 0;
}

// Unified check: should we suppress "undefined variable" warning for this name?
int should_suppress_undef_warning(ParserContext *ctx, const char *name)
{
    if (strcmp(name, "struct") == 0 || strcmp(name, "tv") == 0)
    {
        return 1;
    }

    if (is_extern_symbol(ctx, name))
    {
        return 1;
    }

    int is_all_caps = 1;
    for (const char *p = name; *p; p++)
    {
        if (islower((unsigned char)*p))
        {
            is_all_caps = 0;
            break;
        }
    }
    if (is_all_caps && name[0] != '\0')
    {
        return 1;
    }

    if (ctx->has_external_includes)
    {
        return 1;
    }

    return 0;
}

void register_slice(ParserContext *ctx, const char *type)
{
    if (is_known_generic(ctx, (char *)type))
    {
        return;
    }

    SliceType *c = ctx->used_slices;
    while (c)
    {
        if (strcmp(c->name, type) == 0)
        {
            return;
        }
        c = c->next;
    }
    SliceType *n = xmalloc(sizeof(SliceType));
    n->name = xstrdup(type);
    n->next = ctx->used_slices;
    ctx->used_slices = n;

    // Register Struct Def for Reflection
    char slice_name[MAX_TYPE_NAME_LEN];
    snprintf(slice_name, sizeof(slice_name), "Slice__%s", type);

    ASTNode *len_f = ast_create(NODE_FIELD);
    len_f->field.name = xstrdup("len");
    len_f->field.type = xstrdup("int");
    ASTNode *cap_f = ast_create(NODE_FIELD);
    cap_f->field.name = xstrdup("cap");
    cap_f->field.type = xstrdup("int");
    ASTNode *data_f = ast_create(NODE_FIELD);
    data_f->field.name = xstrdup("data");
    char ptr_type[MAX_TYPE_NAME_LEN];
    snprintf(ptr_type, sizeof(ptr_type), "%s*", type);
    data_f->field.type = xstrdup(ptr_type);

    data_f->next = len_f;
    len_f->next = cap_f;

    ASTNode *def = ast_create(NODE_STRUCT);
    def->strct.name = xstrdup(slice_name);
    def->strct.fields = data_f;

    register_struct_def(ctx, slice_name, def);

    // Backward compatibility: alias Slice_T to Slice__T
    char legacy_name[MAX_VAR_NAME_LEN];
    snprintf(legacy_name, sizeof(legacy_name), "Slice_%s", type);
    if (strcmp(slice_name, legacy_name) != 0)
    {
        register_type_alias(ctx, legacy_name, slice_name, NULL, 0, NULL, (Token){0}, 0);
    }
}

void register_tuple(ParserContext *ctx, const char *sig)
{
    TupleType *c = ctx->used_tuples;
    while (c)
    {
        if (strcmp(c->sig, sig) == 0)
        {
            return;
        }
        c = c->next;
    }
    TupleType *n = xmalloc(sizeof(TupleType));
    n->sig = xstrdup(sig);
    n->next = ctx->used_tuples;
    ctx->used_tuples = n;

    char struct_name[MAX_ERROR_MSG_LEN];
    char *clean_sig = sanitize_mangled_name(sig);
    snprintf(struct_name, sizeof(struct_name), "Tuple__%s", clean_sig);
    free(clean_sig);

    ASTNode *s_def = ast_create(NODE_STRUCT);
    s_def->strct.name = xstrdup(struct_name);

    char *s_sig = xstrdup(sig);
    char *current = s_sig;
    char *next_sep = strstr(current, "__");
    ASTNode *head = NULL, *tail = NULL;
    int i = 0;
    while (current)
    {
        if (next_sep)
        {
            *next_sep = 0;
        }

        ASTNode *f = ast_create(NODE_FIELD);
        char fname[32];
        snprintf(fname, sizeof(fname), "v%d", i++);
        f->field.name = xstrdup(fname);
        f->field.type = xstrdup(current);

        if (!head)
        {
            head = f;
        }
        else
        {
            tail->next = f;
        }
        tail = f;

        if (next_sep)
        {
            current = next_sep + 2;
            next_sep = strstr(current, "__");
        }
        else
        {
            break;
        }
    }
    free(s_sig);
    s_def->strct.fields = head;

    register_struct_def(ctx, struct_name, s_def);
}

void register_struct_def(ParserContext *ctx, const char *name, ASTNode *node)
{
    // In LSP mode, check for existing struct def to avoid duplicates
    if (g_config.mode_lsp)
    {
        StructDef *existing = NULL;
        StructDef *curr = ctx->struct_defs;
        while (curr)
        {
            if (strcmp(curr->name, name) == 0)
            {
                existing = curr;
                break;
            }
            curr = curr->next;
        }
        if (existing)
        {
            existing->node = node;
            // Symbol update will happen below
        }
    }

    StructDef *d = NULL;
    if (g_config.mode_lsp)
    {
        StructDef *curr = ctx->struct_defs;
        while (curr)
        {
            if (strcmp(curr->name, name) == 0)
            {
                d = curr;
                break;
            }
            curr = curr->next;
        }
    }

    if (!d)
    {
        d = xmalloc(sizeof(StructDef));
        d->name = xstrdup(name);
        d->next = ctx->struct_defs;
        ctx->struct_defs = d;
    }

    d->node = node;

    // MISRA Rule 5.7: Tag name shall be a unique identifier
    if (g_config.misra_mode)
    {
        ZenSymbol *all = ctx->all_symbols;
        while (all)
        {
            if ((all->kind == SYM_STRUCT || all->kind == SYM_ENUM) && strcmp(all->name, name) == 0)
            {
                zerror_at(node ? node->token : (Token){0}, "MISRA Rule 5.7");
                break;
            }
            all = all->next;
        }
    }

    ZenSymbol *sym_existing = symbol_lookup_local(ctx->global_scope, name);
    ZenSymbol *sym = NULL;
    if (!sym_existing)
    {
        sym = symbol_add(ctx->global_scope, name,
                         (node && node->type == NODE_ENUM) ? SYM_ENUM : SYM_STRUCT);
    }
    else
    {
        sym = sym_existing;
        // Ensure kind matches if re-registering
        sym->kind = (node && node->type == NODE_ENUM) ? SYM_ENUM : SYM_STRUCT;
    }

    sym->data.node = node;
    sym->link_name = node ? node->link_name : NULL;
    if (node)
    {
        sym->decl_token = node->token;
        if (node->type == NODE_STRUCT)
        {
            sym->is_export = node->strct.is_export;
        }
        else if (node->type == NODE_ENUM)
        {
            sym->is_export = node->enm.is_export;
        }
    }
    sym->type_info = node ? node->type_info : NULL;

    // MISRA Section 5 Auditing
    audit_section_5(ctx, ctx->global_scope, name, sym->link_name, sym->decl_token);

    register_symbol_to_lsp(ctx, sym);
}

ASTNode *find_struct_def(ParserContext *ctx, const char *name)
{
    ZenSymbol *sym = symbol_lookup_kind(ctx->current_scope, name, SYM_STRUCT);
    if (!sym)
    {
        sym = symbol_lookup_kind(ctx->current_scope, name, SYM_ENUM);
    }
    if (sym)
    {
        return sym->data.node;
    }

    extern ASTNode *global_user_structs;
    if (global_user_structs)
    {
        ASTNode *s = global_user_structs;
        while (s)
        {
            if ((s->type == NODE_STRUCT || s->type == NODE_ENUM) &&
                strcmp((s->type == NODE_STRUCT ? s->strct.name : s->enm.name), name) == 0)
            {
                if (s->type == NODE_STRUCT && s->strct.is_incomplete)
                {
                    s = s->next;
                    continue;
                }
                return s;
            }
            s = s->next;
        }
    }

    if (!ctx)
    {
        return NULL;
    }

    Instantiation *i = ctx->instantiations;
    while (i)
    {
        if (strcmp(i->name, name) == 0)
        {
            return i->struct_node;
        }
        i = i->next;
    }

    ASTNode *s = ctx->instantiated_structs;
    while (s)
    {
        if ((s->type == NODE_STRUCT || s->type == NODE_ENUM) &&
            strcmp((s->type == NODE_STRUCT ? s->strct.name : s->enm.name), name) == 0)
        {
            return s;
        }
        s = s->next;
    }

    StructRef *r = ctx->parsed_structs_list;
    while (r)
    {
        if (r->node->type == NODE_STRUCT && strcmp(r->node->strct.name, name) == 0)
        {
            return r->node;
        }
        if (r->node->type == NODE_ENUM && strcmp(r->node->enm.name, name) == 0)
        {
            return r->node;
        }
        r = r->next;
    }

    // Fallback: Search all symbols ever registered (robust for cross-module/early calls)
    ZenSymbol *all = ctx->all_symbols;
    while (all)
    {
        if ((all->kind == SYM_STRUCT || all->kind == SYM_ENUM) && strcmp(all->name, name) == 0)
        {
            if (all->data.node)
            {
                return all->data.node;
            }
        }
        all = all->next;
    }

    // Check manually registered definitions (e.g. Slices)
    StructDef *d = ctx->struct_defs;
    while (d)
    {
        if (strcmp(d->name, name) == 0)
        {
            return d->node;
        }
        d = d->next;
    }

    // Check enums list (for @derive(Eq) and field type lookups)
    StructRef *e = ctx->parsed_enums_list;
    while (e)
    {
        if (e->node->type == NODE_ENUM && strcmp(e->node->enm.name, name) == 0)
        {
            return e->node;
        }
        e = e->next;
    }

    return NULL;
}

ASTNode *find_trait_def(ParserContext *ctx, const char *name)
{
    if (!ctx || !name)
    {
        return NULL;
    }

    StructRef *r = ctx->parsed_globals_list;
    while (r)
    {
        if (r->node && r->node->type == NODE_TRAIT && strcmp(r->node->trait.name, name) == 0)
        {
            return r->node;
        }
        r = r->next;
    }
    return NULL;
}

ASTNode *find_concrete_struct_def(ParserContext *ctx, const char *name)
{
    Instantiation *i = ctx->instantiations;
    while (i)
    {
        if (strcmp(i->name, name) == 0 && i->struct_node && i->struct_node->type == NODE_STRUCT &&
            !i->struct_node->strct.is_template)
        {
            return i->struct_node;
        }
        i = i->next;
    }

    ASTNode *s = ctx->instantiated_structs;
    while (s)
    {
        if (s->type == NODE_STRUCT && !s->strct.is_template && strcmp(s->strct.name, name) == 0)
        {
            return s;
        }
        s = s->next;
    }

    StructRef *r = ctx->parsed_structs_list;
    while (r)
    {
        if (r->node->type == NODE_STRUCT && !r->node->strct.is_template &&
            strcmp(r->node->strct.name, name) == 0)
        {
            return r->node;
        }
        r = r->next;
    }

    StructDef *d = ctx->struct_defs;
    while (d)
    {
        if (d->node && d->node->type == NODE_STRUCT && !d->node->strct.is_template &&
            strcmp(d->name, name) == 0)
        {
            return d->node;
        }
        d = d->next;
    }

    return NULL;
}

Module *find_module(ParserContext *ctx, const char *alias)
{
    Module *m = ctx->modules;
    while (m)
    {
        if (m->alias && strcmp(m->alias, alias) == 0)
        {
            return m;
        }
        m = m->next;
    }
    return NULL;
}

void register_module(ParserContext *ctx, const char *alias, const char *path)
{
    Module *m = xmalloc(sizeof(Module));
    m->alias = alias ? xstrdup(alias) : NULL;
    m->path = xstrdup(path);
    m->base_name = extract_module_name(path);
    m->next = ctx->modules;
    ctx->modules = m;
}

void register_selective_import(ParserContext *ctx, const char *symbol, const char *alias,
                               const char *source_module)
{
    SelectiveImport *si = xmalloc(sizeof(SelectiveImport));
    si->symbol = xstrdup(symbol);
    si->alias = alias ? xstrdup(alias) : NULL;
    si->source_module = xstrdup(source_module);
    si->next = ctx->selective_imports;
    ctx->selective_imports = si;
}

SelectiveImport *find_selective_import(ParserContext *ctx, const char *name)
{
    SelectiveImport *si = ctx->selective_imports;
    while (si)
    {
        if (si->alias && strcmp(si->alias, name) == 0)
        {
            return si;
        }
        if (!si->alias && strcmp(si->symbol, name) == 0)
        {
            return si;
        }
        si = si->next;
    }
    return NULL;
}

char *extract_module_name(const char *path)
{
    const char *slash = strrchr(path, '/');
    const char *backslash = strrchr(path, '\\');
    if (backslash && (!slash || backslash > slash))
    {
        slash = backslash;
    }

    const char *base = slash ? slash + 1 : path;
    const char *dot = strrchr(base, '.');
    int len = dot ? (int)(dot - base) : (int)strlen(base);
    char *name = xmalloc(len + 1);
    strncpy(name, base, len);
    name[len] = 0;

    // Sanitize to ensure valid C identifier
    for (int i = 0; i < len; i++)
    {
        if (!isalnum(name[i]))
        {
            name[i] = '_';
        }
    }

    return name;
}

int is_ident_char(char c)
{
    return isalnum(c) || c == '_';
}

ASTNode *copy_fields(ASTNode *fields)
{
    if (!fields)
    {
        return NULL;
    }
    ASTNode *n = ast_create(NODE_FIELD);
    n->field.name = xstrdup(fields->field.name);
    n->field.type = xstrdup(fields->field.type);
    n->next = copy_fields(fields->next);
    return n;
}

char *replace_in_string(const char *src, const char *old_w, const char *new_w)
{
    if (!src || !old_w || !new_w)
    {
        return src ? xstrdup(src) : NULL;
    }

    // Check for multiple parameters (comma separated)
    if (strchr(old_w, ','))
    {
        char *running_src = xstrdup(src);

        char *p_ptr = (char *)old_w;
        char *c_ptr = (char *)new_w;

        while (*p_ptr && *c_ptr)
        {
            char *p_end = strchr(p_ptr, ',');
            int p_len = p_end ? (int)(p_end - p_ptr) : (int)strlen(p_ptr);

            char *c_end = strchr(c_ptr, ',');
            int c_len = c_end ? (int)(c_end - c_ptr) : (int)strlen(c_ptr);

            char *curr_p = xmalloc(p_len + 1);
            strncpy(curr_p, p_ptr, p_len);
            curr_p[p_len] = 0;

            char *curr_c = xmalloc(c_len + 1);
            strncpy(curr_c, c_ptr, c_len);
            curr_c[c_len] = 0;

            char *next_src = replace_in_string(running_src, curr_p, curr_c);
            free(running_src);
            running_src = next_src;

            free(curr_p);
            free(curr_c);

            if (p_end)
            {
                p_ptr = p_end + 1;
            }
            else
            {
                break;
            }
            if (c_end)
            {
                c_ptr = c_end + 1;
            }
            else
            {
                break;
            }
        }
        return running_src;
    }

    char *result;
    int i, cnt = 0;
    int newWlen = strlen(new_w);
    int oldWlen = strlen(old_w);

    // Pass 1: Count replacements
    int in_string = 0;
    for (i = 0; src[i] != '\0'; i++)
    {
        if (src[i] == '\"' && (i == 0 || src[i - 1] != '\\'))
        {
            in_string = !in_string;
        }

        if (!in_string && strstr(&src[i], old_w) == &src[i])
        {
            // Check boundaries
            int valid = 1;
            if (i > 0 && is_ident_char(src[i - 1]))
            {
                valid = 0;
            }
            if (valid && (is_ident_char(src[i + oldWlen]) || src[i + oldWlen] == '<'))
            {
                valid = 0;
            }

            if (valid)
            {
                cnt++;
                i += oldWlen - 1;
            }
        }
    }

    // Allocate result buffer
    result = (char *)xmalloc(i + cnt * (newWlen - oldWlen) + 1);

    // Pass 2: Perform replacement
    int j = 0;
    in_string = 0;

    int src_idx = 0;

    while (src[src_idx] != '\0')
    {
        if (src[src_idx] == '\"' && (src_idx == 0 || src[src_idx - 1] != '\\'))
        {
            in_string = !in_string;
        }

        int replaced = 0;
        if (!in_string && strstr(&src[src_idx], old_w) == &src[src_idx])
        {
            int valid = 1;
            if (src_idx > 0 && is_ident_char(src[src_idx - 1]))
            {
                valid = 0;
            }
            if (valid && (is_ident_char(src[src_idx + oldWlen]) || src[src_idx + oldWlen] == '<'))
            {
                valid = 0;
            }

            if (valid)
            {
                strcpy(&result[j], new_w);
                j += newWlen;
                src_idx += oldWlen;
                replaced = 1;
            }
        }

        if (!replaced)
        {
            result[j++] = src[src_idx++];
        }
    }
    result[j] = '\0';
    return result;
}

Type *replace_type_formal(Type *t, const char *p, const char *c, const char *os, const char *ns);
// Helper to replace generic params in mangled names (e.g. Option_V_None ->
// Option_int_None)
char *replace_mangled_part(const char *src, const char *param, const char *concrete)
{
    if (!src || !param || !concrete)
    {
        return src ? xstrdup(src) : NULL;
    }

    size_t plen = strlen(param);
    size_t clen = strlen(concrete);
    size_t src_len = strlen(src);

    // Initial estimate for result size
    size_t res_cap = src_len + 512;
    char *result = xmalloc(res_cap);
    result[0] = 0;

    const char *curr = src;
    char *out = result;
    size_t current_len = 0;

    while (*curr)
    {
        // Ensure enough space (including the next character or replacement)
        if (current_len + (clen > 1 ? clen : 1) + 1 >= res_cap)
        {
            res_cap = res_cap * 2 + clen;
            char *new_res = xmalloc(res_cap);
            memcpy(new_res, result, current_len);
            free(result);
            result = new_res;
            out = result + current_len;
        }

        // Check if param matches here
        if (strncmp(curr, param, plen) == 0)
        {
            int valid = 1;
            int has_underscore_boundary = 0;

            if (curr > src)
            {
                if (*(curr - 1) == '_')
                {
                    has_underscore_boundary = 1;
                }
                else if (is_ident_char(*(curr - 1)))
                {
                    valid = 0;
                }
            }

            if (valid && curr[plen] != 0 && curr[plen] != '_' && is_ident_char(curr[plen]))
            {
                if (strncmp(curr + plen, "Ptr", 3) != 0)
                {
                    valid = 0;
                }
            }
            if (valid && curr[plen] == '_')
            {
                has_underscore_boundary = 1;
            }

            if (valid && !has_underscore_boundary)
            {
                // Also allow <, ,, (, [ as boundaries
                char prev = (curr > src) ? *(curr - 1) : 0;
                if (prev == '<' || prev == ',' || prev == '(' || prev == '[' || prev == ' ')
                {
                    // OK
                }
                else
                {
                    valid = 0;
                }
            }

            if (valid)
            {
                // Ensure double underscore boundary for the replacement
                if (curr > src && *(curr - 1) == '_' && (curr == src + 1 || *(curr - 2) != '_'))
                {
                    *out++ = '_';
                    current_len++;
                }

                memcpy(out, concrete, clen);
                out += clen;
                current_len += clen;

                if (curr[plen] == '_' && curr[plen + 1] != '_')
                {
                    *out++ = '_';
                    current_len++;
                }

                curr += plen;
                continue;
            }
        }
        *out++ = *curr++;
        current_len++;
    }
    *out = 0;
    return result;
}

char *replace_type_str(const char *src, const char *param, const char *concrete,
                       const char *old_struct, const char *new_struct)
{
    if (!src)
    {
        return NULL;
    }
    if (!param || !concrete)
    {
        return xstrdup(src);
    }

    // 1. Exact match (base case)
    if (strcmp(src, param) == 0)
    {
        return xstrdup(concrete);
    }

    // 2. Handle simple pointer cases recursively (safe as src shrinks)
    size_t slen = strlen(src);
    if (slen > 1 && src[slen - 1] == '*')
    {
        char *base = xmalloc(slen);
        strncpy(base, src, slen - 1);
        base[slen - 1] = 0;
        char *nb = replace_type_str(base, param, concrete, old_struct, new_struct);
        char *res = xmalloc(strlen(nb) + 2);
        sprintf(res, "%s*", nb);
        free(base);
        free(nb);
        return res;
    }

    // 3. Structural fallback for complex strings (e.g. "Self", "Option<T>")
    char *res = xstrdup(src);

    // Case 3a: Explicit template replacement (e.g. Vec<T> -> Vec__int32_t)
    if (old_struct && new_struct && param)
    {
        char tpl_w[MAX_TYPE_NAME_LEN];
        snprintf(tpl_w, sizeof(tpl_w), "%s<%s>", old_struct, param);
        if (strstr(res, tpl_w))
        {
            char *tmp = replace_in_string(res, tpl_w, new_struct);
            free(res);
            res = tmp;
        }
    }

    // Case 3b: Base struct replacement (e.g. Vec -> Vec__int32_t)
    if (old_struct && new_struct && strstr(res, old_struct))
    {
        char *tmp = replace_in_string(res, old_struct, new_struct);
        free(res);
        res = tmp;
    }

    // 4. Boundary-safe mangled replacement (e.g. "Option_T" or "Option__T")
    // Split multi-param strings (X, Y, Z) and replace each individually
    char *final_res = xstrdup(res);
    if (param && concrete && strchr(param, ','))
    {
        char *p_ptr = (char *)param;
        char *c_ptr = (char *)concrete;
        while (*p_ptr && *c_ptr)
        {
            char *p_end = strchr(p_ptr, ',');
            int p_len = p_end ? (int)(p_end - p_ptr) : (int)strlen(p_ptr);
            char *c_end = strchr(c_ptr, ',');
            int c_len = c_end ? (int)(c_end - c_ptr) : (int)strlen(c_ptr);

            char *p_part = xmalloc(p_len + 1);
            strncpy(p_part, p_ptr, p_len);
            p_part[p_len] = 0;

            char *c_part = xmalloc(c_len + 1);
            strncpy(c_part, c_ptr, c_len);
            c_part[c_len] = 0;

            char *clean_c = sanitize_mangled_name(c_part);
            char *tmp = replace_mangled_part(final_res, p_part, clean_c);
            free(final_res);
            final_res = tmp;

            free(p_part);
            free(c_part);
            free(clean_c);

            if (p_end)
            {
                p_ptr = p_end + 1;
            }
            else
            {
                break;
            }
            if (c_end)
            {
                c_ptr = c_end + 1;
            }
            else
            {
                break;
            }
        }
    }
    else
    {
        char *t1 = replace_in_string(final_res, param, concrete);
        free(final_res);
        final_res = t1;

        char *clean_c = sanitize_mangled_name(concrete);
        char *tmp = replace_mangled_part(final_res, param, clean_c);
        free(final_res);
        final_res = tmp;
        free(clean_c);
    }

    free(res);
    return final_res;
}

ASTNode *copy_ast_replacing(ASTNode *n, const char *p, const char *c, const char *os,
                            const char *ns);

Type *type_from_string_helper(const char *c)
{
    if (!c)
    {
        return NULL;
    }

    // Check for pointer suffix '*'
    size_t len = strlen(c);
    if (len > 0 && c[len - 1] == '*')
    {
        size_t base_len = len - 1;
        char *base = xmalloc(base_len + 1);
        strncpy(base, c, base_len);
        base[base_len] = 0;

        Type *inner = type_from_string_helper(base);
        free(base);

        return type_new_ptr(inner);
    }

    // Check for 'const ' prefix
    if (strncmp(c, "const ", 6) == 0)
    {
        Type *inner = type_from_string_helper(c + 6);
        if (inner)
        {
            inner->is_const = 1;
        }
        return inner;
    }

    if (strncmp(c, "struct ", 7) == 0)
    {
        Type *n = type_new(TYPE_STRUCT);
        n->name = sanitize_mangled_name(c + 7);
        n->is_explicit_struct = 1;
        return n;
    }

    if (strcmp(c, "int") == 0)
    {
        return type_new(TYPE_INT);
    }
    if (strcmp(c, "float") == 0)
    {
        return type_new(TYPE_FLOAT);
    }
    if (strcmp(c, "void") == 0)
    {
        return type_new(TYPE_VOID);
    }
    if (strcmp(c, "string") == 0)
    {
        return type_new(TYPE_STRING);
    }
    if (strcmp(c, "bool") == 0)
    {
        return type_new(TYPE_BOOL);
    }
    if (strcmp(c, "char") == 0)
    {
        return type_new(TYPE_CHAR);
    }
    if (strcmp(c, "I8") == 0 || strcmp(c, "i8") == 0)
    {
        return type_new(TYPE_I8);
    }
    if (strcmp(c, "U8") == 0 || strcmp(c, "u8") == 0)
    {
        return type_new(TYPE_U8);
    }
    if (strcmp(c, "I16") == 0 || strcmp(c, "i16") == 0)
    {
        return type_new(TYPE_I16);
    }
    if (strcmp(c, "U16") == 0 || strcmp(c, "u16") == 0)
    {
        return type_new(TYPE_U16);
    }
    if (strcmp(c, "I32") == 0 || strcmp(c, "i32") == 0 || strcmp(c, "int32_t") == 0)
    {
        return type_new(TYPE_I32);
    }
    if (strcmp(c, "U32") == 0 || strcmp(c, "u32") == 0 || strcmp(c, "uint32_t") == 0)
    {
        return type_new(TYPE_U32);
    }
    if (strcmp(c, "I64") == 0 || strcmp(c, "i64") == 0 || strcmp(c, "int64_t") == 0)
    {
        return type_new(TYPE_I64);
    }
    if (strcmp(c, "U64") == 0 || strcmp(c, "u64") == 0 || strcmp(c, "uint64_t") == 0)
    {
        return type_new(TYPE_U64);
    }
    if (strcmp(c, "float") == 0 || strcmp(c, "f32") == 0)
    {
        return type_new(TYPE_F32);
    }
    if (strcmp(c, "double") == 0 || strcmp(c, "f64") == 0)
    {
        return type_new(TYPE_F64);
    }
    if (strcmp(c, "I128") == 0 || strcmp(c, "i128") == 0)
    {
        return type_new(TYPE_I128);
    }
    if (strcmp(c, "U128") == 0 || strcmp(c, "u128") == 0)
    {
        return type_new(TYPE_U128);
    }
    if (strcmp(c, "rune") == 0)
    {
        return type_new(TYPE_RUNE);
    }
    if (strcmp(c, "uint") == 0)
    {
        return type_new(TYPE_UINT);
    }

    if (strcmp(c, "byte") == 0)
    {
        return type_new(TYPE_BYTE);
    }
    if (strcmp(c, "usize") == 0)
    {
        return type_new(TYPE_USIZE);
    }
    if (strcmp(c, "isize") == 0)
    {
        return type_new(TYPE_ISIZE);
    }

    Type *n = type_new(TYPE_STRUCT);
    n->name = sanitize_mangled_name(c);
    return n;
}

Type *replace_type_formal(Type *t, const char *p, const char *c, const char *os, const char *ns)
{
    if (!t || (uintptr_t)t < 0x10000)
    {
        return NULL;
    }

    // Defensive check: Ensure kind is valid
    if ((int)t->kind < 0 || (int)t->kind > 100) // 100 is a safe upper bound for TypeKind
    {
        return NULL;
    }

    // Exact Match Logic (with multi-param splitting)
    if ((t->kind == TYPE_STRUCT || t->kind == TYPE_GENERIC) && t->name)
    {

        if (p && c && strchr(p, ','))
        {
            char *p_ptr = (char *)p;
            char *c_ptr = (char *)c;
            while (*p_ptr && *c_ptr)
            {
                char *p_end = strchr(p_ptr, ',');
                int p_len = p_end ? (int)(p_end - p_ptr) : (int)strlen(p_ptr);
                char *c_end = strchr(c_ptr, ',');
                int c_len = c_end ? (int)(c_end - c_ptr) : (int)strlen(c_ptr);

                if ((int)strlen(t->name) == p_len && strncmp(t->name, p_ptr, p_len) == 0)
                {
                    char *c_part = xmalloc(c_len + 1);
                    strncpy(c_part, c_ptr, c_len);
                    c_part[c_len] = 0;

                    Type *res = type_from_string_helper(c_part);
                    free(c_part);
                    return res;
                }
                if (p_end)
                {
                    p_ptr = p_end + 1;
                }
                else
                {
                    break;
                }
                if (c_end)
                {
                    c_ptr = c_end + 1;
                }
                else
                {
                    break;
                }
            }
        }
        else if (p && strcmp(t->name, p) == 0)
        {
            return type_from_string_helper(c);
        }
    }

    Type *n = xmalloc(sizeof(Type));
    *n = *t;

    if (t->name)
    {
        if (os && ns && strcmp(t->name, os) == 0)
        {
            n->name = xstrdup(ns);
            n->kind = TYPE_STRUCT;
            n->arg_count = 0;
            n->args = NULL;
        }
        else if (p && c)
        {
            // Suffix Match Logic (with multi-param splitting)
            char p_suffix[4096];
            p_suffix[0] = 0;

            const char *p_ptr = p;
            while (p_ptr && *p_ptr)
            {
                const char *p_next = strchr(p_ptr, ',');
                int sub_len = p_next ? (int)(p_next - p_ptr) : (int)strlen(p_ptr);
                char *sub = xmalloc(sub_len + 1);
                strncpy(sub, p_ptr, sub_len);
                sub[sub_len] = 0;

                char *clean_sub = sanitize_mangled_name(sub);
                strcat(p_suffix, "__");
                strcat(p_suffix, clean_sub);
                free(clean_sub);
                free(sub);

                if (p_next)
                {
                    p_ptr = p_next + 1;
                }
                else
                {
                    break;
                }
            }

            size_t nlen = strlen(t->name);
            size_t slen = strlen(p_suffix);

            int match = 0;
            int found_slen = 0;
            int num_ptr_suffixes = 0;
            if (nlen >= slen && strcmp(t->name + nlen - slen, p_suffix) == 0)
            {
                match = 1;
                found_slen = slen;
            }
            else if (nlen > slen)
            {
                // Try matching with Ptr suffix
                const char *p_match = strstr(t->name, p_suffix);
                while (p_match)
                {
                    const char *after = p_match + slen;
                    int is_all_ptr = 1;
                    if (*after == '\0')
                    {
                        is_all_ptr = 0; // Handled by exact match above
                    }
                    while (*after)
                    {
                        if (strncmp(after, "Ptr", 3) == 0)
                        {
                            after += 3;
                        }
                        else
                        {
                            is_all_ptr = 0;
                            break;
                        }
                    }
                    if (is_all_ptr)
                    {
                        match = 1;
                        found_slen = nlen - (p_match - t->name);
                        num_ptr_suffixes = (nlen - (p_match - t->name) - slen) / 3;
                        break;
                    }
                    p_match = strstr(p_match + 1, p_suffix);
                }
            }

            if (match)
            {
                slen = found_slen;
                char c_suffix[MAX_ERROR_MSG_LEN];
                c_suffix[0] = 0;
                const char *c_ptr = c;
                while (c_ptr && *c_ptr)
                {
                    const char *c_next = strchr(c_ptr, ',');
                    int sub_len = c_next ? (int)(c_next - c_ptr) : (int)strlen(c_ptr);

                    char *sub = xmalloc(sub_len + 1);
                    strncpy(sub, c_ptr, sub_len);
                    sub[sub_len] = 0;

                    char *clean = sanitize_mangled_name(sub);
                    // Standardize: always use __ for mangled part
                    strcat(c_suffix, "__");
                    strcat(c_suffix, clean);
                    free(clean);
                    free(sub);

                    if (c_next)
                    {
                        c_ptr = c_next + 1;
                    }
                    else
                    {
                        break;
                    }
                }

                // Calculate required size more accurately
                size_t c_suffix_len = strlen(c_suffix);
                size_t total_needed =
                    (nlen > slen ? nlen - slen : 0) + c_suffix_len + (num_ptr_suffixes * 3) + 1;
                char *new_name = xmalloc(total_needed);
                if (nlen > slen)
                {
                    strncpy(new_name, t->name, nlen - slen);
                    new_name[nlen - slen] = 0;
                }
                else
                {
                    new_name[0] = 0;
                }

                // Handle underscore merging: ensure exactly two underscores
                char *p_end = new_name + strlen(new_name);
                while (p_end > new_name && *(p_end - 1) == '_')
                {
                    *(--p_end) = '\0';
                }
                strcat(new_name, c_suffix);

                // Restore Ptr suffixes
                for (int k = 0; k < num_ptr_suffixes; k++)
                {
                    strcat(new_name, "Ptr");
                }
                n->name = new_name;
                n->kind = TYPE_STRUCT;
                n->arg_count = 0;
                n->args = NULL;
            }
            else
            {
                n->name = xstrdup(t->name);
            }
        }
        else
        {
            n->name = xstrdup(t->name);
        }
    }

    if (t->kind == TYPE_POINTER || t->kind == TYPE_ARRAY || t->kind == TYPE_FUNCTION ||
        t->kind == TYPE_VECTOR)
    {
        n->inner = replace_type_formal(t->inner, p, c, os, ns);
    }

    if (t->arg_count > 0 && t->args)
    {
        n->args = xmalloc(sizeof(Type *) * t->arg_count);
        for (int i = 0; i < t->arg_count; i++)
        {
            n->args[i] = replace_type_formal(t->args[i], p, c, os, ns);
        }
    }

    return n;
}

ASTNode *copy_ast_replacing(ASTNode *n, const char *p, const char *c, const char *os,
                            const char *ns)
{
    if (!n)
    {
        return NULL;
    }

    ASTNode *new_node = ast_create(n->type);
    ASTNode *old_next =
        new_node->next; // Preserve next if ast_create did something (it doesn't currently)
    *new_node = *n;
    new_node->next = old_next; // Restore next before recursion

    if (n->resolved_type)
    {
        new_node->resolved_type = replace_type_str(n->resolved_type, p, c, os, ns);
    }
    new_node->type_info = replace_type_formal(n->type_info, p, c, os, ns);

    new_node->next = copy_ast_replacing(n->next, p, c, os, ns);

    switch (n->type)
    {
    case NODE_FUNCTION:
        new_node->func.name = n->func.name ? xstrdup(n->func.name) : NULL;
        new_node->func.ret_type = replace_type_str(n->func.ret_type, p, c, os, ns);

        char *tmp_args = n->func.args ? xstrdup(n->func.args) : NULL;
        if (p && c && strchr(p, ','))
        {
            char *p_ptr = (char *)p;
            char *c_ptr = (char *)c;
            while (*p_ptr && *c_ptr)
            {
                char *p_end = strchr(p_ptr, ',');
                int p_len = p_end ? (int)(p_end - p_ptr) : (int)strlen(p_ptr);
                char *c_end = strchr(c_ptr, ',');
                int c_len = c_end ? (int)(c_end - c_ptr) : (int)strlen(c_ptr);

                char *p_part = xmalloc(p_len + 1);
                strncpy(p_part, p_ptr, p_len);
                p_part[p_len] = 0;

                char *c_part = xmalloc(c_len + 1);
                strncpy(c_part, c_ptr, c_len);
                c_part[c_len] = 0;

                char *t1 = replace_in_string(tmp_args, p_part, c_part);
                free(tmp_args);
                tmp_args = t1;

                char *clean_c = sanitize_mangled_name(c_part);
                char *t2 = replace_mangled_part(tmp_args, p_part, clean_c);
                free(tmp_args);
                tmp_args = t2;

                free(p_part);
                free(c_part);
                free(clean_c);

                if (p_end)
                {
                    p_ptr = p_end + 1;
                }
                else
                {
                    break;
                }
                if (c_end)
                {
                    c_ptr = c_end + 1;
                }
                else
                {
                    break;
                }
            }
        }
        else
        {
            char *t1 = replace_in_string(tmp_args, p, c);
            free(tmp_args);
            tmp_args = t1;

            if (p && c)
            {
                char *clean_c = sanitize_mangled_name(c);
                char *t2 = replace_mangled_part(tmp_args, p, clean_c);
                free(tmp_args);
                tmp_args = t2;
                free(clean_c);
            }
        }

        if (os && ns)
        {
            char *tmp2 = replace_in_string(tmp_args, os, ns);
            free(tmp_args);
            tmp_args = tmp2;
        }
        new_node->func.arg_count = n->func.arg_count;
        if (n->func.arg_count > 0 && n->func.arg_types)
        {
            new_node->func.arg_types = xmalloc(sizeof(Type *) * n->func.arg_count);
            for (int i = 0; i < n->func.arg_count; i++)
            {
                new_node->func.arg_types[i] =
                    replace_type_formal(n->func.arg_types[i], p, c, os, ns);
            }
        }
        else
        {
            new_node->func.arg_types = NULL;
        }
        new_node->func.args = tmp_args;

        new_node->func.ret_type_info = replace_type_formal(n->func.ret_type_info, p, c, os, ns);

        // Deep copy default values AST if present
        if (n->func.default_values && n->func.arg_count > 0)
        {
            new_node->func.default_values = xmalloc(sizeof(ASTNode *) * n->func.arg_count);
            // We also need to regenerate the string defaults array based on the substituted ASTs
            // This ensures potential generic params in default values (T{}) are updated (i32{})
            // in the string representation used by codegen.
            char **new_defaults_strs = xmalloc(sizeof(char *) * n->func.arg_count);

            for (int i = 0; i < n->func.arg_count; i++)
            {
                if (n->func.default_values[i])
                {
                    new_node->func.default_values[i] =
                        copy_ast_replacing(n->func.default_values[i], p, c, os, ns);
                    new_defaults_strs[i] = ast_to_string(new_node->func.default_values[i]);
                }
                else
                {
                    new_node->func.default_values[i] = NULL;
                    new_defaults_strs[i] = NULL;
                }
            }
            // Replace the old string-based defaults with our regenerated ones
            // Note: We leak the old 'tmp_args' calculated above, but that's just a single string
            // for valid args The 'defaults' array in func struct is what matters for function
            // definition. Wait, NODE_FUNCTION has char *args (legacy) AND char **defaults (array).
            // parse_and_convert_args populated both.
            // We need to update new_node->func.defaults.
            new_node->func.defaults = new_defaults_strs;
        }

        new_node->func.body = copy_ast_replacing(n->func.body, p, c, os, ns);
        break;
    case NODE_BLOCK:
        new_node->block.statements = copy_ast_replacing(n->block.statements, p, c, os, ns);
        break;
    case NODE_RAW_STMT:
    {
        char *s1 = xstrdup(n->raw_stmt.content);
        if (p && c && strchr(p, ','))
        {
            char *p_ptr = (char *)p;
            char *c_ptr = (char *)c;
            while (*p_ptr && *c_ptr)
            {
                char *p_end = strchr(p_ptr, ',');
                int p_len = p_end ? (int)(p_end - p_ptr) : (int)strlen(p_ptr);
                char *c_end = strchr(c_ptr, ',');
                int c_len = c_end ? (int)(c_end - c_ptr) : (int)strlen(c_ptr);

                char *p_part = xmalloc(p_len + 1);
                strncpy(p_part, p_ptr, p_len);
                p_part[p_len] = 0;

                char *c_part = xmalloc(c_len + 1);
                strncpy(c_part, c_ptr, c_len);
                c_part[c_len] = 0;

                char *t1 = replace_in_string(s1, p_part, c_part);
                free(s1);
                s1 = t1;

                char *clean_c = sanitize_mangled_name(c_part);
                char *t2 = replace_mangled_part(s1, p_part, clean_c);
                free(s1);
                s1 = t2;

                free(p_part);
                free(c_part);
                free(clean_c);

                if (p_end)
                {
                    p_ptr = p_end + 1;
                }
                else
                {
                    break;
                }
                if (c_end)
                {
                    c_ptr = c_end + 1;
                }
                else
                {
                    break;
                }
            }
        }
        else
        {
            char *t1 = replace_in_string(s1, p, c);
            free(s1);
            s1 = t1;

            if (p && c)
            {
                char *clean_c = sanitize_mangled_name(c);
                char *t2 = replace_mangled_part(s1, p, clean_c);
                free(s1);
                s1 = t2;
                free(clean_c);
            }
        }

        if (os && ns)
        {
            char *s2 = replace_in_string(s1, os, ns);
            free(s1);
            s1 = s2;
        }

        new_node->raw_stmt.content = s1;
    }
    break;
    case NODE_VAR_DECL:
        new_node->var_decl.name = n->var_decl.name ? xstrdup(n->var_decl.name) : NULL;
        new_node->var_decl.type_str = replace_type_str(n->var_decl.type_str, p, c, os, ns);
        new_node->var_decl.type_info = replace_type_formal(n->var_decl.type_info, p, c, os, ns);
        new_node->var_decl.init_expr = copy_ast_replacing(n->var_decl.init_expr, p, c, os, ns);
        break;
    case NODE_RETURN:
        new_node->ret.value = copy_ast_replacing(n->ret.value, p, c, os, ns);
        break;
    case NODE_EXPR_BINARY:
        new_node->binary.left = copy_ast_replacing(n->binary.left, p, c, os, ns);
        new_node->binary.right = copy_ast_replacing(n->binary.right, p, c, os, ns);
        new_node->binary.op = n->binary.op ? xstrdup(n->binary.op) : NULL;
        break;
    case NODE_EXPR_UNARY:
        new_node->unary.op = n->unary.op ? xstrdup(n->unary.op) : NULL;
        new_node->unary.operand = copy_ast_replacing(n->unary.operand, p, c, os, ns);
        break;
    case NODE_EXPR_CALL:
        new_node->call.callee = copy_ast_replacing(n->call.callee, p, c, os, ns);
        new_node->call.args = copy_ast_replacing(n->call.args, p, c, os, ns);
        new_node->call.arg_names = n->call.arg_names; // Share pointer (shallow copy)
        new_node->call.arg_count = n->call.arg_count;
        break;
    case NODE_EXPR_VAR:
    {
        char *n1 = n->var_ref.name ? xstrdup(n->var_ref.name) : NULL;
        if (p && c && strchr(p, ','))
        {
            char *p_ptr = (char *)p;
            char *c_ptr = (char *)c;
            while (*p_ptr && *c_ptr)
            {
                char *p_end = strchr(p_ptr, ',');
                int p_len = p_end ? (int)(p_end - p_ptr) : (int)strlen(p_ptr);
                char *c_end = strchr(c_ptr, ',');
                int c_len = c_end ? (int)(c_end - c_ptr) : (int)strlen(c_ptr);

                char *p_part = xmalloc(p_len + 1);
                strncpy(p_part, p_ptr, p_len);
                p_part[p_len] = 0;

                char *c_part = xmalloc(c_len + 1);
                strncpy(c_part, c_ptr, c_len);
                c_part[c_len] = 0;

                char *t1 = replace_in_string(n1, p_part, c_part);
                free(n1);
                n1 = t1;

                char *clean_c = sanitize_mangled_name(c_part);
                char *t2 = replace_mangled_part(n1, p_part, clean_c);
                free(n1);
                n1 = t2;

                free(p_part);
                free(c_part);
                free(clean_c);

                if (p_end)
                {
                    p_ptr = p_end + 1;
                }
                else
                {
                    break;
                }
                if (c_end)
                {
                    c_ptr = c_end + 1;
                }
                else
                {
                    break;
                }
            }
        }
        else
        {
            if (p && c)
            {
                char *t1 = replace_in_string(n1, p, c);
                free(n1);
                n1 = t1;

                char *clean_c = sanitize_mangled_name(c);
                char *n2 = replace_mangled_part(n1, p, clean_c);
                free(clean_c);
                free(n1);
                n1 = n2;
            }
        }

        if (os && ns)
        {
            int os_len = strlen(os);
            int ns_len = strlen(ns);
            // Only replace if it starts with os__ and DOES NOT already start with ns__
            if (strncmp(n1, os, os_len) == 0 && n1[os_len] == '_' && n1[os_len + 1] == '_' &&
                strncmp(n1, ns, ns_len) != 0)
            {
                char *suffix = n1 + os_len;
                char buf[MAX_ERROR_MSG_LEN];
                snprintf(buf, sizeof(buf), "%s%s", ns, suffix);
                char *n3 = merge_underscores(buf);
                free(n1);
                n1 = n3;
            }
        }
        new_node->var_ref.name = n1;
    }
    break;
    case NODE_FIELD:
        new_node->field.name = n->field.name ? xstrdup(n->field.name) : NULL;
        new_node->field.type = replace_type_str(n->field.type, p, c, os, ns);
        break;
    case NODE_EXPR_LITERAL:
        if (n->literal.type_kind == LITERAL_STRING)
        {
            new_node->literal.string_val =
                n->literal.string_val ? xstrdup(n->literal.string_val) : NULL;
        }
        break;
    case NODE_EXPR_MEMBER:
        new_node->member.target = copy_ast_replacing(n->member.target, p, c, os, ns);
        new_node->member.field = n->member.field ? xstrdup(n->member.field) : NULL;
        break;
    case NODE_EXPR_INDEX:
        new_node->index.array = copy_ast_replacing(n->index.array, p, c, os, ns);
        new_node->index.index = copy_ast_replacing(n->index.index, p, c, os, ns);
        break;
    case NODE_EXPR_CAST:
        new_node->cast.target_type = replace_type_str(n->cast.target_type, p, c, os, ns);
        new_node->cast.expr = copy_ast_replacing(n->cast.expr, p, c, os, ns);
        break;
    case NODE_EXPR_STRUCT_INIT:
    {
        char *new_name = replace_type_str(n->struct_init.struct_name, p, c, os, ns);

        int is_ptr = 0;
        size_t len = strlen(new_name);
        if (len > 0 && new_name[len - 1] == '*')
        {
            is_ptr = 1;
        }

        int is_primitive = is_primitive_type_name(new_name);

        if ((is_ptr || is_primitive) && !n->struct_init.fields)
        {
            new_node->type = NODE_EXPR_LITERAL;
            new_node->literal.type_kind = LITERAL_INT;
            new_node->literal.int_val = 0;
            free(new_name);
        }
        else
        {
            new_node->struct_init.struct_name = new_name;
            ASTNode *h = NULL, *t = NULL, *curr = n->struct_init.fields;
            while (curr)
            {
                ASTNode *cp = copy_ast_replacing(curr, p, c, os, ns);
                cp->next = NULL;
                if (!h)
                {
                    h = cp;
                }
                else
                {
                    t->next = cp;
                }
                t = cp;
                curr = curr->next;
            }
            new_node->struct_init.fields = h;
        }
        break;
    }
    case NODE_IF:
        new_node->if_stmt.condition = copy_ast_replacing(n->if_stmt.condition, p, c, os, ns);
        new_node->if_stmt.then_body = copy_ast_replacing(n->if_stmt.then_body, p, c, os, ns);
        new_node->if_stmt.else_body = copy_ast_replacing(n->if_stmt.else_body, p, c, os, ns);
        break;
    case NODE_WHILE:
        new_node->while_stmt.condition = copy_ast_replacing(n->while_stmt.condition, p, c, os, ns);
        new_node->while_stmt.body = copy_ast_replacing(n->while_stmt.body, p, c, os, ns);
        break;
    case NODE_FOR:
        new_node->for_stmt.init = copy_ast_replacing(n->for_stmt.init, p, c, os, ns);
        new_node->for_stmt.condition = copy_ast_replacing(n->for_stmt.condition, p, c, os, ns);
        new_node->for_stmt.step = copy_ast_replacing(n->for_stmt.step, p, c, os, ns);
        new_node->for_stmt.body = copy_ast_replacing(n->for_stmt.body, p, c, os, ns);
        break;
    case NODE_FOR_RANGE:
        new_node->for_range.start = copy_ast_replacing(n->for_range.start, p, c, os, ns);
        new_node->for_range.end = copy_ast_replacing(n->for_range.end, p, c, os, ns);
        new_node->for_range.body = copy_ast_replacing(n->for_range.body, p, c, os, ns);
        break;

    case NODE_MATCH_CASE:
        if (n->match_case.pattern)
        {
            char *s1 = xstrdup(n->match_case.pattern);
            if (p && c && strchr(p, ','))
            {
                char *p_ptr = (char *)p;
                char *c_ptr = (char *)c;
                while (*p_ptr && *c_ptr)
                {
                    char *p_end = strchr(p_ptr, ',');
                    int p_len = p_end ? (int)(p_end - p_ptr) : (int)strlen(p_ptr);
                    char *c_end = strchr(c_ptr, ',');
                    int c_len = c_end ? (int)(c_end - c_ptr) : (int)strlen(c_ptr);

                    char *p_part = xmalloc(p_len + 1);
                    strncpy(p_part, p_ptr, p_len);
                    p_part[p_len] = 0;

                    char *c_part = xmalloc(c_len + 1);
                    strncpy(c_part, c_ptr, c_len);
                    c_part[c_len] = 0;

                    char *t1 = replace_mangled_part(s1, p_part, c_part);
                    free(s1);
                    s1 = t1;

                    free(p_part);
                    free(c_part);

                    if (p_end)
                    {
                        p_ptr = p_end + 1;
                    }
                    else
                    {
                        break;
                    }
                    if (c_end)
                    {
                        c_ptr = c_end + 1;
                    }
                    else
                    {
                        break;
                    }
                }
            }
            else
            {
                char *t1 = replace_in_string(s1, p, c);
                free(s1);
                s1 = t1;
                char *t2 = replace_mangled_part(s1, p, c);
                free(s1);
                s1 = t2;
            }

            if (os && ns)
            {
                char *s2 = replace_in_string(s1, os, ns);
                free(s1);
                s1 = s2;
                char *colons = strstr(s1, "::");
                if (colons)
                {
                    colons[0] = '_';
                    memmove(colons + 1, colons + 2, strlen(colons + 2) + 1);
                }
            }
            new_node->match_case.pattern = s1;
        }
        new_node->match_case.binding_count = n->match_case.binding_count;
        if (n->match_case.binding_names)
        {
            new_node->match_case.binding_names =
                xmalloc(sizeof(char *) * n->match_case.binding_count);
            for (int i = 0; i < n->match_case.binding_count; i++)
            {
                if (n->match_case.binding_names[i])
                {
                    new_node->match_case.binding_names[i] = xstrdup(n->match_case.binding_names[i]);
                }
                else
                {
                    new_node->match_case.binding_names[i] = NULL;
                }
            }
        }
        if (n->match_case.binding_refs)
        {
            new_node->match_case.binding_refs = xmalloc(sizeof(int) * n->match_case.binding_count);
            memcpy(new_node->match_case.binding_refs, n->match_case.binding_refs,
                   sizeof(int) * n->match_case.binding_count);
        }
        new_node->match_case.is_default = n->match_case.is_default;
        new_node->match_case.is_destructuring = n->match_case.is_destructuring;

        new_node->match_case.body = copy_ast_replacing(n->match_case.body, p, c, os, ns);
        if (n->match_case.guard)
        {
            new_node->match_case.guard = copy_ast_replacing(n->match_case.guard, p, c, os, ns);
        }
        break;

    case NODE_IMPL:
        new_node->impl.struct_name = replace_type_str(n->impl.struct_name, p, c, os, ns);
        new_node->impl.methods = copy_ast_replacing(n->impl.methods, p, c, os, ns);
        break;
    case NODE_IMPL_TRAIT:
        new_node->impl_trait.trait_name =
            n->impl_trait.trait_name ? xstrdup(n->impl_trait.trait_name) : NULL;
        new_node->impl_trait.target_type =
            replace_type_str(n->impl_trait.target_type, p, c, os, ns);
        new_node->impl_trait.methods = copy_ast_replacing(n->impl_trait.methods, p, c, os, ns);
        break;
    case NODE_TYPEOF:
    case NODE_EXPR_SIZEOF:
        new_node->size_of.target_type = replace_type_str(n->size_of.target_type, p, c, os, ns);
        new_node->size_of.expr = copy_ast_replacing(n->size_of.expr, p, c, os, ns);
        new_node->size_of.is_type = n->size_of.is_type;
        if (n->size_of.target_type_info)
        {
            new_node->size_of.target_type_info =
                replace_type_formal(n->size_of.target_type_info, p, c, os, ns);
        }
        break;
    case NODE_LAMBDA:
        // Use a new lambda ID for each instantiation to ensure unique C function names
        new_node->lambda.lambda_id = g_parser_ctx->lambda_counter++;
        new_node->lambda.num_params = n->lambda.num_params;
        if (n->lambda.num_params > 0)
        {
            new_node->lambda.param_names = xmalloc(sizeof(char *) * n->lambda.num_params);
            new_node->lambda.param_types = xmalloc(sizeof(char *) * n->lambda.num_params);
            for (int i = 0; i < n->lambda.num_params; i++)
            {
                new_node->lambda.param_names[i] = xstrdup(n->lambda.param_names[i]);
                new_node->lambda.param_types[i] =
                    replace_type_str(n->lambda.param_types[i], p, c, os, ns);
            }
        }
        new_node->lambda.return_type = replace_type_str(n->lambda.return_type, p, c, os, ns);
        new_node->lambda.num_captures = n->lambda.num_captures;
        if (n->lambda.num_captures > 0)
        {
            new_node->lambda.captured_vars = xmalloc(sizeof(char *) * n->lambda.num_captures);
            new_node->lambda.captured_types = xmalloc(sizeof(char *) * n->lambda.num_captures);
            new_node->lambda.captured_types_info = xmalloc(sizeof(Type *) * n->lambda.num_captures);
            if (n->lambda.capture_modes)
            {
                new_node->lambda.capture_modes = xmalloc(sizeof(int) * n->lambda.num_captures);
            }

            for (int i = 0; i < n->lambda.num_captures; i++)
            {
                new_node->lambda.captured_vars[i] = xstrdup(n->lambda.captured_vars[i]);
                new_node->lambda.captured_types[i] =
                    replace_type_str(n->lambda.captured_types[i], p, c, os, ns);
                new_node->lambda.captured_types_info[i] =
                    replace_type_formal(n->lambda.captured_types_info[i], p, c, os, ns);
                if (n->lambda.capture_modes)
                {
                    new_node->lambda.capture_modes[i] = n->lambda.capture_modes[i];
                }
            }
        }
        new_node->lambda.body = copy_ast_replacing(n->lambda.body, p, c, os, ns);
        new_node->lambda.is_bare = n->lambda.is_bare;
        register_lambda(g_parser_ctx, new_node);
        break;
    case NODE_DESTRUCT_VAR:
        if (n->destruct.count > 0)
        {
            new_node->destruct.names = xmalloc(sizeof(char *) * n->destruct.count);
            new_node->destruct.types = xmalloc(sizeof(char *) * n->destruct.count);
            new_node->destruct.type_infos = xmalloc(sizeof(Type *) * n->destruct.count);
            if (n->destruct.field_names)
            {
                new_node->destruct.field_names = xmalloc(sizeof(char *) * n->destruct.count);
            }

            for (int i = 0; i < n->destruct.count; i++)
            {
                new_node->destruct.names[i] = xstrdup(n->destruct.names[i]);
                new_node->destruct.types[i] = replace_type_str(n->destruct.types[i], p, c, os, ns);
                new_node->destruct.type_infos[i] =
                    replace_type_formal(n->destruct.type_infos[i], p, c, os, ns);
                if (n->destruct.field_names && n->destruct.field_names[i])
                {
                    new_node->destruct.field_names[i] = xstrdup(n->destruct.field_names[i]);
                }
                else if (n->destruct.field_names)
                {
                    new_node->destruct.field_names[i] = NULL;
                }
            }
        }
        new_node->destruct.init_expr = copy_ast_replacing(n->destruct.init_expr, p, c, os, ns);
        new_node->destruct.struct_name = replace_type_str(n->destruct.struct_name, p, c, os, ns);
        new_node->destruct.else_block = copy_ast_replacing(n->destruct.else_block, p, c, os, ns);
        break;
    case NODE_MATCH:
        new_node->match_stmt.expr = copy_ast_replacing(n->match_stmt.expr, p, c, os, ns);
        new_node->match_stmt.cases = copy_ast_replacing(n->match_stmt.cases, p, c, os, ns);
        break;
    case NODE_LOOP:
        new_node->loop_stmt.body = copy_ast_replacing(n->loop_stmt.body, p, c, os, ns);
        break;
    case NODE_REPEAT:
        new_node->repeat_stmt.count = n->repeat_stmt.count ? xstrdup(n->repeat_stmt.count) : NULL;
        new_node->repeat_stmt.body = copy_ast_replacing(n->repeat_stmt.body, p, c, os, ns);
        break;
    case NODE_UNLESS:
        new_node->unless_stmt.condition =
            copy_ast_replacing(n->unless_stmt.condition, p, c, os, ns);
        new_node->unless_stmt.body = copy_ast_replacing(n->unless_stmt.body, p, c, os, ns);
        break;
    case NODE_GUARD:
        new_node->guard_stmt.condition = copy_ast_replacing(n->guard_stmt.condition, p, c, os, ns);
        new_node->guard_stmt.body = copy_ast_replacing(n->guard_stmt.body, p, c, os, ns);
        break;
    case NODE_BREAK:
    case NODE_CONTINUE:
        // No members to copy besides next (handled at end)
        break;
    case NODE_EXPR_ARRAY_LITERAL:
        new_node->array_literal.elements =
            copy_ast_replacing(n->array_literal.elements, p, c, os, ns);
        new_node->array_literal.count = n->array_literal.count;
        break;
    case NODE_EXPR_TUPLE_LITERAL:
        new_node->tuple_literal.elements =
            copy_ast_replacing(n->tuple_literal.elements, p, c, os, ns);
        new_node->tuple_literal.count = n->tuple_literal.count;
        break;
    case NODE_EXPR_SLICE:
        new_node->slice.array = copy_ast_replacing(n->slice.array, p, c, os, ns);
        new_node->slice.start = copy_ast_replacing(n->slice.start, p, c, os, ns);
        new_node->slice.end = copy_ast_replacing(n->slice.end, p, c, os, ns);
        break;
    case NODE_ASSERT:
        new_node->assert_stmt.condition =
            copy_ast_replacing(n->assert_stmt.condition, p, c, os, ns);
        new_node->assert_stmt.message =
            n->assert_stmt.message ? xstrdup(n->assert_stmt.message) : NULL;
        break;
    case NODE_DEFER:
        new_node->defer_stmt.stmt = copy_ast_replacing(n->defer_stmt.stmt, p, c, os, ns);
        break;
    case NODE_TERNARY:
        new_node->ternary.cond = copy_ast_replacing(n->ternary.cond, p, c, os, ns);
        new_node->ternary.true_expr = copy_ast_replacing(n->ternary.true_expr, p, c, os, ns);
        new_node->ternary.false_expr = copy_ast_replacing(n->ternary.false_expr, p, c, os, ns);
        break;
    case NODE_ASM:
        new_node->asm_stmt.code = n->asm_stmt.code ? xstrdup(n->asm_stmt.code) : NULL;
        new_node->asm_stmt.is_volatile = n->asm_stmt.is_volatile;
        new_node->asm_stmt.num_outputs = n->asm_stmt.num_outputs;
        new_node->asm_stmt.num_inputs = n->asm_stmt.num_inputs;
        new_node->asm_stmt.num_clobbers = n->asm_stmt.num_clobbers;
        // ASM usually doesn't contain generic parameters in constraints, but we could harden here
        // if needed
        break;
    case NODE_GOTO:
        new_node->goto_stmt.label_name =
            n->goto_stmt.label_name ? xstrdup(n->goto_stmt.label_name) : NULL;
        break;
    case NODE_LABEL:
        new_node->label_stmt.label_name =
            n->label_stmt.label_name ? xstrdup(n->label_stmt.label_name) : NULL;
        break;
    case NODE_DO_WHILE:
        new_node->while_stmt.condition = copy_ast_replacing(n->while_stmt.condition, p, c, os, ns);
        new_node->while_stmt.body = copy_ast_replacing(n->while_stmt.body, p, c, os, ns);
        break;
    case NODE_TRY:
        new_node->try_stmt.expr = copy_ast_replacing(n->try_stmt.expr, p, c, os, ns);
        break;
    case NODE_REFLECTION:
        new_node->reflection.kind = n->reflection.kind;
        new_node->reflection.target_type =
            replace_type_formal(n->reflection.target_type, p, c, os, ns);
        break;
    case NODE_REPL_PRINT:
        new_node->repl_print.expr = copy_ast_replacing(n->repl_print.expr, p, c, os, ns);
        break;
    case NODE_CUDA_LAUNCH:
        new_node->cuda_launch.call = copy_ast_replacing(n->cuda_launch.call, p, c, os, ns);
        new_node->cuda_launch.grid = copy_ast_replacing(n->cuda_launch.grid, p, c, os, ns);
        new_node->cuda_launch.block = copy_ast_replacing(n->cuda_launch.block, p, c, os, ns);
        new_node->cuda_launch.shared_mem =
            copy_ast_replacing(n->cuda_launch.shared_mem, p, c, os, ns);
        new_node->cuda_launch.stream = copy_ast_replacing(n->cuda_launch.stream, p, c, os, ns);
        break;
    case NODE_VA_START:
        new_node->va_start.ap = copy_ast_replacing(n->va_start.ap, p, c, os, ns);
        new_node->va_start.last_arg = copy_ast_replacing(n->va_start.last_arg, p, c, os, ns);
        break;
    case NODE_VA_END:
        new_node->va_end.ap = copy_ast_replacing(n->va_end.ap, p, c, os, ns);
        break;
    case NODE_VA_COPY:
        new_node->va_copy.dest = copy_ast_replacing(n->va_copy.dest, p, c, os, ns);
        new_node->va_copy.src = copy_ast_replacing(n->va_copy.src, p, c, os, ns);
        break;
    case NODE_VA_ARG:
        new_node->va_arg.ap = copy_ast_replacing(n->va_arg.ap, p, c, os, ns);
        new_node->va_arg.type_info = replace_type_formal(n->va_arg.type_info, p, c, os, ns);
        break;
    default:
        break;
    }
    return new_node;
}

// Helper to sanitize type names for mangling (e.g. "int*" -> "intPtr")
char *sanitize_mangled_name(const char *s)
{
    char *buf = xmalloc(strlen(s) * 4 + 1);

    // Skip "struct " prefix if present to avoid "struct_" in mangled names
    if (strncmp(s, "struct ", 7) == 0)
    {
        s += 7;
    }

    char *p = buf;
    while (*s)
    {
        if (*s == '*')
        {
            strcpy(p, "Ptr");
            p += 3;
        }
        else if (*s == '<' || *s == ',' || *s == ' ')
        {
            *p++ = '_';
            *p++ = '_';
        }
        else if (*s == '>' || *s == '&')
        {
            // Skip > and & (often used in references) to keep names clean
        }
        else if ((*s >= 'a' && *s <= 'z') || (*s >= 'A' && *s <= 'Z') || (*s >= '0' && *s <= '9') ||
                 *s == '_')
        {
            *p++ = *s;
        }
        else
        {
            *p++ = '_';
        }
        s++;
    }
    *p = 0;
    return buf;
}

// Helper to unmangle Ptr suffix back to pointer type ("intPtr" -> "int*")
char *unmangle_ptr_suffix(const char *s)
{
    if (!s)
    {
        return NULL;
    }

    size_t len = strlen(s);
    if (len <= 3 || strcmp(s + len - 3, "Ptr") != 0)
    {
        return xstrdup(s); // No Ptr suffix, return as-is
    }

    // Extract base type (everything before "Ptr")
    char *base = xmalloc(len - 2);
    strncpy(base, s, len - 3);
    base[len - 3] = '\0';

    char *result = xmalloc(strlen(base) + 16);

    // Check if base is a primitive type
    if (is_primitive_type_name(base))
    {
        sprintf(result, "%s*", base);
    }
    else
    {
        // Don't unmangle non-primitives ending in Ptr (like Vec_intPtr)
        strcpy(result, s);
    }

    free(base);
    return result;
}

FuncSig *find_func(ParserContext *ctx, const char *name)
{
    ZenSymbol *sym = symbol_lookup_kind(ctx->current_scope, name, SYM_FUNCTION);
    if (sym)
    {
        return sym->data.sig;
    }

    FuncSig *c = ctx->func_registry;
    while (c)
    {
        if (strcmp(c->name, name) == 0)
        {
            return c;
        }
        c = c->next;
    }

    // Fallback: Check current_impl_methods (siblings in the same impl block)
    if (ctx && ctx->current_impl_methods)
    {
        ASTNode *n = ctx->current_impl_methods;
        while (n)
        {
            if (n->type == NODE_FUNCTION && strcmp(n->func.name, name) == 0)
            {
                // Found sibling method. Construct a temporary FuncSig.
                FuncSig *sig = xmalloc(sizeof(FuncSig));
                sig->name = n->func.name;
                sig->decl_token = n->token;
                sig->total_args = n->func.arg_count;
                sig->defaults = n->func.defaults;
                sig->arg_types = n->func.arg_types;
                sig->ret_type = n->func.ret_type_info;
                sig->is_varargs = n->func.is_varargs;
                sig->is_async = n->func.is_async;
                sig->required = 0;
                sig->next = NULL;
                return sig;
            }
            n = n->next;
        }
    }

    return NULL;
}

// Helper function to recursively scan AST for sizeof types AND generic calls to trigger
// instantiation
static void trigger_type_instantiation(ParserContext *ctx, Type *t)
{
    if (!t)
    {
        return;
    }

    // Handle slices
    if (t->kind == TYPE_ARRAY && t->array_size == 0 && t->inner)
    {
        char *inner_str = type_to_string(t->inner);
        register_slice(ctx, inner_str);
        free(inner_str);
    }

    // Handle mangled types (instantiations)
    if (t->name && strchr(t->name, '_'))
    {
        char *type_copy = xstrdup(t->name);
        char *underscore = strchr(type_copy, '_');
        if (underscore)
        {
            char *concrete_arg = underscore;
            while (*concrete_arg == '_')
            {
                concrete_arg++;
            }
            *underscore = '\0';
            char *template_name = type_copy;

            GenericTemplate *gt = ctx->templates;
            int found = 0;
            while (gt)
            {
                if (strcmp(gt->name, template_name) == 0)
                {
                    found = 1;
                    break;
                }
                gt = gt->next;
            }

            if (found)
            {
                char *unmangled = unmangle_ptr_suffix(concrete_arg);
                Token dummy_tok = {0};
                instantiate_generic(ctx, template_name, concrete_arg, unmangled, dummy_tok);
                free(unmangled);
            }
        }
        free(type_copy);
    }

    // Recursive scan
    trigger_type_instantiation(ctx, t->inner);
    if (t->args)
    {
        for (int i = 0; i < t->arg_count; i++)
        {
            trigger_type_instantiation(ctx, t->args[i]);
        }
    }
}

static void trigger_instantiations(ParserContext *ctx, ASTNode *node)
{
    if (!node)
    {
        return;
    }

    // Process type information
    if (node->type_info)
    {
        trigger_type_instantiation(ctx, node->type_info);
    }

    // Process current node
    if (node->type == NODE_EXPR_SIZEOF && node->size_of.target_type)
    {
        const char *type_str = node->size_of.target_type;
        if (strchr(type_str, '_'))
        {
            // Remove trailing '*' or 'Ptr' if present
            char *type_copy = xstrdup(type_str);
            char *star = strchr(type_copy, '*');
            if (star)
            {
                *star = '\0';
            }
            else
            {
                // Check for "Ptr" suffix and remove it
                size_t len = strlen(type_copy);
                if (len > 3 && strcmp(type_copy + len - 3, "Ptr") == 0)
                {
                    type_copy[len - 3] = '\0';
                }
            }

            char *underscore = strchr(type_copy, '_');
            if (underscore)
            {
                char *concrete_arg = underscore;
                while (*concrete_arg == '_')
                {
                    concrete_arg++;
                }
                *underscore = '\0';
                char *template_name = type_copy;

                // Check if this is a known generic template
                GenericTemplate *gt = ctx->templates;
                int found = 0;
                while (gt)
                {
                    if (strcmp(gt->name, template_name) == 0)
                    {
                        found = 1;
                        break;
                    }
                    gt = gt->next;
                }

                if (found)
                {
                    char *unmangled = unmangle_ptr_suffix(concrete_arg);
                    Token dummy_tok = {0};
                    instantiate_generic(ctx, template_name, concrete_arg, unmangled, dummy_tok);
                    free(unmangled);
                }
            }
            free(type_copy);
        }
    }
    else if (node->type == NODE_EXPR_VAR)
    {
        const char *name = node->var_ref.name;
        if (strchr(name, '_'))
        {
            GenericFuncTemplate *t = ctx->func_templates;
            while (t)
            {
                size_t tlen = strlen(t->name);
                if (strncmp(name, t->name, tlen) == 0 && name[tlen] == '_' && name[tlen + 1] == '_')
                {
                    char *template_name = t->name;
                    char *concrete_arg = (char *)name + tlen + 2;

                    char *unmangled = unmangle_ptr_suffix(concrete_arg);
                    instantiate_function_template(ctx, template_name, concrete_arg, unmangled);
                    free(unmangled);
                    break; // Found match, stop searching
                }
                t = t->next;
            }
        }
    }
    else if (node->type == NODE_EXPR_STRUCT_INIT && node->struct_init.struct_name)
    {
        const char *name = node->struct_init.struct_name;
        if (strchr(name, '_'))
        {
            char *type_copy = xstrdup(name);
            char *underscore = strchr(type_copy, '_');
            if (underscore)
            {
                char *concrete_arg = underscore;
                while (*concrete_arg == '_')
                {
                    concrete_arg++;
                }
                *underscore = '\0';
                char *template_name = type_copy;

                GenericTemplate *gt = ctx->templates;
                int found = 0;
                while (gt)
                {
                    if (strcmp(gt->name, template_name) == 0)
                    {
                        found = 1;
                        break;
                    }
                    gt = gt->next;
                }

                if (found)
                {
                    char *unmangled = unmangle_ptr_suffix(concrete_arg);
                    Token dummy_tok = {0};
                    instantiate_generic(ctx, template_name, concrete_arg, unmangled, dummy_tok);
                    free(unmangled);
                }
            }
            free(type_copy);
        }
    }

    switch (node->type)
    {
    case NODE_FUNCTION:
        trigger_instantiations(ctx, node->func.body);
        break;
    case NODE_BLOCK:
        trigger_instantiations(ctx, node->block.statements);
        break;
    case NODE_VAR_DECL:
        trigger_instantiations(ctx, node->var_decl.init_expr);
        break;
    case NODE_RETURN:
        trigger_instantiations(ctx, node->ret.value);
        break;
    case NODE_EXPR_BINARY:
        trigger_instantiations(ctx, node->binary.left);
        trigger_instantiations(ctx, node->binary.right);
        break;
    case NODE_EXPR_UNARY:
        trigger_instantiations(ctx, node->unary.operand);
        break;
    case NODE_EXPR_CALL:
        trigger_instantiations(ctx, node->call.callee);
        trigger_instantiations(ctx, node->call.args);
        break;
    case NODE_EXPR_MEMBER:
        trigger_instantiations(ctx, node->member.target);
        break;
    case NODE_EXPR_INDEX:
        trigger_instantiations(ctx, node->index.array);
        trigger_instantiations(ctx, node->index.index);
        break;
    case NODE_EXPR_CAST:
        trigger_instantiations(ctx, node->cast.expr);
        break;
    case NODE_IF:
        trigger_instantiations(ctx, node->if_stmt.condition);
        trigger_instantiations(ctx, node->if_stmt.then_body);
        trigger_instantiations(ctx, node->if_stmt.else_body);
        break;
    case NODE_WHILE:
        trigger_instantiations(ctx, node->while_stmt.condition);
        trigger_instantiations(ctx, node->while_stmt.body);
        break;
    case NODE_FOR:
        trigger_instantiations(ctx, node->for_stmt.init);
        trigger_instantiations(ctx, node->for_stmt.condition);
        trigger_instantiations(ctx, node->for_stmt.step);
        trigger_instantiations(ctx, node->for_stmt.body);
        break;
    case NODE_FOR_RANGE:
        trigger_instantiations(ctx, node->for_range.start);
        trigger_instantiations(ctx, node->for_range.end);
        trigger_instantiations(ctx, node->for_range.body);
        break;
    case NODE_EXPR_STRUCT_INIT:
        trigger_instantiations(ctx, node->struct_init.fields);
        break;
    case NODE_MATCH:
        trigger_instantiations(ctx, node->match_stmt.expr);
        trigger_instantiations(ctx, node->match_stmt.cases);
        break;
    case NODE_MATCH_CASE:
        trigger_instantiations(ctx, node->match_case.guard);
        trigger_instantiations(ctx, node->match_case.body);
        break;
    case NODE_ASSERT:
        trigger_instantiations(ctx, node->assert_stmt.condition);
        break;
    case NODE_DEFER:
        trigger_instantiations(ctx, node->defer_stmt.stmt);
        break;
    case NODE_UNLESS:
        trigger_instantiations(ctx, node->unless_stmt.condition);
        trigger_instantiations(ctx, node->unless_stmt.body);
        break;
    case NODE_GUARD:
        trigger_instantiations(ctx, node->guard_stmt.condition);
        trigger_instantiations(ctx, node->guard_stmt.body);
        break;
    case NODE_LOOP:
        trigger_instantiations(ctx, node->loop_stmt.body);
        break;
    case NODE_REPEAT:
        trigger_instantiations(ctx, node->repeat_stmt.body);
        break;
    case NODE_DO_WHILE:
        trigger_instantiations(ctx, node->while_stmt.condition);
        trigger_instantiations(ctx, node->while_stmt.body);
        break;
    case NODE_TERNARY:
        trigger_instantiations(ctx, node->ternary.cond);
        trigger_instantiations(ctx, node->ternary.true_expr);
        trigger_instantiations(ctx, node->ternary.false_expr);
        break;
    case NODE_EXPR_ARRAY_LITERAL:
        trigger_instantiations(ctx, node->array_literal.elements);
        break;
    case NODE_EXPR_TUPLE_LITERAL:
        trigger_instantiations(ctx, node->tuple_literal.elements);
        break;
    case NODE_EXPR_SLICE:
        trigger_instantiations(ctx, node->slice.array);
        trigger_instantiations(ctx, node->slice.start);
        trigger_instantiations(ctx, node->slice.end);
        break;
    case NODE_DESTRUCT_VAR:
        trigger_instantiations(ctx, node->destruct.init_expr);
        trigger_instantiations(ctx, node->destruct.else_block);
        break;
    case NODE_LAMBDA:
        trigger_instantiations(ctx, node->lambda.body);
        break;
    case NODE_TRY:
        trigger_instantiations(ctx, node->try_stmt.expr);
        break;
    case NODE_CUDA_LAUNCH:
        trigger_instantiations(ctx, node->cuda_launch.call);
        trigger_instantiations(ctx, node->cuda_launch.grid);
        trigger_instantiations(ctx, node->cuda_launch.block);
        trigger_instantiations(ctx, node->cuda_launch.shared_mem);
        trigger_instantiations(ctx, node->cuda_launch.stream);
        break;
    case NODE_VA_START:
        trigger_instantiations(ctx, node->va_start.ap);
        trigger_instantiations(ctx, node->va_start.last_arg);
        break;
    case NODE_VA_END:
        trigger_instantiations(ctx, node->va_end.ap);
        break;
    case NODE_VA_COPY:
        trigger_instantiations(ctx, node->va_copy.dest);
        trigger_instantiations(ctx, node->va_copy.src);
        break;
    case NODE_VA_ARG:
        trigger_instantiations(ctx, node->va_arg.ap);
        break;
    default:
        break;
    }

    // Visit next sibling
    trigger_instantiations(ctx, node->next);
}

char *instantiate_function_template(ParserContext *ctx, const char *name, const char *concrete_type,
                                    const char *unmangled_type)
{
    GenericFuncTemplate *tpl = find_func_template(ctx, name);
    if (!tpl)
    {
        return NULL;
    }

    char *clean_type = sanitize_mangled_name(concrete_type);

    int is_still_generic = 0;
    if (strlen(clean_type) == 1 && isupper(clean_type[0]))
    {
        is_still_generic = 1;
    }

    if (is_known_generic(ctx, clean_type))
    {
        is_still_generic = 1;
    }

    char buf[MAX_ERROR_MSG_LEN];
    snprintf(buf, sizeof(buf), "%s__%s", name, clean_type);
    char *mangled = merge_underscores(buf);
    free(clean_type);

    if (is_still_generic)
    {
        return mangled;
    }

    if (find_func(ctx, mangled))
    {
        return mangled;
    }

    const char *subst_arg = unmangled_type ? unmangled_type : concrete_type;

    // Scan the original return type for generic struct patterns like "Triple_X_Y_Z"
    // and instantiate them with the concrete types
    if (tpl->func_node && tpl->func_node->func.ret_type)
    {
        const char *ret = tpl->func_node->func.ret_type;

        // Build the param suffix (e.g., for "X,Y,Z" -> "_X_Y_Z")
        size_t suffix_cap = strlen(tpl->generic_param) * 2 + 64;
        char *param_suffix = xmalloc(suffix_cap);
        param_suffix[0] = 0;
        const char *p_ptr = tpl->generic_param;
        while (p_ptr && *p_ptr)
        {
            strcat(param_suffix, "__");
            const char *p_next = strchr(p_ptr, ',');
            int sub_len = p_next ? (int)(p_next - p_ptr) : (int)strlen(p_ptr);
            strncat(param_suffix, p_ptr, sub_len);
            if (p_next)
            {
                p_ptr = p_next + 1;
            }
            else
            {
                break;
            }
        }

        // Check if ret_type ends with param_suffix (e.g., "Triple_X_Y_Z" ends with "_X_Y_Z")
        size_t ret_len = strlen(ret);
        size_t suffix_len = strlen(param_suffix);
        if (ret_len > suffix_len && strcmp(ret + ret_len - suffix_len, param_suffix) == 0)
        {
            // Extract base struct name (e.g., "Triple" from "Triple_X_Y_Z")
            size_t base_len = ret_len - suffix_len;
            char *struct_base = xmalloc(base_len + 1);
            strncpy(struct_base, ret, base_len);
            struct_base[base_len] = 0;

            // Check if it's a known generic template
            GenericTemplate *gt = ctx->templates;
            while (gt && strcmp(gt->name, struct_base) != 0)
            {
                gt = gt->next;
            }
            if (gt)
            {
                // Parse the concrete types from unmangled_type or concrete_type
                const char *types_src = unmangled_type ? unmangled_type : concrete_type;

                // Count params in template
                int template_param_count = 1;
                for (const char *p = tpl->generic_param; *p; p++)
                {
                    if (*p == ',')
                    {
                        template_param_count++;
                    }
                }

                // Split concrete types
                char **args = xmalloc(sizeof(char *) * template_param_count);
                int arg_count = 0;
                const char *types_ptr = types_src;
                while (types_ptr && *types_ptr && arg_count < template_param_count)
                {
                    const char *types_next = strchr(types_ptr, ',');
                    int types_len =
                        types_next ? (int)(types_next - types_ptr) : (int)strlen(types_ptr);

                    args[arg_count] = xmalloc(types_len + 1);
                    strncpy(args[arg_count], types_ptr, types_len);
                    args[arg_count][types_len] = 0;
                    arg_count++;

                    if (types_next)
                    {
                        types_ptr = types_next + 1;
                    }
                    else
                    {
                        break;
                    }
                }

                // Now instantiate the struct with these args
                Token dummy_tok = {0};
                if (arg_count == 1)
                {
                    // Unmangle Ptr suffix if needed (e.g., intPtr -> int*)
                    char *unmangled = xstrdup(args[0]);
                    size_t alen = strlen(args[0]);
                    if (alen > 3 && strcmp(args[0] + alen - 3, "Ptr") == 0)
                    {
                        char *base = xstrdup(args[0]);
                        base[alen - 3] = '\0';
                        free(unmangled);
                        unmangled = xmalloc(strlen(base) + 16);
                        if (is_unmangle_primitive(base))
                        {
                            sprintf(unmangled, "%s*", base);
                        }
                        else
                        {
                            sprintf(unmangled, "struct %s*", base);
                        }
                        free(base);
                    }
                    instantiate_generic(ctx, struct_base, args[0], unmangled, dummy_tok);
                    free(unmangled);
                }
                else if (arg_count > 1)
                {
                    instantiate_generic_multi(ctx, struct_base, args, arg_count, dummy_tok);
                }

                // Cleanup
                for (int i = 0; i < arg_count; i++)
                {
                    free(args[i]);
                }
                free(args);
            }
            free(struct_base);
        }
        free(param_suffix);
    }

    ASTNode *new_fn = copy_ast_replacing(tpl->func_node, tpl->generic_param, subst_arg, NULL, NULL);
    if (!new_fn || new_fn->type != NODE_FUNCTION)
    {
        return NULL;
    }

    free(new_fn->func.name);
    new_fn->func.name = xstrdup(mangled);
    new_fn->func.generic_params = NULL;

    add_instantiated_func(ctx, new_fn);

    register_func(ctx, ctx->global_scope, mangled, new_fn->func.arg_count, new_fn->func.defaults,
                  new_fn->func.arg_types, new_fn->func.ret_type_info, new_fn->func.is_varargs, 0,
                  new_fn->func.pure, new_fn->link_name, new_fn->token, new_fn->func.is_export);

    trigger_instantiations(ctx, new_fn->func.body);

    if (new_fn->func.arg_types)
    {
        for (int i = 0; i < new_fn->func.arg_count; i++)
        {
            Type *at = new_fn->func.arg_types[i];
            if (at && at->kind == TYPE_ARRAY && at->array_size == 0 && at->inner)
            {
                char *inner_str = type_to_string(at->inner);
                register_slice(ctx, inner_str);
                free(inner_str);
            }
        }
    }

    return mangled;
}

char *process_fstring(ParserContext *ctx, const char *content, char ***used_syms, int *count)
{
    (void)used_syms;
    (void)count;
    char *gen = xmalloc(8192); // Increased buffer size

    strcpy(gen, "({ static char _b[4096]; _b[0]=0; char _t[1024]; ");

    char *s = xstrdup(content);
    char *cur = s;

    while (*cur)
    {
        char *brace = cur;
        while (*brace && *brace != '{')
        {
            brace++;
        }

        if (brace > cur)
        {
            char tmp = *brace;
            *brace = 0;
            strcat(gen, "strcat(_b, \"");
            strcat(gen, cur);
            strcat(gen, "\"); ");
            *brace = tmp;
        }

        if (*brace == 0)
        {
            break;
        }

        char *p = brace + 1;
        char *colon = NULL;
        int depth = 1;

        while (*p && depth > 0)
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
                colon = p;
            }
            if (depth == 0)
            {
                break;
            }
            p++;
        }

        *p = 0;
        char *expr_str = brace + 1;
        char *fmt = NULL;
        if (colon)
        {
            *colon = 0;
            fmt = colon + 1;
        }

        // Parse expression fully to handle default arguments etc.
        Lexer expr_lex;
        lexer_init(&expr_lex, expr_str);
        ASTNode *expr_node = parse_expression(ctx, &expr_lex);

        // Codegen expression to temporary buffer
        char *code_buffer = NULL;
        {
            Emitter saved = ctx->emitter;
            emitter_init_buffer(&ctx->emitter);
            codegen_expression(ctx, expr_node);
            code_buffer = emitter_take_string(&ctx->emitter);
            ctx->emitter = saved;
        }

        if (fmt)
        {
            strcat(gen, "sprintf(_t, \"%");
            strcat(gen, fmt);
            strcat(gen, "\", ");
            if (code_buffer)
            {
                strcat(gen, code_buffer);
            }
            else
            {
                strcat(gen, expr_str); // Fallback
            }
            strcat(gen, "); strcat(_b, _t); ");
        }
        else
        {
            strcat(gen, "sprintf(_t, _z_str(");
            if (code_buffer)
            {
                strcat(gen, code_buffer);
            }
            else
            {
                strcat(gen, expr_str);
            }
            strcat(gen, "), ");
            if (code_buffer)
            {
                strcat(gen, code_buffer);
            }
            else
            {
                strcat(gen, expr_str);
            }
            strcat(gen, "); strcat(_b, _t); ");
        }

        if (code_buffer)
        {
            free(code_buffer);
        }

        cur = p + 1;
    }

    strcat(gen, "_b; })");
    free(s);
    return gen;
}

void register_impl(ParserContext *ctx, const char *trait, const char *strct)
{
    ImplReg *r = xmalloc(sizeof(ImplReg));
    r->trait = xstrdup(trait);
    r->strct = xstrdup(strct);
    r->next = ctx->registered_impls;
    ctx->registered_impls = r;
}

int check_impl(ParserContext *ctx, const char *trait, const char *strct)
{
    ImplReg *r = ctx->registered_impls;
    while (r)
    {
        if (strcmp(r->trait, trait) == 0 && strcmp(r->strct, strct) == 0)
        {
            return 1;
        }
        r = r->next;
    }

    r = ctx->registered_impls;
    while (r)
    {
        char *base_reg = xstrdup(r->strct);
        char *ptr2 = strchr(base_reg, '<');
        if (ptr2)
        {
            *ptr2 = 0;
            size_t blen = strlen(base_reg);
            if (strncmp(strct, base_reg, blen) == 0 && strct[blen] == '_')
            {
                if (strcmp(r->trait, trait) == 0)
                {
                    free(base_reg);
                    return 1;
                }
            }
        }
        free(base_reg);
        r = r->next;
    }

    return 0;
}

static int is_unmangle_primitive(const char *base)
{
    return (strcmp(base, "int") == 0 || strcmp(base, "uint") == 0 || strcmp(base, "char") == 0 ||
            strcmp(base, "bool") == 0 || strcmp(base, "void") == 0 || strcmp(base, "byte") == 0 ||
            strcmp(base, "rune") == 0 || strcmp(base, "float") == 0 ||
            strcmp(base, "double") == 0 || strcmp(base, "f32") == 0 || strcmp(base, "f64") == 0 ||
            strcmp(base, "size_t") == 0 || strcmp(base, "usize") == 0 ||
            strcmp(base, "isize") == 0 || strcmp(base, "ptrdiff_t") == 0 ||
            strncmp(base, "i8", 2) == 0 || strncmp(base, "u8", 2) == 0 ||
            strncmp(base, "int8", 4) == 0 || strncmp(base, "int16", 5) == 0 ||
            strncmp(base, "int32", 5) == 0 || strncmp(base, "int64", 5) == 0 ||
            strncmp(base, "uint8", 5) == 0 || strncmp(base, "uint16", 6) == 0 ||
            strncmp(base, "uint32", 6) == 0 || strncmp(base, "uint64", 6) == 0);
}

void register_template(ParserContext *ctx, const char *name, ASTNode *node)
{
    GenericTemplate *t = xcalloc(1, sizeof(GenericTemplate));
    t->name = xstrdup(name);
    t->struct_node = node;
    t->next = ctx->templates;
    ctx->templates = t;
}

ASTNode *copy_fields_replacing(ParserContext *ctx, ASTNode *fields, const char *param,
                               const char *concrete)
{
    if (!fields)
    {
        return NULL;
    }
    ASTNode *n = ast_create(NODE_FIELD);
    n->field.name = xstrdup(fields->field.name);

    // Replace strings
    n->field.type = replace_type_str(fields->field.type, param, concrete, NULL, NULL);

    // Replace formal types (Deep Copy)
    n->type_info = replace_type_formal(fields->type_info, param, concrete, NULL, NULL);

    if (n->field.type && strchr(n->field.type, '_'))
    {
        // Parse potential generic: e.g. "MapEntry_int" -> instantiate("MapEntry",
        // "int")
        char *underscore = strrchr(n->field.type, '_');
        if (underscore && underscore > n->field.type)
        {
            // Remove trailing '*' if present
            char *type_copy = xstrdup(n->field.type);
            char *star = strchr(type_copy, '*');
            if (star)
            {
                *star = '\0';
            }

            underscore = strrchr(type_copy, '_');
            if (underscore)
            {
                *underscore = '\0';
                char *template_name = type_copy;
                char *concrete_arg = underscore + 1;

                // Check if this is actually a known generic template
                GenericTemplate *gt = ctx->templates;
                int found = 0;
                while (gt)
                {
                    if (strcmp(gt->name, template_name) == 0)
                    {
                        found = 1;
                        break;
                    }
                    gt = gt->next;
                }

                if (found)
                {
                    char *unmangled = unmangle_ptr_suffix(concrete_arg);
                    if (concrete)
                    {
                        char *clean_concrete = sanitize_mangled_name(concrete);
                        if (strcmp(concrete_arg, clean_concrete) == 0)
                        {
                            free(unmangled);
                            unmangled = xstrdup(concrete);
                        }
                        free(clean_concrete);
                    }

                    instantiate_generic(ctx, template_name, concrete_arg, unmangled, fields->token);
                    free(unmangled);
                }
            }
            free(type_copy);
        }
    }

    // Additional check: if type_info is a pointer to a struct with a mangled name,
    // instantiate that struct as well (fixes cases like RcInner<T>* where the
    // string check above might not catch it)
    if (n->type_info && n->type_info->kind == TYPE_POINTER && n->type_info->inner)
    {
        Type *inner = n->type_info->inner;
        if (inner->kind == TYPE_STRUCT && inner->name && strchr(inner->name, '_'))
        {
            // Extract template name by checking against known templates
            // We can't use strrchr because types like "Inner_int32_t" have multiple underscores
            char *template_name = NULL;
            char *concrete_arg = NULL;

            // Try each known template to see if the type name starts with it
            GenericTemplate *gt = ctx->templates;
            while (gt)
            {
                size_t tlen = strlen(gt->name);
                // Check if name starts with template name followed by double underscore
                if (strncmp(inner->name, gt->name, tlen) == 0 && inner->name[tlen] == '_' &&
                    inner->name[tlen + 1] == '_')
                {
                    template_name = gt->name;
                    concrete_arg =
                        inner->name + tlen + 2; // Skip template name and double underscore
                    break;
                }
                gt = gt->next;
            }

            if (template_name && concrete_arg)
            {
                char *unmangled = unmangle_ptr_suffix(concrete_arg);
                if (concrete)
                {
                    char *clean_concrete = sanitize_mangled_name(concrete);
                    if (strcmp(concrete_arg, clean_concrete) == 0)
                    {
                        free(unmangled);
                        unmangled = xstrdup(concrete);
                    }
                    free(clean_concrete);
                }
                instantiate_generic(ctx, template_name, concrete_arg, unmangled, fields->token);
                free(unmangled);
            }
        }
    }

    n->next = copy_fields_replacing(ctx, fields->next, param, concrete);
    return n;
}

void instantiate_methods(ParserContext *ctx, GenericImplTemplate *it,
                         const char *mangled_struct_name, const char *arg,
                         const char *unmangled_arg)
{
    if (check_impl(ctx, "Methods", mangled_struct_name))
    {
        return; // Simple dedupe check
    }

    ASTNode *backup_next = it->impl_node->next;
    it->impl_node->next = NULL; // Break link to isolate node

    // Use unmangled_arg if provided, otherwise arg
    char *raw = (char *)(unmangled_arg ? unmangled_arg : arg);
    char *subst_arg = unmangle_ptr_suffix(raw);

    ASTNode *new_impl = copy_ast_replacing(it->impl_node, it->generic_param, subst_arg,
                                           it->struct_name, mangled_struct_name);

    // Also replace mangled template name (both List__G and List_G)
    if (strchr(it->struct_name, '<'))
    {
        char *sanitized = sanitize_mangled_name(it->struct_name);
        if (strcmp(sanitized, it->struct_name) != 0)
        {
            ASTNode *tmp = copy_ast_replacing(new_impl, NULL, NULL, sanitized, mangled_struct_name);
            new_impl = tmp;
        }

        char *old_sanitized = xstrdup(sanitized);
        char *double_underscore = strstr(old_sanitized, "__");
        if (double_underscore)
        {
            memmove(double_underscore, double_underscore + 1, strlen(double_underscore + 1) + 1);
        }

        if (strcmp(old_sanitized, it->struct_name) != 0 && strcmp(old_sanitized, sanitized) != 0)
        {
            ASTNode *tmp =
                copy_ast_replacing(new_impl, NULL, NULL, old_sanitized, mangled_struct_name);
            new_impl = tmp;
        }

        free(old_sanitized);
        free(sanitized);
    }
    free(subst_arg);
    it->impl_node->next = backup_next; // Restore

    ASTNode *meth = NULL;

    if (new_impl->type == NODE_IMPL)
    {
        new_impl->impl.struct_name = xstrdup(mangled_struct_name);
        meth = new_impl->impl.methods;
    }
    else if (new_impl->type == NODE_IMPL_TRAIT)
    {
        new_impl->impl_trait.target_type = xstrdup(mangled_struct_name);
        meth = new_impl->impl_trait.methods;
    }

    while (meth)
    {
        // Standardize: ensure __ between type and method
        // If it's already correctly mangled (e.g. Vec__int32_t__with_capacity), skip
        size_t mlen = strlen(mangled_struct_name);
        int correctly_mangled = (strncmp(meth->func.name, mangled_struct_name, mlen) == 0 &&
                                 meth->func.name[mlen] == '_' && meth->func.name[mlen + 1] == '_');

        if (!correctly_mangled)
        {
            // Find the method part in the original name (e.g. "with_capacity" in
            // "Vec_with_capacity")
            char *original_method = meth->func.name;
            if (strncmp(original_method, it->struct_name, strlen(it->struct_name)) == 0)
            {
                original_method += strlen(it->struct_name);
            }
            while (*original_method == '_')
            {
                original_method++;
            }

            char *temp = xmalloc(strlen(mangled_struct_name) + strlen(original_method) + 3);
            sprintf(temp, "%s__%s", mangled_struct_name, original_method);
            char *new_name = merge_underscores(temp);
            free(temp);
            free(meth->func.name);
            meth->func.name = new_name;
        }

        register_func(ctx, ctx->global_scope, meth->func.name, meth->func.arg_count,
                      meth->func.defaults, meth->func.arg_types, meth->func.ret_type_info,
                      meth->func.is_varargs, (meth->type == NODE_FUNCTION && meth->func.is_async),
                      meth->func.pure, meth->link_name, meth->token, meth->func.is_export);

        // Handle generic return types in methods (e.g., Option<T> -> Option_int)
        if (meth->func.ret_type &&
            (strchr(meth->func.ret_type, '_') || strchr(meth->func.ret_type, '<')))
        {
            GenericTemplate *gt = ctx->templates;

            while (gt)
            {
                size_t tlen = strlen(gt->name);
                char delim = meth->func.ret_type[tlen];
                if (strncmp(meth->func.ret_type, gt->name, tlen) == 0 &&
                    (delim == '_' || delim == '<'))
                {
                    // Found matching template prefix
                    const char *type_arg = meth->func.ret_type + tlen;
                    while (*type_arg == '_' || *type_arg == '<')
                    {
                        type_arg++;
                    }

                    // Simple approach: instantiate 'Template' with 'Arg'.
                    char *clean_arg = xstrdup(type_arg);
                    if (delim == '<')
                    {
                        char *closer = strrchr(clean_arg, '>');
                        if (closer)
                        {
                            *closer = 0;
                        }
                    }

                    // Unmangle Ptr suffix if present (e.g., intPtr -> int*)
                    char *inner_unmangled_arg = xstrdup(clean_arg);
                    size_t alen = strlen(clean_arg);
                    if (alen > 3 && strcmp(clean_arg + alen - 3, "Ptr") == 0)
                    {
                        char *base = xstrdup(clean_arg);
                        base[alen - 3] = '\0';
                        free(inner_unmangled_arg);
                        inner_unmangled_arg = xmalloc(strlen(base) + 16);
                        // Check if base is a primitive type
                        if (is_unmangle_primitive(base))
                        {
                            sprintf(inner_unmangled_arg, "%s*", base);
                        }
                        else
                        {
                            sprintf(inner_unmangled_arg, "struct %s*", base);
                        }
                        free(base);
                    }

                    instantiate_generic(ctx, gt->name, clean_arg, inner_unmangled_arg, meth->token);
                    free(clean_arg);
                }
                gt = gt->next;
            }
        }

        trigger_instantiations(ctx, meth->func.body);

        meth = meth->next;
    }
    add_instantiated_func(ctx, new_impl);
}

static void register_enum_constructor(ParserContext *ctx, const char *m, const char *var_name,
                                      int tag_id, Type *payload, Token token, int is_export)
{
    size_t mangled_var_sz = strlen(m) + strlen(var_name) + 3;
    char *mangled_var = xmalloc(mangled_var_sz);
    snprintf(mangled_var, mangled_var_sz, "%s__%s", m, var_name);
    register_enum_variant(ctx, m, mangled_var, tag_id);

    Type *ret_t = type_new(TYPE_ENUM);
    ret_t->name = xstrdup(m);

    if (payload)
    {
        Type **at = xmalloc(sizeof(Type *));
        at[0] = payload;
        register_func(ctx, ctx->global_scope, mangled_var, 1, NULL, at, ret_t, 0, 0, 0, NULL, token,
                      is_export);
    }
    else
    {
        register_func(ctx, ctx->global_scope, mangled_var, 0, NULL, NULL, ret_t, 0, 0, 0, NULL,
                      token, is_export);
    }
    free(mangled_var);
}

void instantiate_generic(ParserContext *ctx, const char *tpl, const char *arg,
                         const char *unmangled_arg, Token token)
{
    // Ignore generic placeholders
    if (strlen(arg) == 1 && isupper(arg[0]))
    {
        return;
    }
    if (strcmp(arg, "T") == 0)
    {
        return;
    }

    char *clean_arg = sanitize_mangled_name(arg);
    char *m = xmalloc(strlen(tpl) + strlen(clean_arg) + 4);
    strcpy(m, tpl);
    char *m_end = m + strlen(m);
    while (m_end > m && *(m_end - 1) == '_')
    {
        *(--m_end) = '\0';
    }
    strcat(m, "__");
    strcat(m, clean_arg);
    free(clean_arg);

    Instantiation *c = ctx->instantiations;
    while (c)
    {
        if (strcmp(c->name, m) == 0)
        {
            free(m);
            return; // Already instantiated, DO NOTHING.
        }
        c = c->next;
    }

    GenericTemplate *t = ctx->templates;
    while (t)
    {
        if (strcmp(t->name, tpl) == 0)
        {
            break;
        }
        t = t->next;
    }
    if (!t)
    {
        zpanic_at(token, "Unknown generic: %s", tpl);
    }

    Instantiation *ni = xcalloc(1, sizeof(Instantiation));
    ni->name = xstrdup(m);
    ni->template_name = xstrdup(tpl);
    ni->concrete_arg = xstrdup(arg);
    ni->unmangled_arg = unmangled_arg ? xstrdup(unmangled_arg)
                                      : xstrdup(arg); // Fallback to arg if unmangled is generic
    ni->struct_node = NULL;                           // Placeholder to break cycles
    ni->next = ctx->instantiations;
    ctx->instantiations = ni;

    ASTNode *struct_node_copy = NULL;

    if (t->struct_node->type == NODE_STRUCT)
    {
        ASTNode *i = ast_create(NODE_STRUCT);
        i->strct.name = xstrdup(m);
        i->strct.is_template = 0;
        i->strct.is_export = t->struct_node->strct.is_export;

        // Copy type attributes (e.g. has_drop)
        i->type_info = type_new(TYPE_STRUCT);
        i->type_info->name = xstrdup(m);
        if (t->struct_node->type_info)
        {
            i->type_info->traits = t->struct_node->type_info->traits;
            i->type_info->is_restrict = t->struct_node->type_info->is_restrict;
        }
        i->strct.is_packed = t->struct_node->strct.is_packed;
        i->strct.is_union = t->struct_node->strct.is_union;
        i->strct.align = t->struct_node->strct.align;
        if (t->struct_node->strct.parent)
        {
            i->strct.parent = xstrdup(t->struct_node->strct.parent);
        }
        const char *gp = (t->struct_node->strct.generic_param_count > 0)
                             ? t->struct_node->strct.generic_params[0]
                             : "T";
        const char *subst_arg = unmangled_arg ? unmangled_arg : arg;
        i->strct.fields = copy_fields_replacing(ctx, t->struct_node->strct.fields, gp, subst_arg);
        struct_node_copy = i;
        register_struct_def(ctx, m, i);

        // Register slice types used in the instantiated struct's fields
        ASTNode *fld = i->strct.fields;
        while (fld)
        {
            if (fld->field.type && strncmp(fld->field.type, "Slice__", 7) == 0)
            {
                register_slice(ctx, fld->field.type + 7);
            }
            fld = fld->next;
        }
    }
    else if (t->struct_node->type == NODE_ENUM)
    {
        ASTNode *i = ast_create(NODE_ENUM);
        i->enm.name = xstrdup(m);
        i->enm.is_template = 0;
        i->enm.is_export = t->struct_node->enm.is_export;

        // Copy type attributes (e.g. has_drop)
        i->type_info = type_new(TYPE_ENUM);
        i->type_info->name = xstrdup(m);
        if (t->struct_node->type_info)
        {
            i->type_info->traits = t->struct_node->type_info->traits;
        }

        ASTNode *h = 0, *tl = 0;
        ASTNode *v = t->struct_node->enm.variants;
        while (v)
        {
            ASTNode *nv = ast_create(NODE_ENUM_VARIANT);
            nv->variant.name = xstrdup(v->variant.name);
            nv->variant.tag_id = v->variant.tag_id;
            const char *subst_arg = unmangled_arg ? unmangled_arg : arg;
            nv->variant.payload = replace_type_formal(
                v->variant.payload, t->struct_node->enm.generic_param, subst_arg, NULL, NULL);

            register_enum_constructor(ctx, m, nv->variant.name, nv->variant.tag_id,
                                      nv->variant.payload, token, i->enm.is_export);

            if (!h)
            {
                h = nv;
            }
            else
            {
                tl->next = nv;
            }
            tl = nv;
            v = v->next;
        }
        i->enm.variants = h;
        struct_node_copy = i;
    }

    ni->struct_node = struct_node_copy;

    if (struct_node_copy)
    {
        struct_node_copy->next = ctx->instantiated_structs;
        ctx->instantiated_structs = struct_node_copy;
    }

    GenericImplTemplate *it = ctx->impl_templates;
    while (it)
    {
        if (strcmp(it->struct_name, tpl) == 0)
        {
            instantiate_methods(ctx, it, m, arg, unmangled_arg);
        }
        it = it->next;
    }
    free(m);
}

static void free_field_list(ASTNode *fields)
{
    while (fields)
    {
        ASTNode *next = fields->next;
        if (fields->field.name)
        {
            free(fields->field.name);
        }
        if (fields->field.type)
        {
            free(fields->field.type);
        }
        free(fields);
        fields = next;
    }
}

void instantiate_generic_multi(ParserContext *ctx, const char *tpl, char **args, int arg_count,
                               Token token)
{
    // Build mangled name from all args
    size_t m_len = strlen(tpl) + 1;
    for (int i = 0; i < arg_count; i++)
    {
        char *clean = sanitize_mangled_name(args[i]);
        m_len += 2 + strlen(clean);
        free(clean);
    }
    char *m = xmalloc(m_len + 1);
    strcpy(m, tpl);
    char *m_end = m + strlen(m);
    while (m_end > m && *(m_end - 1) == '_')
    {
        *(--m_end) = '\0';
    }
    for (int i = 0; i < arg_count; i++)
    {
        char *clean = sanitize_mangled_name(args[i]);
        strcat(m, "__");
        strcat(m, clean);
        free(clean);
    }

    // Check if already instantiated
    Instantiation *c = ctx->instantiations;
    while (c)
    {
        if (strcmp(c->name, m) == 0)
        {
            free(m);
            return; // Already done
        }
        c = c->next;
    }

    // Find the template
    GenericTemplate *t = ctx->templates;
    while (t)
    {
        if (strcmp(t->name, tpl) == 0)
        {
            break;
        }
        t = t->next;
    }
    if (!t)
    {
        zpanic_at(token, "Unknown generic: %s", tpl);
    }

    // Register instantiation first (to break cycles)
    Instantiation *ni = xcalloc(1, sizeof(Instantiation));
    ni->name = xstrdup(m);
    ni->template_name = xstrdup(tpl);
    ni->concrete_arg = (arg_count > 0) ? xstrdup(args[0]) : xstrdup("T");

    // For multi-param, build a comma-separated string for unmangled_arg
    size_t u_len = 0;
    for (int i = 0; i < arg_count; i++)
    {
        u_len += strlen(args[i]) + 1;
    }
    char *u_buf = xmalloc(u_len + 1);
    u_buf[0] = 0;
    for (int i = 0; i < arg_count; i++)
    {
        if (i > 0)
        {
            strcat(u_buf, ",");
        }
        strcat(u_buf, args[i]);
    }
    ni->unmangled_arg = u_buf;

    ni->struct_node = NULL;
    ni->next = ctx->instantiations;
    ctx->instantiations = ni;

    if (t->struct_node->type == NODE_STRUCT)
    {
        ASTNode *i = ast_create(NODE_STRUCT);
        i->strct.name = xstrdup(m);
        i->strct.is_template = 0;
        i->strct.is_export = t->struct_node->strct.is_export;

        // Copy struct attributes
        i->strct.is_packed = t->struct_node->strct.is_packed;
        i->strct.is_union = t->struct_node->strct.is_union;
        i->strct.align = t->struct_node->strct.align;
        if (t->struct_node->strct.parent)
        {
            i->strct.parent = xstrdup(t->struct_node->strct.parent);
        }

        // Copy fields with sequential substitutions for each param
        ASTNode *fields = t->struct_node->strct.fields;
        int param_count = t->struct_node->strct.generic_param_count;

        if (param_count > 0 && arg_count > 0)
        {
            // First substitution
            i->strct.fields = copy_fields_replacing(
                ctx, fields, t->struct_node->strct.generic_params[0], args[0]);

            // Subsequent substitutions (for params B, C, etc.)
            for (int j = 1; j < param_count && j < arg_count; j++)
            {
                ASTNode *prev_fields = i->strct.fields;
                ASTNode *tmp = copy_fields_replacing(
                    ctx, prev_fields, t->struct_node->strct.generic_params[j], args[j]);
                free_field_list(prev_fields);
                i->strct.fields = tmp;
            }
        }
        else
        {
            i->strct.fields = copy_fields_replacing(ctx, fields, "T", "int");
        }

        ni->struct_node = i;
        register_struct_def(ctx, m, i);

        i->next = ctx->instantiated_structs;
        ctx->instantiated_structs = i;
    }
    else if (t->struct_node->type == NODE_ENUM)
    {
        ASTNode *i = ast_create(NODE_ENUM);
        i->enm.name = xstrdup(m);
        i->enm.is_template = 0;
        i->enm.is_export = t->struct_node->enm.is_export;

        // Copy type attributes
        i->type_info = type_new(TYPE_ENUM);
        i->type_info->name = xstrdup(m);
        if (t->struct_node->type_info)
        {
            i->type_info->traits = t->struct_node->type_info->traits;
        }

        ASTNode *h = 0, *tl = 0;
        ASTNode *v = t->struct_node->enm.variants;

        // Construct comma-separated concrete args string
        size_t c_args_len = 1;
        for (int j = 0; j < arg_count; j++)
        {
            c_args_len += strlen(args[j]) + 1;
        }
        char *c_args = xmalloc(c_args_len);
        c_args[0] = 0;
        for (int j = 0; j < arg_count; j++)
        {
            if (j > 0)
            {
                strcat(c_args, ",");
            }
            strcat(c_args, args[j]);
        }

        while (v)
        {
            ASTNode *nv = ast_create(NODE_ENUM_VARIANT);
            nv->variant.name = xstrdup(v->variant.name);
            nv->variant.tag_id = v->variant.tag_id;

            // Use multi-parameter substitution for payload
            Type *payload = v->variant.payload;
            nv->variant.payload = NULL;
            if (payload)
            {
                nv->variant.payload = replace_type_formal(
                    payload, t->struct_node->enm.generic_param, c_args, NULL, NULL);
            }

            register_enum_constructor(ctx, m, nv->variant.name, nv->variant.tag_id,
                                      nv->variant.payload, token, i->enm.is_export);

            if (!h)
            {
                h = nv;
            }
            else
            {
                tl->next = nv;
            }
            tl = nv;
            v = v->next;
        }
        free(c_args);
        i->enm.variants = h;
        ni->struct_node = i;
        register_struct_def(ctx, m, i);

        i->next = ctx->instantiated_structs;
        ctx->instantiated_structs = i;
    }
    free(m);
}

int is_file_imported(ParserContext *ctx, const char *p)
{
    if (!p)
    {
        return 0;
    }
    ImportedFile *c = ctx->imported_files;
    while (c)
    {
        if (strcmp(c->path, p) == 0)
        {
            return 1;
        }
        c = c->next;
    }
    return 0;
}

void mark_file_imported(ParserContext *ctx, const char *p)
{
    ImportedFile *f = xmalloc(sizeof(ImportedFile));
    f->path = xstrdup(p);
    f->next = ctx->imported_files;
    ctx->imported_files = f;
}

char *parse_condition_raw(ParserContext *ctx, Lexer *l)
{
    (void)ctx; // suppress unused parameter warning
    Token t = lexer_peek(l);
    if (t.type == TOK_LPAREN)
    {
        Token op = lexer_next(l);
        const char *s = op.start;
        int d = 1;
        while (d > 0)
        {
            t = lexer_next(l);
            if (t.type == TOK_EOF)
            {
                zpanic_at(t, "Unterminated condition");
            }
            if (t.type == TOK_LPAREN)
            {
                d++;
            }
            if (t.type == TOK_RPAREN)
            {
                d--;
            }
        }
        const char *cs = s + 1;
        int len = t.start - cs;
        char *c = xmalloc(len + 1);
        strncpy(c, cs, len);
        c[len] = 0;
        return c;
    }
    else
    {
        const char *start = l->src + l->pos;
        while (1)
        {
            t = lexer_peek(l);
            if (t.type == TOK_LBRACE || t.type == TOK_EOF)
            {
                break;
            }
            lexer_next(l);
        }
        int len = (l->src + l->pos) - start;
        if (len == 0)
        {
            zpanic_at(lexer_peek(l), "Empty condition or missing body");
        }
        char *c = xmalloc(len + 1);
        strncpy(c, start, len);
        c[len] = 0;
        return c;
    }
}

typedef struct
{
    char *final_struct;
    char *final_cast;
} MixinResolution;

static MixinResolution resolve_mixin_method(ParserContext *ctx, const char *struct_name,
                                            const char *method_name, int is_ptr)
{
    MixinResolution res = {xstrdup(struct_name), NULL};

    char target_func_raw[MAX_FUNC_NAME_LEN];
    sprintf(target_func_raw, "%s__%s", struct_name, method_name);
    char *target_func = merge_underscores(target_func_raw);

    if (!find_func(ctx, target_func))
    {
        ASTNode *mixin_def = find_struct_def(ctx, struct_name);
        if (mixin_def && mixin_def->type == NODE_STRUCT && mixin_def->strct.used_structs)
        {
            for (int k = 0; k < mixin_def->strct.used_struct_count; k++)
            {
                char mixin_func_raw[128];
                sprintf(mixin_func_raw, "%s__%s", mixin_def->strct.used_structs[k], method_name);
                char *mixin_func = merge_underscores(mixin_func_raw);
                if (find_func(ctx, mixin_func))
                {
                    free(res.final_struct);
                    res.final_struct = xstrdup(mixin_def->strct.used_structs[k]);
                    char cast_buf[128];
                    if (is_ptr)
                    {
                        sprintf(cast_buf, "(%s*)", res.final_struct);
                    }
                    else
                    {
                        sprintf(cast_buf, "(%s*)&", res.final_struct);
                    }
                    res.final_cast = xstrdup(cast_buf);
                    free(mixin_func);
                    break;
                }
                free(mixin_func);
            }
        }
    }
    free(target_func);
    return res;
}

static MixinResolution resolve_method_from_type_str(ParserContext *ctx, const char *vtype,
                                                    const char *method)
{
    char ptr_check[64];
    strncpy(ptr_check, vtype, 63);
    ptr_check[63] = 0;
    int is_ptr = (strchr(ptr_check, '*') != NULL);
    if (is_ptr)
    {
        char *p = strchr(ptr_check, '*');
        if (p)
        {
            *p = 0;
        }
    }
    return resolve_mixin_method(ctx, ptr_check, method, is_ptr);
}

char *rewrite_expr_methods(ParserContext *ctx, char *raw)
{
    if (!raw)
    {
        return NULL;
    }

    int in_expr = 0;
    char *result = xmalloc(strlen(raw) * 4 + 100);
    char *dest = result;
    char *src = raw;

    while (*src)
    {
        if (strncmp(src, "#{", 2) == 0)
        {
            in_expr = 1;
            src += 2;
            *dest++ = '(';
            continue;
        }

        if (in_expr && *src == '}')
        {
            in_expr = 0;
            *dest++ = ')';
            src++;
            continue;
        }

        if (in_expr && *src == '.')
        {
            char acc[64];
            int i = 0;
            char *back = src - 1;
            while (back >= raw && (isalnum(*back) || *back == '_'))
            {
                back--;
            }
            back++;
            while (back < src && i < 63)
            {
                acc[i++] = *back++;
            }
            acc[i] = 0;

            char *vtype = find_symbol_type(ctx, acc);
            if (!vtype)
            {
                *dest++ = *src++;
                continue;
            }

            char method[64];
            i = 0;
            src++;
            while (isalnum(*src) || *src == '_')
            {
                method[i++] = *src++;
            }
            method[i] = 0;

            // Check for field access
            char *base_t = xstrdup(vtype);
            char *pc = strchr(base_t, '*');
            int is_ptr_type = (pc != NULL);
            if (pc)
            {
                *pc = 0;
            }

            // Resolve type alias if exists (for example: Vec2f -> Vec2_float)
            const char *resolved_type = find_type_alias(ctx, base_t);
            if (resolved_type)
            {
                free(base_t);
                base_t = xstrdup(resolved_type);
            }

            ASTNode *def = find_struct_def(ctx, base_t);
            int is_field = 0;
            if (def && (def->type == NODE_STRUCT))
            {
                ASTNode *f = def->strct.fields;
                while (f)
                {
                    if (strcmp(f->field.name, method) == 0)
                    {
                        is_field = 1;
                        break;
                    }
                    f = f->next;
                }
            }
            free(base_t);

            if (is_field)
            {
                dest -= strlen(acc);
                if (is_ptr_type)
                {
                    dest += sprintf(dest, "(%s)->%s", acc, method);
                }
                else
                {
                    dest += sprintf(dest, "(%s).%s", acc, method);
                }
                continue;
            }

            if (*src == '(')
            {
                dest -= strlen(acc);
                int paren_depth = 0;
                src++;
                paren_depth++;

                int is_ptr = (strchr(vtype, '*') != NULL);

                // Mixin Lookup Logic
                MixinResolution res = resolve_method_from_type_str(ctx, vtype, method);
                char *final_cast = res.final_cast;
                char *final_method = xstrdup(method);
                char *final_struct = res.final_struct;

                if (final_cast)
                {
                    // Mixin call: Foo__method((Foo*)&obj
                    char call_buf[MAX_ERROR_MSG_LEN];
                    snprintf(call_buf, sizeof(call_buf), "%s__%s", final_struct, final_method);
                    char *mangled_call = merge_underscores(call_buf);

                    dest += sprintf(dest, "%s(%s%s", mangled_call, final_cast, acc);
                    free(final_cast);
                }
                else
                {
                    // Standard call
                    char call_buf[MAX_ERROR_MSG_LEN];
                    snprintf(call_buf, sizeof(call_buf), "%s__%s", final_struct, final_method);
                    char *mangled_call = merge_underscores(call_buf);

                    dest += sprintf(dest, "%s(%s%s", mangled_call, is_ptr ? "" : "&", acc);
                }
                free(final_struct);
                free(final_method);

                int has_args = 0;
                while (*src && paren_depth > 0)
                {
                    if (!isspace(*src))
                    {
                        has_args = 1;
                    }
                    if (*src == '(')
                    {
                        paren_depth++;
                    }
                    if (*src == ')')
                    {
                        paren_depth--;
                    }
                    if (paren_depth == 0)
                    {
                        break;
                    }
                    *dest++ = *src++;
                }

                if (has_args)
                {
                    *dest++ = ')';
                }
                else
                {
                    *dest++ = ')';
                }

                src++;
                continue;
            }
            else
            {
                dest -= strlen(acc);
                int is_ptr = (strchr(vtype, '*') != NULL);
                // Mixin Lookup Logic (No Parens)
                MixinResolution res = resolve_method_from_type_str(ctx, vtype, method);
                char *final_cast = res.final_cast;
                char *final_method = xstrdup(method);
                char *final_struct = res.final_struct;

                if (final_cast)
                {
                    char call_buf[MAX_ERROR_MSG_LEN];
                    snprintf(call_buf, sizeof(call_buf), "%s__%s", final_struct, final_method);
                    char *mangled_call = merge_underscores(call_buf);

                    dest += sprintf(dest, "%s(%s%s)", mangled_call, final_cast, acc);
                    free(final_cast);
                }
                else
                {
                    char call_buf[MAX_ERROR_MSG_LEN];
                    snprintf(call_buf, sizeof(call_buf), "%s__%s", final_struct, final_method);
                    char *mangled_call = merge_underscores(call_buf);

                    dest += sprintf(dest, "%s(%s%s)", mangled_call, is_ptr ? "" : "&", acc);
                }
                free(final_struct);
                free(final_method);
                continue;
            }
        }

        if (!in_expr && strncmp(src, "::", 2) == 0)
        {
            char acc[64];
            int i = 0;
            char *back = src - 1;
            while (back >= raw && (isalnum(*back) || *back == '_'))
            {
                back--;
            }
            back++;
            while (back < src && i < 63)
            {
                acc[i++] = *back++;
            }
            acc[i] = 0;

            src += 2;
            char field[64];
            i = 0;
            while (isalnum(*src) || *src == '_')
            {
                field[i++] = *src++;
            }
            field[i] = 0;

            dest -= strlen(acc);

            Module *mod = find_module(ctx, acc);
            if (mod && mod->is_c_header)
            {
                dest += sprintf(dest, "%s", field);
            }
            else
            {
                ASTNode *sdef = find_struct_def(ctx, acc);
                if (sdef && sdef->type == NODE_ENUM)
                {
                    // For Enums, check if it's a variant
                    int is_variant = 0;
                    ASTNode *v = sdef->enm.variants;
                    while (v)
                    {
                        if (strcmp(v->variant.name, field) == 0)
                        {
                            is_variant = 1;
                            break;
                        }
                        v = v->next;
                    }
                    if (is_variant)
                    {
                        dest += sprintf(dest, "%s__%s", acc, field);
                    }
                    else
                    {
                        // Static method on Enum
                        dest += sprintf(dest, "%s__%s", acc, field);
                    }
                }
                else if (sdef || !mod)
                {
                    // Struct static method, or Type static method
                    dest += sprintf(dest, "%s__%s", acc, field);
                }
                else
                {
                    // Module function
                    dest += sprintf(dest, "%s__%s", acc, field);
                }
            }
            continue;
        }

        if (in_expr && isalpha(*src))
        {
            char tok[128];
            int i = 0;
            while ((isalnum(*src) || *src == '_') && i < 127)
            {
                tok[i++] = *src++;
            }
            tok[i] = 0;

            while (*src == ' ' || *src == '\t')
            {
                src++;
            }

            if (strncmp(src, "::", 2) == 0)
            {
                src += 2;
                char func_name[128];
                snprintf(func_name, sizeof(func_name), "%s", tok);
                char method[64];
                i = 0;
                while (isalnum(*src) || *src == '_')
                {
                    method[i++] = *src++;
                }
                method[i] = 0;

                while (*src == ' ' || *src == '\t')
                {
                    src++;
                }

                if (*src == '(')
                {
                    src++;

                    char mangled[MAX_MANGLED_NAME_LEN];

                    const char *aliased = find_type_alias(ctx, func_name);
                    const char *use_name = aliased ? aliased : func_name;

                    Module *mod = find_module(ctx, use_name);
                    if (mod)
                    {
                        if (mod->is_c_header)
                        {
                            snprintf(mangled, sizeof(mangled), "%s", method);
                        }
                        else
                        {
                            char mangled_raw[MAX_MANGLED_NAME_LEN];
                            snprintf(mangled_raw, sizeof(mangled_raw), "%s__%s", mod->base_name,
                                     method);
                            char *mangled_merged = merge_underscores(mangled_raw);
                            strncpy(mangled, mangled_merged, sizeof(mangled) - 1);
                            mangled[sizeof(mangled) - 1] = 0;
                        }
                    }
                    else
                    {
                        ASTNode *sdef = find_struct_def(ctx, use_name);
                        if (sdef)
                        {
                            char mangled_raw[MAX_MANGLED_NAME_LEN];
                            snprintf(mangled_raw, sizeof(mangled_raw), "%s__%s", use_name, method);
                            char *mangled_merged = merge_underscores(mangled_raw);
                            strncpy(mangled, mangled_merged, sizeof(mangled) - 1);
                            mangled[sizeof(mangled) - 1] = 0;
                        }
                        else
                        {
                            char mangled_raw[MAX_MANGLED_NAME_LEN];
                            snprintf(mangled_raw, sizeof(mangled_raw), "%s__%s", use_name, method);
                            char *mangled_merged = merge_underscores(mangled_raw);
                            strncpy(mangled, mangled_merged, sizeof(mangled) - 1);
                            mangled[sizeof(mangled) - 1] = 0;
                        }
                    }

                    if (*src == ')')
                    {
                        dest += sprintf(dest, "%s()", mangled);
                        src++;
                    }
                    else
                    {
                        FuncSig *sig = find_func(ctx, func_name);
                        if (sig)
                        {
                            dest += sprintf(dest, "%s(&(%s){0}", mangled, func_name);
                            while (*src && *src != ')')
                            {
                                *dest++ = *src++;
                            }
                            *dest++ = ')';
                            if (*src == ')')
                            {
                                src++;
                            }
                        }
                        else
                        {
                            dest += sprintf(dest, "%s(", mangled);
                            while (*src && *src != ')')
                            {
                                *dest++ = *src++;
                            }
                            *dest++ = ')';
                            if (*src == ')')
                            {
                                src++;
                            }
                        }
                    }
                    continue;
                }
            }

            strcpy(dest, tok);
            dest += strlen(tok);
            continue;
        }

        *dest++ = *src++;
    }

    *dest = 0;
    return result;
}

char *consume_and_rewrite(ParserContext *ctx, Lexer *l)
{
    char *r = consume_until_semicolon(l);
    char *rw = rewrite_expr_methods(ctx, r);
    free(r);
    return rw;
}

char *parse_and_convert_args(ParserContext *ctx, Lexer *l, char ***defaults_out,
                             ASTNode ***default_values_out, int *count_out, Type ***types_out,
                             char ***names_out, int *is_varargs_out, char ***ctype_overrides_out)
{
    Token t = lexer_next(l);
    if (t.type != TOK_LPAREN)
    {
        zpanic_at(t, "Expected '(' in function args");
    }

    size_t buf_size = 8192;
    char *buf = xmalloc(buf_size);
    buf[0] = 0;
    int count = 0;
    int max_args = 16;
    char **defaults = xcalloc(max_args, sizeof(char *));
    ASTNode **default_values = xcalloc(max_args, sizeof(ASTNode *));
    Type **types = xcalloc(max_args, sizeof(Type *));
    char **names = xcalloc(max_args, sizeof(char *));
    char **ctype_overrides = xcalloc(max_args, sizeof(char *));

    // Initial 16 entries already zeroed by xcalloc

    if (lexer_peek(l).type != TOK_RPAREN)
    {
        while (1)
        {
            if (count >= max_args)
            {
                int new_max = max_args * 2;
                defaults = xrealloc(defaults, sizeof(char *) * new_max);
                memset(defaults + max_args, 0, sizeof(char *) * (new_max - max_args));
                default_values = xrealloc(default_values, sizeof(ASTNode *) * new_max);
                memset(default_values + max_args, 0, sizeof(ASTNode *) * (new_max - max_args));
                types = xrealloc(types, sizeof(Type *) * new_max);
                memset(types + max_args, 0, sizeof(Type *) * (new_max - max_args));
                names = xrealloc(names, sizeof(char *) * new_max);
                memset(names + max_args, 0, sizeof(char *) * (new_max - max_args));
                ctype_overrides = xrealloc(ctype_overrides, sizeof(char *) * new_max);
                memset(ctype_overrides + max_args, 0, sizeof(char *) * (new_max - max_args));
                max_args = new_max;
            }

            // Check for @ctype("...") before parameter
            char *ctype_override = NULL;
            if (lexer_peek(l).type == TOK_AT)
            {
                lexer_next(l); // eat @
                Token attr = lexer_next(l);
                if (attr.type == TOK_IDENT && attr.len == 5 && strncmp(attr.start, "ctype", 5) == 0)
                {
                    if (lexer_next(l).type != TOK_LPAREN)
                    {
                        zpanic_at(lexer_peek(l), "Expected ( after @ctype");
                    }
                    Token ctype_tok = lexer_next(l);
                    if (ctype_tok.type != TOK_STRING)
                    {
                        zpanic_at(ctype_tok, "@ctype requires a string argument");
                    }
                    // Extract string content (strip quotes)
                    ctype_override = xmalloc(ctype_tok.len - 1);
                    strncpy(ctype_override, ctype_tok.start + 1, ctype_tok.len - 2);
                    ctype_override[ctype_tok.len - 2] = 0;
                    if (lexer_next(l).type != TOK_RPAREN)
                    {
                        zpanic_at(lexer_peek(l), "Expected ) after @ctype string");
                    }
                }
                else
                {
                    zpanic_at(attr, "Unknown parameter attribute @%.*s", attr.len, attr.start);
                }
            }

            int is_const_param = 0;
            Token param_tok = lexer_next(l);

            if (is_token(param_tok, "const"))
            {
                is_const_param = 1;
                param_tok = lexer_next(l);
            }

            // Handle 'self'
            if (is_token(param_tok, "self"))
            {
                names[count] = xstrdup("self");
                if (ctx->current_impl_struct)
                {
                    char *buf_type = xmalloc(strlen(ctx->current_impl_struct) + 2);
                    sprintf(buf_type, "%s*", ctx->current_impl_struct);

                    if (is_primitive_type_name(ctx->current_impl_struct))
                    {
                        // Primitives: self is a pointer in signature and body
                        TypeKind pk = get_primitive_type_kind(ctx->current_impl_struct);
                        Type *bt = type_new(pk);
                        if (pk == TYPE_STRUCT)
                        { // Fallback if get_primitive_type_kind failed for some reason
                            bt->name = xstrdup(ctx->current_impl_struct);
                        }
                        bt->is_const = is_const_param;
                        Type *ptr = type_new_ptr(bt);

                        add_symbol(ctx, "self", buf_type, ptr, 0);
                        types[count] = ptr;
                    }
                    else
                    {
                        // Structs: self is a pointer in signature and body
                        Type *st = type_new(TYPE_STRUCT);
                        st->name = xstrdup(ctx->current_impl_struct);
                        st->is_const = is_const_param;
                        Type *ptr = type_new_ptr(st);

                        add_symbol(ctx, "self", buf_type, ptr, 0);
                        types[count] = ptr;
                    }
                    free(buf_type);
                    if (is_const_param)
                    {
                        strcat(buf, "const void* self");
                    }
                    else
                    {
                        strcat(buf, "void* self");
                    }
                }
                else
                {
                    if (is_const_param)
                    {
                        strcat(buf, "const void* self");
                    }
                    else
                    {
                        strcat(buf, "void* self");
                    }
                    Type *void_type = type_new(TYPE_VOID);
                    void_type->is_const = is_const_param;
                    types[count] = type_new_ptr(void_type);
                    add_symbol(ctx, "self", is_const_param ? "const void*" : "void*", types[count],
                               0);
                }
                ctype_overrides[count] = ctype_override;
                count++;
            }
            else
            {
                if (param_tok.type != TOK_IDENT)
                {
                    zpanic_at(lexer_peek(l), "Expected arg name");
                }
                check_identifier(ctx, param_tok);
                char *name = token_strdup(param_tok);
                names[count] = name; // Store name
                if (lexer_next(l).type != TOK_COLON)
                {
                    zpanic_at(lexer_peek(l), "Expected ':'");
                }

                Type *arg_type = parse_type_formal(ctx, l);
                if (is_const_param)
                {
                    arg_type->is_const = 1;
                }
                char *type_str = type_to_string(arg_type);

                add_symbol(ctx, name, type_str, arg_type, 0);
                types[count] = arg_type;

                if (strlen(buf) > 0)
                {
                    strcat(buf, ", ");
                }

                // Ensure buf has enough space before appending
                size_t needed = strlen(buf) + strlen(type_str) + strlen(name) + 32;
                if (needed > buf_size)
                {
                    while (needed > buf_size)
                    {
                        buf_size *= 2;
                    }
                    buf = xrealloc(buf, buf_size);
                }

                char *fn_ptr = strstr(type_str, "(*)");
                if (get_inner_type(arg_type)->kind == TYPE_FUNCTION)
                {
                    strcat(buf, "z_closure_T ");
                    strcat(buf, name);
                }
                else if (fn_ptr)
                {
                    // Inject name into function pointer: int (*)(int) -> int (*name)(int)
                    int prefix_len = fn_ptr - type_str;
                    strncat(buf, type_str, prefix_len);
                    strcat(buf, " (*");
                    strcat(buf, name);
                    strcat(buf, ")");
                    strcat(buf, fn_ptr + 3); // Skip "(*)"
                }
                else
                {
                    // Use @ctype override if present
                    if (ctype_override)
                    {
                        strcat(buf, ctype_override);
                    }
                    else
                    {
                        strcat(buf, type_str);
                    }
                    strcat(buf, " ");
                    strcat(buf, name);
                }

                ctype_overrides[count] = ctype_override;
                count++;

                if (lexer_peek(l).type == TOK_OP && is_token(lexer_peek(l), "="))
                {
                    lexer_next(l); // consume =

                    // Parse the expression into an AST node
                    ASTNode *def_node = parse_expression(ctx, l);

                    // Store both the AST node and the reconstructed string for legacy support
                    default_values[count - 1] = def_node;
                    defaults[count - 1] = ast_to_string(def_node);
                }
            }
            if (lexer_peek(l).type == TOK_COMMA)
            {
                lexer_next(l);
                // Check if next is ...
                if (lexer_peek(l).type == TOK_ELLIPSIS)
                {
                    lexer_next(l);
                    if (is_varargs_out)
                    {
                        *is_varargs_out = 1;
                    }
                    if (strlen(buf) > 0)
                    {
                        strcat(buf, ", ");
                    }
                    strcat(buf, "...");
                    break; // Must be last
                }
            }
            else
            {
                break;
            }
        }
    }
    if (lexer_next(l).type != TOK_RPAREN)
    {
        zpanic_at(lexer_peek(l), "Expected ')' after args");
    }

    *defaults_out = defaults;
    *default_values_out = default_values;
    *count_out = count;
    *types_out = types;
    *names_out = names;
    if (ctype_overrides_out)
    {
        *ctype_overrides_out = ctype_overrides;
    }
    return buf;
}

// Helper to find similar symbol name in current scope
char *find_similar_symbol(ParserContext *ctx, const char *name)
{
    if (!ctx->current_scope)
    {
        return NULL;
    }

    const char *best_match = NULL;
    int best_dist = 999;

    // Check local scopes
    Scope *s = ctx->current_scope;
    while (s)
    {
        ZenSymbol *sym = s->symbols;
        while (sym)
        {
            int dist = levenshtein(name, sym->name);
            if (dist < best_dist && dist <= 3)
            {
                best_dist = dist;
                best_match = sym->name;
            }
            sym = sym->next;
        }
        s = s->parent;
    }

    // Check builtins/globals if any (simplified)
    return best_match ? xstrdup(best_match) : NULL;
}

static const char *get_closest_type_hint(ParserContext *ctx, const char *name)
{
    int best_dist = 4;
    const char *best = NULL;

    StructDef *def = ctx->struct_defs;
    while (def)
    {
        int dist = levenshtein(name, def->name);
        if (dist < best_dist)
        {
            best_dist = dist;
            best = def->name;
        }
        def = def->next;
    }

    StructRef *er = ctx->parsed_enums_list;
    while (er)
    {
        if (er->node && er->node->type == NODE_ENUM)
        {
            int dist = levenshtein(name, er->node->enm.name);
            if (dist < best_dist)
            {
                best_dist = dist;
                best = er->node->enm.name;
            }
        }
        er = er->next;
    }

    TypeAlias *ta = ctx->type_aliases;
    while (ta)
    {
        int dist = levenshtein(name, ta->alias);
        if (dist < best_dist)
        {
            best_dist = dist;
            best = ta->alias;
        }
        ta = ta->next;
    }

    return best;
}

void register_plugin(ParserContext *ctx, const char *name, const char *alias)
{
    // Try to find existing (built-in) or already loaded plugin
    ZPlugin *plugin = zptr_find_plugin(name);

    // If not found, try to load it dynamically
    if (!plugin)
    {
#ifdef ZC_STATIC_PLUGINS
        plugin = zptr_find_plugin(name);
        if (!plugin && strchr(name, '/'))
        {
            const char *last_slash = strrchr(name, '/');
            plugin = zptr_find_plugin(last_slash + 1);
        }
#endif
        if (!plugin)
        {
            char path[MAX_PATH_LEN];
            snprintf(path, sizeof(path), "%s%s", name, z_get_plugin_ext());
            plugin = zptr_load_plugin(path);
        }

        if (!plugin && !strchr(name, '/'))
        {
            char path[MAX_PATH_LEN];
            snprintf(path, sizeof(path), "%s%s%s", z_get_run_prefix(), name, z_get_plugin_ext());
            plugin = zptr_load_plugin(path);
        }

        // Fallback for system-wide plugins
        if (!plugin)
        {
            char path[MAX_PATH_LEN];
            // Try full name first
            snprintf(path, sizeof(path), ZEN_SHARE_DIR "/plugins/%s%s", name, z_get_plugin_ext());
            plugin = zptr_load_plugin(path);

            // If it was a path like "plugins/name", try matching just the base "name" in system dir
            if (!plugin && strchr(name, '/'))
            {
                const char *last_slash = strrchr(name, '/');
                snprintf(path, sizeof(path), ZEN_SHARE_DIR "/plugins/%s%s", last_slash + 1,
                         z_get_plugin_ext());
                plugin = zptr_load_plugin(path);
            }
        }
    }

    if (!plugin)
    {
        fprintf(stderr,
                COLOR_RED "Error:" COLOR_RESET " Could not load plugin '%s'\n"
                          "       Tried built-ins and dynamic loading (.so)\n",
                name);
        if (g_config.mode_lsp)
        {
            // Register alias anyway to avoid redundant "Unknown plugin" noise
            ImportedPlugin *p = xmalloc(sizeof(ImportedPlugin));
            p->name = xstrdup(name);
            p->alias = alias ? xstrdup(alias) : NULL;
            p->next = ctx->imported_plugins;
            ctx->imported_plugins = p;
            return;
        }
        exit(1);
    }

    ImportedPlugin *p = xmalloc(sizeof(ImportedPlugin));
    p->name = xstrdup(plugin->name); // Use the plugin's internal name
    p->alias = alias ? xstrdup(alias) : NULL;
    p->next = ctx->imported_plugins;
    ctx->imported_plugins = p;
}

const char *resolve_plugin(ParserContext *ctx, const char *name_or_alias)
{
    for (ImportedPlugin *p = ctx->imported_plugins; p; p = p->next)
    {
        // Check if it matches the alias
        if (p->alias && strcmp(p->alias, name_or_alias) == 0)
        {
            return p->name;
        }
        // Check if it matches the name
        if (strcmp(p->name, name_or_alias) == 0)
        {
            return p->name;
        }
    }
    return NULL; // Plugin not found
}

// Type Validation
void register_type_usage(ParserContext *ctx, const char *name, Token t)
{
    if (ctx->is_speculative)
    {
        return;
    }

    TypeUsage *u = xmalloc(sizeof(TypeUsage));
    u->name = xstrdup(name);
    u->location = t;
    u->next = ctx->pending_type_validations;
    ctx->pending_type_validations = u;
}

int validate_types(ParserContext *ctx)
{
    int errors = 0;
    TypeUsage *u = ctx->pending_type_validations;
    while (u)
    {
        ASTNode *def = find_struct_def(ctx, u->name);
        if (!def)
        {
            const char *alias = find_type_alias(ctx, u->name);
            if (!alias)
            {
                SelectiveImport *si = find_selective_import(ctx, u->name);
                if (!si && !is_extern_symbol(ctx, u->name))
                {
                    if (!is_trait(u->name) && TYPE_UNKNOWN == find_primitive_kind(u->name))
                    {
                        // Check dynamic whitelist from zenc.json
                        int whitelisted = 0;
                        if (g_config.c_type_whitelist)
                        {
                            char **ptr = g_config.c_type_whitelist;
                            while (*ptr)
                            {
                                if (strcmp(u->name, *ptr) == 0)
                                {
                                    whitelisted = 1;
                                    break;
                                }
                                ptr++;
                            }
                        }

                        if (whitelisted)
                        {
                            u = u->next;
                            continue;
                        }

                        if (!g_config.mode_lsp && !g_config.mode_doc)
                        {
                            char msg[MAX_SHORT_MSG_LEN];
                            snprintf(msg, sizeof(msg),
                                     "Unknown type '%s' (assuming external C struct)", u->name);
                            const char *hint = get_closest_type_hint(ctx, u->name);
                            if (hint)
                            {
                                char help[MAX_MANGLED_NAME_LEN];
                                snprintf(help, sizeof(help), "Did you mean '%s'?", hint);
                                zwarn_with_suggestion(u->location, msg, help);
                            }
                            else
                            {
                                zwarn_at(u->location, "%s", msg);
                            }
                        }
                    }
                }
            }
        }
        u = u->next;
    }
    return errors == 0;
}

void propagate_vector_inner_types(ParserContext *ctx)
{
    StructRef *ref = ctx->parsed_structs_list;
    while (ref)
    {
        ASTNode *strct = ref->node;
        if (strct && strct->type == NODE_STRUCT && strct->type_info &&
            strct->type_info->kind == TYPE_VECTOR && !strct->type_info->inner)
        {
            if (strct->strct.fields && strct->strct.fields->type_info)
            {
                strct->type_info->inner = strct->strct.fields->type_info;
            }
        }
        ref = ref->next;
    }
}

void propagate_drop_traits(ParserContext *ctx)
{
    int changed = 1;
    while (changed)
    {
        changed = 0;

        // Process regular structs
        StructRef *ref = ctx->parsed_structs_list;
        while (ref)
        {
            ASTNode *strct = ref->node;
            if (strct && strct->type == NODE_STRUCT && strct->type_info &&
                !strct->type_info->traits.has_drop)
            {
                ASTNode *field = strct->strct.fields;
                while (field)
                {
                    Type *ft = field->type_info;
                    if (ft)
                    {
                        if (ft->kind == TYPE_VECTOR)
                        {
                            strct->type_info->traits.has_drop = 1;
                            changed = 1;
                            break;
                        }
                        if (ft->kind == TYPE_FUNCTION && ft->traits.has_drop && !ft->is_raw)
                        {
                            strct->type_info->traits.has_drop = 1;
                            changed = 1;
                            break;
                        }
                        if (ft->kind == TYPE_STRUCT && ft->name)
                        {
                            ASTNode *fdef = find_struct_def(ctx, ft->name);
                            if (fdef && fdef->type_info && fdef->type_info->traits.has_drop)
                            {
                                strct->type_info->traits.has_drop = 1;
                                changed = 1;
                                break;
                            }
                        }
                    }
                    field = field->next;
                }
            }
            ref = ref->next;
        }

        // Process instantiated templates
        ASTNode *ins = ctx->instantiated_structs;
        while (ins)
        {
            if (ins->type == NODE_STRUCT && ins->type_info && !ins->type_info->traits.has_drop)
            {
                ASTNode *field = ins->strct.fields;
                while (field)
                {
                    Type *ft = field->type_info;
                    if (ft)
                    {
                        if (ft->kind == TYPE_VECTOR)
                        {
                            ins->type_info->traits.has_drop = 1;
                            changed = 1;
                            break;
                        }
                        if (ft->kind == TYPE_FUNCTION && ft->traits.has_drop && !ft->is_raw)
                        {
                            ins->type_info->traits.has_drop = 1;
                            changed = 1;
                            break;
                        }
                        if (ft->kind == TYPE_STRUCT && ft->name)
                        {
                            ASTNode *fdef = find_struct_def(ctx, ft->name);
                            if (fdef && fdef->type_info && fdef->type_info->traits.has_drop)
                            {
                                ins->type_info->traits.has_drop = 1;
                                changed = 1;
                                break;
                            }
                        }
                    }
                    field = field->next;
                }
            }
            ins = ins->next;
        }
    }
}

const char *normalize_type_name(const char *name)
{
    if (!name)
    {
        return NULL;
    }

    return get_primitive_c_name(name);
}

int is_reserved_keyword(Token t)
{
    // Lexer-level keywords
    switch (t.type)
    {
    case TOK_TEST:
    case TOK_ASSERT:
    case TOK_SIZEOF:
    case TOK_DEF:
    case TOK_DEFER:
    case TOK_AUTOFREE:
    case TOK_USE:
    case TOK_TRAIT:
    case TOK_IMPL:
    case TOK_AND:
    case TOK_OR:
    case TOK_FOR:
    case TOK_COMPTIME:
    case TOK_UNION:
    case TOK_ASM:
    case TOK_VOLATILE:
    case TOK_ASYNC:
    case TOK_AWAIT:
    case TOK_ALIAS:
    case TOK_OPAQUE:
        return 1;
    default:
        break;
    }

    if (t.type == TOK_IDENT)
    {
        static const char *pseudo_keywords[] = {
            "let",   "var",      "static", "const",  "return", "if",    "else",   "while",
            "break", "continue", "loop",   "repeat", "unless", "guard", "launch", "do",
            "goto",  "plugin",   "fn",     "struct", "enum",   "self",  NULL};

        for (int i = 0; pseudo_keywords[i] != NULL; i++)
        {
            if (t.len == (int)strlen(pseudo_keywords[i]) &&
                strncmp(t.start, pseudo_keywords[i], t.len) == 0)
            {
                return 1;
            }
        }
    }

    return 0;
}

void check_identifier(ParserContext *ctx, Token t)
{
    (void)ctx;
    if (is_reserved_keyword(t))
    {
        char buf[MAX_SHORT_MSG_LEN];
        char name[64];
        int len = t.len < 63 ? t.len : 63;
        strncpy(name, t.start, len);
        name[len] = 0;
        snprintf(buf, sizeof(buf), "Cannot use reserved keyword '%s' as an identifier", name);
        zpanic_at(t, "%s", buf);
    }
}

static void audit_section_5(ParserContext *ctx, Scope *scope, const char *name,
                            const char *link_name, Token tok)
{
    if (!scope || !name)
    {
        return;
    }
    if (strcmp(name, "it") == 0 || strcmp(name, "self") == 0)
    {
        return;
    }

    Scope *p = scope;
    int limit = (p == ctx->global_scope) ? 31 : 63;

    while (p)
    {
        ZenSymbol *sh = p->symbols;
        while (sh)
        {
            // Rule 5.3: Shadowing
            if (p != scope && strcmp(sh->name, name) == 0 && !ctx->silent_warnings)
            {
                if (g_config.misra_mode)
                {
                    zerror_at(tok, "MISRA Rule 5.3");
                }
                else
                {
                    warn_shadowing(tok, name);
                }
            }

            // Rules 5.1/5.2: Distinctness
            if (g_config.misra_mode)
            {
                const char *actual_name = link_name ? link_name : name;
                const char *sh_actual_name = sh->link_name ? sh->link_name : sh->name;

                // For distinctness, we check if they are different names but collide
                if (strcmp(sh_actual_name, actual_name) != 0)
                {
                    misra_check_identifier_collision(tok, sh_actual_name, actual_name, limit);
                }
            }

            sh = sh->next;
        }
        p = p->parent;
    }
}

static void sync_type_linkage(ParserContext *ctx, Type *t)
{
    if (!t)
    {
        return;
    }
    if ((t->kind == TYPE_STRUCT || t->kind == TYPE_ENUM) && !t->link_name && t->name)
    {
        ASTNode *def = find_struct_def(ctx, t->name);
        if (def && def->link_name)
        {
            t->link_name = xstrdup(def->link_name);
        }
    }
    if (t->inner)
    {
        sync_type_linkage(ctx, t->inner);
    }
    for (int i = 0; i < t->arg_count; i++)
    {
        sync_type_linkage(ctx, t->args[i]);
    }
}

static void sync_link_names_recursive(ParserContext *ctx, ASTNode *node)
{
    if (!node)
    {
        return;
    }

    // Sync type_info if present
    if (node->type_info)
    {
        sync_type_linkage(ctx, node->type_info);
    }

    // Node-specific sync
    switch (node->type)
    {
    case NODE_FUNCTION:
        if (node->func.ret_type_info)
        {
            sync_type_linkage(ctx, node->func.ret_type_info);
        }
        if (node->func.arg_types)
        {
            for (int i = 0; i < node->func.arg_count; i++)
            {
                sync_type_linkage(ctx, node->func.arg_types[i]);
            }
        }
        sync_link_names_recursive(ctx, node->func.body);
        break;
    case NODE_STRUCT:
        sync_link_names_recursive(ctx, node->strct.fields);
        break;
    case NODE_VAR_DECL:
        sync_link_names_recursive(ctx, node->var_decl.init_expr);
        break;
    case NODE_BLOCK:
        sync_link_names_recursive(ctx, node->block.statements);
        break;
    case NODE_IF:
        sync_link_names_recursive(ctx, node->if_stmt.condition);
        sync_link_names_recursive(ctx, node->if_stmt.then_body);
        sync_link_names_recursive(ctx, node->if_stmt.else_body);
        break;
    case NODE_RETURN:
        sync_link_names_recursive(ctx, node->ret.value);
        break;
    case NODE_EXPR_CALL:
        sync_link_names_recursive(ctx, node->call.callee);
        sync_link_names_recursive(ctx, node->call.args);
        break;
    case NODE_EXPR_BINARY:
        sync_link_names_recursive(ctx, node->binary.left);
        sync_link_names_recursive(ctx, node->binary.right);
        break;
    case NODE_EXPR_UNARY:
        sync_link_names_recursive(ctx, node->unary.operand);
        break;
    case NODE_EXPR_MEMBER:
        sync_link_names_recursive(ctx, node->member.target);
        break;
    case NODE_EXPR_CAST:
        sync_link_names_recursive(ctx, node->cast.expr);
        break;
    case NODE_ROOT:
        sync_link_names_recursive(ctx, node->root.children);
        break;
    default:
        break;
    }

    sync_link_names_recursive(ctx, node->next);
}

void sync_all_link_names(ParserContext *ctx, ASTNode *root)
{
    sync_link_names_recursive(ctx, root);
}
