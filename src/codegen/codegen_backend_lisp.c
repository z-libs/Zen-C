// SPDX-License-Identifier: MIT
// Lisp transpiler backend
#include "codegen_backend.h"
#include "codegen.h"
#include "../parser/parser.h"
#include "../ast/ast.h"
#include <string.h>
#include <ctype.h>

static const char *current_function = NULL;
static void lisp_indent(ParserContext *ctx, int depth)
{
    for (int i = 0; i < depth; i++)
    {
        emitter_printf(&ctx->cg.emitter, "  ");
    }
}

static void lisp_escape_str(ParserContext *ctx, const char *s)
{
    if (!s)
    {
        emitter_printf(&ctx->cg.emitter, "\"\"");
        return;
    }
    emitter_printf(&ctx->cg.emitter, "\"");
    for (const char *p = s; *p; p++)
    {
        char c = *p;
        switch (c)
        {
        case '"':
            emitter_printf(&ctx->cg.emitter, "\\\"");
            break;
        case '\\':
            emitter_printf(&ctx->cg.emitter, "\\\\");
            break;
        case '\n':
            emitter_printf(&ctx->cg.emitter, "~%");
            break;
        default:
            emitter_printf(&ctx->cg.emitter, "%c", c);
            break;
        }
    }
    emitter_printf(&ctx->cg.emitter, "\"");
}

static void lisp_emit_expr(ParserContext *ctx, ASTNode *node, int depth);
static void lisp_emit_stmt(ParserContext *ctx, ASTNode *node, int depth, int *first);
static void lisp_emit_stmts(ParserContext *ctx, ASTNode *stmts, int depth);

static const char *lisp_op(const char *zen_op)
{
    if (!zen_op)
    {
        return "?";
    }
    if (strcmp(zen_op, "+") == 0)
    {
        return "+";
    }
    if (strcmp(zen_op, "-") == 0)
    {
        return "-";
    }
    if (strcmp(zen_op, "*") == 0)
    {
        return "*";
    }
    if (strcmp(zen_op, "/") == 0)
    {
        return "/";
    }
    if (strcmp(zen_op, "%") == 0)
    {
        return "mod";
    }
    if (strcmp(zen_op, "==") == 0)
    {
        return "=";
    }
    if (strcmp(zen_op, "!=") == 0)
    {
        return "/=";
    }
    if (strcmp(zen_op, "<") == 0)
    {
        return "<";
    }
    if (strcmp(zen_op, "<=") == 0)
    {
        return "<=";
    }
    if (strcmp(zen_op, ">") == 0)
    {
        return ">";
    }
    if (strcmp(zen_op, ">=") == 0)
    {
        return ">=";
    }
    if (strcmp(zen_op, "&&") == 0)
    {
        return "and";
    }
    if (strcmp(zen_op, "||") == 0)
    {
        return "or";
    }
    if (strcmp(zen_op, "&") == 0)
    {
        return "logand";
    }
    if (strcmp(zen_op, "|") == 0)
    {
        return "logior";
    }
    if (strcmp(zen_op, "^") == 0)
    {
        return "logxor";
    }
    if (strcmp(zen_op, "<<") == 0)
    {
        return "ash";
    }
    if (strcmp(zen_op, ">>") == 0)
    {
        return "ash"; // assumes unsigned
    }
    if (strcmp(zen_op, "..") == 0)
    {
        return ".."; // range literal
    }
    return zen_op;
}

static const char *lisp_callee_name(ASTNode *callee)
{
    if (!callee)
    {
        return NULL;
    }
    if (callee->type == NODE_EXPR_VAR && callee->var_ref.name)
    {
        const char *name = callee->var_ref.name;
        // Strip mangle prefix: Vec__int32_t__new -> new
        const char *last = strrchr(name, '_');
        if (last && last > name && *(last - 1) == '_')
        {
            return last + 1;
        }
        return name;
    }
    if (callee->type == NODE_EXPR_MEMBER && callee->member.field)
    {
        return callee->member.field;
    }
    return NULL;
}

// Forward declarations
static void lisp_emit_call(ParserContext *ctx, ASTNode *node, int depth);

static void lisp_emit_expr(ParserContext *ctx, ASTNode *node, int depth)
{
    if (!node)
    {
        emitter_printf(&ctx->cg.emitter, "nil");
        return;
    }
    switch (node->type)
    {
    case NODE_EXPR_LITERAL:
    {
        switch (node->literal.type_kind)
        {
        case LITERAL_INT:
            emitter_printf(&ctx->cg.emitter, "%llu", node->literal.int_val);
            break;
        case LITERAL_FLOAT:
            emitter_printf(&ctx->cg.emitter, "%g", node->literal.float_val);
            break;
        case LITERAL_STRING:
        case LITERAL_RAW_STRING:
            lisp_escape_str(ctx, node->literal.string_val);
            break;
        case LITERAL_CHAR:
            emitter_printf(&ctx->cg.emitter, "#\\%c", (char)node->literal.int_val);
            break;
        }
        break;
    }
    case NODE_EXPR_VAR:
        if (node->var_ref.name)
        {
            emitter_printf(&ctx->cg.emitter, "%s", node->var_ref.name);
        }
        else
        {
            emitter_printf(&ctx->cg.emitter, "?var");
        }
        break;
    case NODE_EXPR_BINARY:
    {
        const char *op = lisp_op(node->binary.op);
        if (strcmp(op, "..") == 0)
        {
            emitter_printf(&ctx->cg.emitter, "(cons ");
            lisp_emit_expr(ctx, node->binary.left, depth);
            emitter_printf(&ctx->cg.emitter, " ");
            lisp_emit_expr(ctx, node->binary.right, depth);
            emitter_printf(&ctx->cg.emitter, ")");
        }
        else
        {
            emitter_printf(&ctx->cg.emitter, "(%s ", op);
            lisp_emit_expr(ctx, node->binary.left, depth);
            emitter_printf(&ctx->cg.emitter, " ");
            lisp_emit_expr(ctx, node->binary.right, depth);
            emitter_printf(&ctx->cg.emitter, ")");
        }
        break;
    }
    case NODE_EXPR_UNARY:
    {
        const char *op = node->unary.op;
        if (strcmp(op, "!") == 0)
        {
            emitter_printf(&ctx->cg.emitter, "(not ");
            lisp_emit_expr(ctx, node->unary.operand, depth);
            emitter_printf(&ctx->cg.emitter, ")");
        }
        else if (strcmp(op, "-") == 0)
        {
            emitter_printf(&ctx->cg.emitter, "(- ");
            lisp_emit_expr(ctx, node->unary.operand, depth);
            emitter_printf(&ctx->cg.emitter, ")");
        }
        else if (strcmp(op, "*") == 0)
        {
            emitter_printf(&ctx->cg.emitter, "(deref ");
            lisp_emit_expr(ctx, node->unary.operand, depth);
            emitter_printf(&ctx->cg.emitter, ")");
        }
        else if (strcmp(op, "&") == 0)
        {
            emitter_printf(&ctx->cg.emitter, "(ref ");
            lisp_emit_expr(ctx, node->unary.operand, depth);
            emitter_printf(&ctx->cg.emitter, ")");
        }
        else
        {
            emitter_printf(&ctx->cg.emitter, "(%s ", op);
            lisp_emit_expr(ctx, node->unary.operand, depth);
            emitter_printf(&ctx->cg.emitter, ")");
        }
        break;
    }
    case NODE_EXPR_CALL:
        lisp_emit_call(ctx, node, depth);
        break;
    case NODE_EXPR_MEMBER:
        if (node->member.field)
        {
            // Numeric field → tuple index (nth N obj)
            if (node->member.field[0] >= '0' && node->member.field[0] <= '9')
            {
                emitter_printf(&ctx->cg.emitter, "(nth %s ", node->member.field);
                if (node->member.target)
                {
                    lisp_emit_expr(ctx, node->member.target, depth);
                }
                else
                {
                    emitter_printf(&ctx->cg.emitter, "nil");
                }
                emitter_printf(&ctx->cg.emitter, ")");
            }
            else
            {
                // Symbolic field → slot access
                emitter_printf(&ctx->cg.emitter, "(%s-", node->member.field);
                if (node->member.target)
                {
                    lisp_emit_expr(ctx, node->member.target, depth);
                }
                else
                {
                    emitter_printf(&ctx->cg.emitter, "nil");
                }
                emitter_printf(&ctx->cg.emitter, ")");
            }
        }
        break;
    case NODE_EXPR_INDEX:
    {
        emitter_printf(&ctx->cg.emitter, "(aref ");
        lisp_emit_expr(ctx, node->index.array, depth);
        emitter_printf(&ctx->cg.emitter, " ");
        lisp_emit_expr(ctx, node->index.index, depth);
        emitter_printf(&ctx->cg.emitter, ")");
        break;
    }
    case NODE_EXPR_CAST:
        emitter_printf(&ctx->cg.emitter, "(the ");
        if (node->type_info)
        {
            char *tn = type_to_string(node->type_info);
            if (tn)
            {
                emitter_printf(&ctx->cg.emitter, "%s ", tn);
                zfree(tn);
            }
        }
        if (node->cast.expr)
        {
            lisp_emit_expr(ctx, node->cast.expr, depth);
        }
        else
        {
            emitter_printf(&ctx->cg.emitter, "nil");
        }
        emitter_printf(&ctx->cg.emitter, ")");
        break;
    case NODE_EXPR_SIZEOF:
        emitter_printf(&ctx->cg.emitter, "(error \"sizeof in lisp?\")");
        break;
    case NODE_TYPEOF:
        emitter_printf(&ctx->cg.emitter, "(type-of ");
        if (node->size_of.expr)
        {
            lisp_emit_expr(ctx, node->size_of.expr, depth);
        }
        else
        {
            emitter_printf(&ctx->cg.emitter, "nil");
        }
        emitter_printf(&ctx->cg.emitter, ")");
        break;
    case NODE_EXPR_STRUCT_INIT:
    {
        emitter_printf(&ctx->cg.emitter, "(list");
        for (ASTNode *f = node->struct_init.fields; f; f = f->next)
        {
            emitter_printf(&ctx->cg.emitter, " ");
            if (f->type == NODE_EXPR_BINARY && f->binary.op && strcmp(f->binary.op, "=") == 0)
            {
                lisp_emit_expr(ctx, f->binary.right, depth);
            }
            else
            {
                lisp_emit_expr(ctx, f, depth);
            }
        }
        emitter_printf(&ctx->cg.emitter, ")");
        break;
    }
    case NODE_EXPR_ARRAY_LITERAL:
    case NODE_EXPR_TUPLE_LITERAL:
    {
        ASTNode *elems = node->array_literal.elements;
        if (elems)
        {
            emitter_printf(&ctx->cg.emitter, "(list");
            for (ASTNode *e = elems; e; e = e->next)
            {
                emitter_printf(&ctx->cg.emitter, " ");
                lisp_emit_expr(ctx, e, depth);
            }
            emitter_printf(&ctx->cg.emitter, ")");
        }
        else
        {
            emitter_printf(&ctx->cg.emitter, "nil");
        }
        break;
    }
    case NODE_TERNARY:
    {
        emitter_printf(&ctx->cg.emitter, "(if ");
        lisp_emit_expr(ctx, node->ternary.cond, depth);
        emitter_printf(&ctx->cg.emitter, " ");
        lisp_emit_expr(ctx, node->ternary.true_expr, depth);
        emitter_printf(&ctx->cg.emitter, " ");
        lisp_emit_expr(ctx, node->ternary.false_expr, depth);
        emitter_printf(&ctx->cg.emitter, ")");
        break;
    }
    case NODE_LAMBDA:
    {
        emitter_printf(&ctx->cg.emitter, "(lambda (");
        for (int i = 0; i < node->lambda.num_params; i++)
        {
            if (i > 0)
            {
                emitter_printf(&ctx->cg.emitter, " ");
            }
            if (node->lambda.param_names && node->lambda.param_names[i])
            {
                emitter_printf(&ctx->cg.emitter, "%s", node->lambda.param_names[i]);
            }
            else
            {
                emitter_printf(&ctx->cg.emitter, "p%d", i);
            }
        }
        emitter_printf(&ctx->cg.emitter, ")\n");
        lisp_indent(ctx, depth + 1);
        if (node->lambda.body)
        {
            // Unwrap block bodies to emit contents directly
            ASTNode *body = node->lambda.body;
            if (body->type == NODE_BLOCK && body->block.statements)
            {
                int bf = 1;
                for (ASTNode *s = body->block.statements; s; s = s->next)
                {
                    if (!bf)
                    {
                        emitter_printf(&ctx->cg.emitter, "\n");
                        lisp_indent(ctx, depth + 2);
                    }
                    bf = 0;
                    // Unwrap return statements in expression lambdas
                    if (s->type == NODE_RETURN && s->ret.value)
                    {
                        lisp_emit_expr(ctx, s->ret.value, depth + 2);
                    }
                    else
                    {
                        int sbf = 0;
                        lisp_emit_stmt(ctx, s, depth + 2, &sbf);
                    }
                }
            }
            else
            {
                int bf = 1;
                lisp_emit_stmt(ctx, body, depth + 1, &bf);
            }
        }
        emitter_printf(&ctx->cg.emitter, ")");
        break;
    }
    default:
        emitter_printf(&ctx->cg.emitter, "(unhandled-expr)");
        break;
    }
}

static void lisp_emit_call(ParserContext *ctx, ASTNode *node, int depth)
{
    (void)depth;
    const char *name = lisp_callee_name(node->call.callee);
    if (name)
    {
        emitter_printf(&ctx->cg.emitter, "(%s", name);
    }
    else
    {
        emitter_printf(&ctx->cg.emitter, "(call");
    }
    // Detect println patterns in raw calls
    if (name && strcmp(name, "println") == 0)
    {
        // Already handled as RAW; shouldn't reach here normally
        emitter_printf(&ctx->cg.emitter, " ...)");
        return;
    }
    // Determine if first arg is a method self-arg (UNARY &)
    int skip_first = 0;
    if (node->call.args && node->call.args->type == NODE_EXPR_UNARY && node->call.args->unary.op &&
        strcmp(node->call.args->unary.op, "&") == 0)
    {
        skip_first = 1;
    }
    int argn = 0;
    for (ASTNode *a = node->call.args; a; a = a->next)
    {
        if (skip_first && argn == 0)
        {
            argn++;
            continue;
        }
        emitter_printf(&ctx->cg.emitter, " ");
        lisp_emit_expr(ctx, a, depth);
        argn++;
    }
    if (skip_first)
    {
        // Emit the self arg as first argument
        emitter_printf(&ctx->cg.emitter, " ");
        lisp_emit_expr(ctx, node->call.args, depth);
    }
    emitter_printf(&ctx->cg.emitter, ")");
}

static int lisp_raw_is_print(const char *raw)
{
    if (!raw)
    {
        return 0;
    }
    const char *p = raw;
    while (*p && strchr(" ({", *p))
    {
        p++;
    }
    return strncmp(p, "fprintf(stdout,", 15) == 0;
}

// Try to parse a string literal from C: "hello"
// Returns length of parsed string including quotes, 0 on failure
static int parse_c_string(const char *s, char *out, int out_max)
{
    if (!s || *s != '"')
    {
        return 0;
    }
    int si = 1; // skip opening quote
    int oi = 0;
    while (s[si] && s[si] != '"')
    {
        if (s[si] == '\\')
        {
            si++;
            if (!s[si])
            {
                return 0;
            }
            switch (s[si])
            {
            case 'n':
                if (oi < out_max - 1)
                {
                    out[oi++] = '\n';
                }
                break;
            case 't':
                if (oi < out_max - 1)
                {
                    out[oi++] = '\t';
                }
                break;
            case '\\':
                if (oi < out_max - 1)
                {
                    out[oi++] = '\\';
                }
                break;
            case '"':
                if (oi < out_max - 1)
                {
                    out[oi++] = '"';
                }
                break;
            default:
                if (oi < out_max - 1)
                {
                    out[oi++] = s[si];
                }
                break;
            }
            si++;
        }
        else
        {
            if (oi < out_max - 1)
            {
                out[oi++] = s[si];
            }
            si++;
        }
    }
    if (s[si] != '"')
    {
        return 0;
    }
    out[oi] = '\0';
    return si + 1; // include closing quote
}

// Emit (princ ...) or (format t ...) for a print/println RAW node
// We emit each fprintf segment as a separate Lisp output call for simplicity.
static void lisp_emit_print(ParserContext *ctx, const char *raw, int depth)
{
    (void)depth;
    const char *p = raw;
    while (*p && strchr(" ({", *p))
    {
        p++;
    }
    int emitted = 0;
    int has_newline = 0;
    while (*p)
    {
        if (*p == '#')
        {
            while (*p && *p != '\n')
            {
                p++;
            }
            if (*p)
            {
                p++;
            }
            continue;
        }
        while (*p && strchr("; \n\t", *p))
        {
            p++;
        }
        if (!*p || *p == '0' || *p == ')')
        {
            break;
        }
        if (strncmp(p, "fprintf(stdout,", 15) == 0)
        {
            p += 15;
            while (*p && strchr(" \t\n", *p))
            {
                p++;
            }
            if (!*p)
            {
                break;
            }
            if (*p == '"')
            {
                char first_str[64];
                int n = parse_c_string(p, first_str, sizeof(first_str));
                if (n > 0)
                {
                    p += n;
                    while (*p && strchr(", \t\n", *p))
                    {
                        p++;
                    }
                    if (*p == '"')
                    {
                        // Pattern: "%s", "value" — first is format, second is value
                        char lit[2048];
                        n = parse_c_string(p, lit, sizeof(lit));
                        if (n > 0)
                        {
                            p += n;
                            if (strcmp(lit, "\n") == 0)
                            {
                                has_newline = 1;
                            }
                            else
                            {
                                if (emitted)
                                {
                                    emitter_printf(&ctx->cg.emitter, "\n");
                                    lisp_indent(ctx, depth + 1);
                                }
                                emitter_printf(&ctx->cg.emitter, "(princ ");
                                lisp_escape_str(ctx, lit);
                                emitter_printf(&ctx->cg.emitter, ")");
                                emitted++;
                            }
                        }
                    }
                    else if (strcmp(first_str, "\n") == 0)
                    {
                        // Pattern: "\n" — direct newline
                        has_newline = 1;
                    }
                }
            }
            else if (strncmp(p, "_z_str(", 7) == 0)
            {
                p += 7;
                while (*p && strchr(" \t\n", *p))
                {
                    p++;
                }
                char expr[256];
                int ei = 0;
                int parens = 0;
                while (*p && ei < 255)
                {
                    if (*p == ')' && parens == 0)
                    {
                        break;
                    }
                    if (*p == '#')
                    {
                        while (*p && *p != '\n')
                        {
                            p++;
                        }
                        if (*p)
                        {
                            p++;
                        }
                        continue;
                    }
                    if (*p == '(')
                    {
                        parens++;
                    }
                    if (*p == ')')
                    {
                        parens--;
                    }
                    expr[ei++] = *p++;
                }
                expr[ei] = '\0';
                if (*p == ')')
                {
                    p++;
                }
                while (*p && strchr(" \t\n,", *p))
                {
                    p++;
                }
                if (strncmp(p, "_z_arg(", 7) == 0)
                {
                    p += 7;
                    while (*p && *p != ')')
                    {
                        p++;
                    }
                    if (*p == ')')
                    {
                        p++;
                    }
                }
                if (emitted)
                {
                    emitter_printf(&ctx->cg.emitter, "\n");
                    lisp_indent(ctx, depth + 1);
                }
                emitter_printf(&ctx->cg.emitter, "(princ ");
                // Replace commas with spaces, collapse multiple spaces
                char cleaned[256];
                int ci = 0;
                int last_was_space = 0;
                for (char *e = expr; *e; e++)
                {
                    char c = (*e == ',') ? ' ' : *e;
                    if (c == ' ')
                    {
                        if (last_was_space)
                        {
                            continue;
                        }
                        last_was_space = 1;
                    }
                    else
                    {
                        last_was_space = 0;
                    }
                    if (ci < 255)
                    {
                        cleaned[ci++] = c;
                    }
                }
                cleaned[ci] = '\0';
                // Trim leading/trailing spaces
                char *start = cleaned;
                while (*start == ' ')
                {
                    start++;
                }
                char *end = start + strlen(start);
                while (end > start && *(end - 1) == ' ')
                {
                    end--;
                }
                *end = '\0';
                // Convert C-style func(args) to Lisp (func args)
                // Only if it looks like a function call (identifier followed by paren)
                int is_call = 0;
                char *scan = start;
                while (*scan && (isalnum(*scan) || *scan == '_' || *scan == '-'))
                {
                    scan++;
                }
                if (*scan == '(')
                {
                    is_call = 1;
                }
                if (is_call)
                {
                    emitter_printf(&ctx->cg.emitter, "(");
                    // emit the identifier part (before the paren)
                    for (char *e = start; *e && *e != '('; e++)
                    {
                        emitter_printf(&ctx->cg.emitter, "%c", *e);
                    }
                    // find matching close paren and emit inner contents
                    char *inner = scan + 1;
                    // Trim trailing close paren
                    char *end_mark = start + strlen(start) - 1;
                    while (end_mark > inner && *end_mark == ')')
                    {
                        end_mark--;
                    }
                    *(end_mark + 1) = '\0';
                    emitter_printf(&ctx->cg.emitter, " %s)", inner);
                }
                else
                {
                    emitter_printf(&ctx->cg.emitter, "%s", start);
                }
                emitter_printf(&ctx->cg.emitter, ")");
                emitted++;
            }
            while (*p && *p != ';')
            {
                p++;
            }
        }
        else
        {
            p++;
        }
    }
    if (has_newline)
    {
        if (emitted)
        {
            emitter_printf(&ctx->cg.emitter, "\n");
            lisp_indent(ctx, depth + 1);
        }
        emitter_printf(&ctx->cg.emitter, "(terpri)");
        emitted++;
    }
    if (!emitted)
    {
        emitter_printf(&ctx->cg.emitter, "nil");
    }
}

static void lisp_emit_stmt(ParserContext *ctx, ASTNode *node, int depth, int *first)
{
    if (!node)
    {
        return;
    }
    if (!*first)
    {
        emitter_printf(&ctx->cg.emitter, "\n");
        lisp_indent(ctx, depth);
    }
    *first = 0;

    switch (node->type)
    {
    case NODE_BLOCK:
        lisp_emit_stmts(ctx, node->block.statements, depth);
        break;
    case NODE_VAR_DECL:
    {
        const char *vname = node->var_decl.name ? node->var_decl.name : "?";
        if (node->var_decl.init_expr)
        {
            emitter_printf(&ctx->cg.emitter, "(setf %s ", vname);
            lisp_emit_expr(ctx, node->var_decl.init_expr, depth);
            emitter_printf(&ctx->cg.emitter, ")");
        }
        break;
    }
    case NODE_EXPR_BINARY:
    {
        // Assignment: a = b
        if (node->binary.op && strcmp(node->binary.op, "=") == 0 && node->binary.left &&
            node->binary.right)
        {
            emitter_printf(&ctx->cg.emitter, "(setf ");
            lisp_emit_expr(ctx, node->binary.left, depth);
            emitter_printf(&ctx->cg.emitter, " ");
            lisp_emit_expr(ctx, node->binary.right, depth);
            emitter_printf(&ctx->cg.emitter, ")");
        }
        else
        {
            lisp_emit_expr(ctx, node, depth);
        }
        break;
    }
    case NODE_EXPR_CALL:
        lisp_emit_call(ctx, node, depth);
        break;
    case NODE_EXPR_UNARY:
    case NODE_EXPR_LITERAL:
    case NODE_EXPR_VAR:
    case NODE_EXPR_MEMBER:
    case NODE_EXPR_INDEX:
    case NODE_EXPR_CAST:
    case NODE_TYPEOF:
    case NODE_EXPR_TUPLE_LITERAL:
    case NODE_EXPR_ARRAY_LITERAL:
    case NODE_EXPR_SLICE:
        lisp_emit_expr(ctx, node, depth);
        break;
    case NODE_RETURN:
    {
        const char *fn = current_function ? current_function : "nil";
        emitter_printf(&ctx->cg.emitter, "(return-from %s", fn);
        if (node->ret.value)
        {
            emitter_printf(&ctx->cg.emitter, " ");
            lisp_emit_expr(ctx, node->ret.value, depth);
        }
        emitter_printf(&ctx->cg.emitter, ")");
        break;
    }
    case NODE_IF:
    {
        emitter_printf(&ctx->cg.emitter, "(if ");
        lisp_emit_expr(ctx, node->if_stmt.condition, depth);
        emitter_printf(&ctx->cg.emitter, "\n");
        lisp_indent(ctx, depth + 1);
        int bf = 1;
        lisp_emit_stmt(ctx, node->if_stmt.then_body, depth + 1, &bf);
        if (node->if_stmt.else_body)
        {
            emitter_printf(&ctx->cg.emitter, "\n");
            lisp_indent(ctx, depth + 1);
            bf = 1;
            lisp_emit_stmt(ctx, node->if_stmt.else_body, depth + 1, &bf);
        }
        emitter_printf(&ctx->cg.emitter, ")");
        break;
    }
    case NODE_WHILE:
    {
        emitter_printf(&ctx->cg.emitter, "(loop while ");
        lisp_emit_expr(ctx, node->while_stmt.condition, depth);
        emitter_printf(&ctx->cg.emitter, " do\n");
        lisp_indent(ctx, depth + 1);
        int bf = 1;
        lisp_emit_stmt(ctx, node->while_stmt.body, depth + 1, &bf);
        emitter_printf(&ctx->cg.emitter, ")");
        break;
    }
    case NODE_FOR:
    {
        // (progn init (loop while condition do body step))
        emitter_printf(&ctx->cg.emitter, "(progn");
        if (node->for_stmt.init)
        {
            emitter_printf(&ctx->cg.emitter, "\n");
            lisp_indent(ctx, depth + 1);
            int bf = 1;
            lisp_emit_stmt(ctx, node->for_stmt.init, depth + 1, &bf);
        }
        emitter_printf(&ctx->cg.emitter, "\n");
        lisp_indent(ctx, depth + 1);
        emitter_printf(&ctx->cg.emitter, "(loop while ");
        if (node->for_stmt.condition)
        {
            lisp_emit_expr(ctx, node->for_stmt.condition, depth);
        }
        else
        {
            emitter_printf(&ctx->cg.emitter, "t");
        }
        emitter_printf(&ctx->cg.emitter, " do\n");
        lisp_indent(ctx, depth + 2);
        int bf = 1;
        lisp_emit_stmt(ctx, node->for_stmt.body, depth + 2, &bf);
        if (node->for_stmt.step)
        {
            emitter_printf(&ctx->cg.emitter, "\n");
            lisp_indent(ctx, depth + 2);
            emitter_printf(&ctx->cg.emitter, "(");
            // Emit step as var = var op val
            lisp_emit_expr(ctx, node->for_stmt.step, depth);
            emitter_printf(&ctx->cg.emitter, ")");
        }
        emitter_printf(&ctx->cg.emitter, ")");
        emitter_printf(&ctx->cg.emitter, ")");
        break;
    }
    case NODE_FOR_RANGE:
    {
        emitter_printf(&ctx->cg.emitter, "(loop for %s",
                       node->for_range.var_name ? node->for_range.var_name : "?");
        if (node->for_range.start)
        {
            emitter_printf(&ctx->cg.emitter, " from ");
            lisp_emit_expr(ctx, node->for_range.start, depth);
        }
        if (node->for_range.end)
        {
            emitter_printf(&ctx->cg.emitter, " %s ", node->for_range.is_inclusive ? "to" : "below");
            lisp_emit_expr(ctx, node->for_range.end, depth);
        }
        if (node->for_range.step)
        {
            emitter_printf(&ctx->cg.emitter, " by %s", node->for_range.step);
        }
        emitter_printf(&ctx->cg.emitter, " do\n");
        lisp_indent(ctx, depth + 1);
        int bf = 1;
        lisp_emit_stmt(ctx, node->for_range.body, depth + 1, &bf);
        emitter_printf(&ctx->cg.emitter, ")");
        break;
    }
    case NODE_LOOP:
    {
        emitter_printf(&ctx->cg.emitter, "(loop do\n");
        lisp_indent(ctx, depth + 1);
        int bf = 1;
        lisp_emit_stmt(ctx, node->loop_stmt.body, depth + 1, &bf);
        emitter_printf(&ctx->cg.emitter, ")");
        break;
    }
    case NODE_BREAK:
        emitter_printf(&ctx->cg.emitter, "(return)");
        break;
    case NODE_CONTINUE:
        emitter_printf(&ctx->cg.emitter, "(continue)");
        break;
    case NODE_MATCH:
    {
        emitter_printf(&ctx->cg.emitter, "(cond");
        const char *expr_var = "__match_val";
        int need_var = 1;
        // For simple variable references, no need for a temp variable
        if (node->match_stmt.expr && node->match_stmt.expr->type == NODE_EXPR_VAR)
        {
            expr_var = node->match_stmt.expr->var_ref.name ? node->match_stmt.expr->var_ref.name
                                                           : "__match_val";
            need_var = 0;
        }
        if (need_var)
        {
            emitter_printf(&ctx->cg.emitter, "\n");
            lisp_indent(ctx, depth + 1);
            emitter_printf(&ctx->cg.emitter, "((let ((%s ", expr_var);
            lisp_emit_expr(ctx, node->match_stmt.expr, depth);
            emitter_printf(&ctx->cg.emitter, "))");
        }
        for (ASTNode *c = node->match_stmt.cases; c; c = c->next)
        {
            emitter_printf(&ctx->cg.emitter, "\n");
            lisp_indent(ctx, depth + 1);
            emitter_printf(&ctx->cg.emitter, "(");
            if (c->match_case.is_default ||
                (c->match_case.pattern && strcmp(c->match_case.pattern, "_") == 0))
            {
                emitter_printf(&ctx->cg.emitter, "t");
            }
            else if (c->match_case.pattern)
            {
                emitter_printf(&ctx->cg.emitter, "(= %s %s)", expr_var, c->match_case.pattern);
            }
            else
            {
                emitter_printf(&ctx->cg.emitter, "t");
            }
            emitter_printf(&ctx->cg.emitter, "\n");
            lisp_indent(ctx, depth + 2);
            int bf = 1;
            lisp_emit_stmt(ctx, c->match_case.body, depth + 2, &bf);
            emitter_printf(&ctx->cg.emitter, ")");
        }
        if (need_var)
        {
            emitter_printf(&ctx->cg.emitter, "))");
        }
        emitter_printf(&ctx->cg.emitter, ")");
        break;
    }
    case NODE_RAW_STMT:
    {
        if (node->raw_stmt.content && lisp_raw_is_print(node->raw_stmt.content))
        {
            emitter_printf(&ctx->cg.emitter, "(progn\n");
            lisp_emit_print(ctx, node->raw_stmt.content, depth + 1);
            emitter_printf(&ctx->cg.emitter, ")");
        }
        else
        {
            emitter_printf(&ctx->cg.emitter, "(error \"raw C: ");
            if (node->raw_stmt.content)
            {
                for (const char *p = node->raw_stmt.content; *p; p++)
                {
                    char c = *p;
                    if (c == '"')
                    {
                        emitter_printf(&ctx->cg.emitter, "\\\"");
                    }
                    else if (c == '\\')
                    {
                        emitter_printf(&ctx->cg.emitter, "\\\\");
                    }
                    else if (c == '\n')
                    {
                        emitter_printf(&ctx->cg.emitter, "\\n");
                    }
                    else if (c >= 32 && c < 127)
                    {
                        emitter_printf(&ctx->cg.emitter, "%c", c);
                    }
                }
            }
            emitter_printf(&ctx->cg.emitter, "\")");
        }
        break;
    }
    case NODE_ASSERT:
    {
        const char *fn = current_function ? current_function : "nil";
        emitter_printf(&ctx->cg.emitter, "(if (not ");
        if (node->assert_stmt.condition)
        {
            lisp_emit_expr(ctx, node->assert_stmt.condition, depth);
        }
        else
        {
            emitter_printf(&ctx->cg.emitter, "nil");
        }
        emitter_printf(&ctx->cg.emitter, ") (return-from %s 1))", fn);
        break;
    }
    case NODE_EXPECT:
        // Non-fatal assert — just a warning
        break;
    case NODE_REPL_PRINT:
        // Implicit REPL print of expression value
        emitter_printf(&ctx->cg.emitter, "(print ");
        if (node->ret.value)
        {
            lisp_emit_expr(ctx, node->ret.value, depth);
        }
        emitter_printf(&ctx->cg.emitter, ")");
        break;
    case NODE_INCLUDE:
    case NODE_IMPORT:
        break;
    default:
        emitter_printf(&ctx->cg.emitter, "; unhandled stmt type %d", node->type);
        break;
    }
}

static void lisp_emit_stmts(ParserContext *ctx, ASTNode *stmts, int depth)
{
    if (!stmts)
    {
        return;
    }
    // Count statements
    int count = 0;
    for (ASTNode *s = stmts; s; s = s->next)
    {
        count++;
    }
    if (count > 1)
    {
        emitter_printf(&ctx->cg.emitter, "(progn\n");
        int first = 1;
        for (ASTNode *s = stmts; s; s = s->next)
        {
            if (!first)
            {
                emitter_printf(&ctx->cg.emitter, "\n");
                lisp_indent(ctx, depth + 1);
            }
            first = 0;
            lisp_emit_stmt(ctx, s, depth + 1, &first);
        }
        emitter_printf(&ctx->cg.emitter, ")");
    }
    else
    {
        int first = 1;
        lisp_emit_stmt(ctx, stmts, depth + 1, &first);
    }
}

static void lisp_emit_func(ParserContext *ctx, ASTNode *node, int depth, int *first)
{
    if (!*first)
    {
        emitter_printf(&ctx->cg.emitter, "\n\n");
        lisp_indent(ctx, depth);
    }
    *first = 0;

    const char *fname = node->func.name ? node->func.name : "anonymous";
    current_function = fname;
    emitter_printf(&ctx->cg.emitter, "(defun %s (", fname);
    // Parameters
    for (int i = 0; i < node->func.arg_count; i++)
    {
        if (i > 0)
        {
            emitter_printf(&ctx->cg.emitter, " ");
        }
        if (node->func.param_names && node->func.param_names[i])
        {
            emitter_printf(&ctx->cg.emitter, "%s", node->func.param_names[i]);
        }
        else
        {
            emitter_printf(&ctx->cg.emitter, "p%d", i);
        }
    }
    emitter_printf(&ctx->cg.emitter, ")\n");
    lisp_indent(ctx, depth + 1);
    if (node->func.body)
    {
        if (node->func.body->type == NODE_BLOCK)
        {
            // Scan for variable declarations to emit as (let (vars...) ...)
            ASTNode *stmts = node->func.body->block.statements;
            char var_names[128][64];
            int var_count = 0;
            for (ASTNode *s = stmts; s; s = s->next)
            {
                if (s->type == NODE_VAR_DECL && s->var_decl.name && var_count < 128)
                {
                    strncpy(var_names[var_count], s->var_decl.name, 63);
                    var_names[var_count][63] = '\0';
                    var_count++;
                }
            }
            if (var_count > 0)
            {
                emitter_printf(&ctx->cg.emitter, "(let (");
                for (int i = 0; i < var_count; i++)
                {
                    if (i > 0)
                    {
                        emitter_printf(&ctx->cg.emitter, " ");
                    }
                    emitter_printf(&ctx->cg.emitter, "%s", var_names[i]);
                }
                emitter_printf(&ctx->cg.emitter, ")\n");
                lisp_indent(ctx, depth + 2);
            }
            emitter_printf(&ctx->cg.emitter, "(declare (ignorable ");
            for (int i = 0; i < node->func.arg_count; i++)
            {
                if (i > 0)
                {
                    emitter_printf(&ctx->cg.emitter, " ");
                }
                emitter_printf(&ctx->cg.emitter, "%s",
                               node->func.param_names ? node->func.param_names[i] : "?");
            }
            if (var_count > 0)
            {
                emitter_printf(&ctx->cg.emitter, " ");
                for (int i = 0; i < var_count; i++)
                {
                    if (i > 0 || node->func.arg_count > 0)
                    {
                        emitter_printf(&ctx->cg.emitter, " ");
                    }
                    emitter_printf(&ctx->cg.emitter, "%s", var_names[i]);
                }
            }
            emitter_printf(&ctx->cg.emitter, "))\n");
            lisp_indent(ctx, depth + (var_count > 0 ? 2 : 1));
            lisp_emit_stmts(ctx, stmts, depth + (var_count > 0 ? 2 : 1));
            if (var_count > 0)
            {
                emitter_printf(&ctx->cg.emitter, ")");
            }
        }
        else
        {
            int bf = 1;
            lisp_emit_stmt(ctx, node->func.body, depth + 1, &bf);
        }
    }
    emitter_printf(&ctx->cg.emitter, ")");
    current_function = NULL;
}

static void lisp_emit_root(ParserContext *ctx, ASTNode *node, int depth)
{
    if (!node || node->type != NODE_ROOT)
    {
        return;
    }
    int first = 1;
    int has_main = 0;
    for (ASTNode *c = node->root.children; c; c = c->next)
    {
        switch (c->type)
        {
        case NODE_FUNCTION:
            lisp_emit_func(ctx, c, depth, &first);
            if (c->func.name && strcmp(c->func.name, "main") == 0)
            {
                has_main = 1;
            }
            break;
        case NODE_IMPORT:
            break;
        default:
            break;
        }
    }
    // Call main at the bottom
    if (has_main)
    {
        emitter_printf(&ctx->cg.emitter, "\n\n(main)\n");
    }
}

static void lisp_emit_program(ParserContext *ctx, ASTNode *root)
{
    // Emit a shebang and a quick header
    emitter_printf(&ctx->cg.emitter, "#!/usr/bin/env sbcl --script\n");
    lisp_emit_root(ctx, root, 0);
    emitter_printf(&ctx->cg.emitter, "\n");
}

static void lisp_emit_preamble(ParserContext *ctx)
{
    (void)ctx;
}

static const BackendOptAlias lisp_aliases[] = {
    {"--backend-full-content", "full-content", NULL},
    {NULL, NULL, NULL},
};

static const CodegenBackend lisp_backend = {
    .name = "lisp",
    .extension = ".lisp",
    .emit_program = lisp_emit_program,
    .emit_preamble = lisp_emit_preamble,
    .needs_cc = 0,
    .aliases = lisp_aliases,
};

void codegen_register_lisp_backend(void)
{
    codegen_register_backend(&lisp_backend);
}
