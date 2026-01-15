
#include "zprep_plugin.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void skip_whitespace(const char **p)
{
    while (**p && isspace(**p))
    {
        (*p)++;
    }
}

static void parse_lisp_expr(const char **p, const ZApi *api, int depth);

static void parse_atom(const char **p, const ZApi *api)
{
    FILE *out = api->out;
    const char *start = *p;

    // Check for unexpected closer
    if (**p == ')')
    {
        fprintf(stderr, "Error: Unexpected ')' at %s:%d\n", api->filename, api->current_line);
        exit(1);
    }

    // Number
    if (isdigit(**p) || (**p == '-' && isdigit((*p)[1])))
    {
        if (**p == '-')
        {
            (*p)++;
        }
        while (**p && isdigit(**p))
        {
            (*p)++;
        }
        fprintf(out, "l_num(");
        fwrite(start, 1, *p - start, out);
        fprintf(out, ")");
        return;
    }

    if (**p == '"')
    {
        (*p)++;
        while (**p && **p != '"')
        {
            if (**p == '\\')
            {
                (*p)++;
            }
            (*p)++;
        }
        if (**p == '"')
        {
            (*p)++;
        }
        fprintf(out, "l_nil()");
        return;
    }

    // Symbol
    while (**p && !isspace(**p) && **p != ')' && **p != '(')
    {
        (*p)++;
    }
    fwrite(start, 1, *p - start, out);
}

static void parse_lisp_list(const char **p, const ZApi *api, int depth)
{
    FILE *out = api->out;
    (*p)++; // consume (
    skip_whitespace(p);

    if (!**p)
    {
        fprintf(stderr, "Error: Unclosed parenthesis (unexpected EOF) at %s:%d\n", api->filename,
                api->current_line);
        exit(1);
    }

    const char *op_start = *p;
    while (**p && !isspace(**p) && **p != ')' && **p != '(')
    {
        (*p)++;
    }
    int op_len = *p - op_start;

    // Arithmetic
    if (op_len == 1 && strchr("+-*/", *op_start))
    {
        char op_char = *op_start;
        char *func = "l_add";
        if (op_char == '-')
        {
            func = "l_sub";
        }
        if (op_char == '*')
        {
            func = "l_mul";
        }
        if (op_char == '/')
        {
            func = "l_div";
        }

        fprintf(out, "%s(", func);
        skip_whitespace(p);
        parse_lisp_expr(p, api, depth + 1);
        fprintf(out, ", ");
        skip_whitespace(p);
        parse_lisp_expr(p, api, depth + 1);
        fprintf(out, ")");
        skip_whitespace(p);
        while (**p && **p != ')')
        {
            if (**p == '(')
            {
                int d = 1;
                (*p)++;
                while (d > 0 && **p)
                {
                    if (**p == '(')
                    {
                        d++;
                    }
                    if (**p == ')')
                    {
                        d--;
                    }
                    (*p)++;
                }
            }
            else
            {
                while (**p && !isspace(**p) && **p != ')')
                {
                    (*p)++;
                }
            }
            skip_whitespace(p);
        }
    }
    else if ((op_len == 1 && strchr("<>", *op_start)) || (op_len == 2 && strchr("=!", *op_start)))
    {
        char func[32] = "l_eq";
        if (op_len == 1 && *op_start == '<')
        {
            strcpy(func, "l_lt");
        }
        else if (op_len == 1 && *op_start == '>')
        {
            strcpy(func, "l_gt");
        }

        fprintf(out, "%s(", func);
        skip_whitespace(p);
        parse_lisp_expr(p, api, depth + 1);
        fprintf(out, ", ");
        skip_whitespace(p);
        parse_lisp_expr(p, api, depth + 1);
        fprintf(out, ")");
    }
    // Cons/Car/Cdr
    else if (op_len == 4 && strncmp(op_start, "cons", 4) == 0)
    {
        fprintf(out, "l_cons(");
        skip_whitespace(p);
        parse_lisp_expr(p, api, depth + 1);
        fprintf(out, ", ");
        skip_whitespace(p);
        parse_lisp_expr(p, api, depth + 1);
        fprintf(out, ")");
    }
    else if (op_len == 3 && strncmp(op_start, "car", 3) == 0)
    {
        fprintf(out, "l_car(");
        skip_whitespace(p);
        parse_lisp_expr(p, api, depth + 1);
        fprintf(out, ")");
    }
    else if (op_len == 3 && strncmp(op_start, "cdr", 3) == 0)
    {
        fprintf(out, "l_cdr(");
        skip_whitespace(p);
        parse_lisp_expr(p, api, depth + 1);
        fprintf(out, ")");
    }
    // List - stub
    else if (op_len == 4 && strncmp(op_start, "list", 4) == 0)
    {
        fprintf(out, "l_nil()");
    }
    // print/println
    else if (strncmp(op_start, "print", 5) == 0)
    {
        skip_whitespace(p);
        if (**p == '"')
        {
            // Raw string print
            fprintf(out, "printf(\"");
            (*p)++;
            while (**p && **p != '"')
            {
                if (**p == '\\')
                {
                    fprintf(out, "\\");
                    (*p)++;
                    if (**p)
                    {
                        fprintf(out, "%c", *(*p)++);
                    }
                }
                else
                {
                    fprintf(out, "%c", *(*p)++);
                }
            }
            fprintf(out, "%s\")", op_start[5] == 'l' ? "\\n" : "");
            if (**p == '"')
            {
                (*p)++;
            }
        }
        else
        {
            // LVal print
            fprintf(out, "l_print(");
            parse_lisp_expr(p, api, depth + 1);
            fprintf(out, ");%s", op_start[5] == 'l' ? " printf(\"\\n\");" : "");
        }
    }
    // if
    else if (op_len == 2 && strncmp(op_start, "if", 2) == 0)
    {
        fprintf(out, "(l_truthy(");
        skip_whitespace(p);
        parse_lisp_expr(p, api, depth + 1);
        fprintf(out, ") ? ");
        skip_whitespace(p);
        parse_lisp_expr(p, api, depth + 1);
        fprintf(out, " : ");
        skip_whitespace(p);
        if (**p != ')')
        {
            parse_lisp_expr(p, api, depth + 1);
        }
        else
        {
            fprintf(out, "l_nil()");
        }
        fprintf(out, ")");
    }
    // let
    else if (op_len == 3 && strncmp(op_start, "let", 3) == 0)
    {
        fprintf(out, "({\n");
        skip_whitespace(p);
        // Bindings...
        if (**p == '(')
        {
            (*p)++;
            skip_whitespace(p);
            while (**p && **p != ')')
            {
                if (**p == '(')
                {
                    (*p)++;
                    skip_whitespace(p);
                    const char *vstart = *p;
                    while (**p && !isspace(**p) && **p != ')')
                    {
                        (*p)++;
                    }
                    fprintf(out, "LVal ");
                    fwrite(vstart, 1, *p - vstart, out);
                    fprintf(out, " = ");
                    skip_whitespace(p);
                    parse_lisp_expr(p, api, depth + 1);
                    fprintf(out, ";\n");
                    skip_whitespace(p);
                    if (**p == ')')
                    {
                        (*p)++;
                    }
                }
                skip_whitespace(p);
            }
            if (!**p)
            {
                fprintf(stderr, "Error: Unclosed let bindings at %s:%d\n", api->filename,
                        api->current_line);
                exit(1);
            }
            if (**p == ')')
            {
                (*p)++;
            }
        }
        // Body...
        skip_whitespace(p);
        while (**p && **p != ')')
        {
            parse_lisp_expr(p, api, depth + 1);
            fprintf(out, ";\n");
            skip_whitespace(p);
        }
        fprintf(out, "})");
    }
    // defun
    else if (op_len == 5 && strncmp(op_start, "defun", 5) == 0)
    {
        skip_whitespace(p);
        const char *nstart = *p;
        while (**p && !isspace(**p) && **p != '(')
        {
            (*p)++;
        }
        fprintf(out, "auto LVal ");
        fwrite(nstart, 1, *p - nstart, out);
        fprintf(out, "(");
        skip_whitespace(p);
        (*p)++; // (
        skip_whitespace(p);
        int first = 1;
        while (**p && **p != ')')
        {
            if (!first)
            {
                fprintf(out, ", ");
            }
            first = 0;
            fprintf(out, "LVal ");
            const char *astart = *p;
            while (**p && !isspace(**p) && **p != ')')
            {
                (*p)++;
            }
            fwrite(astart, 1, *p - astart, out);
            skip_whitespace(p);
        }
        (*p)++; // )
        fprintf(out, ") {\n return ({\n");
        skip_whitespace(p);
        while (**p && **p != ')')
        {
            parse_lisp_expr(p, api, depth + 1);
            fprintf(out, ";\n");
            skip_whitespace(p);
        }
        fprintf(out, "});\n}");
    }
    // Function call
    else
    {
        fwrite(op_start, 1, op_len, out);
        fprintf(out, "(");
        skip_whitespace(p);
        int first = 1;
        while (**p && **p != ')')
        {
            if (!first)
            {
                fprintf(out, ", ");
            }
            first = 0;
            parse_lisp_expr(p, api, depth + 1);
            skip_whitespace(p);
        }
        fprintf(out, ")");
    }

    // consume remaining args or check for close
    while (**p && **p != ')')
    {
        (*p)++;
    }

    if (!**p)
    {
        fprintf(stderr, "Error: Unclosed parenthesis (end of list) at %s:%d\n", api->filename,
                api->current_line);
        exit(1);
    }
    if (**p == ')')
    {
        (*p)++;
    }
}

static void parse_lisp_expr(const char **p, const ZApi *api, int depth)
{
    skip_whitespace(p);
    if (!**p)
    {
        return; // Should not happen if called correctly
    }
    if (**p == '(')
    {
        parse_lisp_list(p, api, depth);
    }
    else
    {
        parse_atom(p, api);
    }
}

void lisp_transpile(const char *input_body, const ZApi *api)
{
    FILE *out = api->out;
    const char *p = input_body;

    static int runtime_emitted = 0;
    if (!runtime_emitted && api->hoist_out)
    {
        FILE *h = api->hoist_out;
        fprintf(h, "/* Lisp Runtime */\n");
        fprintf(h, "typedef enum { L_NUM, L_PAIR, L_NIL } LType;\n");
        fprintf(h, "typedef struct LVal { LType type; union { long num; struct { "
                   "struct LVal *car; "
                   "struct LVal *cdr; } pair; }; } *LVal;\n");
        fprintf(h, "static struct LVal _nil = { L_NIL }; static LVal LNIL = &_nil;\n");
        fprintf(h, "static LVal nil = &_nil;\n"); // Use static for file scope
        fprintf(h, "static LVal l_num(long n) { LVal v = malloc(sizeof(struct LVal)); "
                   "v->type=L_NUM; v->num=n; return v; }\n");
        fprintf(h, "static LVal l_nil() { return LNIL; }\n");
        fprintf(h, "static LVal l_cons(LVal a, LVal b) { LVal v = "
                   "malloc(sizeof(struct LVal)); "
                   "v->type=L_PAIR; v->pair.car=a; v->pair.cdr=b; return v; }\n");
        fprintf(h, "static LVal l_car(LVal v) { return (v && v->type==L_PAIR) ? "
                   "v->pair.car : LNIL; }\n");
        fprintf(h, "static LVal l_cdr(LVal v) { return (v && v->type==L_PAIR) ? "
                   "v->pair.cdr : LNIL; }\n");
        fprintf(h, "static int l_truthy(LVal v) { return (v && v->type!=L_NIL); }\n");
        fprintf(h, "static LVal l_add(LVal a, LVal b) { long "
                   "x=(a&&a->type==L_NUM)?a->num:0; long "
                   "y=(b&&b->type==L_NUM)?b->num:0; return l_num(x+y); }\n");
        fprintf(h, "static LVal l_sub(LVal a, LVal b) { long "
                   "x=(a&&a->type==L_NUM)?a->num:0; long "
                   "y=(b&&b->type==L_NUM)?b->num:0; return l_num(x-y); }\n");
        fprintf(h, "static LVal l_mul(LVal a, LVal b) { long "
                   "x=(a&&a->type==L_NUM)?a->num:0; long "
                   "y=(b&&b->type==L_NUM)?b->num:0; return l_num(x*y); }\n");
        fprintf(h, "static LVal l_div(LVal a, LVal b) { long "
                   "x=(a&&a->type==L_NUM)?a->num:0; long "
                   "y=(b&&b->type==L_NUM)?b->num:0; return l_num(y?x/y:0); }\n");
        fprintf(h, "static LVal l_lt(LVal a, LVal b) { long "
                   "x=(a&&a->type==L_NUM)?a->num:0; long "
                   "y=(b&&b->type==L_NUM)?b->num:0; return (x<y)?l_num(1):LNIL; }\n");
        fprintf(h, "static void l_print(LVal v) { \n");
        fprintf(h, "  if(!v || v->type==L_NIL) printf(\"nil\");\n");
        fprintf(h, "  else if(v->type==L_NUM) printf(\"%%ld\", v->num);\n");
        fprintf(h, "  else if(v->type==L_PAIR) { printf(\"(\"); "
                   "l_print(v->pair.car); printf(\" . "
                   "\"); l_print(v->pair.cdr); printf(\")\"); }\n");
        fprintf(h, "}\n");

        runtime_emitted = 1;
    }

    fprintf(out, "({\n");

    while (*p)
    {
        skip_whitespace(&p);
        if (!*p)
        {
            break;
        }

        if (*p == ')')
        {
            fprintf(stderr, "Error: Unexpected ')' at top level at %s:%d\n", api->filename,
                    api->current_line);
            exit(1);
        }

        fprintf(out, "    ");
        parse_lisp_expr(&p, api, 1);
        fprintf(out, ";\n");
    }
    fprintf(out, "})\n");
}

ZPlugin lisp_plugin = {.name = "lisp", .fn = lisp_transpile};

PLUGINAPI ZPlugin *z_plugin_init(void)
{
    return &lisp_plugin;
}
