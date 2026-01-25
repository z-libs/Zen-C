
#include "parser.h"
#include "zprep.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static ASTNode *generate_derive_impls(ParserContext *ctx, ASTNode *strct, char **traits, int count);

// Main parsing entry point
ASTNode *parse_program_nodes(ParserContext *ctx, Lexer *l)
{
    ASTNode *h = 0, *tl = 0;
    while (1)
    {
        skip_comments(l);
        Token t = lexer_peek(l);
        if (t.type == TOK_EOF)
        {
            break;
        }

        if (t.type == TOK_COMPTIME)
        {
            ASTNode *gen = parse_comptime(ctx, l);
            if (gen)
            {
                if (!h)
                {
                    h = gen;
                }
                else
                {
                    tl->next = gen;
                }
                if (!tl)
                {
                    tl = gen;
                }
                while (tl->next)
                {
                    tl = tl->next;
                }
            }
            continue;
        }

        ASTNode *s = 0;

        int attr_must_use = 0;
        int attr_deprecated = 0;
        int attr_inline = 0;
        int attr_pure = 0;
        int attr_noreturn = 0;
        int attr_cold = 0;
        int attr_hot = 0;
        int attr_packed = 0;
        int attr_align = 0;
        int attr_noinline = 0;
        int attr_constructor = 0;
        int attr_destructor = 0;
        int attr_unused = 0;
        int attr_weak = 0;
        int attr_export = 0;
        int attr_comptime = 0;
        int attr_cuda_global = 0; // @global -> __global__
        int attr_cuda_device = 0; // @device -> __device__
        int attr_cuda_host = 0;   // @host -> __host__
        char *deprecated_msg = NULL;
        char *attr_section = NULL;

        char *derived_traits[32];
        int derived_count = 0;

        while (t.type == TOK_AT)
        {
            lexer_next(l);
            Token attr = lexer_next(l);
            if (attr.type != TOK_IDENT && attr.type != TOK_COMPTIME)
            {
                zpanic_at(attr, "Expected attribute name after @");
            }

            if (0 == strncmp(attr.start, "must_use", 8) && 8 == attr.len)
            {
                attr_must_use = 1;
            }
            else if (0 == strncmp(attr.start, "deprecated", 10) && 10 == attr.len)
            {
                attr_deprecated = 1;
                if (lexer_peek(l).type == TOK_LPAREN)
                {
                    lexer_next(l);
                    Token msg = lexer_next(l);
                    if (msg.type == TOK_STRING)
                    {
                        deprecated_msg = xmalloc(msg.len - 1);
                        strncpy(deprecated_msg, msg.start + 1, msg.len - 2);
                        deprecated_msg[msg.len - 2] = 0;
                    }
                    if (lexer_next(l).type != TOK_RPAREN)
                    {
                        zpanic_at(lexer_peek(l), "Expected ) after deprecated message");
                    }
                }
            }
            else if (0 == strncmp(attr.start, "inline", 6) && 6 == attr.len)
            {
                attr_inline = 1;
            }
            else if (0 == strncmp(attr.start, "noinline", 8) && 8 == attr.len)
            {
                attr_noinline = 1;
            }
            else if (0 == strncmp(attr.start, "pure", 4) && 4 == attr.len)
            {
                attr_pure = 1;
            }
            else if (0 == strncmp(attr.start, "noreturn", 8) && 8 == attr.len)
            {
                attr_noreturn = 1;
            }
            else if (0 == strncmp(attr.start, "cold", 4) && 4 == attr.len)
            {
                attr_cold = 1;
            }
            else if (0 == strncmp(attr.start, "hot", 3) && 3 == attr.len)
            {
                attr_hot = 1;
            }
            else if (0 == strncmp(attr.start, "constructor", 11) && 11 == attr.len)
            {
                attr_constructor = 1;
            }
            else if (0 == strncmp(attr.start, "destructor", 10) && 10 == attr.len)
            {
                attr_destructor = 1;
            }
            else if (0 == strncmp(attr.start, "unused", 6) && 6 == attr.len)
            {
                attr_unused = 1;
            }
            else if (0 == strncmp(attr.start, "weak", 4) && 4 == attr.len)
            {
                attr_weak = 1;
            }
            else if (0 == strncmp(attr.start, "export", 6) && 6 == attr.len)
            {
                attr_export = 1;
            }
            else if (0 == strncmp(attr.start, "comptime", 8) && 8 == attr.len)
            {
                attr_comptime = 1;
            }
            else if (0 == strncmp(attr.start, "section", 7) && 7 == attr.len)
            {
                if (lexer_peek(l).type == TOK_LPAREN)
                {
                    lexer_next(l);
                    Token sec = lexer_next(l);
                    if (sec.type == TOK_STRING)
                    {
                        attr_section = xmalloc(sec.len - 1);
                        strncpy(attr_section, sec.start + 1, sec.len - 2);
                        attr_section[sec.len - 2] = 0;
                    }
                    if (lexer_next(l).type != TOK_RPAREN)
                    {
                        zpanic_at(lexer_peek(l), "Expected ) after section name");
                    }
                }
                else
                {
                    zpanic_at(lexer_peek(l), "@section requires a name: @section(\"name\")");
                }
            }
            else if (0 == strncmp(attr.start, "packed", 6) && 6 == attr.len)
            {
                attr_packed = 1;
            }
            else if (0 == strncmp(attr.start, "align", 5) && 5 == attr.len)
            {
                if (lexer_peek(l).type == TOK_LPAREN)
                {
                    lexer_next(l);
                    Token num = lexer_next(l);
                    if (num.type == TOK_INT)
                    {
                        attr_align = atoi(num.start);
                    }
                    if (lexer_next(l).type != TOK_RPAREN)
                    {
                        zpanic_at(lexer_peek(l), "Expected ) after align value");
                    }
                }
                else
                {
                    zpanic_at(lexer_peek(l), "@align requires a value: @align(N)");
                }
            }
            else if (0 == strncmp(attr.start, "derive", 6) && 6 == attr.len)
            {
                if (lexer_peek(l).type == TOK_LPAREN)
                {
                    lexer_next(l);
                    while (1)
                    {
                        Token t = lexer_next(l);
                        if (t.type != TOK_IDENT)
                        {
                            zpanic_at(t, "Expected trait name in @derive");
                        }
                        if (derived_count < 32)
                        {
                            derived_traits[derived_count++] = token_strdup(t);
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
                    if (lexer_next(l).type != TOK_RPAREN)
                    {
                        zpanic_at(lexer_peek(l), "Expected ) after derive traits");
                    }
                }
                else
                {
                    zpanic_at(lexer_peek(l), "@derive requires traits: @derive(Debug, Clone)");
                }
            }
            else
            {
                // Checking for CUDA attributes...
                if (0 == strncmp(attr.start, "global", 6) && 6 == attr.len)
                {
                    attr_cuda_global = 1;
                }
                else if (0 == strncmp(attr.start, "device", 6) && 6 == attr.len)
                {
                    attr_cuda_device = 1;
                }
                else if (0 == strncmp(attr.start, "host", 4) && 4 == attr.len)
                {
                    attr_cuda_host = 1;
                }
                else
                {
                    zwarn_at(attr, "Unknown attribute: %.*s", attr.len, attr.start);
                }
            }

            t = lexer_peek(l);
        }

        if (t.type == TOK_PREPROC)
        {
            lexer_next(l);
            char *content = xmalloc(t.len + 2);
            strncpy(content, t.start, t.len);
            content[t.len] = '\n';
            content[t.len + 1] = 0;
            s = ast_create(NODE_RAW_STMT);
            s->raw_stmt.content = content;
        }
        else if (t.type == TOK_DEF)
        {
            s = parse_def(ctx, l);
        }
        else if (t.type == TOK_IDENT)
        {
            // Inline function: inline fn name(...) { }
            if (0 == strncmp(t.start, "inline", 6) && 6 == t.len)
            {
                lexer_next(l);
                Token next = lexer_peek(l);
                if (next.type == TOK_IDENT && 2 == next.len && 0 == strncmp(next.start, "fn", 2))
                {
                    s = parse_function(ctx, l, 0);
                    attr_inline = 1;
                }
                else
                {
                    zpanic_at(next, "Expected 'fn' after 'inline'");
                }
            }
            else if (0 == strncmp(t.start, "fn", 2) && 2 == t.len)
            {
                s = parse_function(ctx, l, 0);
            }
            else if (0 == strncmp(t.start, "struct", 6) && 6 == t.len)
            {
                s = parse_struct(ctx, l, 0);
                if (s && s->type == NODE_STRUCT)
                {
                    s->strct.is_packed = attr_packed;
                    s->strct.align = attr_align;

                    if (derived_count > 0)
                    {
                        ASTNode *impls =
                            generate_derive_impls(ctx, s, derived_traits, derived_count);
                        s->next = impls;
                    }
                }
            }
            else if (0 == strncmp(t.start, "enum", 4) && 4 == t.len)
            {
                s = parse_enum(ctx, l);
                if (s && s->type == NODE_ENUM)
                {
                    if (derived_count > 0)
                    {
                        ASTNode *impls =
                            generate_derive_impls(ctx, s, derived_traits, derived_count);
                        s->next = impls;
                    }
                }
            }
            else if (t.len == 4 && strncmp(t.start, "impl", 4) == 0)
            {
                s = parse_impl(ctx, l);
            }
            else if (t.len == 5 && strncmp(t.start, "trait", 5) == 0)
            {
                s = parse_trait(ctx, l);
            }
            else if (t.len == 7 && strncmp(t.start, "include", 7) == 0)
            {
                s = parse_include(ctx, l);
            }
            else if (t.len == 6 && strncmp(t.start, "import", 6) == 0)
            {
                s = parse_import(ctx, l);
            }
            else if (t.len == 3 && strncmp(t.start, "let", 3) == 0)
            {
                s = parse_var_decl(ctx, l);
            }
            else if (t.len == 3 && strncmp(t.start, "var", 3) == 0)
            {
                zpanic_at(t, "'var' is deprecated. Use 'let' instead.");
            }
            else if (t.len == 5 && strncmp(t.start, "const", 5) == 0)
            {
                zpanic_at(t, "'const' for declarations is deprecated. Use 'def' for constants or "
                             "'let x: const T' for read-only variables.");
            }
            else if (t.len == 6 && strncmp(t.start, "extern", 6) == 0)
            {
                lexer_next(l);

                Token peek = lexer_peek(l);
                if (peek.type == TOK_IDENT && peek.len == 2 && strncmp(peek.start, "fn", 2) == 0)
                {
                    s = parse_function(ctx, l, 0);
                }
                else
                {
                    while (1)
                    {
                        Token sym = lexer_next(l);
                        if (sym.type != TOK_IDENT)
                        {
                            break;
                        }

                        char *name = token_strdup(sym);
                        register_extern_symbol(ctx, name);

                        Token next = lexer_peek(l);
                        if (next.type == TOK_COMMA)
                        {
                            lexer_next(l);
                        }
                        else
                        {
                            break;
                        }
                    }

                    if (lexer_peek(l).type == TOK_SEMICOLON)
                    {
                        lexer_next(l);
                    }
                    continue;
                }
            }
            else if (0 == strncmp(t.start, "type", 4) && 4 == t.len)
            {
                s = parse_type_alias(ctx, l);
            }
            else if (0 == strncmp(t.start, "raw", 3) && 3 == t.len)
            {
                lexer_next(l);
                if (lexer_peek(l).type != TOK_LBRACE)
                {
                    zpanic_at(lexer_peek(l), "Expected { after raw");
                }
                lexer_next(l);

                const char *start = l->src + l->pos;

                int depth = 1;
                while (depth > 0)
                {
                    Token t = lexer_next(l);
                    if (t.type == TOK_EOF)
                    {
                        zpanic_at(t, "Unexpected EOF in raw block");
                    }
                    if (t.type == TOK_LBRACE)
                    {
                        depth++;
                    }
                    if (t.type == TOK_RBRACE)
                    {
                        depth--;
                    }
                }

                const char *end = l->src + l->pos - 1;
                size_t len = end - start;

                char *content = xmalloc(len + 1);
                memcpy(content, start, len);
                content[len] = 0;

                s = ast_create(NODE_RAW_STMT);
                s->raw_stmt.content = content;
            }
            else
            {
                lexer_next(l);
            }
        }
        else if (t.type == TOK_ALIAS)
        {
            s = parse_type_alias(ctx, l);
        }
        else if (t.type == TOK_ASYNC)
        {
            lexer_next(l);
            Token next = lexer_peek(l);
            if (0 == strncmp(next.start, "fn", 2) && 2 == next.len)
            {
                s = parse_function(ctx, l, 1);
                if (s)
                {
                    s->func.is_async = 1;
                }
            }
            else
            {
                zpanic_at(next, "Expected 'fn' after 'async'");
            }
        }

        else if (t.type == TOK_UNION)
        {
            s = parse_struct(ctx, l, 1);
        }
        else if (t.type == TOK_TRAIT)
        {
            s = parse_trait(ctx, l);
        }
        else if (t.type == TOK_IMPL)
        {
            s = parse_impl(ctx, l);
        }
        else if (t.type == TOK_TEST)
        {
            s = parse_test(ctx, l);
        }
        else
        {
            lexer_next(l);
        }

        if (s && s->type == NODE_FUNCTION)
        {
            s->func.must_use = attr_must_use;
            s->func.is_inline = attr_inline || s->func.is_inline;
            s->func.noinline = attr_noinline;
            s->func.constructor = attr_constructor;
            s->func.destructor = attr_destructor;
            s->func.unused = attr_unused;
            s->func.weak = attr_weak;
            s->func.is_export = attr_export;
            s->func.cold = attr_cold;
            s->func.hot = attr_hot;
            s->func.noreturn = attr_noreturn;
            s->func.pure = attr_pure;
            s->func.section = attr_section;
            s->func.is_comptime = attr_comptime;
            s->func.cuda_global = attr_cuda_global;
            s->func.cuda_device = attr_cuda_device;
            s->func.cuda_host = attr_cuda_host;

            if (attr_deprecated && s->func.name)
            {
                register_deprecated_func(ctx, s->func.name, deprecated_msg);
            }

            if (attr_must_use && s->func.name)
            {
                FuncSig *sig = find_func(ctx, s->func.name);
                if (sig)
                {
                    sig->must_use = 1;
                }
            }
        }

        if (s)
        {
            if (!h)
            {
                h = s;
            }
            else
            {
                tl->next = s;
            }
            tl = s;
            while (tl->next)
            {
                tl = tl->next;
            }
        }
    }
    return h;
}

ASTNode *parse_program(ParserContext *ctx, Lexer *l)
{
    g_parser_ctx = ctx;
    enter_scope(ctx);
    register_builtins(ctx);

    ASTNode *r = ast_create(NODE_ROOT);
    r->root.children = parse_program_nodes(ctx, l);
    return r;
}

static ASTNode *generate_derive_impls(ParserContext *ctx, ASTNode *strct, char **traits, int count)
{
    ASTNode *head = NULL, *tail = NULL;
    char *name = strct->strct.name;

    for (int i = 0; i < count; i++)
    {
        char *trait = traits[i];
        char *code = NULL;

        if (0 == strcmp(trait, "Clone"))
        {
            code = xmalloc(1024);
            sprintf(code, "impl %s { fn clone(self) -> %s { return *self; } }", name, name);
        }
        else if (0 == strcmp(trait, "Eq"))
        {
            char body[4096];
            body[0] = 0;

            if (strct->type == NODE_ENUM)
            {
                // Simple Enum equality (tag comparison)
                // Generate Eq impl for Enum

                sprintf(body, "return self.tag == other.tag;");
            }
            else
            {
                ASTNode *f = strct->strct.fields;
                int first = 1;
                strcat(body, "return ");
                while (f)
                {
                    if (f->type == NODE_FIELD)
                    {
                        char *fn = f->field.name;
                        char *ft = f->field.type;
                        if (!first)
                        {
                            strcat(body, " && ");
                        }
                        char cmp[256];

                        // Detect pointer using type_info OR string check (fallback)
                        int is_ptr = 0;
                        if (f->type_info && f->type_info->kind == TYPE_POINTER)
                        {
                            is_ptr = 1;
                        }
                        // Fallback: check if type string ends with '*'
                        if (!is_ptr && ft && strchr(ft, '*'))
                        {
                            is_ptr = 1;
                        }

                        // Only look up struct def for non-pointer types
                        ASTNode *fdef = is_ptr ? NULL : find_struct_def(ctx, ft);

                        if (!is_ptr && fdef && fdef->type == NODE_ENUM)
                        {
                            // Enum field: compare tags
                            sprintf(cmp, "self.%s.tag == other.%s.tag", fn, fn);
                        }
                        else if (!is_ptr && fdef && fdef->type == NODE_STRUCT)
                        {
                            // Struct field: use __eq function
                            sprintf(cmp, "%s__eq(&self.%s, &other.%s)", ft, fn, fn);
                        }
                        else
                        {
                            // Primitive, POINTER, or unknown: use ==
                            sprintf(cmp, "self.%s == other.%s", fn, fn);
                        }
                        strcat(body, cmp);
                        first = 0;
                    }
                    f = f->next;
                }
                if (first)
                {
                    strcat(body, "true");
                }
                strcat(body, ";");
            }
            code = xmalloc(4096 + 1024);
            // Updated signature: other is a pointer T*
            sprintf(code, "impl %s { fn eq(self, other: %s*) -> bool { %s } }", name, name, body);
        }
        else if (0 == strcmp(trait, "Debug"))
        {
            // Simplistic Debug for now, I know.
            code = xmalloc(1024);
            sprintf(code, "impl %s { fn to_string(self) -> char* { return \"%s { ... }\"; } }",
                    name, name);
        }
        else if (0 == strcmp(trait, "Copy"))
        {
            // Marker trait for Copy/Move semantics
            code = xmalloc(1024);
            sprintf(code, "impl Copy for %s {}", name);
        }

        if (code)
        {
            Lexer tmp;
            lexer_init(&tmp, code);
            ASTNode *impl = parse_impl(ctx, &tmp);
            if (impl)
            {
                if (!head)
                {
                    head = impl;
                }
                else
                {
                    tail->next = impl;
                }
                tail = impl;
            }
        }
    }
    return head;
}
