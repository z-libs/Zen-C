
#include "../codegen/codegen.h"
#include "parser.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void instantiate_methods(ParserContext *ctx, GenericImplTemplate *it,
                         const char *mangled_struct_name, const char *arg);

Token expect(Lexer *l, ZTokenType type, const char *msg)
{
    Token t = lexer_next(l);
    if (t.type != type)
    {
        zpanic_at(t, "Expected %s, but got '%.*s'", msg, t.len, t.start);
        return (Token){type, t.start, 0, t.line, t.col};
    }
    return t;
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

void skip_comments(Lexer *l)
{
    while (lexer_peek(l).type == TOK_COMMENT)
    {
        lexer_next(l);
    }
}

// C reserved words that conflict with C when used as identifiers.
// TODO: We gotta work on these.
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
    Scope *s = xmalloc(sizeof(Scope));
    s->symbols = 0;
    s->parent = ctx->current_scope;
    ctx->current_scope = s;
}

void exit_scope(ParserContext *ctx)
{
    if (!ctx->current_scope)
    {
        return;
    }

    // Check for unused variables
    Symbol *sym = ctx->current_scope->symbols;
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

void add_symbol(ParserContext *ctx, const char *n, const char *t, Type *type_info)
{
    add_symbol_with_token(ctx, n, t, type_info, (Token){0});
}

void add_symbol_with_token(ParserContext *ctx, const char *n, const char *t, Type *type_info,
                           Token tok)
{
    if (!ctx->current_scope)
    {
        enter_scope(ctx);
    }

    if (n[0] != '_' && ctx->current_scope->parent && strcmp(n, "it") != 0 && strcmp(n, "self") != 0)
    {
        Scope *p = ctx->current_scope->parent;
        while (p)
        {
            Symbol *sh = p->symbols;
            while (sh)
            {
                if (strcmp(sh->name, n) == 0)
                {
                    warn_shadowing(tok, n);
                    break;
                }
                sh = sh->next;
            }
            if (sh)
            {
                break; // found it
            }
            p = p->parent;
        }
    }
    Symbol *s = xmalloc(sizeof(Symbol));
    s->name = xstrdup(n);
    s->type_name = t ? xstrdup(t) : NULL;
    s->type_info = type_info;
    s->is_mutable = 1;
    s->is_used = 0;
    s->decl_token = tok;
    s->is_const_value = 0;
    s->next = ctx->current_scope->symbols;
    ctx->current_scope->symbols = s;

    // LSP: Also add to flat list (for persistent access after scope exit)
    Symbol *lsp_copy = xmalloc(sizeof(Symbol));
    *lsp_copy = *s;
    lsp_copy->next = ctx->all_symbols;
    ctx->all_symbols = lsp_copy;
}

Type *find_symbol_type_info(ParserContext *ctx, const char *n)
{
    if (!ctx->current_scope)
    {
        return NULL;
    }
    Scope *s = ctx->current_scope;
    while (s)
    {
        Symbol *sym = s->symbols;
        while (sym)
        {
            if (strcmp(sym->name, n) == 0)
            {
                return sym->type_info;
            }
            sym = sym->next;
        }
        s = s->parent;
    }
    return NULL;
}

char *find_symbol_type(ParserContext *ctx, const char *n)
{
    if (!ctx->current_scope)
    {
        return NULL;
    }
    Scope *s = ctx->current_scope;
    while (s)
    {
        Symbol *sym = s->symbols;
        while (sym)
        {
            if (strcmp(sym->name, n) == 0)
            {
                return sym->type_name;
            }
            sym = sym->next;
        }
        s = s->parent;
    }
    return NULL;
}

Symbol *find_symbol_entry(ParserContext *ctx, const char *n)
{
    if (!ctx->current_scope)
    {
        return NULL;
    }
    Scope *s = ctx->current_scope;
    while (s)
    {
        Symbol *sym = s->symbols;
        while (sym)
        {
            if (strcmp(sym->name, n) == 0)
            {
                return sym;
            }
            sym = sym->next;
        }
        s = s->parent;
    }
    return NULL;
}

// LSP: Search flat symbol list (works after scopes are destroyed).
Symbol *find_symbol_in_all(ParserContext *ctx, const char *n)
{
    Symbol *sym = ctx->all_symbols;
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

void register_func(ParserContext *ctx, const char *name, int count, char **defaults,
                   Type **arg_types, Type *ret_type, int is_varargs, int is_async, Token decl_token)
{
    FuncSig *f = xmalloc(sizeof(FuncSig));
    f->name = xstrdup(name);
    f->decl_token = decl_token;
    f->total_args = count;
    f->defaults = defaults;
    f->arg_types = arg_types;
    f->ret_type = ret_type;
    f->is_varargs = is_varargs;
    f->is_async = is_async;
    f->must_use = 0; // Default: can discard result
    f->next = ctx->func_registry;
    ctx->func_registry = f;
}

void register_func_template(ParserContext *ctx, const char *name, const char *param, ASTNode *node)
{
    GenericFuncTemplate *t = xmalloc(sizeof(GenericFuncTemplate));
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
    ctx->known_generics[ctx->known_generics_count++] = strdup(name);
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
            instantiate_methods(ctx, t, inst->name, inst->concrete_arg);
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

void add_to_enum_list(ParserContext *ctx, ASTNode *node)
{
    StructRef *r = xmalloc(sizeof(StructRef));
    r->node = node;
    r->next = ctx->parsed_enums_list;
    ctx->parsed_enums_list = r;
}

void add_to_func_list(ParserContext *ctx, ASTNode *node)
{
    StructRef *r = xmalloc(sizeof(StructRef));
    r->node = node;
    r->next = ctx->parsed_funcs_list;
    ctx->parsed_funcs_list = r;
}

void add_to_impl_list(ParserContext *ctx, ASTNode *node)
{
    StructRef *r = xmalloc(sizeof(StructRef));
    r->node = node;
    r->next = ctx->parsed_impls_list;
    ctx->parsed_impls_list = r;
}

void add_to_global_list(ParserContext *ctx, ASTNode *node)
{
    StructRef *r = xmalloc(sizeof(StructRef));
    r->node = node;
    r->next = ctx->parsed_globals_list;
    ctx->parsed_globals_list = r;
}

void register_builtins(ParserContext *ctx)
{
    Type *t = type_new(TYPE_BOOL);
    t->is_const = 1;
    add_symbol(ctx, "true", "bool", t);

    t = type_new(TYPE_BOOL);
    t->is_const = 1;
    add_symbol(ctx, "false", "bool", t);

    // Register 'free'
    Type *void_t = type_new(TYPE_VOID);
    add_symbol(ctx, "free", "void", void_t);

    // Register common libc functions to avoid warnings
    add_symbol(ctx, "strdup", "string", type_new(TYPE_STRING));
    add_symbol(ctx, "malloc", "void*", type_new_ptr(void_t));
    add_symbol(ctx, "realloc", "void*", type_new_ptr(void_t));
    add_symbol(ctx, "calloc", "void*", type_new_ptr(void_t));
    add_symbol(ctx, "puts", "int", type_new(TYPE_INT));
    add_symbol(ctx, "printf", "int", type_new(TYPE_INT));
    add_symbol(ctx, "strcmp", "int", type_new(TYPE_INT));
    add_symbol(ctx, "strlen", "int", type_new(TYPE_INT));
    add_symbol(ctx, "strcpy", "string", type_new(TYPE_STRING));
    add_symbol(ctx, "strcat", "string", type_new(TYPE_STRING));
    add_symbol(ctx, "exit", "void", void_t);

    // File I/O
    add_symbol(ctx, "fopen", "void*", type_new_ptr(void_t));
    add_symbol(ctx, "fclose", "int", type_new(TYPE_INT));
    add_symbol(ctx, "fread", "usize", type_new(TYPE_USIZE));
    add_symbol(ctx, "fwrite", "usize", type_new(TYPE_USIZE));
    add_symbol(ctx, "fseek", "int", type_new(TYPE_INT));
    add_symbol(ctx, "ftell", "long", type_new(TYPE_I64));
    add_symbol(ctx, "rewind", "void", void_t);
    add_symbol(ctx, "fprintf", "int", type_new(TYPE_INT));
    add_symbol(ctx, "sprintf", "int", type_new(TYPE_INT));
    add_symbol(ctx, "feof", "int", type_new(TYPE_INT));
    add_symbol(ctx, "ferror", "int", type_new(TYPE_INT));
    add_symbol(ctx, "usleep", "int", type_new(TYPE_INT));
}

void add_instantiated_func(ParserContext *ctx, ASTNode *fn)
{
    fn->next = ctx->instantiated_funcs;
    ctx->instantiated_funcs = fn;
}

void register_enum_variant(ParserContext *ctx, const char *ename, const char *vname, int tag)
{
    EnumVariantReg *r = xmalloc(sizeof(EnumVariantReg));
    r->enum_name = xstrdup(ename);
    r->variant_name = xstrdup(vname);
    r->tag_id = tag;
    r->next = ctx->enum_variants;
    ctx->enum_variants = r;
}

EnumVariantReg *find_enum_variant(ParserContext *ctx, const char *vname)
{
    EnumVariantReg *r = ctx->enum_variants;
    while (r)
    {
        if (strcmp(r->variant_name, vname) == 0)
        {
            return r;
        }
        r = r->next;
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

void register_var_mutability(ParserContext *ctx, const char *name, int is_mutable)
{
    VarMutability *v = xmalloc(sizeof(VarMutability));
    v->name = xstrdup(name);
    v->is_mutable = is_mutable;
    v->next = ctx->var_mutability_table;
    ctx->var_mutability_table = v;
}

int is_var_mutable(ParserContext *ctx, const char *name)
{
    for (VarMutability *v = ctx->var_mutability_table; v; v = v->next)
    {
        if (strcmp(v->name, name) == 0)
        {
            return v->is_mutable;
        }
    }
    return 1;
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
    char slice_name[256];
    sprintf(slice_name, "Slice_%s", type);

    ASTNode *len_f = ast_create(NODE_FIELD);
    len_f->field.name = xstrdup("len");
    len_f->field.type = xstrdup("int");
    ASTNode *cap_f = ast_create(NODE_FIELD);
    cap_f->field.name = xstrdup("cap");
    cap_f->field.type = xstrdup("int");
    ASTNode *data_f = ast_create(NODE_FIELD);
    data_f->field.name = xstrdup("data");
    char ptr_type[256];
    sprintf(ptr_type, "%s*", type);
    data_f->field.type = xstrdup(ptr_type);

    data_f->next = len_f;
    len_f->next = cap_f;

    ASTNode *def = ast_create(NODE_STRUCT);
    def->strct.name = xstrdup(slice_name);
    def->strct.fields = data_f;

    register_struct_def(ctx, slice_name, def);
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
}

void register_struct_def(ParserContext *ctx, const char *name, ASTNode *node)
{
    StructDef *d = xmalloc(sizeof(StructDef));
    d->name = xstrdup(name);
    d->node = node;
    d->next = ctx->struct_defs;
    ctx->struct_defs = d;
}

ASTNode *find_struct_def(ParserContext *ctx, const char *name)
{
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
        if (s->type == NODE_STRUCT && strcmp(s->strct.name, name) == 0)
        {
            return s;
        }
        s = s->next;
    }

    StructRef *r = ctx->parsed_structs_list;
    while (r)
    {
        if (strcmp(r->node->strct.name, name) == 0)
        {
            return r->node;
        }
        r = r->next;
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

Module *find_module(ParserContext *ctx, const char *alias)
{
    Module *m = ctx->modules;
    while (m)
    {
        if (strcmp(m->alias, alias) == 0)
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
    m->alias = xstrdup(alias);
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
    const char *base = slash ? slash + 1 : path;
    const char *dot = strrchr(base, '.');
    int len = dot ? (int)(dot - base) : (int)strlen(base);
    char *name = xmalloc(len + 1);
    strncpy(name, base, len);
    name[len] = 0;
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

    char *result;
    int i, cnt = 0;
    int newWlen = strlen(new_w);
    int oldWlen = strlen(old_w);

    for (i = 0; src[i] != '\0'; i++)
    {
        if (strstr(&src[i], old_w) == &src[i])
        {
            // Check boundaries to ensure we match whole words only
            int valid = 1;

            // Check preceding character
            if (i > 0 && is_ident_char(src[i - 1]))
            {
                valid = 0;
            }

            // Check following character
            if (valid && is_ident_char(src[i + oldWlen]))
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

    i = 0;
    while (*src)
    {
        if (strstr(src, old_w) == src)
        {
            int valid = 1;

            // Check boundary relative to the *new* result buffer built so far
            if (i > 0 && is_ident_char(result[i - 1]))
            {
                valid = 0;
            }

            // Check boundary relative to the *original* source string
            if (valid && is_ident_char(src[oldWlen]))
            {
                valid = 0;
            }

            if (valid)
            {
                strcpy(&result[i], new_w);
                i += newWlen;
                src += oldWlen;
            }
            else
            {
                result[i++] = *src++;
            }
        }
        else
        {
            result[i++] = *src++;
        }
    }
    result[i] = '\0';
    return result;
}

char *replace_type_str(const char *src, const char *param, const char *concrete,
                       const char *old_struct, const char *new_struct)
{
    if (!src)
    {
        return NULL;
    }

    if (strcmp(src, param) == 0)
    {
        return xstrdup(concrete);
    }

    if (old_struct && new_struct && strcmp(src, old_struct) == 0)
    {
        return xstrdup(new_struct);
    }

    if (old_struct && new_struct && param)
    {
        char *mangled = xmalloc(strlen(old_struct) + strlen(param) + 2);
        sprintf(mangled, "%s_%s", old_struct, param);
        if (strcmp(src, mangled) == 0)
        {
            free(mangled);
            return xstrdup(new_struct);
        }
        free(mangled);
    }

    if (param && concrete && src)
    {
        char suffix[256];
        sprintf(suffix, "_%s", param);
        size_t slen = strlen(src);
        size_t plen = strlen(suffix);
        if (slen > plen && strcmp(src + slen - plen, suffix) == 0)
        {
            // Ends with _T -> Replace suffix with _int (sanitize for pointers like
            // JsonValue*)
            char *clean_concrete = sanitize_mangled_name(concrete);
            char *ret = xmalloc(slen - plen + strlen(clean_concrete) + 2);
            strncpy(ret, src, slen - plen);
            ret[slen - plen] = 0;
            strcat(ret, "_");
            strcat(ret, clean_concrete);
            free(clean_concrete);
            return ret;
        }
    }

    size_t len = strlen(src);
    if (len > 1 && src[len - 1] == '*')
    {
        char *base = xmalloc(len);
        strncpy(base, src, len - 1);
        base[len - 1] = 0;

        char *new_base = replace_type_str(base, param, concrete, old_struct, new_struct);
        free(base);

        if (strcmp(new_base, base) != 0)
        {
            char *ret = xmalloc(strlen(new_base) + 2);
            sprintf(ret, "%s*", new_base);
            free(new_base);
            return ret;
        }
        free(new_base);
    }

    if (strncmp(src, "Slice_", 6) == 0)
    {
        char *base = xstrdup(src + 6);
        char *new_base = replace_type_str(base, param, concrete, old_struct, new_struct);
        free(base);

        if (strcmp(new_base, base) != 0)
        {
            char *ret = xmalloc(strlen(new_base) + 7);
            sprintf(ret, "Slice_%s", new_base);
            free(new_base);
            return ret;
        }
        free(new_base);
    }

    return xstrdup(src);
}

ASTNode *copy_ast_replacing(ASTNode *n, const char *p, const char *c, const char *os,
                            const char *ns);

Type *replace_type_formal(Type *t, const char *p, const char *c, const char *os, const char *ns)
{
    if (!t)
    {
        return NULL;
    }

    if ((t->kind == TYPE_STRUCT || t->kind == TYPE_GENERIC) && t->name && strcmp(t->name, p) == 0)
    {
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

        if (strcmp(c, "I8") == 0)
        {
            return type_new(TYPE_I8);
        }
        if (strcmp(c, "U8") == 0)
        {
            return type_new(TYPE_U8);
        }
        if (strcmp(c, "I16") == 0)
        {
            return type_new(TYPE_I16);
        }
        if (strcmp(c, "U16") == 0)
        {
            return type_new(TYPE_U16);
        }
        if (strcmp(c, "I32") == 0)
        {
            return type_new(TYPE_I32);
        }
        if (strcmp(c, "U32") == 0)
        {
            return type_new(TYPE_U32);
        }
        if (strcmp(c, "I64") == 0)
        {
            return type_new(TYPE_I64);
        }
        if (strcmp(c, "U64") == 0)
        {
            return type_new(TYPE_U64);
        }
        if (strcmp(c, "F32") == 0)
        {
            return type_new(TYPE_F32);
        }
        if (strcmp(c, "F64") == 0)
        {
            return type_new(TYPE_F64);
        }

        if (strcmp(c, "usize") == 0)
        {
            return type_new(TYPE_USIZE);
        }
        if (strcmp(c, "isize") == 0)
        {
            return type_new(TYPE_ISIZE);
        }
        if (strcmp(c, "byte") == 0)
        {
            return type_new(TYPE_BYTE);
        }
        if (strcmp(c, "I128") == 0)
        {
            return type_new(TYPE_I128);
        }
        if (strcmp(c, "U128") == 0)
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

        Type *n = type_new(TYPE_STRUCT);
        n->name = sanitize_mangled_name(c);
        return n;
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
            char suffix[256];
            sprintf(suffix, "_%s", p); // e.g. "_T"
            size_t nlen = strlen(t->name);
            size_t slen = strlen(suffix);

            if (nlen > slen && strcmp(t->name + nlen - slen, suffix) == 0)
            {
                // It ends in _T. Replace with _int (c), sanitizing for pointers
                char *clean_c = sanitize_mangled_name(c);
                char *new_name = xmalloc(nlen - slen + strlen(clean_c) + 2);
                strncpy(new_name, t->name, nlen - slen);
                new_name[nlen - slen] = 0;
                strcat(new_name, "_");
                strcat(new_name, clean_c);
                free(clean_c);
                n->name = new_name;
                // Ensure it's concrete to prevent double mangling later
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

    if (t->kind == TYPE_POINTER || t->kind == TYPE_ARRAY)
    {
        n->inner = replace_type_formal(t->inner, p, c, os, ns);
    }

    if (n->arg_count > 0 && t->args)
    {
        n->args = xmalloc(sizeof(Type *) * t->arg_count);
        for (int i = 0; i < t->arg_count; i++)
        {
            n->args[i] = replace_type_formal(t->args[i], p, c, os, ns);
        }
    }

    return n;
}

// Helper to replace generic params in mangled names (e.g. Option_V_None ->
// Option_int_None)
char *replace_mangled_part(const char *src, const char *param, const char *concrete)
{
    if (!src || !param || !concrete)
    {
        return src ? xstrdup(src) : NULL;
    }

    char *result = xmalloc(4096); // Basic buffer for simplicity
    result[0] = 0;

    const char *curr = src;
    char *out = result;
    int plen = strlen(param);

    while (*curr)
    {
        // Check if param matches here
        if (strncmp(curr, param, plen) == 0)
        {
            // Check boundaries: Must be delimited by quoted boundaries, OR
            // underscores, OR string ends
            int valid = 1;

            // Check Prev: Start of string OR Underscore
            if (curr > src)
            {
                if (*(curr - 1) != '_' && is_ident_char(*(curr - 1)))
                {
                    valid = 0;
                }
            }

            // Check Next: End of string OR Underscore
            if (valid && curr[plen] != 0 && curr[plen] != '_' && is_ident_char(curr[plen]))
            {
                valid = 0;
            }

            if (valid)
            {
                strcpy(out, concrete);
                out += strlen(concrete);
                curr += plen;
                continue;
            }
        }
        *out++ = *curr++;
    }
    *out = 0;
    return xstrdup(result);
}

ASTNode *copy_ast_replacing(ASTNode *n, const char *p, const char *c, const char *os,
                            const char *ns)
{
    if (!n)
    {
        return NULL;
    }

    ASTNode *new_node = xmalloc(sizeof(ASTNode));
    *new_node = *n;

    if (n->resolved_type)
    {
        new_node->resolved_type = replace_type_str(n->resolved_type, p, c, os, ns);
    }
    new_node->type_info = replace_type_formal(n->type_info, p, c, os, ns);

    new_node->next = copy_ast_replacing(n->next, p, c, os, ns);

    switch (n->type)
    {
    case NODE_FUNCTION:
        new_node->func.name = xstrdup(n->func.name);
        new_node->func.ret_type = replace_type_str(n->func.ret_type, p, c, os, ns);

        char *tmp_args = replace_in_string(n->func.args, p, c);
        if (os && ns)
        {
            char *tmp2 = replace_in_string(tmp_args, os, ns);
            free(tmp_args);
            tmp_args = tmp2;
        }
        if (p && c)
        {
            char *clean_c = sanitize_mangled_name(c);
            char *tmp3 = replace_mangled_part(tmp_args, p, clean_c);
            free(clean_c);
            free(tmp_args);
            tmp_args = tmp3;
        }
        new_node->func.args = tmp_args;

        new_node->func.ret_type_info = replace_type_formal(n->func.ret_type_info, p, c, os, ns);
        if (n->func.arg_types)
        {
            new_node->func.arg_types = xmalloc(sizeof(Type *) * n->func.arg_count);
            for (int i = 0; i < n->func.arg_count; i++)
            {
                new_node->func.arg_types[i] =
                    replace_type_formal(n->func.arg_types[i], p, c, os, ns);
            }
        }

        new_node->func.body = copy_ast_replacing(n->func.body, p, c, os, ns);
        break;
    case NODE_BLOCK:
        new_node->block.statements = copy_ast_replacing(n->block.statements, p, c, os, ns);
        break;
    case NODE_RAW_STMT:
    {
        char *s1 = replace_in_string(n->raw_stmt.content, p, c);
        if (os && ns)
        {
            char *s2 = replace_in_string(s1, os, ns);
            free(s1);
            s1 = s2;
        }

        if (p && c)
        {
            char *clean_c = sanitize_mangled_name(c);
            char *s3 = replace_mangled_part(s1, p, clean_c);
            free(clean_c);
            free(s1);
            s1 = s3;
        }

        new_node->raw_stmt.content = s1;
    }
    break;
    case NODE_VAR_DECL:
        new_node->var_decl.name = xstrdup(n->var_decl.name);
        new_node->var_decl.type_str = replace_type_str(n->var_decl.type_str, p, c, os, ns);
        new_node->var_decl.init_expr = copy_ast_replacing(n->var_decl.init_expr, p, c, os, ns);
        break;
    case NODE_RETURN:
        new_node->ret.value = copy_ast_replacing(n->ret.value, p, c, os, ns);
        break;
    case NODE_EXPR_BINARY:
        new_node->binary.left = copy_ast_replacing(n->binary.left, p, c, os, ns);
        new_node->binary.right = copy_ast_replacing(n->binary.right, p, c, os, ns);
        new_node->binary.op = xstrdup(n->binary.op);
        break;
    case NODE_EXPR_UNARY:
        new_node->unary.op = xstrdup(n->unary.op);
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
        char *n1 = xstrdup(n->var_ref.name);
        if (p && c)
        {
            char *clean_c = sanitize_mangled_name(c);
            char *n2 = replace_mangled_part(n1, p, clean_c);
            free(clean_c);
            free(n1);
            n1 = n2;
        }
        new_node->var_ref.name = n1;
    }
    break;
    case NODE_FIELD:
        new_node->field.name = xstrdup(n->field.name);
        new_node->field.type = replace_type_str(n->field.type, p, c, os, ns);
        break;
    case NODE_EXPR_LITERAL:
        if (n->literal.type_kind == 2)
        {
            new_node->literal.string_val = xstrdup(n->literal.string_val);
        }
        break;
    case NODE_EXPR_MEMBER:
        new_node->member.target = copy_ast_replacing(n->member.target, p, c, os, ns);
        new_node->member.field = xstrdup(n->member.field);
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
        new_node->struct_init.struct_name =
            replace_type_str(n->struct_init.struct_name, p, c, os, ns);
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
        break;
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

    case NODE_MATCH_CASE:
        if (n->match_case.pattern)
        {
            char *s1 = replace_in_string(n->match_case.pattern, p, c);
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
    default:
        break;
    }
    return new_node;
}

// Helper to sanitize type names for mangling (e.g. "int*" -> "intPtr")
char *sanitize_mangled_name(const char *s)
{
    char *buf = xmalloc(strlen(s) * 4 + 1);
    char *p = buf;
    while (*s)
    {
        if (*s == '*')
        {
            strcpy(p, "Ptr");
            p += 3;
        }
        else if (*s == ' ')
        {
            *p++ = '_';
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

FuncSig *find_func(ParserContext *ctx, const char *name)
{
    FuncSig *c = ctx->func_registry;
    while (c)
    {
        if (strcmp(c->name, name) == 0)
        {
            return c;
        }
        c = c->next;
    }
    return NULL;
}

char *instantiate_function_template(ParserContext *ctx, const char *name, const char *concrete_type)
{
    GenericFuncTemplate *tpl = find_func_template(ctx, name);
    if (!tpl)
    {
        return NULL;
    }

    char *clean_type = sanitize_mangled_name(concrete_type);
    char *mangled = xmalloc(strlen(name) + strlen(clean_type) + 2);
    sprintf(mangled, "%s_%s", name, clean_type);
    free(clean_type);

    if (find_func(ctx, mangled))
    {
        return mangled;
    }

    ASTNode *new_fn =
        copy_ast_replacing(tpl->func_node, tpl->generic_param, concrete_type, NULL, NULL);
    if (!new_fn || new_fn->type != NODE_FUNCTION)
    {
        return NULL;
    }
    free(new_fn->func.name);
    new_fn->func.name = xstrdup(mangled);

    register_func(ctx, mangled, new_fn->func.arg_count, new_fn->func.defaults,
                  new_fn->func.arg_types, new_fn->func.ret_type_info, new_fn->func.is_varargs, 0,
                  new_fn->token);

    add_instantiated_func(ctx, new_fn);
    return mangled;
}

char *process_fstring(ParserContext *ctx, const char *content)
{
    (void)ctx; // suppress unused parameter warning
    char *gen = xmalloc(4096);

    strcpy(gen, "({ static char _b[1024]; _b[0]=0; char _t[128]; ");

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
        char *expr = brace + 1;
        char *fmt = NULL;
        if (colon)
        {
            *colon = 0;
            fmt = colon + 1;
        }

        if (fmt)
        {
            strcat(gen, "sprintf(_t, \"%");
            strcat(gen, fmt);
            strcat(gen, "\", ");
            strcat(gen, expr);
            strcat(gen, "); strcat(_b, _t); ");
        }
        else
        {
            strcat(gen, "sprintf(_t, _z_str(");
            strcat(gen, expr);
            strcat(gen, "), ");
            strcat(gen, expr);
            strcat(gen, "); strcat(_b, _t); ");
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
    return 0;
}

void register_template(ParserContext *ctx, const char *name, ASTNode *node)
{
    GenericTemplate *t = xmalloc(sizeof(GenericTemplate));
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
                    instantiate_generic(ctx, template_name, concrete_arg);
                }
            }
            free(type_copy);
        }
    }

    n->next = copy_fields_replacing(ctx, fields->next, param, concrete);
    return n;
}

void instantiate_methods(ParserContext *ctx, GenericImplTemplate *it,
                         const char *mangled_struct_name, const char *arg)
{
    if (check_impl(ctx, "Methods", mangled_struct_name))
    {
        return; // Simple dedupe check
    }

    ASTNode *backup_next = it->impl_node->next;
    it->impl_node->next = NULL; // Break link to isolate node
    ASTNode *new_impl = copy_ast_replacing(it->impl_node, it->generic_param, arg, it->struct_name,
                                           mangled_struct_name);
    it->impl_node->next = backup_next; // Restore

    new_impl->impl.struct_name = xstrdup(mangled_struct_name);
    ASTNode *meth = new_impl->impl.methods;
    while (meth)
    {
        char *suffix = strchr(meth->func.name, '_');
        if (suffix)
        {
            char *new_name = xmalloc(strlen(mangled_struct_name) + strlen(suffix) + 1);
            sprintf(new_name, "%s%s", mangled_struct_name, suffix);
            free(meth->func.name);
            meth->func.name = new_name;
            register_func(ctx, new_name, meth->func.arg_count, meth->func.defaults,
                          meth->func.arg_types, meth->func.ret_type_info, meth->func.is_varargs, 0,
                          meth->token);
        }

        // Handle generic return types in methods (e.g., Option<T> -> Option_int)
        if (meth->func.ret_type && strchr(meth->func.ret_type, '_'))
        {
            char *ret_copy = xstrdup(meth->func.ret_type);
            char *underscore = strrchr(ret_copy, '_');
            if (underscore && underscore > ret_copy)
            {
                *underscore = '\0';
                char *template_name = ret_copy;

                // Check if this looks like a generic (e.g., "Option_V" or "Result_V")
                GenericTemplate *gt = ctx->templates;
                while (gt)
                {
                    if (strcmp(gt->name, template_name) == 0)
                    {
                        // Found matching template, instantiate it
                        instantiate_generic(ctx, template_name, arg);
                        break;
                    }
                    gt = gt->next;
                }
            }
            free(ret_copy);
        }

        meth = meth->next;
    }
    add_instantiated_func(ctx, new_impl);
}

void instantiate_generic(ParserContext *ctx, const char *tpl, const char *arg)
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
    char m[256];
    sprintf(m, "%s_%s", tpl, clean_arg);
    free(clean_arg);

    Instantiation *c = ctx->instantiations;
    while (c)
    {
        if (strcmp(c->name, m) == 0)
        {
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
        zpanic("Unknown generic: %s", tpl);
    }

    Instantiation *ni = xmalloc(sizeof(Instantiation));
    ni->name = xstrdup(m);
    ni->template_name = xstrdup(tpl);
    ni->concrete_arg = xstrdup(arg);
    ni->struct_node = NULL; // Placeholder to break cycles
    ni->next = ctx->instantiations;
    ctx->instantiations = ni;

    ASTNode *struct_node_copy = NULL;

    if (t->struct_node->type == NODE_STRUCT)
    {
        ASTNode *i = ast_create(NODE_STRUCT);
        i->strct.name = xstrdup(m);
        i->strct.is_template = 0;
        i->strct.fields = copy_fields_replacing(ctx, t->struct_node->strct.fields,
                                                t->struct_node->strct.generic_param, arg);
        struct_node_copy = i;
        register_struct_def(ctx, m, i);
    }
    else if (t->struct_node->type == NODE_ENUM)
    {
        ASTNode *i = ast_create(NODE_ENUM);
        i->enm.name = xstrdup(m);
        i->enm.is_template = 0;
        ASTNode *h = 0, *tl = 0;
        ASTNode *v = t->struct_node->enm.variants;
        while (v)
        {
            ASTNode *nv = ast_create(NODE_ENUM_VARIANT);
            nv->variant.name = xstrdup(v->variant.name);
            nv->variant.tag_id = v->variant.tag_id;
            nv->variant.payload = replace_type_formal(
                v->variant.payload, t->struct_node->enm.generic_param, arg, NULL, NULL);
            char mangled_var[512];
            sprintf(mangled_var, "%s_%s", m, nv->variant.name);
            register_enum_variant(ctx, m, mangled_var, nv->variant.tag_id);
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
            instantiate_methods(ctx, it, m, arg);
        }
        it = it->next;
    }
}

int is_file_imported(ParserContext *ctx, const char *p)
{
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
                zpanic("Unterminated condition");
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
            zpanic("Empty condition or missing body");
        }
        char *c = xmalloc(len + 1);
        strncpy(c, start, len);
        c[len] = 0;
        return c;
    }
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

                char ptr_check[64];
                strcpy(ptr_check, vtype);
                int is_ptr = (strchr(ptr_check, '*') != NULL);
                if (is_ptr)
                {
                    char *p = strchr(ptr_check, '*');
                    if (p)
                    {
                        *p = 0;
                    }
                }

                dest += sprintf(dest, "%s_%s(%s%s", ptr_check, method, is_ptr ? "" : "&", acc);

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
                char ptr_check[64];
                strcpy(ptr_check, vtype);
                int is_ptr = (strchr(ptr_check, '*') != NULL);
                if (is_ptr)
                {
                    char *p = strchr(ptr_check, '*');
                    if (p)
                    {
                        *p = 0;
                    }
                }
                dest += sprintf(dest, "%s_%s(%s%s)", ptr_check, method, is_ptr ? "" : "&", acc);
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
                dest += sprintf(dest, "%s_%s", acc, field);
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

                    char mangled[256];
                    snprintf(mangled, sizeof(mangled), "%s_%s", func_name, method);

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

char *parse_and_convert_args(ParserContext *ctx, Lexer *l, char ***defaults_out, int *count_out,
                             Type ***types_out, char ***names_out, int *is_varargs_out)
{
    if (lexer_next(l).type != TOK_LPAREN)
    {
        zpanic("Expected '(' in function args");
    }

    char *buf = xmalloc(1024);
    buf[0] = 0;
    int count = 0;
    char **defaults = xmalloc(sizeof(char *) * 16);
    Type **types = xmalloc(sizeof(Type *) * 16);
    char **names = xmalloc(sizeof(char *) * 16);

    for (int i = 0; i < 16; i++)
    {
        defaults[i] = NULL;
        types[i] = NULL;
        names[i] = NULL;
    }

    if (lexer_peek(l).type != TOK_RPAREN)
    {
        while (1)
        {
            Token t = lexer_next(l);
            // Handle 'self'
            if (t.type == TOK_IDENT && strncmp(t.start, "self", 4) == 0 && t.len == 4)
            {
                names[count] = xstrdup("self");
                if (ctx->current_impl_struct)
                {
                    Type *st = NULL;
                    // Check for primitives to avoid creating struct int*
                    if (strcmp(ctx->current_impl_struct, "int") == 0)
                    {
                        st = type_new(TYPE_INT);
                    }
                    else if (strcmp(ctx->current_impl_struct, "float") == 0)
                    {
                        st = type_new(TYPE_F32);
                    }
                    else if (strcmp(ctx->current_impl_struct, "char") == 0)
                    {
                        st = type_new(TYPE_CHAR);
                    }
                    else if (strcmp(ctx->current_impl_struct, "bool") == 0)
                    {
                        st = type_new(TYPE_BOOL);
                    }
                    else if (strcmp(ctx->current_impl_struct, "string") == 0)
                    {
                        st = type_new(TYPE_STRING);
                    }
                    // Add other primitives as needed
                    else
                    {
                        st = type_new(TYPE_STRUCT);
                        st->name = xstrdup(ctx->current_impl_struct);
                    }
                    Type *pt = type_new_ptr(st);

                    char buf_type[256];
                    sprintf(buf_type, "%s*", ctx->current_impl_struct);
                    // Register 'self' with actual type in symbol table
                    add_symbol(ctx, "self", buf_type, pt);

                    types[count] = pt;

                    strcat(buf, "void* self");
                }
                else
                {
                    strcat(buf, "void* self");
                    types[count] = type_new_ptr(type_new(TYPE_VOID));
                    add_symbol(ctx, "self", "void*", types[count]);
                }
                count++;
            }
            else
            {
                if (t.type != TOK_IDENT)
                {
                    zpanic("Expected arg name");
                }
                char *name = token_strdup(t);
                names[count] = name; // Store name
                if (lexer_next(l).type != TOK_COLON)
                {
                    zpanic("Expected ':'");
                }

                Type *arg_type = parse_type_formal(ctx, l);
                char *type_str = type_to_string(arg_type);

                add_symbol(ctx, name, type_str, arg_type);
                types[count] = arg_type;

                if (strlen(buf) > 0)
                {
                    strcat(buf, ", ");
                }

                char *fn_ptr = strstr(type_str, "(*)");
                if (arg_type->kind == TYPE_FUNCTION)
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
                    strcat(buf, type_str);
                    strcat(buf, " ");
                    strcat(buf, name);
                }

                count++;

                if (lexer_peek(l).type == TOK_OP && is_token(lexer_peek(l), "="))
                {
                    lexer_next(l);
                    Token val = lexer_next(l);
                    defaults[count - 1] = token_strdup(val);
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
        zpanic("Expected ')' after args");
    }

    *defaults_out = defaults;
    *count_out = count;
    *types_out = types;
    *names_out = names;
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
        Symbol *sym = s->symbols;
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

void register_plugin(ParserContext *ctx, const char *name, const char *alias)
{
    ImportedPlugin *p = xmalloc(sizeof(ImportedPlugin));
    p->name = xstrdup(name);
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
