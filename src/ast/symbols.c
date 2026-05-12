// SPDX-License-Identifier: MIT
#include "../arena.h"
#include "symbols.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Scope *symbol_scope_create(Scope *parent, const char *name)
{
    Scope *s = xmalloc(sizeof(Scope));
    memset(s, 0, sizeof(Scope));
    s->parent = parent;
    if (name)
    {
        s->name = xstrdup(name);
    }
    return s;
}

void symbol_scope_free(Scope *s)
{
    if (!s)
    {
        return;
    }

    ZenSymbol *sym = s->symbols;
    while (sym)
    {
        ZenSymbol *next = sym->next;
        if (sym->name)
        {
            zfree(sym->name);
        }
        if (sym->cfg_condition)
        {
            zfree(sym->cfg_condition);
        }

        if (sym->kind == SYM_ALIAS)
        {
            if (sym->data.alias.original_type)
            {
                zfree(sym->data.alias.original_type);
            }
        }
        else if (sym->kind == SYM_CONSTANT)
        {
            if (sym->data.constant.str_val)
            {
                zfree(sym->data.constant.str_val);
            }
        }
        else if (sym->kind == SYM_MODULE)
        {
            if (sym->data.module.path)
            {
                zfree(sym->data.module.path);
            }
            if (sym->data.module.alias_name)
            {
                zfree(sym->data.module.alias_name);
            }
        }

        zfree(sym);
        sym = next;
    }

    if (s->name)
    {
        zfree(s->name);
    }
    zfree(s);
}

ZenSymbol *symbol_add(Scope *s, const char *name, SymbolKind kind)
{
    if (!s || !name)
    {
        return NULL;
    }

    ZenSymbol *sym = xmalloc(sizeof(ZenSymbol));
    memset(sym, 0, sizeof(ZenSymbol));
    sym->name = xstrdup(name);
    sym->kind = kind;

    sym->next = s->symbols;
    s->symbols = sym;

    return sym;
}

ZenSymbol *symbol_lookup_local(Scope *s, const char *name)
{
    if (!s || !name)
    {
        return NULL;
    }

    ZenSymbol *curr = s->symbols;
    while (curr)
    {
        if (curr->name && strcmp(curr->name, name) == 0)
        {
            return curr;
        }
        curr = curr->next;
    }
    return NULL;
}

ZenSymbol *symbol_lookup(Scope *s, const char *name)
{
    if (!name)
    {
        return NULL;
    }

    Scope *curr_scope = s;
    while (curr_scope)
    {
        ZenSymbol *sym = symbol_lookup_local(curr_scope, name);
        if (sym)
        {
            return sym;
        }
        curr_scope = curr_scope->parent;
    }
    return NULL;
}

ZenSymbol *symbol_lookup_kind(Scope *s, const char *name, SymbolKind kind)
{
    if (!name)
    {
        return NULL;
    }

    Scope *curr_scope = s;
    while (curr_scope)
    {
        ZenSymbol *sym = curr_scope->symbols;
        while (sym)
        {
            if (sym->kind == kind && sym->name && strcmp(sym->name, name) == 0)
            {
                return sym;
            }
            sym = sym->next;
        }
        curr_scope = curr_scope->parent;
    }
    return NULL;
}
