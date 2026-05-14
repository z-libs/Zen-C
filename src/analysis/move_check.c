// SPDX-License-Identifier: MIT
#include "../constants.h"
#include "move_check.h"
#include "../parser/parser.h"
#include "typecheck.h"
#include "../diagnostics/diagnostics.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

extern ParserContext *g_parser_ctx;

#include "zprep.h"

MoveState *move_state_create(MoveState *parent)
{
    MoveState *s = xmalloc(sizeof(MoveState));
    s->entries = NULL;
    s->parent = parent;
    return s;
}

MoveState *move_state_clone(MoveState *src)
{
    if (!src)
    {
        return NULL;
    }
    MoveState *new_state = move_state_create(src->parent);

    MoveEntry *curr = src->entries;
    MoveEntry **tail = &new_state->entries;
    while (curr)
    {
        MoveEntry *new_entry = xmalloc(sizeof(MoveEntry));
        new_entry->symbol_name = xstrdup(curr->symbol_name);
        new_entry->status = curr->status;
        new_entry->moved_at = curr->moved_at;
        new_entry->next = NULL;
        *tail = new_entry;
        tail = &new_entry->next;
        curr = curr->next;
    }
    return new_state;
}

void move_state_free(MoveState *state)
{
    if (!state)
    {
        return;
    }
    MoveEntry *e = state->entries;
    while (e)
    {
        MoveEntry *next = e->next;
        zfree(e->symbol_name);
        zfree(e);
        e = next;
    }
    zfree(state);
}

char *get_node_path(ASTNode *node, int depth)
{
    if (!node || depth > 32)
    {
        return NULL;
    }
    RECURSION_GUARD_TOKEN(g_parser_ctx, node->token, NULL);

    char *path = NULL;

    if (node->type == NODE_EXPR_VAR)
    {
        path = xstrdup(node->var_ref.name);
    }
    else if (node->type == NODE_EXPR_MEMBER)
    {
        char *target_path = get_node_path(node->member.target, depth + 1);
        if (target_path)
        {
            char buffer[MAX_ERROR_MSG_LEN];
            snprintf(buffer, sizeof(buffer), "%s.%s", target_path, node->member.field);
            zfree(target_path);
            path = xstrdup(buffer);
        }
    }

    RECURSION_EXIT(g_parser_ctx);
    return path;
}

void mark_moved_in_state(MoveState *state, const char *path, Token t)
{
    if (!state || !path)
    {
        return;
    }

    MoveEntry *e = state->entries;
    while (e)
    {
        if (strcmp(e->symbol_name, path) == 0)
        {
            e->status = MOVE_STATE_MOVED;
            e->moved_at = t;
            return;
        }
        e = e->next;
    }

    MoveEntry *new_entry = xmalloc(sizeof(MoveEntry));
    new_entry->symbol_name = xstrdup(path);
    new_entry->status = MOVE_STATE_MOVED;
    new_entry->moved_at = t;
    new_entry->next = state->entries;
    state->entries = new_entry;
}

MoveStatus get_move_status(MoveState *state, const char *path)
{
    if (!path)
    {
        return MOVE_STATE_VALID;
    }
    MoveState *s = state;
    while (s)
    {
        MoveEntry *e = s->entries;
        while (e)
        {
            if (strcmp(e->symbol_name, path) == 0)
            {
                return e->status;
            }
            e = e->next;
        }
        s = s->parent;
    }
    return MOVE_STATE_VALID;
}

void move_state_merge(MoveState *target, MoveState *a, MoveState *b)
{
    if (!target)
    {
        return;
    }

    if (a)
    {
        MoveEntry *e = a->entries;
        while (e)
        {
            MoveStatus status_a = e->status;
            MoveStatus status_b = b ? get_move_status(b, e->symbol_name) : MOVE_STATE_VALID;

            if (status_a == MOVE_STATE_MOVED || status_b == MOVE_STATE_MOVED)
            {
                mark_moved_in_state(target, e->symbol_name, e->moved_at);
            }
            e = e->next;
        }
    }

    if (b)
    {
        MoveEntry *e = b->entries;
        while (e)
        {
            MoveStatus status_b = e->status;
            MoveStatus status_a = a ? get_move_status(a, e->symbol_name) : MOVE_STATE_VALID;

            if (status_b == MOVE_STATE_MOVED || status_a == MOVE_STATE_MOVED)
            {
                mark_moved_in_state(target, e->symbol_name, e->moved_at);
            }
            e = e->next;
        }
    }
}

void move_state_merge_into(MoveState **target, MoveState *src)
{
    if (!src)
    {
        return;
    }

    if (!*target)
    {
        *target = move_state_clone(src);
        return;
    }

    MoveEntry *e = src->entries;
    while (e)
    {
        if (e->status == MOVE_STATE_MOVED)
        {
            mark_moved_in_state(*target, e->symbol_name, e->moved_at);
        }
        e = e->next;
    }
}

int is_type_copy(ParserContext *ctx, Type *t)
{
    if (!t)
    {
        return 1; // Default to Copy for unknown types
    }
    RECURSION_GUARD_TOKEN(ctx, (Token){0}, 1);

    int result = 1;
    if (t->traits.has_drop)
    {
        result = 0;
    }
    else if (t->name && check_impl(ctx, "Drop", t->name))
    {
        result = 0;
    }
    else
    {
        switch (t->kind)
        {
        case TYPE_INT:
        case TYPE_I8:
        case TYPE_I16:
        case TYPE_I32:
        case TYPE_I64:
        case TYPE_U8:
        case TYPE_U16:
        case TYPE_U32:
        case TYPE_U64:
        case TYPE_F32:
        case TYPE_F64:
        case TYPE_BOOL:
        case TYPE_CHAR:
        case TYPE_VOID:
        case TYPE_POINTER:
        case TYPE_FUNCTION:
            result = 1;
            break;
        case TYPE_ENUM:
        case TYPE_BITINT:
        case TYPE_UBITINT:
            result = 1;
            break;
        case TYPE_STRUCT:
            if (check_impl(ctx, "Copy", t->name))
            {
                result = 1;
            }
            else if (check_impl(ctx, "Drop", t->name))
            {
                result = 0;
            }
            else if (t->name)
            {
                ASTNode *def = find_struct_def(ctx, t->name);
                if (def && def->type_info && def->type_info->traits.has_drop)
                {
                    result = 0;
                }
                else
                {
                    result = 1;
                }
            }
            else
            {
                result = 1;
            }
            break;
        case TYPE_ARRAY:
            result = is_type_copy(ctx, t->inner);
            break;
        case TYPE_ALIAS:
            if (t->alias.is_opaque_alias)
            {
                result = 1;
            }
            else
            {
                result = is_type_copy(ctx, t->inner);
            }
            break;
        default:
            result = 1;
            break;
        }
    }
    RECURSION_EXIT(ctx);
    return result;
}

void check_path_validity(TypeChecker *tc, const char *path, Token t)
{
    if (!path)
    {
        return;
    }

    // Check Flow-Sensitive State
    ParserContext *ctx = tc ? tc->pctx : (g_parser_ctx);
    MoveStatus status = MOVE_STATE_VALID;
    const char *moved_path = path;

    if (ctx && ctx->move_state)
    {
        status = get_move_status(ctx->move_state, path);

        // Also check parent paths (e.g., if "zc.features" is moved, then "zc.features.len" is
        // invalid)
        if (status == MOVE_STATE_VALID)
        {
            char *parent = xstrdup(path);
            while (status == MOVE_STATE_VALID)
            {
                char *dot = strrchr(parent, '.');
                if (!dot)
                {
                    break;
                }
                *dot = 0;
                status = get_move_status(ctx->move_state, parent);
                if (status == MOVE_STATE_MOVED || status == MOVE_STATE_MAYBE_MOVED)
                {
                    moved_path = parent;
                    break;
                }
            }
            zfree(parent);
        }
    }

    if (status == MOVE_STATE_MOVED || status == MOVE_STATE_MAYBE_MOVED)
    {
        if (tc && tc->in_loop_pass2)
        {
            if (t.line == 0)
            {
                return;
            }
            if (tc->loop_start_state)
            {
                MoveStatus start_status = get_move_status(tc->loop_start_state, path);
                if (start_status == MOVE_STATE_MOVED || start_status == MOVE_STATE_MAYBE_MOVED)
                {
                    return;
                }
            }
        }

        char msg[MAX_ERROR_MSG_LEN];
        snprintf(msg, 255, "Use of moved value '%s'", moved_path);

        const char *hints[] = {"This type owns resources and cannot be implicitly copied",
                               "Consider using a reference ('&') to borrow the value instead",
                               NULL};
        tc_move_error_with_hints(tc, t, msg, hints);
    }
}

void check_use_validity(TypeChecker *tc, ASTNode *use_node)
{
    if (!use_node)
    {
        return;
    }

    char *path = get_node_path(use_node, 0);
    if (!path && use_node->type == NODE_EXPR_VAR)
    {
        path = xstrdup(use_node->var_ref.name);
    }

    if (!path)
    {
        return;
    }

    check_path_validity(tc, path, use_node->token);
    zfree(path);
}

void mark_symbol_moved(ParserContext *ctx, ZenSymbol *sym, ASTNode *context_node)
{
    if (!context_node)
    {
        return;
    }

    Type *t = context_node->type_info ? context_node->type_info : (sym ? sym->type_info : NULL);
    int copy = is_type_copy(ctx, t);

    if (t && ctx && !copy)
    {
        if (sym)
        {
            sym->is_moved = 1;
        }

        if (ctx->move_state)
        {
            char *path = get_node_path(context_node, 0);
            if (!path && sym)
            {
                path = xstrdup(sym->name);
            }

            if (path)
            {
                mark_moved_in_state(ctx->move_state, path, context_node->token);
                zfree(path);
            }
        }
    }
}

void mark_valid_in_state(MoveState *state, const char *path, Token t)
{
    if (!state || !path)
    {
        return;
    }

    MoveEntry *e = state->entries;
    while (e)
    {
        if (strcmp(e->symbol_name, path) == 0)
        {
            e->status = MOVE_STATE_VALID;
            e->moved_at = t;
            return;
        }
        e = e->next;
    }

    MoveEntry *new_entry = xmalloc(sizeof(MoveEntry));
    new_entry->symbol_name = xstrdup(path);
    new_entry->status = MOVE_STATE_VALID;
    new_entry->moved_at = t;
    new_entry->next = state->entries;
    state->entries = new_entry;
}

void mark_symbol_valid(ParserContext *ctx, ZenSymbol *sym, ASTNode *context_node)
{
    if (sym)
    {
        sym->is_moved = 0;
    }

    if (ctx && ctx->move_state)
    {
        char *path = get_node_path(context_node, 0);
        if (!path && sym)
        {
            path = xstrdup(sym->name);
        }

        if (path)
        {
            mark_valid_in_state(ctx->move_state, path,
                                context_node ? context_node->token
                                             : (sym ? sym->decl_token : (Token){0}));
            zfree(path);
        }
    }
}

void collect_paths_from_node(ASTNode *node, char ***paths, int *count)
{
    if (!node)
    {
        return;
    }

    char *path = get_node_path(node, 0);
    if (path)
    {
        *paths = xrealloc(*paths, sizeof(char *) * (*count + 1));
        (*paths)[*count] = path;
        (*count)++;
    }

    switch (node->type)
    {
    case NODE_EXPR_MEMBER:
        collect_paths_from_node(node->member.target, paths, count);
        break;
    case NODE_EXPR_INDEX:
        collect_paths_from_node(node->index.array, paths, count);
        collect_paths_from_node(node->index.index, paths, count);
        break;
    case NODE_EXPR_CALL:
        collect_paths_from_node(node->call.callee, paths, count);
        {
            ASTNode *arg = node->call.args;
            while (arg)
            {
                collect_paths_from_node(arg, paths, count);
                arg = arg->next;
            }
        }
        break;
    case NODE_EXPR_BINARY:
        collect_paths_from_node(node->binary.left, paths, count);
        collect_paths_from_node(node->binary.right, paths, count);
        break;
    case NODE_EXPR_UNARY:
        collect_paths_from_node(node->unary.operand, paths, count);
        break;
    case NODE_EXPR_CAST:
        collect_paths_from_node(node->cast.expr, paths, count);
        break;
    case NODE_EXPR_ARRAY_LITERAL:
    {
        ASTNode *elem = node->array_literal.elements;
        while (elem)
        {
            collect_paths_from_node(elem, paths, count);
            elem = elem->next;
        }
        break;
    }
    case NODE_EXPR_STRUCT_INIT:
    {
        ASTNode *field = node->struct_init.fields;
        while (field)
        {
            collect_paths_from_node(field->var_decl.init_expr, paths, count);
            field = field->next;
        }
        break;
    }
    case NODE_TERNARY:
        collect_paths_from_node(node->ternary.cond, paths, count);
        collect_paths_from_node(node->ternary.true_expr, paths, count);
        collect_paths_from_node(node->ternary.false_expr, paths, count);
        break;
    default:
        break;
    }
}
