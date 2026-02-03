
#include "lsp_index.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

LSPIndex *lsp_index_new()
{
    return calloc(1, sizeof(LSPIndex));
}

void lsp_index_free(LSPIndex *idx)
{
    if (!idx)
    {
        return;
    }
    LSPRange *c = idx->head;
    while (c)
    {
        LSPRange *n = c->next;
        if (c->hover_text)
        {
            free(c->hover_text);
        }
        free(c);
        c = n;
    }
    free(idx);
}

void lsp_index_add(LSPIndex *idx, LSPRange *r)
{
    if (!idx->head)
    {
        idx->head = r;
        idx->tail = r;
    }
    else
    {
        idx->tail->next = r;
        idx->tail = r;
    }
}

void lsp_index_add_def(LSPIndex *idx, Token t, const char *hover, ASTNode *node)
{
    if (t.line <= 0)
    {
        return;
    }
    LSPRange *r = calloc(1, sizeof(LSPRange));
    r->type = RANGE_DEFINITION;
    r->start_line = t.line - 1;
    r->start_col = t.col - 1;
    r->end_line = t.line - 1;
    r->end_col = t.col - 1 + t.len;
    if (hover)
    {
        r->hover_text = zc_strdup(hover);
    }
    r->node = node;

    lsp_index_add(idx, r);
}

void lsp_index_add_ref(LSPIndex *idx, Token t, Token def_t, ASTNode *node)
{
    if (t.line <= 0 || def_t.line <= 0)
    {
        return;
    }
    LSPRange *r = calloc(1, sizeof(LSPRange));
    r->type = RANGE_REFERENCE;
    r->start_line = t.line - 1;
    r->start_col = t.col - 1;
    r->end_line = t.line - 1;
    r->end_col = t.col - 1 + t.len;

    r->def_line = def_t.line - 1;
    r->def_col = def_t.col - 1;
    r->node = node;

    lsp_index_add(idx, r);
}

LSPRange *lsp_find_at(LSPIndex *idx, int line, int col)
{
    LSPRange *curr = idx->head;
    LSPRange *best = NULL;

    while (curr)
    {
        if (line >= curr->start_line && line <= curr->end_line)
        {
            if (line == curr->start_line && col < curr->start_col)
            {
                curr = curr->next;
                continue;
            }

            if (line == curr->end_line && col > curr->end_col)
            {
                curr = curr->next;
                continue;
            }

            best = curr;
        }
        curr = curr->next;
    }
    return best;
}

// Walker.

void lsp_walk_node(LSPIndex *idx, ASTNode *node)
{
    if (!node)
    {
        return;
    }

    // Definition logic.
    if (node->type == NODE_FUNCTION)
    {
        char hover[256];
        sprintf(hover, "fn %s(...) -> %s", node->func.name,
                node->func.ret_type ? node->func.ret_type : "void");
        lsp_index_add_def(idx, node->token, hover, node);

        // Recurse body.
        lsp_walk_node(idx, node->func.body);
    }
    else if (node->type == NODE_VAR_DECL)
    {
        char hover[256];
        sprintf(hover, "var %s", node->var_decl.name);
        lsp_index_add_def(idx, node->token, hover, node);

        lsp_walk_node(idx, node->var_decl.init_expr);
    }
    else if (node->type == NODE_CONST)
    {
        char hover[256];
        sprintf(hover, "const %s", node->var_decl.name);
        lsp_index_add_def(idx, node->token, hover, node);

        lsp_walk_node(idx, node->var_decl.init_expr);
    }
    else if (node->type == NODE_STRUCT)
    {
        char hover[256];
        if (node->strct.is_opaque)
        {
            sprintf(hover, "opaque struct %s", node->strct.name);
        }
        else
        {
            sprintf(hover, "struct %s", node->strct.name);
        }
        lsp_index_add_def(idx, node->token, hover, node);
    }
    else if (node->type == NODE_ENUM)
    {
        char hover[256];
        sprintf(hover, "enum %s", node->enm.name);
        lsp_index_add_def(idx, node->token, hover, node);
    }
    else if (node->type == NODE_TYPE_ALIAS)
    {
        char hover[256];
        sprintf(hover, "alias %s = %s", node->type_alias.alias, node->type_alias.original_type);
        lsp_index_add_def(idx, node->token, hover, node);
    }
    else if (node->type == NODE_TRAIT)
    {
        char hover[256];
        sprintf(hover, "trait %s", node->trait.name);
        lsp_index_add_def(idx, node->token, hover, node);
    }

    // Reference logic.
    if (node->definition_token.line > 0 && node->definition_token.line != node->token.line)
    {
        // It has a definition!
        lsp_index_add_ref(idx, node->token, node->definition_token, node);
    }
    else if (node->definition_token.line > 0)
    {
        lsp_index_add_ref(idx, node->token, node->definition_token, node);
    }

    // General recursion.

    switch (node->type)
    {
    case NODE_ROOT:
        lsp_walk_node(idx, node->root.children);
        break;
    case NODE_BLOCK:
        lsp_walk_node(idx, node->block.statements);
        break;
    case NODE_IF:
        lsp_walk_node(idx, node->if_stmt.condition);
        lsp_walk_node(idx, node->if_stmt.then_body);
        lsp_walk_node(idx, node->if_stmt.else_body);
        break;
    case NODE_WHILE:
        lsp_walk_node(idx, node->while_stmt.condition);
        lsp_walk_node(idx, node->while_stmt.body);
        break;
    case NODE_RETURN:
        lsp_walk_node(idx, node->ret.value);
        break;
    case NODE_EXPR_BINARY:
        lsp_walk_node(idx, node->binary.left);
        lsp_walk_node(idx, node->binary.right);
        break;
    case NODE_EXPR_CALL:
        lsp_walk_node(idx, node->call.callee);
        lsp_walk_node(idx, node->call.args);
        break;
    case NODE_MATCH:
        lsp_walk_node(idx, node->match_stmt.expr);
        lsp_walk_node(idx, node->match_stmt.cases);
        break;
    case NODE_MATCH_CASE:
        lsp_walk_node(idx, node->match_case.guard);
        lsp_walk_node(idx, node->match_case.body);
        break;
    case NODE_FOR:
        lsp_walk_node(idx, node->for_stmt.init);
        lsp_walk_node(idx, node->for_stmt.condition);
        lsp_walk_node(idx, node->for_stmt.step);
        lsp_walk_node(idx, node->for_stmt.body);
        break;
    case NODE_FOR_RANGE:
        lsp_walk_node(idx, node->for_range.start);
        lsp_walk_node(idx, node->for_range.end);
        lsp_walk_node(idx, node->for_range.body);
        break;
    case NODE_LOOP:
        lsp_walk_node(idx, node->loop_stmt.body);
        break;
    case NODE_DEFER:
        lsp_walk_node(idx, node->defer_stmt.stmt);
        break;
    default:
        break;
    }

    // Walk next sibling.
    lsp_walk_node(idx, node->next);
}

void lsp_build_index(LSPIndex *idx, ASTNode *root)
{
    lsp_walk_node(idx, root);
}
