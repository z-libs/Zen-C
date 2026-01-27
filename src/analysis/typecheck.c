
#include "typecheck.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ** Internal Helpers **

static void tc_error(TypeChecker *tc, Token t, const char *msg)
{
    fprintf(stderr, "Type Error at %s:%d:%d: %s\n", g_current_filename, t.line, t.col, msg);
    tc->error_count++;
}

static void tc_enter_scope(TypeChecker *tc)
{
    Scope *s = malloc(sizeof(Scope));
    if (!s) return;
    s->symbols = NULL;
    s->parent = tc->current_scope;
    tc->current_scope = s;
}

static void tc_exit_scope(TypeChecker *tc)
{
    if (!tc->current_scope)
    {
        return;
    }
    Scope *old = tc->current_scope;
    tc->current_scope = old->parent;

    ZenSymbol *sym = old->symbols;
    while (sym)
    {
        ZenSymbol *next = sym->next;
        free(sym);
        sym = next;
    }
    free(old);
}

static void tc_add_symbol(TypeChecker *tc, const char *name, Type *type, Token t)
{
    ZenSymbol *s = malloc(sizeof(ZenSymbol));
    memset(s, 0, sizeof(ZenSymbol));
    s->name = strdup(name);
    s->type_info = type;
    s->decl_token = t;
    s->next = tc->current_scope->symbols;
    tc->current_scope->symbols = s;
}

static ZenSymbol *tc_lookup(TypeChecker *tc, const char *name)
{
    Scope *s = tc->current_scope;
    while (s)
    {
        ZenSymbol *curr = s->symbols;
        while (curr)
        {
            if (0 == strcmp(curr->name, name))
            {
                return curr;
            }
            curr = curr->next;
        }
        s = s->parent;
    }
    return NULL;
}

// ** Move Semantics Helpers **

static int is_safe_to_copy(TypeChecker *tc, Type *t)
{
    // Use parser's helper if available, or simple heuristic
    return is_type_copy(tc->pctx, t);
}

static void check_use_validity(TypeChecker *tc, ASTNode *var_node, ZenSymbol *sym)
{
    if (!sym || !var_node)
    {
        return;
    }

    if (sym->is_moved)
    {
        char msg[256];
        snprintf(
            msg, 255,
            "Use of moved value '%s'. This type owns resources and cannot be implicitly copied.",
            sym->name);
        tc_error(tc, var_node->token, msg);
    }
}

static void mark_symbol_moved(TypeChecker *tc, ZenSymbol *sym, ASTNode *context_node)
{
    (void)context_node;
    if (!sym)
    {
        return;
    }

    // Only move if type is NOT Copy
    Type *t = sym->type_info;
    if (t && !is_safe_to_copy(tc, t))
    {
        sym->is_moved = 1;
    }
}

static void mark_symbol_valid(TypeChecker *tc, ZenSymbol *sym)
{
    (void)tc;
    if (sym)
    {
        sym->is_moved = 0;
    }
}

// ** Node Checkers **

static void check_node(TypeChecker *tc, ASTNode *node);

static void check_expr_binary(TypeChecker *tc, ASTNode *node)
{
    check_node(tc, node->binary.left);
    check_node(tc, node->binary.right);

    // Assignment Logic for Moves
    if (strcmp(node->binary.op, "=") == 0)
    {
        // If RHS is a var, it might Move
        if (node->binary.right->type == NODE_EXPR_VAR)
        {
            ZenSymbol *rhs_sym = tc_lookup(tc, node->binary.right->var_ref.name);
            if (rhs_sym)
            {
                mark_symbol_moved(tc, rhs_sym, node);
            }
        }

        // LHS is being (re-)initialized, so it becomes Valid.
        if (node->binary.left->type == NODE_EXPR_VAR)
        {
            ZenSymbol *lhs_sym = tc_lookup(tc, node->binary.left->var_ref.name);
            if (lhs_sym)
            {
                mark_symbol_valid(tc, lhs_sym);
            }
        }
    }
}

static void check_expr_call(TypeChecker *tc, ASTNode *node)
{
    check_node(tc, node->call.callee);

    // Check arguments
    ASTNode *arg = node->call.args;
    while (arg)
    {
        check_node(tc, arg);

        // If argument is passed by VALUE, and it's a variable, it MOVES.
        // If passed by ref (UNARY '&'), the child was checked but Is Not A Var Node itself.
        if (arg->type == NODE_EXPR_VAR)
        {
            ZenSymbol *sym = tc_lookup(tc, arg->var_ref.name);
            if (sym)
            {
                mark_symbol_moved(tc, sym, node);
            }
        }

        arg = arg->next;
    }
}

static void check_block(TypeChecker *tc, ASTNode *block)
{
    tc_enter_scope(tc);
    ASTNode *stmt = block->block.statements;
    while (stmt)
    {
        check_node(tc, stmt);
        stmt = stmt->next;
    }
    tc_exit_scope(tc);
}

static int check_type_compatibility(TypeChecker *tc, Type *target, Type *value, Token t)
{
    if (!target || !value)
    {
        return 1; // Can't check
    }

    // Simple equality check for now... This will be changed.
    if (!type_eq(target, value))
    {

        // For now we have strict equality on structure.

        // In Zen C (like C), void* is generic.
        if (TYPE_POINTER == target->kind && TYPE_VOID == target->inner->kind)
        {
            return 1;
        }
        if (TYPE_POINTER == value->kind && TYPE_VOID == value->inner->kind)
        {
            return 1;
        }

        // Exception: integer promotion/demotion.

        if (is_integer_type(target) && is_integer_type(value))
        {
            return 1;
        }

        char *t_str = type_to_string(target);
        char *v_str = type_to_string(value);
        char msg[256];
        snprintf(msg, 255, "Type mismatch: expected '%s', got '%s'", t_str, v_str);
        tc_error(tc, t, msg);
        free(t_str);
        free(v_str);
        return 0;
    }
    return 1;
}

static void check_var_decl(TypeChecker *tc, ASTNode *node)
{
    if (node->var_decl.init_expr)
    {
        check_node(tc, node->var_decl.init_expr);

        Type *decl_type = node->type_info;
        Type *init_type = node->var_decl.init_expr->type_info;

        if (decl_type && init_type)
        {
            check_type_compatibility(tc, decl_type, init_type, node->token);
        }

        // Move Analysis: If initializing from another variable, it moves.
        if (node->var_decl.init_expr->type == NODE_EXPR_VAR)
        {
            ZenSymbol *init_sym = tc_lookup(tc, node->var_decl.init_expr->var_ref.name);
            if (init_sym)
            {
                mark_symbol_moved(tc, init_sym, node);
            }
        }
    }

    // If type is not explicit, we should ideally infer it from init_expr.
    Type *t = node->type_info;
    if (!t && node->var_decl.init_expr)
    {
        t = node->var_decl.init_expr->type_info;
    }

    tc_add_symbol(tc, node->var_decl.name, t, node->token);
}

static void check_function(TypeChecker *tc, ASTNode *node)
{
    // Just to suppress the warning.
    (void)tc_error;

    tc->current_func = node;
    tc_enter_scope(tc);

    for (int i = 0; i < node->func.arg_count; i++)
    {
        if (node->func.param_names && node->func.param_names[i])
        {
            tc_add_symbol(tc, node->func.param_names[i], NULL, (Token){0});
        }
    }

    check_node(tc, node->func.body);

    tc_exit_scope(tc);
    tc->current_func = NULL;
}

static void check_expr_var(TypeChecker *tc, ASTNode *node)
{
    ZenSymbol *sym = tc_lookup(tc, node->var_ref.name);
    if (!sym)
    {
        // Check global functions/contexts if not found in locals
        // This is a naive check.
        // We really want to warn here if it's truly unknown.
    }
    if (sym && sym->type_info)
    {
        node->type_info = sym->type_info;
    }

    // Check for Use-After-Move
    check_use_validity(tc, node, sym);
}

static void check_node(TypeChecker *tc, ASTNode *node)
{
    if (!node)
    {
        return;
    }

    switch (node->type)
    {
    case NODE_ROOT:
        check_node(tc, node->root.children);
        break;
    case NODE_BLOCK:
        check_block(tc, node);
        break;
    case NODE_VAR_DECL:
        check_var_decl(tc, node);
        break;
    case NODE_FUNCTION:
        check_function(tc, node);
        break;
    case NODE_EXPR_VAR:
        check_expr_var(tc, node);
        break;
    case NODE_RETURN:
        if (node->ret.value)
        {
            check_node(tc, node->ret.value);
        }

        break;

    // Control flow with nested nodes.
    case NODE_IF:
        check_node(tc, node->if_stmt.condition);
        check_node(tc, node->if_stmt.then_body);
        check_node(tc, node->if_stmt.else_body);
        break;
    case NODE_WHILE:
        check_node(tc, node->while_stmt.condition);
        check_node(tc, node->while_stmt.body);
        break;
    case NODE_FOR:
        tc_enter_scope(tc); // For loop init variable is scoped
        check_node(tc, node->for_stmt.init);
        check_node(tc, node->for_stmt.condition);
        check_node(tc, node->for_stmt.step);
        check_node(tc, node->for_stmt.body);
        tc_exit_scope(tc);
        break;
    case NODE_EXPR_BINARY:
        check_expr_binary(tc, node);
        break;
    case NODE_EXPR_CALL:
        check_expr_call(tc, node);
        break;
    default:
        // Generic recursion for lists and other nodes.
        // Special case for Return to trigger move?
        if (node->type == NODE_RETURN && node->ret.value)
        {
            // If returning a variable by value, it is moved.
            if (node->ret.value->type == NODE_EXPR_VAR)
            {
                ZenSymbol *sym = tc_lookup(tc, node->ret.value->var_ref.name);
                if (sym)
                {
                    mark_symbol_moved(tc, sym, node);
                }
            }
        }
        break;
    }

    if (node->next)
    {
        check_node(tc, node->next);
    }
}

// ** Entry Point **

int check_program(ParserContext *ctx, ASTNode *root)
{
    TypeChecker tc = {0};
    tc.pctx = ctx;

    printf("[TypeCheck] Starting semantic analysis...\n");
    check_node(&tc, root);

    if (tc.error_count > 0)
    {
        printf("[TypeCheck] Found %d errors.\n", tc.error_count);
        return 1;
    }
    printf("[TypeCheck] Passed.\n");
    return 0;
}
