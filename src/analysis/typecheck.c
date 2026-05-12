#include "../utils/colors.h"
#include "typecheck_internal.h"
#include "../constants.h"

#include "typecheck.h"
#include "comptime_interpreter.h"
#include "diagnostics/diagnostics.h"
#include "move_check.h"
#include "platform/misra.h"
#include <ctype.h>
#include <string.h>

// External helpers from parser
char *resolve_struct_name_from_type(ParserContext *ctx, Type *t, int *is_ptr_out,
                                    char **allocated_out);
FuncSig *find_func(ParserContext *ctx, const char *name);
ASTNode *find_trait_def(ParserContext *ctx, const char *name);
Type *type_from_string_helper(const char *c);
Type *resolve_alias(Type *t);
int integer_type_width(Type *t);
char *merge_underscores(const char *in);
int eval_const_int_expr(ASTNode *node, ParserContext *ctx, long long *out_val);
int tc_expr_has_side_effects(ASTNode *node);
int is_expression_invariant(TypeChecker *tc, ASTNode *node, int *val);

// ** Internal Helpers **

int tc_expr_has_side_effects(ASTNode *node)
{
    if (!node)
    {
        return 0;
    }

    switch (node->type)
    {
    case NODE_EXPR_CALL:
        // Function calls are always considered to have potential side effects in MISRA
        return 1;

    case NODE_EXPR_BINARY:
        // Assignment operators are side effects
        if (node->binary.op && strstr(node->binary.op, "="))
        {
            return 1;
        }
        return tc_expr_has_side_effects(node->binary.left) ||
               tc_expr_has_side_effects(node->binary.right);

    case NODE_EXPR_UNARY:
        // Increment/Decrement are side effects (prefix and postfix)
        if (node->unary.op &&
            (strcmp(node->unary.op, "++") == 0 || strcmp(node->unary.op, "--") == 0 ||
             strcmp(node->unary.op, "_post++") == 0 || strcmp(node->unary.op, "_post--") == 0))
        {
            return 1;
        }
        return tc_expr_has_side_effects(node->unary.operand);

    case NODE_RAW_STMT:
        // Inline assembly always has potential side effects
        return 1;

    case NODE_EXPR_STRUCT_INIT:
    {
        ASTNode *f = node->struct_init.fields;
        while (f)
        {
            if (tc_expr_has_side_effects(f->var_decl.init_expr))
            {
                return 1;
            }
            f = f->next;
        }
        return 0;
    }

    case NODE_EXPR_ARRAY_LITERAL:
    {
        ASTNode *e = node->array_literal.elements;
        while (e)
        {
            if (tc_expr_has_side_effects(e))
            {
                return 1;
            }
            e = e->next;
        }
        return 0;
    }

    case NODE_EXPR_TUPLE_LITERAL:
    {
        ASTNode *e = node->tuple_literal.elements;
        while (e)
        {
            if (tc_expr_has_side_effects(e))
            {
                return 1;
            }
            e = e->next;
        }
        return 0;
    }

    case NODE_EXPR_MEMBER:
        return tc_expr_has_side_effects(node->member.target);

    case NODE_EXPR_INDEX:
        return tc_expr_has_side_effects(node->index.array) ||
               tc_expr_has_side_effects(node->index.index);

    case NODE_EXPR_CAST:
        return tc_expr_has_side_effects(node->cast.expr);

    case NODE_EXPR_SLICE:
        return tc_expr_has_side_effects(node->slice.array) ||
               tc_expr_has_side_effects(node->slice.start) ||
               tc_expr_has_side_effects(node->slice.end);

    default:
        // Most other nodes (LITERAL, VAR, SIZEOF itself if nested) are side-effect free
        return 0;
    }
}

int is_expression_invariant(TypeChecker *tc, ASTNode *node, int *val);

void collect_symbols(ASTNode *node, SymbolSet *reads, SymbolSet *writes)
{
    if (!node)
    {
        return;
    }

    switch (node->type)
    {
    case NODE_EXPR_BINARY:
        if (node->binary.op && strstr(node->binary.op, "="))
        {
            // LHS is a write
            if (node->binary.left->type == NODE_EXPR_VAR)
            {
                if (writes->count < 32)
                {
                    writes->syms[writes->count++] = node->binary.left->var_ref.symbol;
                }
            }
            collect_symbols(node->binary.left, reads, writes); // In case of complex LHS
            collect_symbols(node->binary.right, reads, writes);
        }
        else
        {
            collect_symbols(node->binary.left, reads, writes);
            collect_symbols(node->binary.right, reads, writes);
        }
        break;

    case NODE_EXPR_UNARY:
        if (node->unary.op &&
            (strcmp(node->unary.op, "++") == 0 || strcmp(node->unary.op, "--") == 0 ||
             strcmp(node->unary.op, "_post++") == 0 || strcmp(node->unary.op, "_post--") == 0))
        {
            if (node->unary.operand->type == NODE_EXPR_VAR)
            {
                if (writes->count < 32)
                {
                    writes->syms[writes->count++] = node->unary.operand->var_ref.symbol;
                }
            }
        }
        collect_symbols(node->unary.operand, reads, writes);
        break;

    case NODE_EXPR_VAR:
        if (reads->count < 32)
        {
            reads->syms[reads->count++] = node->var_ref.symbol;
        }
        break;

    case NODE_EXPR_CALL:
        collect_symbols(node->call.callee, reads, writes);
        ASTNode *arg = node->call.args;
        while (arg)
        {
            collect_symbols(arg, reads, writes);
            arg = arg->next;
        }
        break;

    default:
        // Generic traversal? For now just handle these.
        break;
    }
}
void check_side_effect_collision(TypeChecker *tc, ASTNode *left, ASTNode *right, Token token)
{
    CompilerConfig *cfg = &tc->pctx->compiler->config;
    if (!cfg->misra_mode || !left || !right)
    {
        return;
    }

    SymbolSet l_reads = {0}, l_writes = {0};
    SymbolSet r_reads = {0}, r_writes = {0};

    collect_symbols(left, &l_reads, &l_writes);
    collect_symbols(right, &r_reads, &r_writes);

    // Rule 13.2: Modification collision
    for (int i = 0; i < l_writes.count; i++)
    {
        ZenSymbol *s = l_writes.syms[i];
        if (!s)
        {
            continue;
        }

        // Check against other writes
        for (int j = 0; j < r_writes.count; j++)
        {
            if (s == r_writes.syms[j])
            {
                tc_error(tc, token, "MISRA Rule 13.2: symbol modified in multiple sub-expressions");
                return;
            }
        }

        // Check against other reads
        for (int j = 0; j < r_reads.count; j++)
        {
            if (s == r_reads.syms[j])
            {
                tc_error(tc, token,
                         "MISRA Rule 13.2: symbol both read and modified in same expression");
                return;
            }
        }
    }

    // Vice versa for r_writes against l_reads
    for (int i = 0; i < r_writes.count; i++)
    {
        ZenSymbol *s = r_writes.syms[i];
        if (!s)
        {
            continue;
        }

        for (int j = 0; j < l_reads.count; j++)
        {
            if (s == l_reads.syms[j])
            {
                tc_error(tc, token,
                         "MISRA Rule 13.2: symbol both read and modified in same expression");
                return;
            }
        }
    }
}

void check_all_args_side_effects(TypeChecker *tc, ASTNode *receiver, ASTNode *args, Token token)
{
    CompilerConfig *cfg = &tc->pctx->compiler->config;
    if (!cfg->misra_mode)
    {
        return;
    }

    SymbolSet reads = {0}, writes = {0};

    if (receiver)
    {
        collect_symbols(receiver, &reads, &writes);
    }

    ASTNode *arg = args;
    while (arg)
    {
        SymbolSet next_reads = {0}, next_writes = {0};
        collect_symbols(arg, &next_reads, &next_writes);

        // Check against cumulative sets
        for (int i = 0; i < next_writes.count; i++)
        {
            ZenSymbol *s = next_writes.syms[i];
            for (int r = 0; r < reads.count; r++)
            {
                if (s == reads.syms[r])
                {
                    tc_error(tc, token, "MISRA Rule 13.2: argument read and modified in same call");
                    return;
                }
            }
            for (int w = 0; w < writes.count; w++)
            {
                if (s == writes.syms[w])
                {
                    tc_error(tc, token, "MISRA Rule 13.2: symbol modified in multiple arguments");
                    return;
                }
            }
        }
        for (int i = 0; i < next_reads.count; i++)
        {
            ZenSymbol *s = next_reads.syms[i];
            for (int w = 0; w < writes.count; w++)
            {
                if (s == writes.syms[w])
                {
                    tc_error(tc, token, "MISRA Rule 13.2: argument read and modified in same call");
                    return;
                }
            }
        }

        // Add to cumulative sets
        for (int i = 0; i < next_reads.count && reads.count < 32; i++)
        {
            reads.syms[reads.count++] = next_reads.syms[i];
        }
        for (int i = 0; i < next_writes.count && writes.count < 32; i++)
        {
            writes.syms[writes.count++] = next_writes.syms[i];
        }

        arg = arg->next;
    }
}

// Internal MISRA helpers moved to platform/misra.c

void tc_error(TypeChecker *tc, Token t, const char *msg)
{
    if (tc->move_checks_only)
    {
        return;
    }
    zerror_at(t, "%s", msg);
}

int is_expression_invariant(TypeChecker *tc, ASTNode *node, int *val)
{
    if (!node)
    {
        return 0;
    }
    long long out;
    if (eval_const_int_expr(node, tc->pctx, &out))
    {
        if (val)
        {
            *val = (int)out;
        }
        return 1;
    }
    return 0;
}

// Global recursion guard

void tc_error_with_hints(TypeChecker *tc, Token t, const char *msg, const char *const *hints)
{
    if (tc->move_checks_only)
    {
        return;
    }
    zerror_with_hints(t, msg, hints);
}

void tc_move_error_with_hints(TypeChecker *tc, Token t, const char *msg, const char *const *hints)
{
    (void)tc;
    zerror_with_hints(t, msg, hints);
}

int is_char_type(Type *t)
{
    if (!t)
    {
        return 0;
    }
    Type *res = resolve_alias(t);
    return (res->kind == TYPE_CHAR || res->kind == TYPE_C_CHAR || res->kind == TYPE_C_UCHAR);
}

// tc_check_misra_10_4 moved to misra_check_binary_op_essential_types in misra.c

void tc_enter_scope(TypeChecker *tc)
{
    tc->current_depth++;
    enter_scope(tc->pctx);
}

void tc_exit_scope(TypeChecker *tc)
{
    if (tc->current_depth > 0)
    {
        tc->current_depth--;
    }
    exit_scope(tc->pctx);
}

void tc_add_symbol(TypeChecker *tc, const char *name, Type *type, Token t, int is_immutable)
{
    CompilerConfig *cfg = &tc->pctx->compiler->config;
    if (cfg->misra_mode)
    {
        misra_check_shadowing(tc->pctx, name, t);
        misra_check_typographic_ambiguity(tc->pctx, name, t);
    }
    add_symbol_with_token(tc->pctx, name, NULL, type, t, 0);
    ZenSymbol *sym = symbol_lookup(tc->pctx->current_scope, name);
    if (sym)
    {
        sym->is_immutable = is_immutable;
        sym->scope_depth = tc->current_depth;
    }
}

ZenSymbol *tc_lookup(TypeChecker *tc, const char *name)
{
    ZenSymbol *sym = symbol_lookup(tc->pctx->current_scope, name);
    if (sym)
    {
        sym->is_used = 1;
    }
    return sym;
}

void mark_type_as_used(TypeChecker *tc, Type *t)
{
    if (!t)
    {
        return;
    }

    // Unroll pointers, arrays, vectors
    Type *curr = t;
    while (curr &&
           (curr->kind == TYPE_POINTER || curr->kind == TYPE_ARRAY || curr->kind == TYPE_VECTOR))
    {
        curr = curr->inner;
    }

    if (!curr)
    {
        return;
    }

    if (curr->kind == TYPE_STRUCT || curr->kind == TYPE_ENUM)
    {
        if (curr->name)
        {
            ZenSymbol *sym =
                symbol_lookup_kind(tc->pctx->global_scope, curr->name,
                                   (curr->kind == TYPE_STRUCT) ? SYM_STRUCT : SYM_ENUM);
            if (sym)
            {
                sym->is_used = 1;
            }
        }
    }
    else if (curr->kind == TYPE_ALIAS)
    {
        if (curr->name)
        {
            ZenSymbol *sym = symbol_lookup_kind(tc->pctx->global_scope, curr->name, SYM_ALIAS);
            if (sym)
            {
                sym->is_used = 1;
            }
        }
    }

    // Generic arguments
    for (int i = 0; i < curr->arg_count; i++)
    {
        mark_type_as_used(tc, curr->args[i]);
    }

    // Function type return and args
    if (curr->kind == TYPE_FUNCTION)
    {
        mark_type_as_used(tc, curr->inner);
        for (int i = 0; i < curr->arg_count; i++)
        {
            mark_type_as_used(tc, curr->args[i]);
        }
    }
}

// Internal MISRA helpers moved to platform/misra.c

int get_asm_register_size(Type *t)
{
    if (!t)
    {
        return 0;
    }
    if (t->kind == TYPE_F64 || t->kind == TYPE_I64 || t->kind == TYPE_U64 ||
        (t->kind == TYPE_STRUCT && t->name &&
         (0 == strcmp(t->name, "int64_t") || 0 == strcmp(t->name, "uint64_t"))))
    {
        return 64;
    }
    if (t->kind == TYPE_I128 || t->kind == TYPE_U128)
    {
        return 128;
    }
    return 32;
}

int integer_type_width(Type *t)
{
    if (!t)
    {
        return 0;
    }
    switch (t->kind)
    {
    case TYPE_I8:
    case TYPE_U8:
    case TYPE_BYTE:
    case TYPE_C_CHAR:
    case TYPE_C_UCHAR:
        return 8;
    case TYPE_I16:
    case TYPE_U16:
    case TYPE_C_SHORT:
    case TYPE_C_USHORT:
        return 16;
    case TYPE_I32:
    case TYPE_U32:
    case TYPE_INT:
    case TYPE_UINT:
    case TYPE_RUNE:
    case TYPE_C_INT:
    case TYPE_C_UINT:
        return 32;
    case TYPE_I64:
    case TYPE_U64:
    case TYPE_ISIZE:
    case TYPE_USIZE:
    case TYPE_C_LONG:
    case TYPE_C_ULONG:
    case TYPE_C_LONGLONG:
    case TYPE_C_ULONGLONG:
        return 64;
    case TYPE_I128:
    case TYPE_U128:
        return 128;
    default:
        return 0;
    }
}

// ============================================================================
// EXPRESSION CHECKERS
// ============================================================================

void check_node(TypeChecker *tc, ASTNode *node, int depth);
void check_node(TypeChecker *tc, ASTNode *node, int depth)
{
    if (!node || !tc)
    {
        return;
    }
    RECURSION_GUARD_TOKEN(tc->pctx, node->token, );
    if (depth > 1024)
    {
        tc_error(tc, node->token, "Expression too deep");
        RECURSION_EXIT(tc->pctx);
        return;
    }

    switch (node->type)
    {
    case NODE_ROOT:
    {
        ASTNode *child = node->root.children;
        while (child)
        {
            check_node(tc, child, depth + 1);
            child = child->next;
        }
    }
    break;
    case NODE_BLOCK:
        check_block(tc, node, depth + 1);
        break;
    case NODE_VAR_DECL:
        check_var_decl(tc, node, depth + 1);
        break;
    case NODE_FUNCTION:
        check_function(tc, node, depth + 1);
        break;
    case NODE_TRAIT:
        tc_check_trait(tc, node, depth + 1);
        break;
    case NODE_IMPL:
        tc_check_impl(tc, node, depth + 1);
        break;
    case NODE_IMPL_TRAIT:
        tc_check_impl_trait(tc, node, depth + 1);
        break;
    case NODE_IMPORT:
        check_node(tc, node->import_stmt.module_root, depth + 1);
        break;
    case NODE_EXPR_VAR:
        check_expr_var(tc, node);
        break;
    case NODE_EXPR_LITERAL:
        check_expr_literal(tc, node);
        break;
    case NODE_RETURN:
        tc->func_return_count++;
        if (node->ret.value)
        {
            check_node(tc, node->ret.value, depth + 1);
        }
        // Check return type compatibility with function
        if (tc->current_func)
        {
            const char *ret_type = tc->current_func->func.ret_type;
            int func_is_void = !ret_type || strcmp(ret_type, "void") == 0;

            if (func_is_void && node->ret.value)
            {
                tc_error(tc, node->token, "Return with value in void function");
            }
            else if (!func_is_void && !node->ret.value)
            {
                char msg[MAX_SHORT_MSG_LEN];
                const char *rule = tc->pctx->config->misra_mode ? "MISRA Rule 2.1: " : "";
                snprintf(msg, sizeof(msg), "%sReturn without value in function returning '%s'",
                         rule, ret_type);
                tc_error(tc, node->token, msg);
            }
            else if (node->ret.value && tc->current_func->func.ret_type_info)
            {
                apply_implicit_struct_pointer_conversions(tc, &node->ret.value,
                                                          tc->current_func->func.ret_type_info);
                check_type_compatibility(tc, tc->current_func->func.ret_type_info,
                                         node->ret.value->type_info, node->token, node->ret.value,
                                         0);
            }
        }
        tc->is_unreachable = 1;
        break;

    // Control flow with nested nodes.
    case NODE_IF:
    {
        int old_stmt_ctx = tc->is_stmt_context;
        tc->is_stmt_context = 0;
        check_node(tc, node->if_stmt.condition, depth + 1);
        tc->is_stmt_context = old_stmt_ctx;

        // Validate condition is boolean-compatible
        if (node->if_stmt.condition && node->if_stmt.condition->type_info)
        {
            Type *cond_type = resolve_alias(node->if_stmt.condition->type_info);
            if (tc->pctx->config->misra_mode)
            {
                misra_check_condition_boolean(tc->pctx, node->if_stmt.condition->type_info,
                                              node->if_stmt.condition->token);
                int inv;
                if (is_expression_invariant(tc, node->if_stmt.condition, &inv))
                {
                    misra_check_invariant_condition(tc->pctx, node->if_stmt.condition->token);
                }
            }
            else if (cond_type->kind != TYPE_BOOL && !is_integer_type(cond_type) &&
                     cond_type->kind != TYPE_POINTER && cond_type->kind != TYPE_STRING)
            {
                const char *hints[] = {"If conditions must be boolean, integer, or pointer", NULL};
                tc_error_with_hints(tc, node->if_stmt.condition->token,
                                    "Condition must be a truthy type", hints);
            }
        }

        MoveState *initial_state = tc->pctx->move_state;
        int initial_unreachable = tc->is_unreachable;

        misra_check_compound_body(tc->pctx, node->if_stmt.then_body, "if");
        if (node->if_stmt.else_body)
        {
            if (node->if_stmt.else_body->type == NODE_IF)
            {
                misra_check_terminal_else(tc->pctx, node);
            }
            else
            {
                misra_check_compound_body(tc->pctx, node->if_stmt.else_body, "else");
            }
        }

        if (initial_state)
        {
            tc->pctx->move_state = move_state_clone(initial_state);
        }
        check_node(tc, node->if_stmt.then_body, depth + 1);
        MoveState *after_then = tc->pctx->move_state;
        int then_unreachable = tc->is_unreachable;

        MoveState *after_else = NULL;
        int else_unreachable = initial_unreachable;
        tc->is_unreachable = initial_unreachable; // Reset for else branch

        if (node->if_stmt.else_body)
        {
            if (initial_state)
            {
                tc->pctx->move_state = move_state_clone(initial_state);
            }
            check_node(tc, node->if_stmt.else_body, depth + 1);
            after_else = tc->pctx->move_state;
            else_unreachable = tc->is_unreachable;
        }

        tc->pctx->move_state = initial_state;

        if (initial_state)
        {
            MoveState *merge_a = then_unreachable ? NULL : after_then;
            MoveState *merge_b =
                else_unreachable ? NULL : (node->if_stmt.else_body ? after_else : initial_state);

            // Only merge reachable paths
            move_state_merge(initial_state, merge_a, merge_b);

            if (after_then)
            {
                move_state_free(after_then);
            }
            if (after_else)
            {
                move_state_free(after_else);
            }
        }

        tc->is_unreachable = then_unreachable && else_unreachable;
        break;
    }
    case NODE_MATCH:
        check_node(tc, node->match_stmt.expr, depth + 1);
        misra_check_match_stmt(tc->pctx, node);
        // Visit each match case
        {
            MoveState *match_initial_state = tc->pctx->move_state;
            MoveState *merged_state = NULL;
            int match_initial_unreachable = tc->is_unreachable;
            int all_unreachable = 1;

            ASTNode *mcase = node->match_stmt.cases;
            int has_default = 0;
            int clause_count = 0;

            while (mcase)
            {
                if (mcase->type == NODE_MATCH_CASE)
                {
                    if (mcase->match_case.is_default)
                    {
                        has_default = 1;
                    }
                    clause_count++;

                    if (match_initial_state)
                    {
                        tc->pctx->move_state = move_state_clone(match_initial_state);
                    }
                    tc->is_unreachable = match_initial_unreachable;

                    tc_enter_scope(tc);
                    if (mcase->match_case.binding_count > 0)
                    {
                        for (int i = 0; i < mcase->match_case.binding_count; i++)
                        {
                            char *bname = mcase->match_case.binding_names[i];
                            if (bname)
                            {
                                // For now, we use UNSAFE_ANY as the binding type
                                // In a more complete implementation, we'd infer it from the enum
                                // payload
                                Type *bt = type_new(TYPE_UNSAFE_ANY);
                                if (mcase->match_case.binding_refs &&
                                    mcase->match_case.binding_refs[i])
                                {
                                    bt = type_new_ptr(bt);
                                }
                                tc_add_symbol(tc, bname, bt, mcase->token, 0);
                            }
                        }
                    }

                    check_node(tc, mcase->match_case.body, depth + 1);
                    tc_exit_scope(tc);

                    // MISRA Rule 16.3: An unconditional break or return shall terminate every
                    // switch-clause
                    if (tc->pctx->config->misra_mode && !tc->is_unreachable &&
                        mcase->match_case.body && mcase->match_case.body->type == NODE_BLOCK &&
                        mcase->match_case.body->block.statements)
                    {
                        tc_error(tc, mcase->token,
                                 "MISRA Rule 16.3: match case must end in break or return");
                    }

                    if (!tc->is_unreachable)
                    {
                        all_unreachable = 0;
                        if (tc->pctx->move_state)
                        {
                            move_state_merge_into(&merged_state, tc->pctx->move_state);
                        }
                    }

                    if (tc->pctx->move_state && tc->pctx->move_state != match_initial_state)
                    {
                        move_state_free(tc->pctx->move_state);
                    }
                }
                mcase = mcase->next;
            }

            if (!has_default)
            {
                all_unreachable = 0;
                if (match_initial_state)
                {
                    move_state_merge_into(&merged_state, match_initial_state);
                }

                if (!tc->pctx->config->misra_mode)
                {
                    const char *hints[] = {"Add a default '_' case to handle all possibilities",
                                           NULL};
                    tc_error_with_hints(tc, node->token,
                                        "Match may not be exhaustive (no default case)", hints);
                }

                misra_check_match_stmt(tc->pctx, node);
            }

            if (match_initial_state && merged_state)
            {
                tc->pctx->move_state = merged_state;
            }
            else if (!merged_state)
            {
                tc->pctx->move_state = match_initial_state;
            }

            tc->is_unreachable = all_unreachable;
        }
        break;
    case NODE_STRUCT:
    case NODE_ENUM:
    case NODE_TYPE_ALIAS:
        if (node->type == NODE_STRUCT)
        {
            misra_check_reserved_identifier(tc->pctx, node->strct.name, node->token);
            misra_check_struct_decl(tc->pctx, node);
            if (node->strct.is_union)
            {
                misra_check_union(tc->pctx, node->token);
            }
        }
        else if (node->type == NODE_ENUM)
        {
            misra_check_reserved_identifier(tc->pctx, node->enm.name, node->token);
        }
        else if (node->type == NODE_TYPE_ALIAS)
        {
            misra_check_reserved_identifier(tc->pctx, node->type_alias.alias, node->token);
        }
        break;
    case NODE_WHILE:
    case NODE_FOR:
        check_loop_passes(tc, node, depth + 1);
        break;
    case NODE_EXPR_BINARY:
        check_expr_binary(tc, node, depth + 1);
        break;
    case NODE_EXPR_UNARY:
        check_expr_unary(tc, node, depth + 1);
        break;
    case NODE_EXPR_CALL:
        check_expr_call(tc, node, depth + 1);
        break;
    case NODE_EXPR_INDEX:
        check_node(tc, node->index.array, depth + 1);
        check_node(tc, node->index.index, depth + 1);

        if (node->index.array->type_info)
        {
            Type *t = node->index.array->type_info;
            int is_ptr = 0;
            if (t->kind == TYPE_POINTER && t->inner && t->inner->kind == TYPE_STRUCT)
            {
                t = t->inner;
                is_ptr = 1;
            }

            if (t->kind == TYPE_STRUCT && t->name)
            {
                size_t tname_len = strlen(t->name);
                char *mangled_idx = xmalloc(tname_len + sizeof("__index"));
                snprintf(mangled_idx, tname_len + sizeof("__index"), "%s__index", t->name);
                char *mangled_get = xmalloc(tname_len + sizeof("__get"));
                snprintf(mangled_get, tname_len + sizeof("__get"), "%s__get", t->name);

                FuncSig *sig = find_func(tc->pctx, mangled_idx);
                char *method_name = NULL;
                if (sig)
                {
                    method_name = "index";
                }
                else
                {
                    sig = find_func(tc->pctx, mangled_get);
                    if (sig)
                    {
                        method_name = "get";
                    }
                }

                if (method_name)
                {
                    ASTNode *array = node->index.array;
                    ASTNode *idx = node->index.index;

                    node->type = NODE_EXPR_CALL;
                    memset(&node->call, 0, sizeof(node->call));

                    ASTNode *callee = ast_create(NODE_EXPR_MEMBER);
                    callee->token = node->token;
                    callee->member.target = array;
                    callee->member.field = xstrdup(method_name);
                    callee->member.is_pointer_access = is_ptr;

                    node->call.callee = callee;
                    node->call.args = idx;

                    check_expr_call(tc, node, depth + 1);
                    zfree(mangled_idx);
                    zfree(mangled_get);
                    break;
                }
                zfree(mangled_idx);
                zfree(mangled_get);
            }
            if (t->kind == TYPE_ARRAY || t->kind == TYPE_POINTER || t->kind == TYPE_VECTOR)
            {
                if (t->kind == TYPE_VECTOR && !t->inner && t->name)
                {
                    ASTNode *def = find_struct_def(tc->pctx, t->name);
                    if (def && def->type == NODE_STRUCT && def->strct.fields)
                    {
                        t->inner = def->strct.fields->type_info;
                    }
                }
                // Propagate lifetime from array/slice to the indexed element
                node->type_info = type_clone(t->inner);
                if (node->type_info && node->index.array->type_info)
                {
                    node->type_info->lifetime_depth = node->index.array->type_info->lifetime_depth;
                }
            }
        }

        // Validate index is integer
        if (node->index.index && node->index.index->type_info)
        {
            if (!is_integer_type(node->index.index->type_info))
            {
                const char *hints[] = {"Array indices must be integers", NULL};
                tc_error_with_hints(tc, node->index.index->token, "Non-integer array index", hints);
            }
        }
        break;
    case NODE_EXPR_MEMBER:
        check_node(tc, node->member.target, depth + 1);
        if (node->member.target && node->member.target->type_info)
        {
            Type *target_type = get_inner_type(node->member.target->type_info);
            if (target_type->kind == TYPE_STRUCT && target_type->name)
            {
                if (tc->pctx->config->misra_mode)
                {
                    ZenSymbol *struct_sym =
                        symbol_lookup_kind(tc->pctx->global_scope, target_type->name, SYM_STRUCT);
                    if (struct_sym)
                    {
                        struct_sym->is_dereferenced = 1;
                    }
                }
                ASTNode *struct_def = find_struct_def(tc->pctx, target_type->name);
                if (struct_def)
                {
                    ASTNode *field = struct_def->strct.fields;
                    while (field)
                    {
                        if (field->type == NODE_FIELD && field->field.name &&
                            strcmp(field->field.name, node->member.field) == 0)
                        {
                            // Propagate lifetime from struct container to the member access result
                            node->type_info = type_clone(field->type_info);
                            if (node->type_info && node->member.target->type_info)
                            {
                                // Depth must be at least that of the container.
                                // (If field itself is static/global, it will be 0, but container's
                                // depth will override)
                                node->type_info->lifetime_depth =
                                    node->member.target->type_info->lifetime_depth;
                            }
                            break;
                        }
                        field = field->next;
                    }
                }
            }

            if (!node->type_info)
            {
                int is_ptr = 0;
                char *alloc_name = NULL;
                char *struct_name =
                    resolve_struct_name_from_type(tc->pctx, target_type, &is_ptr, &alloc_name);

                if (struct_name)
                {
                    char buf[MAX_ERROR_MSG_LEN];
                    snprintf(buf, sizeof(buf), "%s__%s", struct_name, node->member.field);
                    char *mangled = merge_underscores(buf);

                    FuncSig *sig = find_func(tc->pctx, mangled);
                    if (sig)
                    {
                        node->type_info = sig->ret_type;
                    }
                    zfree(mangled);
                }
                if (alloc_name)
                {
                    zfree(alloc_name);
                }
            }
        }
        if (!node->type_info)
        {
            // Fallback for failed lookups
            node->type_info = type_new(TYPE_UNKNOWN);
        }

        if (!tc->is_assign_lhs)
        {
            check_use_validity(tc, node);
        }
        break;
    case NODE_DEFER:
        // Check the deferred statement
        check_node(tc, node->defer_stmt.stmt, depth + 1);
        break;
    case NODE_GUARD:
        // Guard clause: if !condition return
        {
            int old_stmt_ctx = tc->is_stmt_context;
            tc->is_stmt_context = 0;
            check_node(tc, node->guard_stmt.condition, depth + 1);
            tc->is_stmt_context = old_stmt_ctx;
        }
        if (node->guard_stmt.condition && node->guard_stmt.condition->type_info)
        {
            Type *cond_type = resolve_alias(node->guard_stmt.condition->type_info);
            if (tc->pctx->config->misra_mode)
            {
                misra_check_condition_boolean(tc->pctx, node->guard_stmt.condition->type_info,
                                              node->guard_stmt.condition->token);
            }
            else if (cond_type->kind != TYPE_BOOL && !is_integer_type(cond_type) &&
                     cond_type->kind != TYPE_POINTER && cond_type->kind != TYPE_STRING)
            {
                const char *hints[] = {"Guard conditions must be boolean, integer, or pointer",
                                       NULL};
                tc_error_with_hints(tc, node->guard_stmt.condition->token,
                                    "Condition must be a truthy type", hints);
            }
        }
        check_node(tc, node->guard_stmt.body, depth + 1);
        break;
    case NODE_UNLESS:
        // Unless is like if !condition
        {
            int old_stmt_ctx = tc->is_stmt_context;
            tc->is_stmt_context = 0;
            check_node(tc, node->unless_stmt.condition, depth + 1);
            tc->is_stmt_context = old_stmt_ctx;
        }
        if (node->unless_stmt.condition && node->unless_stmt.condition->type_info)
        {
            Type *cond_type = resolve_alias(node->unless_stmt.condition->type_info);
            if (tc->pctx->config->misra_mode)
            {
                misra_check_condition_boolean(tc->pctx, node->unless_stmt.condition->type_info,
                                              node->unless_stmt.condition->token);
            }
            else if (cond_type->kind != TYPE_BOOL && !is_integer_type(cond_type) &&
                     cond_type->kind != TYPE_POINTER && cond_type->kind != TYPE_STRING)
            {
                const char *hints[] = {"Unless conditions must be boolean, integer, or pointer",
                                       NULL};
                tc_error_with_hints(tc, node->unless_stmt.condition->token,
                                    "Condition must be a truthy type", hints);
            }
        }
        check_node(tc, node->unless_stmt.body, depth + 1);
        break;
    case NODE_EXPECT:
    case NODE_ASSERT:
        // Check assert/expect condition
        {
            int old_stmt_ctx = tc->is_stmt_context;
            tc->is_stmt_context = 0;
            check_node(tc, node->assert_stmt.condition, depth + 1);
            tc->is_stmt_context = old_stmt_ctx;
        }
        if (node->assert_stmt.condition && node->assert_stmt.condition->type_info)
        {
            Type *cond_type = resolve_alias(node->assert_stmt.condition->type_info);
            if (tc->pctx->config->misra_mode)
            {
                misra_check_condition_boolean(tc->pctx, node->assert_stmt.condition->type_info,
                                              node->assert_stmt.condition->token);
            }
            else if (cond_type->kind != TYPE_BOOL && !is_integer_type(cond_type) &&
                     cond_type->kind != TYPE_POINTER && cond_type->kind != TYPE_STRING)
            {
                const char *hints[] = {
                    "Assert/expect conditions must be boolean, integer, or pointer", NULL};
                tc_error_with_hints(tc, node->assert_stmt.condition->token,
                                    "Assert/expect condition must be a truthy type", hints);
            }
        }
        break;
    case NODE_TEST:
    {
        MoveState *prev_move_state = tc->pctx->move_state;
        tc->pctx->move_state = move_state_create(NULL);

        check_node(tc, node->test_stmt.body, depth + 1);

        move_state_free(tc->pctx->move_state);
        tc->pctx->move_state = prev_move_state;
        break;
    }

    case NODE_EXPR_CAST:
        // Check the expression being cast
        check_node(tc, node->cast.expr, depth + 1);
        // Could add cast safety checks here (e.g., narrowing, pointer-to-int)
        if (node->cast.expr && node->cast.expr->type_info && node->cast.target_type)
        {
            Type *source_type = resolve_alias(node->cast.expr->type_info);
            Type *target_type = type_from_string_helper(node->cast.target_type);

            if (tc->pctx->config->misra_mode && target_type)
            {
                misra_check_cast(tc->pctx, target_type, source_type, node->token,
                                 is_composite_expression(node->cast.expr));
                misra_check_pointer_conversion(tc->pctx, target_type, source_type, node->token);
                misra_check_void_ptr_cast(tc->pctx, target_type, source_type, node->token);
                if (target_type->kind == TYPE_POINTER)
                {
                    misra_check_null_pointer_constant(tc->pctx, node, node->token);
                }
            }

            // Warn on pointer-to-integer casts (potential data loss)
            if (source_type->kind == TYPE_POINTER)
            {
                const char *target = node->cast.target_type;
                if (strcmp(target, "i8") == 0 || strcmp(target, "i16") == 0 ||
                    strcmp(target, "u8") == 0 || strcmp(target, "u16") == 0)
                {
                    const char *hints[] = {"Pointer-to-small-integer casts may lose address bits",
                                           NULL};
                    tc_error_with_hints(tc, node->token, "Potentially unsafe pointer cast", hints);
                }
            }
            node->type_info = target_type;
            mark_type_as_used(tc, target_type);
        }
        break;
    case NODE_EXPR_ARRAY_LITERAL:
    {
        misra_check_initializer_side_effects(tc->pctx, node);
        ASTNode *elem = node->array_literal.elements;
        Type *elem_type = NULL;
        int count = 0;
        while (elem)
        {
            check_node(tc, elem, depth + 1);
            if (!elem_type && elem->type_info && elem->type_info->kind != TYPE_UNKNOWN)
            {
                elem_type = elem->type_info;
            }
            count++;
            elem = elem->next;
        }
        if (elem_type)
        {
            node->type_info = type_new_array(elem_type, count);
        }
        else
        {
            node->type_info = type_new_array(type_new(TYPE_UNKNOWN), count);
        }
    }
    break;
    case NODE_EXPR_TUPLE_LITERAL:
    {
        misra_check_initializer_side_effects(tc->pctx, node);
        ASTNode *elem = node->tuple_literal.elements;
        while (elem)
        {
            check_node(tc, elem, depth + 1);
            elem = elem->next;
        }
    }
    break;
    case NODE_EXPR_STRUCT_INIT:
        misra_check_initializer_side_effects(tc->pctx, node);
        check_struct_init(tc, node, depth + 1);
        break;
    case NODE_LOOP:
    case NODE_REPEAT:
        check_loop_passes(tc, node, depth + 1);
        break;
    case NODE_TERNARY:
        check_node(tc, node->ternary.cond, depth + 1);
        check_node(tc, node->ternary.true_expr, depth + 1);
        check_node(tc, node->ternary.false_expr, depth + 1);
        // Validate condition
        if (node->ternary.cond && node->ternary.cond->type_info)
        {
            Type *t = node->ternary.cond->type_info;
            if (tc->pctx->config->misra_mode)
            {
                misra_check_condition_boolean(tc->pctx, node->ternary.cond->type_info,
                                              node->ternary.cond->token);
                int inv;
                if (is_expression_invariant(tc, node->ternary.cond, &inv))
                {
                    misra_check_invariant_condition(tc->pctx, node->ternary.cond->token);
                }
            }
            else if (t->kind != TYPE_BOOL && !is_integer_type(t) && t->kind != TYPE_POINTER)
            {
                tc_error(tc, node->ternary.cond->token, "Ternary condition must be truthy");
            }
        }
        // Validate branch compatibility
        if (node->ternary.true_expr && node->ternary.false_expr)
        {
            Type *t1 = node->ternary.true_expr->type_info;
            Type *t2 = node->ternary.false_expr->type_info;
            if (t1 && t2)
            {
                // Loose compatibility check
                if (!check_type_compatibility(tc, t1, t2, node->token, NULL, 0))
                {
                    // Error reported by check_type_compatibility
                }
                else
                {
                    node->type_info = t1; // Inherit type
                }
            }
        }
        break;
    case NODE_ASM:
        for (int i = 0; i < node->asm_stmt.num_outputs; i++)
        {
            ZenSymbol *sym = tc_lookup(tc, node->asm_stmt.outputs[i]);
            if (!sym)
            {
                char msg[MAX_SHORT_MSG_LEN];
                if (tc->pctx->config->misra_mode)
                {
                    snprintf(msg, sizeof(msg),
                             "Undefined output variable in inline assembly: '%s' (MISRA Rule 17.3)",
                             node->asm_stmt.outputs[i]);
                }
                else
                {
                    snprintf(msg, sizeof(msg), "Undefined output variable in inline assembly: '%s'",
                             node->asm_stmt.outputs[i]);
                }
                tc_error(tc, node->token, msg);
            }
            else if (sym->type_info)
            {
                int width = get_asm_register_size(sym->type_info);
                if (width > node->asm_stmt.register_size)
                {
                    node->asm_stmt.register_size = width;
                }
            }
        }
        for (int i = 0; i < node->asm_stmt.num_inputs; i++)
        {
            ZenSymbol *sym = tc_lookup(tc, node->asm_stmt.inputs[i]);
            if (!sym)
            {
                char msg[MAX_SHORT_MSG_LEN];
                if (tc->pctx->config->misra_mode)
                {
                    snprintf(msg, sizeof(msg),
                             "Undefined input variable in inline assembly: '%s' (MISRA Rule 17.3)",
                             node->asm_stmt.inputs[i]);
                }
                else
                {
                    snprintf(msg, sizeof(msg), "Undefined input variable in inline assembly: '%s'",
                             node->asm_stmt.inputs[i]);
                }
                tc_error(tc, node->token, msg);
            }
            else if (sym->type_info)
            {
                int width = get_asm_register_size(sym->type_info);
                if (width > node->asm_stmt.register_size)
                {
                    node->asm_stmt.register_size = width;
                }
            }
        }
        if (node->asm_stmt.register_size > 64)
        {
            char msg[MAX_SHORT_MSG_LEN];
            snprintf(msg, sizeof(msg),
                     "Unsupported register size is required in inline assembly: %i bits",
                     node->asm_stmt.register_size);
            tc_error(tc, node->token, msg);
        }
        break;
    case NODE_LAMBDA:
        check_expr_lambda(tc, node, depth + 1);
        break;
    case NODE_EXPR_SIZEOF:
        if (node->size_of.expr)
        {
            check_node(tc, node->size_of.expr, depth + 1);

            if (tc->pctx->config->misra_mode && tc_expr_has_side_effects(node->size_of.expr))
            {
                misra_check_side_effects_sizeof(tc->pctx, node->size_of.expr);
            }
        }
        node->type_info = type_new(TYPE_I32);
        break;
    case NODE_FOR_RANGE:
        check_loop_passes(tc, node, depth + 1);
        break;
    case NODE_EXPR_SLICE:
        // Check slice target and indices
        check_node(tc, node->slice.array, depth + 1);
        check_node(tc, node->slice.start, depth + 1);
        check_node(tc, node->slice.end, depth + 1);
        break;
    case NODE_DESTRUCT_VAR:
        if (node->destruct.init_expr)
        {
            check_node(tc, node->destruct.init_expr, depth + 1);
        }
        break;
    case NODE_DO_WHILE:
        check_loop_passes(tc, node, depth + 1);
        break;
    case NODE_BREAK:
        if (tc->loop_break_count > 0)
        {
            misra_check_iteration_termination(tc->pctx, node->token);
        }
        tc->loop_break_count++;

        if (tc->move_checks_only)
        {
            // No-op
        }
        else if (tc->pctx->move_state)
        {
            move_state_merge_into(&tc->loop_break_state, tc->pctx->move_state);
        }
        tc->is_unreachable = 1;
        break;
    case NODE_GOTO:
        if (tc->pctx->config->misra_mode)
        {
            ZenSymbol *lbl = tc_lookup(tc, node->goto_stmt.label_name);
            if (lbl && lbl->decl_token.line != 0)
            {
                misra_check_goto_constraint(tc->pctx, node->token, lbl->decl_token);
            }
        }
        misra_check_goto(tc->pctx, node->token);
        tc->is_unreachable = 1;
        break;

    case NODE_CONTINUE:
        if (tc->pctx->move_state)
        {
            move_state_merge_into(&tc->loop_continue_state, tc->pctx->move_state);
        }
        tc->is_unreachable = 1;
        break;
    case NODE_VA_START:
    case NODE_VA_END:
    case NODE_VA_COPY:
    case NODE_VA_ARG:
        misra_check_stdarg(tc->pctx, node->token);
        break;

    case NODE_RAW_STMT:
        misra_check_raw_block(tc->pctx, node->token);
        break;
    case NODE_PREPROC_DIRECTIVE:
        // Rule Zen 1.4 is already handled by parser_audit_preprocessor
        break;
    case NODE_PLUGIN:
        misra_check_plugin_block(tc->pctx, node->token);
        break;
    case NODE_LABEL:
        if (tc->pctx->config->misra_mode)
        {
            ZenSymbol *lbl =
                symbol_add(tc->pctx->current_scope, node->label_stmt.label_name, SYM_LABEL);
            if (lbl)
            {
                lbl->decl_token = node->token;
            }
        }
        break;
    case NODE_COMPTIME:
    {
        // Register comptime builtins for the body
        register_comptime_builtins(tc->pctx);

        // Type-check the comptime body
        ASTNode *stmt = node->comptime.body;
        while (stmt)
        {
            check_node(tc, stmt, depth + 1);
            stmt = stmt->next;
        }

        // Interpret the comptime body
        char *output = interpret_comptime(tc->pctx, node->comptime.body, g_current_filename);
        if (!output)
        {
            break;
        }

        // Parse generated source code
        if (output[0])
        {
            Lexer out_l;
            lexer_init(&out_l, output, tc->pctx->config);
            node->comptime.generated = parse_program_nodes(tc->pctx, &out_l);

            // Type-check generated nodes
            ASTNode *gen = node->comptime.generated;
            while (gen)
            {
                check_node(tc, gen, depth + 1);
                gen = gen->next;
            }
        }
        zfree(output);
        break;
    }

    default:
        // Generic recursion for lists and other nodes.
        // Special case for Return to trigger move?
        if (node->type == NODE_RETURN && node->ret.value)
        {
            // If returning a value, check if it can be moved.
            check_move_for_rvalue(tc, node->ret.value);
        }
        break;
    }
    RECURSION_EXIT(tc->pctx);
}

void infer_node_lifetime(TypeChecker *tc, ASTNode *node)
{
    if (!node || node->type != NODE_FUNCTION)
    {
        return;
    }

    FuncSig *fsig = find_func(tc->pctx, node->func.name);
    if (!fsig)
    {
        return;
    }

    // Default to local argument scope (depth 1)
    int inferred_depth = 1;
    int ptr_param_count = 0;
    int self_depth = -1;
    int elide_idx = -1;

    for (int i = 0; i < fsig->total_args; i++)
    {
        Type *t = (fsig->arg_types && fsig->arg_types[i]) ? fsig->arg_types[i] : NULL;
        if (t && t->kind == TYPE_POINTER)
        {
            ptr_param_count++;
            // Parameters are always at least depth 1 (argument scope)
            if (t->lifetime_depth == 0)
            {
                t->lifetime_depth = 1;
            }

            if (node->func.param_names && node->func.param_names[i] &&
                strcmp(node->func.param_names[i], "self") == 0)
            {
                self_depth = t->lifetime_depth;
                elide_idx = i;
            }
        }
    }

    if (self_depth != -1)
    {
        inferred_depth = self_depth;
    }
    else if (ptr_param_count == 1)
    {
        for (int i = 0; i < fsig->total_args; i++)
        {
            Type *t = (fsig->arg_types && fsig->arg_types[i]) ? fsig->arg_types[i] : NULL;
            if (t && t->kind == TYPE_POINTER)
            {
                inferred_depth = t->lifetime_depth;
                elide_idx = i;
                break;
            }
        }
    }

    node->func.elide_from_idx = elide_idx;
    fsig->elide_from_idx = elide_idx;

    // Update the return type depth in the signature
    if (fsig->ret_type && fsig->ret_type->kind == TYPE_POINTER)
    {
        fsig->ret_type->lifetime_depth = inferred_depth;
    }

    // Also update AST node if types are already resolved there
    if (node->func.ret_type_info && node->func.ret_type_info->kind == TYPE_POINTER)
    {
        node->func.ret_type_info->lifetime_depth = inferred_depth;
    }
}

void check_program_prepass(TypeChecker *tc, ASTNode *root, int depth)
{
    if (!root || root->type != NODE_ROOT)
    {
        return;
    }
    RECURSION_GUARD_TOKEN(tc->pctx, root->token, );

    if (depth > 64)
    {
        RECURSION_EXIT(tc->pctx);
        return;
    }

    ASTNode *n = root->root.children;
    while (n)
    {
        if (n->type == NODE_ROOT)
        {
            check_program_prepass(tc, n, depth + 1);
        }
        else if (n->type == NODE_FUNCTION)
        {
            infer_node_lifetime(tc, n);
        }
        else if (n->type == NODE_IMPL)
        {
            ASTNode *method = n->impl.methods;
            while (method)
            {
                if (method->type == NODE_FUNCTION)
                {
                    infer_node_lifetime(tc, method);
                }
                method = method->next;
            }
        }
        else if (n->type == NODE_IMPL_TRAIT)
        {
            ASTNode *method = n->impl_trait.methods;
            while (method)
            {
                if (method->type == NODE_FUNCTION)
                {
                    infer_node_lifetime(tc, method);
                }
                method = method->next;
            }
        }
        else if (n->type == NODE_IMPORT)
        {
            // Imports are conceptually ROOTs of their own module
            check_program_prepass(tc, n->import_stmt.module_root, depth + 1);
        }
        n = n->next;
    }
    RECURSION_EXIT(tc->pctx);
}

// ** Entry Point **

int check_program(ParserContext *ctx, ASTNode *root)
{
    TypeChecker tc = {0};
    tc.pctx = ctx;
    ctx->current_scope = ctx->global_scope;

    if (!ctx->move_state)
    {
        ctx->move_state = move_state_create(NULL);
    }

    check_program_prepass(&tc, root, 0);

    check_node(&tc, root, 0);
    root->is_checked = 1;

    // Fixed-point iteration to handle secondary instantiations
    int changed = 1;
    while (changed)
    {
        changed = 0;
        ASTNode *inst_func = ctx->instantiated_funcs;
        while (inst_func)
        {
            if (!inst_func->is_checked)
            {
                check_node(&tc, inst_func, 0);
                inst_func->is_checked = 1;
                changed = 1;
                // Restart from head to catch newly added instantiations that might be before us
                break;
            }
            inst_func = inst_func->next;
        }
    }

    if (ctx->move_state)
    {
        move_state_free(ctx->move_state);
        ctx->move_state = NULL;
    }

    if (tc.pctx->config->misra_mode)
    {
        misra_audit_unused_symbols(tc.pctx);
        misra_audit_block_scope(tc.pctx);
        misra_audit_identifier_uniqueness(tc.pctx);
    }

    if (g_error_count > 0)
    {
        fprintf(stderr,
                COLOR_BOLD COLOR_RED "     error" COLOR_RESET
                                     ": semantic analysis found %d error%s\n",
                g_error_count, g_error_count == 1 ? "" : "s");
        return 1;
    }

    return 0;
}

int check_moves_only(ParserContext *ctx, ASTNode *root)
{
    TypeChecker tc = {0};
    tc.pctx = ctx;
    tc.move_checks_only = 1;
    ctx->current_scope = ctx->global_scope;

    if (!ctx->move_state)
    {
        ctx->move_state = move_state_create(NULL);
    }

    check_node(&tc, root, 0);

    if (ctx->move_state)
    {
        move_state_free(ctx->move_state);
        ctx->move_state = NULL;
    }

    return g_error_count;
}
