#include "../constants.h"

#include "typecheck.h"
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
static int integer_type_width(Type *t);
char *merge_underscores(const char *in);
int eval_const_int_expr(ASTNode *node, ParserContext *ctx, long long *out_val);
int tc_expr_has_side_effects(ASTNode *node);
static int is_expression_invariant(TypeChecker *tc, ASTNode *node, int *val);

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

static int is_expression_invariant(TypeChecker *tc, ASTNode *node, int *val);

typedef struct
{
    ZenSymbol *syms[32];
    int count;
} SymbolSet;

static void collect_symbols(ASTNode *node, SymbolSet *reads, SymbolSet *writes)
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
static void check_side_effect_collision(TypeChecker *tc, ASTNode *left, ASTNode *right, Token token)
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

static void check_all_args_side_effects(TypeChecker *tc, ASTNode *receiver, ASTNode *args,
                                        Token token)
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

static int is_expression_invariant(TypeChecker *tc, ASTNode *node, int *val)
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

static int is_char_type(Type *t)
{
    if (!t)
    {
        return 0;
    }
    Type *res = resolve_alias(t);
    return (res->kind == TYPE_CHAR || res->kind == TYPE_C_CHAR || res->kind == TYPE_C_UCHAR);
}

// tc_check_misra_10_4 moved to misra_check_binary_op_essential_types in misra.c

static void tc_enter_scope(TypeChecker *tc)
{
    tc->current_depth++;
    enter_scope(tc->pctx);
}

static void tc_exit_scope(TypeChecker *tc)
{
    if (tc->current_depth > 0)
    {
        tc->current_depth--;
    }
    exit_scope(tc->pctx);
}

static void tc_add_symbol(TypeChecker *tc, const char *name, Type *type, Token t, int is_immutable)
{
    CompilerConfig *cfg = &tc->pctx->compiler->config;
    if (cfg->misra_mode)
    {
        misra_check_shadowing(tc, name, t);
        misra_check_typographic_ambiguity(tc, name, t);
    }
    add_symbol_with_token(tc->pctx, name, NULL, type, t, 0);
    ZenSymbol *sym = symbol_lookup(tc->pctx->current_scope, name);
    if (sym)
    {
        sym->is_immutable = is_immutable;
        sym->scope_depth = tc->current_depth;
    }
}

static ZenSymbol *tc_lookup(TypeChecker *tc, const char *name)
{
    ZenSymbol *sym = symbol_lookup(tc->pctx->current_scope, name);
    if (sym)
    {
        sym->is_used = 1;
    }
    return sym;
}

static void mark_type_as_used(TypeChecker *tc, Type *t)
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

static int get_asm_register_size(Type *t)
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

static int integer_type_width(Type *t)
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

// ** Node Checkers **

static void check_node(TypeChecker *tc, ASTNode *node, int depth);
static void check_expr_lambda(TypeChecker *tc, ASTNode *node, int depth);
static void apply_implicit_struct_pointer_conversions(TypeChecker *tc, ASTNode **expr_ptr,
                                                      Type *expected_type);
static int check_type_compatibility(TypeChecker *tc, Type *target, Type *value, Token t,
                                    ASTNode *value_node, int is_call_arg);

static void check_move_for_rvalue(TypeChecker *tc, ASTNode *rvalue)
{
    if (!rvalue || !rvalue->type_info)
    {
        return;
    }

    if (is_type_copy(tc->pctx, rvalue->type_info))
    {
        return;
    }

    if (rvalue->type == NODE_EXPR_VAR)
    {
        ZenSymbol *sym = tc_lookup(tc, rvalue->var_ref.name);
        if (sym)
        {
            mark_symbol_moved(tc->pctx, sym, rvalue);
        }
    }
    else if (rvalue->type == NODE_EXPR_UNARY && strcmp(rvalue->unary.op, "*") == 0)
    {
        const char *hints[] = {"This type owns resources and cannot be implicitly copied",
                               "Consider borrowing value via references or implementing Copy",
                               NULL};
        tc_move_error_with_hints(tc, rvalue->token, "Cannot move out of a borrowed reference",
                                 hints);
    }
    else if (rvalue->type == NODE_EXPR_MEMBER)
    {
        // Now allowed, but will be tracked by path
        mark_symbol_moved(tc->pctx, NULL, rvalue);
    }
    else if (rvalue->type == NODE_EXPR_INDEX)
    {
        const char *hints[] = {"Cannot move an element out of an array or slice.", NULL};
        tc_move_error_with_hints(tc, rvalue->token, "Cannot move out of an index expression",
                                 hints);
    }
}

Type *resolve_alias(Type *t)
{
    while (t && t->kind == TYPE_ALIAS && t->inner)
    {
        t = t->inner;
    }
    return t;
}

static void check_expr_unary(TypeChecker *tc, ASTNode *node, int depth)
{
    check_node(tc, node->unary.operand, depth + 1);

    Type *operand_type = node->unary.operand->type_info;
    const char *op = node->unary.op;

    if (!operand_type)
    {
        return;
    }

    // Logical NOT: !
    if (strcmp(op, "!") == 0)
    {
        node->type_info = type_new(TYPE_BOOL);
        return;
    }

    // Numeric negation: -
    if (strcmp(op, "-") == 0)
    {
        if (!is_integer_type(operand_type) && !is_float_type(operand_type))
        {
            const char *hints[] = {"Negation requires a numeric operand", NULL};
            tc_error_with_hints(tc, node->token, "Cannot negate non-numeric type", hints);
        }
        else
        {
            node->type_info = operand_type;
        }
        return;
    }

    // Dereference: *
    if (strcmp(op, "*") == 0)
    {
        if (operand_type->kind == TYPE_UNKNOWN)
        {
            node->type_info = type_new(TYPE_UNKNOWN);
            return;
        }

        Type *resolved = resolve_alias(operand_type);
        if (resolved->kind != TYPE_POINTER && resolved->kind != TYPE_STRING)
        {
            const char *hints[] = {"Only pointers can be dereferenced", NULL};
            tc_error_with_hints(tc, node->token, "Cannot dereference non-pointer type", hints);
        }
        else if (resolved->kind == TYPE_STRING)
        {
            node->type_info = type_new(TYPE_CHAR);
        }
        else if (resolved->inner)
        {
            node->type_info = resolved->inner;
            CompilerConfig *cfg = &tc->pctx->compiler->config;
            if (cfg->misra_mode)
            {
                misra_check_file_dereference(tc, operand_type, node->token);
            }
        }
        return;
    }

    // Bitwise NOT: ~
    if (strcmp(op, "~") == 0)
    {
        if (!is_integer_type(operand_type))
        {
            const char *hints[] = {"Bitwise NOT requires an integer operand", NULL};
            tc_error_with_hints(tc, node->token, "Cannot apply ~ to non-integer type", hints);
        }
        else
        {
            misra_check_bitwise_operand(tc, node->unary.operand->type_info, node->token);
            node->type_info = operand_type;
        }
        return;
    }

    if (strcmp(op, "++") == 0 || strcmp(op, "--") == 0 || strcmp(op, "_post++") == 0 ||
        strcmp(op, "_post--") == 0)
    {
        misra_check_inc_dec_result_used(tc, node->token);
        // Track as a write
        if (node->unary.operand->type == NODE_EXPR_VAR)
        {
            ZenSymbol *s = tc_lookup(tc, node->unary.operand->var_ref.name);
            if (s)
            {
                s->is_written_to = 1;
            }
        }
        else if (node->unary.operand->type == NODE_EXPR_UNARY &&
                 strcmp(node->unary.operand->unary.op, "*") == 0)
        {
            ASTNode *inner = node->unary.operand->unary.operand;
            if (inner->type == NODE_EXPR_VAR)
            {
                ZenSymbol *s = tc_lookup(tc, inner->var_ref.name);
                if (s)
                {
                    s->is_written_to = 1;
                }
            }
        }
        if (!is_integer_type(operand_type) && !is_float_type(operand_type) &&
            operand_type->kind != TYPE_POINTER)
        {
            tc_error(tc, node->token, "Increment/decrement requires a numeric or pointer operand");
        }
        node->type_info = operand_type;
        return;
    }

    // Address-of: &
    if (strcmp(op, "&") == 0 || strcmp(op, "&_rval") == 0)
    {
        node->type_info = type_new_ptr(operand_type);
        // Record provenance depth for escape analysis
        if (node->unary.operand->type == NODE_EXPR_VAR)
        {
            ZenSymbol *sym = tc_lookup(tc, node->unary.operand->var_ref.name);
            if (sym)
            {
                node->type_info->lifetime_depth = sym->scope_depth;
            }
        }
        else if (strcmp(op, "&_rval") == 0)
        {
            // R-values are temporaries, their life is the current code block
            if (node->unary.operand->type_info)
            {
                node->type_info->lifetime_depth = node->unary.operand->type_info->lifetime_depth;
            }
            else
            {
                node->type_info->lifetime_depth = tc->current_depth;
            }
        }
        else
        {
            // For complex targets (e.g. &array[0]), inherit from the operand if it has a lifetime
            if (operand_type)
            {
                node->type_info->lifetime_depth = operand_type->lifetime_depth;
            }
        }
        return;
    }
}

static void check_expr_binary(TypeChecker *tc, ASTNode *node, int depth)
{
    const char *op = node->binary.op;
    Type *contextual_type = node->type_info;

    if (strcmp(op, "=") == 0 ||
        (strlen(op) > 1 && op[strlen(op) - 1] == '=' && strcmp(op, "==") != 0 &&
         strcmp(op, "!=") != 0 && strcmp(op, "<=") != 0 && strcmp(op, ">=") != 0))
    {
        if (!tc->is_stmt_context)
        {
            misra_check_assignment_result_used(tc, node->token);
        }

        int old_is_assign_lhs = tc->is_assign_lhs;
        int old_stmt_ctx = tc->is_stmt_context;
        tc->is_assign_lhs = 1;
        tc->is_stmt_context = 0;
        check_node(tc, node->binary.left, depth + 1);
        tc->is_assign_lhs = old_is_assign_lhs;

        if (node->binary.left->type_info && node->binary.right->type == NODE_LAMBDA)
        {
            node->binary.right->type_info = node->binary.left->type_info;
        }
        check_node(tc, node->binary.right, depth + 1);
        tc->is_stmt_context = old_stmt_ctx;

        // Rule 13.2: Side effect collision detection for assignments
        if (g_config.misra_mode)
        {
            SymbolSet l_reads = {0}, l_writes = {0};
            SymbolSet r_reads = {0}, r_writes = {0};
            collect_symbols(node->binary.left, &l_reads, &l_writes);
            collect_symbols(node->binary.right, &r_reads, &r_writes);

            // Treat LHS of assignment as a write
            if (node->binary.left->type == NODE_EXPR_VAR)
            {
                int already_in = 0;
                for (int k = 0; k < l_writes.count; k++)
                {
                    if (l_writes.syms[k] == node->binary.left->var_ref.symbol)
                    {
                        already_in = 1;
                        break;
                    }
                }
                if (!already_in && l_writes.count < 32)
                {
                    l_writes.syms[l_writes.count++] = node->binary.left->var_ref.symbol;
                }
            }

            // Double write collision: i = i++
            int match = 0;
            for (int i = 0; i < l_writes.count; i++)
            {
                ZenSymbol *s = l_writes.syms[i];
                if (!s)
                {
                    continue;
                }
                for (int j = 0; j < r_writes.count; j++)
                {
                    if (s == r_writes.syms[j])
                    {
                        tc_error(tc, node->token,
                                 "MISRA Rule 13.2: symbol modified multiple times in assignment");
                        match = 1;
                        break;
                    }
                }
                if (match)
                {
                    break;
                }

                // Also check if LHS write conflicts with RHS reads (if RHS also writes it)
                // e.g. i = i++ is already covered by double write.
                // But what about i = i + (i++)?
                // r_writes={i}, r_reads={i, i}.
            }
        }
    }
    else
    {
        int old_stmt_ctx = tc->is_stmt_context;
        tc->is_stmt_context = 0;
        check_node(tc, node->binary.left, depth + 1);
        check_node(tc, node->binary.right, depth + 1);
        tc->is_stmt_context = old_stmt_ctx;

        // Rule 13.2: Side effect collision detection for binary operators
        if (strcmp(op, "&&") != 0 && strcmp(op, "||") != 0 && strcmp(op, ",") != 0)
        {
            check_side_effect_collision(tc, node->binary.left, node->binary.right, node->token);
        }
        else if (g_config.misra_mode && (strcmp(op, "&&") == 0 || strcmp(op, "||") == 0))
        {
            // Rule 13.5: The RHS of && or || shall not contain persistent side effects
            SymbolSet r_reads = {0}, r_writes = {0};
            collect_symbols(node->binary.right, &r_reads, &r_writes);
            if (r_writes.count > 0)
            {
                tc_error(tc, node->binary.right->token,
                         "MISRA Rule 13.5: persistent side effect in logical RHS");
            }
        }
    }

    Type *left_type = node->binary.left->type_info;
    Type *right_type = node->binary.right->type_info;

    // Assignment Logic for Moves (and type compatibility)
    if (strcmp(op, "=") == 0)
    {
        // Check type compatibility for assignment
        if (left_type && right_type)
        {
            apply_implicit_struct_pointer_conversions(tc, &node->binary.right, left_type);
            right_type = node->binary.right->type_info;
            check_type_compatibility(tc, left_type, right_type, node->binary.right->token,
                                     node->binary.right, 0);

            // Mark LHS as written to
            if (node->binary.left->type == NODE_EXPR_VAR)
            {
                ZenSymbol *s = tc_lookup(tc, node->binary.left->var_ref.name);
                if (s)
                {
                    s->is_written_to = 1;
                }
            }
            else if (node->binary.left->type == NODE_EXPR_UNARY &&
                     strcmp(node->binary.left->unary.op, "*") == 0)
            {
                ASTNode *inner = node->binary.left->unary.operand;
                if (inner->type == NODE_EXPR_VAR)
                {
                    ZenSymbol *s = tc_lookup(tc, inner->var_ref.name);
                    if (s)
                    {
                        s->is_written_to = 1;
                    }
                }
            }
            // Also handle array indexing as write
            else if (node->binary.left->type == NODE_EXPR_INDEX)
            {
                ASTNode *arr = node->binary.left->index.array;
                if (arr->type == NODE_EXPR_VAR)
                {
                    ZenSymbol *s = tc_lookup(tc, arr->var_ref.name);
                    if (s)
                    {
                        s->is_written_to = 1;
                    }
                }
            }
            // Also handle member access as write
            else if (node->binary.left->type == NODE_EXPR_MEMBER)
            {
                ASTNode *target = node->binary.left->member.target;
                // Follow the member chain to the base variable
                while (target && target->type == NODE_EXPR_MEMBER)
                {
                    target = target->member.target;
                }
                if (target && target->type == NODE_EXPR_VAR)
                {
                    ZenSymbol *s = tc_lookup(tc, target->var_ref.name);
                    if (s)
                    {
                        s->is_written_to = 1;
                    }
                }
            }
        }

        // If RHS is moving a non-copy value, check validity and mark moved
        check_move_for_rvalue(tc, node->binary.right);

        // LHS is being (re-)initialized, so it becomes Valid.
        if (node->binary.left->type == NODE_EXPR_VAR)
        {
            ZenSymbol *lhs_sym = tc_lookup(tc, node->binary.left->var_ref.name);
            if (lhs_sym)
            {
                if (g_config.misra_mode && node->binary.left &&
                    node->binary.left->type == NODE_EXPR_VAR)
                {
                    misra_check_param_modified(tc, node->binary.left, node->binary.left->token);
                }
                if (lhs_sym->is_immutable)
                {
                    tc_error(tc, node->binary.left->token, "Cannot assign to immutable variable");
                }

                if (g_config.misra_mode)
                {
                    misra_check_pointer_conversion(tc, left_type, right_type, node->token);
                }
                mark_symbol_valid(tc->pctx, lhs_sym, node->binary.left);
            }
        }

        // Result type is same as LHS
        node->type_info = left_type;
        return;
    }

    // Arithmetic operators: +, -, *, /, %
    if (strcmp(op, "+") == 0 || strcmp(op, "-") == 0 || strcmp(op, "*") == 0 ||
        strcmp(op, "/") == 0 || strcmp(op, "%") == 0)
    {
        // Division by zero detection for / and %
        if ((strcmp(op, "/") == 0 || strcmp(op, "%") == 0) && node->binary.right &&
            node->binary.right->type == NODE_EXPR_LITERAL)
        {
            LiteralKind kind = node->binary.right->literal.type_kind;
            if (kind == LITERAL_INT && node->binary.right->literal.int_val == 0)
            {
                const char *hints[] = {"Division by zero is undefined behavior", NULL};
                tc_error_with_hints(tc, node->binary.right->token, "Division by zero detected",
                                    hints);
            }
            else if (kind == LITERAL_FLOAT && node->binary.right->literal.float_val == 0.0)
            {
                const char *hints[] = {"Division by zero results in infinity or NaN", NULL};
                tc_error_with_hints(tc, node->binary.right->token, "Division by zero detected",
                                    hints);
            }
        }

        if (left_type && right_type)
        {
            Type *lhs_resolved = resolve_alias(left_type);
            Type *rhs_resolved = resolve_alias(right_type);

            // Pointer Arithmetic
            if (lhs_resolved->kind == TYPE_POINTER || lhs_resolved->kind == TYPE_STRING)
            {
                misra_check_pointer_arithmetic(tc, lhs_resolved, node->token);

                // Ptr - Ptr -> isize
                if (strcmp(op, "-") == 0 &&
                    (rhs_resolved->kind == TYPE_POINTER || rhs_resolved->kind == TYPE_STRING))
                {
                    node->type_info = type_new(TYPE_ISIZE);
                    return;
                }
                // Ptr + Int -> Ptr
                // Ptr - Int -> Ptr
                if ((strcmp(op, "+") == 0 || strcmp(op, "-") == 0) && is_integer_type(rhs_resolved))
                {
                    node->type_info = left_type;
                    return;
                }
            }
            // Int + Ptr -> Ptr
            if (strcmp(op, "+") == 0 && is_integer_type(lhs_resolved) &&
                (rhs_resolved->kind == TYPE_POINTER || rhs_resolved->kind == TYPE_STRING))
            {
                misra_check_pointer_arithmetic(tc, rhs_resolved, node->token);
                node->type_info = right_type;
                return;
            }

            int left_numeric = is_integer_type(left_type) || is_float_type(left_type) ||
                               left_type->kind == TYPE_VECTOR;
            int right_numeric = is_integer_type(right_type) || is_float_type(right_type) ||
                                right_type->kind == TYPE_VECTOR;

            if (!left_numeric || !right_numeric)
            {
                if (left_type->kind == TYPE_UNKNOWN || right_type->kind == TYPE_UNKNOWN)
                {
                    node->type_info = type_new(TYPE_UNKNOWN);
                    return;
                }

                char msg[MAX_SHORT_MSG_LEN];
                snprintf(msg, sizeof(msg), "Operator '%s' requires numeric operands", op);
                const char *hints[] = {
                    "Arithmetic operators can only be used with integer, float, or vector types",
                    NULL};
                tc_error_with_hints(tc, node->token, msg, hints);
            }
            else if (left_type->kind == TYPE_VECTOR || right_type->kind == TYPE_VECTOR)
            {
                if (left_type->kind != right_type->kind || !type_eq(left_type, right_type))
                {
                    tc_error(tc, node->token,
                             "Vector operation requires operands of same vector type");
                }
                node->type_info = left_type;
                return;
            }
            else
            {
                // Rule 10.4: Balancing
                misra_check_binary_op_essential_types(tc, node->binary.left, node->binary.right,
                                                      node->token);

                // MISRA Rule 12.4: Evaluation of constant expressions shall not lead to unsigned
                // wrap Use the contextually pushed down type (stored in node->type_info before this
                // function) or fall back to the inferred operand type.
                Type *target_type = contextual_type ? contextual_type : left_type;

                if (g_config.misra_mode && target_type && is_integer_type(target_type))
                {
                    long long lval, rval;
                    if (eval_const_int_expr(node->binary.left, tc->pctx, &lval) &&
                        eval_const_int_expr(node->binary.right, tc->pctx, &rval))
                    {
                        long long res = 0;
                        if (strcmp(op, "+") == 0)
                        {
                            res = lval + rval;
                        }
                        else if (strcmp(op, "-") == 0)
                        {
                            res = lval - rval;
                        }
                        else if (strcmp(op, "*") == 0)
                        {
                            res = lval * rval;
                        }

                        if (strcmp(op, "+") == 0 || strcmp(op, "-") == 0 || strcmp(op, "*") == 0)
                        {
                            misra_check_unsigned_wrap(tc, op, lval, rval, res, target_type,
                                                      node->token);
                        }
                    }
                }

                // Rule 10.2: Character arithmetic
                misra_check_char_arithmetic(tc, left_type, right_type, op, node->token);

                // Result type: Only infer if not already set by context
                if (!node->type_info)
                {
                    if (is_float_type(left_type) || is_float_type(right_type))
                    {
                        node->type_info = type_new(TYPE_F64);
                    }
                    else
                    {
                        node->type_info = left_type;
                    }
                }
            }
        }
        return;
    }

    // Comparison operators: ==, !=, <, >, <=, >=
    if (strcmp(op, "==") == 0 || strcmp(op, "!=") == 0 || strcmp(op, "<") == 0 ||
        strcmp(op, ">") == 0 || strcmp(op, "<=") == 0 || strcmp(op, ">=") == 0)
    {
        // Result is always bool
        node->type_info = type_new(TYPE_BOOL);

        // Operands should be comparable
        if (left_type && right_type)
        {
            // Rule 10.4: Balancing
            misra_check_binary_op_essential_types(tc, node->binary.left, node->binary.right,
                                                  node->token);

            misra_check_string_compare(tc, left_type, right_type, node->token);

            if (!type_eq(left_type, right_type))
            {
                // Allow comparison between numeric types
                int left_numeric = is_integer_type(left_type) || is_float_type(left_type);
                int right_numeric = is_integer_type(right_type) || is_float_type(right_type);

                if (!left_numeric || !right_numeric)
                {
                    if ((left_type && left_type->kind == TYPE_UNKNOWN) ||
                        (right_type && right_type->kind == TYPE_UNKNOWN))
                    {
                        node->type_info = type_new(TYPE_BOOL);
                        return;
                    }
                    char msg[MAX_SHORT_MSG_LEN];
                    snprintf(msg, sizeof(msg), "Cannot compare '%s' with incompatible types", op);
                    const char *hints[] = {"Ensure both operands have the same or compatible types",
                                           NULL};
                    tc_error_with_hints(tc, node->token, msg, hints);
                }
            }
        }
        return;
    }

    // Logical operators: &&, ||
    if (strcmp(op, "&&") == 0 || strcmp(op, "||") == 0)
    {
        node->type_info = type_new(TYPE_BOOL);
        // Could validate that operands are boolean-like, but C is lax here
        return;
    }

    // Bitwise operators: &, |, ^, <<, >>
    if (strcmp(op, "&") == 0 || strcmp(op, "|") == 0 || strcmp(op, "^") == 0 ||
        strcmp(op, "<<") == 0 || strcmp(op, ">>") == 0)
    {
        // MISRA Rule 12.2: Shift amount validation for << and >>
        if (g_config.misra_mode && (strcmp(op, "<<") == 0 || strcmp(op, ">>") == 0))
        {
            long long shift_amt;
            if (eval_const_int_expr(node->binary.right, tc->pctx, &shift_amt))
            {
                int width = integer_type_width(left_type);
                misra_check_shift_amount(tc, shift_amt, width, node->token);
            }
        }
        else if ((strcmp(op, "<<") == 0 || strcmp(op, ">>") == 0) && node->binary.right &&
                 node->binary.right->type == NODE_EXPR_LITERAL &&
                 node->binary.right->literal.type_kind == LITERAL_INT)
        {
            // Legacy/Non-MISRA warnings
            unsigned long long shift_amt = node->binary.right->literal.int_val;
            if (shift_amt >= 64)
            {
                const char *hints[] = {"Shift amount exceeds bit width, result is undefined", NULL};
                tc_error_with_hints(tc, node->binary.right->token, "Shift amount too large", hints);
            }
            else if (shift_amt >= 32 && left_type &&
                     (left_type->kind == TYPE_INT || left_type->kind == TYPE_UINT ||
                      left_type->kind == TYPE_I32 || left_type->kind == TYPE_U32 ||
                      left_type->kind == TYPE_C_INT || left_type->kind == TYPE_C_UINT))
            {
                const char *hints[] = {
                    "Shift amount >= 32 is undefined behavior for 32-bit integers", NULL};
                tc_error_with_hints(tc, node->binary.right->token,
                                    "Shift amount exceeds 32-bit type width", hints);
            }
        }

        if (left_type && right_type)
        {
            if ((!is_integer_type(left_type) && left_type->kind != TYPE_VECTOR) ||
                (!is_integer_type(right_type) && right_type->kind != TYPE_VECTOR))
            {
                char msg[MAX_SHORT_MSG_LEN];
                snprintf(msg, sizeof(msg),
                         "Bitwise operator '%s' requires integer or vector operands", op);
                const char *hints[] = {"Bitwise operators only work on integer or vector types",
                                       NULL};
                tc_error_with_hints(tc, node->token, msg, hints);
            }
            else if (left_type->kind == TYPE_VECTOR || right_type->kind == TYPE_VECTOR)
            {
                if (left_type->kind != right_type->kind || !type_eq(left_type, right_type))
                {
                    tc_error(tc, node->token, "Vector bitwise operation requires same vector type");
                }
                node->type_info = left_type;
            }
            else
            {
                if (g_config.misra_mode)
                {
                    misra_check_bitwise_operand(tc, left_type, node->token);
                    misra_check_bitwise_operand(tc, right_type, node->token);
                    // Rule 10.4: Balancing for &, |, ^
                    if (strcmp(op, "&") == 0 || strcmp(op, "|") == 0 || strcmp(op, "^") == 0)
                    {
                        misra_check_binary_op_essential_types(tc, node->binary.left,
                                                              node->binary.right, node->token);
                    }
                }
                node->type_info = left_type;
            }
        }
        return;
    }
}

static void check_expr_call(TypeChecker *tc, ASTNode *node, int depth)
{
    check_node(tc, node->call.callee, depth + 1);

    const char *func_name = NULL;
    FuncSig *sig = NULL;

    // Check if the function exists (for simple direct calls)
    if (node->call.callee && node->call.callee->type == NODE_EXPR_VAR)
    {
        func_name = node->call.callee->var_ref.name;

        // Look up function signature
        sig = find_func(tc->pctx, func_name);

        if (g_config.misra_mode)
        {
            Token t = node->call.callee->token;
            if (t.line == 0)
            {
                t = node->token;
            }
            misra_check_banned_function(tc, func_name, t);
        }

        if (g_config.misra_mode && tc->current_func)
        {
            if (strcmp(func_name, tc->current_func->func.name) == 0)
            {
                Token t = node->call.callee->token;
                if (t.line == 0)
                {
                    t = node->token;
                }
                misra_check_recursion(tc, t);
            }
        }

        if (!sig)
        {
            // Check if it's a built-in macro injected by the compiler
            if (strcmp(func_name, "_z_str") == 0)
            {
                // _z_str is a generic format macro from ZC_C_GENERIC_STR
                check_node(tc, node->call.args, depth + 1); // Still check the argument
                node->type_info = type_new(TYPE_STRING);
                return;
            }

            // Check local scope first, then global symbols
            ZenSymbol *sym = tc_lookup(tc, func_name);
            if (!sym)
            {
                ZenSymbol *global_sym = find_symbol_in_all(tc->pctx, func_name);
                if (!global_sym && !should_suppress_undef_warning(tc->pctx, func_name))
                {
                    char msg[MAX_SHORT_MSG_LEN];
                    if (g_config.misra_mode)
                    {
                        snprintf(msg, sizeof(msg), "Undefined function '%s' (MISRA Rule 17.3)",
                                 func_name);
                    }
                    else
                    {
                        snprintf(msg, sizeof(msg), "Undefined function '%s'", func_name);
                    }
                    const char *hints[] = {"Check if the function is defined or imported", NULL};
                    tc_error_with_hints(tc, node->call.callee->token, msg, hints);
                }
            }
        }
    }
    else if (node->call.callee && node->call.callee->type == NODE_EXPR_MEMBER)
    {
        if (node->call.callee->type_info && node->call.callee->type_info->name)
        {
            func_name = node->call.callee->type_info->name;
            sig = find_func(tc->pctx, func_name);
        }

        // Trait method resolution fallback
        if (!sig && node->call.callee->member.target && node->call.callee->member.target->type_info)
        {
            Type *target_type = get_inner_type(node->call.callee->member.target->type_info);
            if (target_type->name && is_trait(target_type->name))
            {
                ASTNode *trait_def = find_trait_def(tc->pctx, target_type->name);
                if (trait_def)
                {
                    ASTNode *method = trait_def->trait.methods;
                    while (method)
                    {
                        if (strcmp(method->func.name, node->call.callee->member.field) == 0)
                        {
                            // Correctly resolve return type for trait method
                            node->type_info = method->func.ret_type_info;
                            break;
                        }
                        method = method->next;
                    }
                }
            }
        }
    }

    // Count arguments
    int arg_count = 0;
    ASTNode *arg = node->call.args;
    while (arg)
    {
        arg_count++;
        arg = arg->next;
    }

    // Member call (a.b()) counts as +1 arg (the receiver)
    if (node->call.callee && node->call.callee->type == NODE_EXPR_MEMBER)
    {
        arg_count++;
    }

    // Enforce @pure constraint
    if (tc->current_func && tc->current_func->func.pure)
    {
        if (!sig || !sig->is_pure)
        {
            // Allow _z_str? Wait, _z_str is a compiler macro, it's not strictly "pure", but it's
            // safe.
            if (!func_name || strcmp(func_name, "_z_str") != 0)
            {
                char msg[MAX_SHORT_MSG_LEN];
                snprintf(msg, sizeof(msg),
                         "Pure function '%s' cannot call non-pure or dynamic function '%s'",
                         tc->current_func->func.name, func_name ? func_name : "unknown");
                const char *hints[] = {
                    "Mark the called function as @pure, or remove @pure from the caller", NULL};
                tc_error_with_hints(tc, node->call.callee->token, msg, hints);
            }
        }
    }

    // Validate argument count
    if (sig)
    {
        int min_args = sig->total_args;
        if (sig->defaults)
        {
            min_args = 0;
            for (int i = 0; i < sig->total_args; i++)
            {
                if (!sig->defaults[i])
                {
                    min_args++;
                }
            }
        }

        if (arg_count < min_args)
        {
            char msg[MAX_SHORT_MSG_LEN];
            snprintf(msg, sizeof(msg), "Too few arguments: '%s' expects at least %d, got %d",
                     func_name, min_args, arg_count);

            const char *hints[] = {"Check the function signature for required parameters", NULL};
            tc_error_with_hints(tc, node->token, msg, hints);
        }
        else if (arg_count > sig->total_args && !sig->is_varargs)
        {
            char msg[MAX_SHORT_MSG_LEN];
            snprintf(msg, sizeof(msg), "Too many arguments: '%s' expects %d, got %d", func_name,
                     sig->total_args, arg_count);

            const char *hints[] = {
                "Remove extra arguments or check if you meant to call a different function", NULL};
            tc_error_with_hints(tc, node->token, msg, hints);
        }
    }

    // Check argument types
    arg = node->call.args;
    int sig_arg_idx = 0;

    // For member calls, the first signature argument is the receiver
    if (node->call.callee && node->call.callee->type == NODE_EXPR_MEMBER)
    {
        if (sig && sig->total_args > 0 && sig->arg_types && sig->arg_types[0])
        {
            Type *expected_rec = sig->arg_types[0];
            Type *actual_rec = node->call.callee->member.target->type_info;

            // Allow T to T* for method receivers
            if (expected_rec->kind == TYPE_POINTER && actual_rec &&
                actual_rec->kind != TYPE_POINTER && type_eq(expected_rec->inner, actual_rec))
            {
                // OK: Compiler will take address
            }
            else
            {
                check_type_compatibility(tc, expected_rec, actual_rec, node->call.callee->token,
                                         NULL, 1);
            }
        }
        sig_arg_idx = 1;
    }

    while (arg)
    {
        Type *expected = NULL;
        if (sig && sig_arg_idx < sig->total_args && sig->arg_types && sig->arg_types[sig_arg_idx])
        {
            expected = sig->arg_types[sig_arg_idx];
        }
        else if (!sig && node->call.callee->type_info)
        {
            Type *callee_t = get_inner_type(node->call.callee->type_info);
            if (callee_t->kind == TYPE_FUNCTION && sig_arg_idx < callee_t->arg_count &&
                callee_t->args)
            {
                expected = callee_t->args[sig_arg_idx];
            }
        }

        // Propagate expected type to lambda for inference
        if (arg->type == NODE_LAMBDA && expected)
        {
            arg->type_info = expected;
        }

        check_node(tc, arg, depth + 1);

        // Validate type against signature
        Type *actual = arg->type_info;
        if (expected && actual)
        {
            Type *e_resolved = get_inner_type(expected);
            Type *a_resolved = get_inner_type(actual);

            if (e_resolved->kind == TYPE_UNKNOWN && a_resolved->kind != TYPE_UNKNOWN)
            {
                // Backward type inference: we passed an actual type to a lambda taking unknown
                *e_resolved = *a_resolved;
            }
            else if (e_resolved->kind == TYPE_FUNCTION && a_resolved->kind == TYPE_FUNCTION)
            {
                for (int j = 0; j < e_resolved->arg_count && j < a_resolved->arg_count; j++)
                {
                    if (a_resolved->args && a_resolved->args[j] &&
                        a_resolved->args[j]->kind == TYPE_UNKNOWN && e_resolved->args &&
                        e_resolved->args[j] && e_resolved->args[j]->kind != TYPE_UNKNOWN)
                    {
                        *a_resolved->args[j] = *e_resolved->args[j];
                    }
                }
                if (a_resolved->inner && a_resolved->inner->kind == TYPE_UNKNOWN &&
                    e_resolved->inner)
                {
                    *a_resolved->inner = *e_resolved->inner;
                }
            }

            // Rule 17.5: Array parameter sizes must match.
            if (g_config.misra_mode && e_resolved->kind == TYPE_ARRAY &&
                a_resolved->kind == TYPE_ARRAY)
            {
                if (e_resolved->array_size != a_resolved->array_size)
                {
                    misra_check_array_param_size(tc, e_resolved->array_size, a_resolved->array_size,
                                                 arg->token);
                }
            }

            check_type_compatibility(tc, expected, actual, arg->token, arg, 1);
        }

        // If argument is passed by VALUE, check if it can be moved.
        check_move_for_rvalue(tc, arg);

        arg = arg->next;
        sig_arg_idx++;
    }

    // Propagate return type from function signature
    if (sig && sig->ret_type)
    {
        if (!node->type_info)
        {
            // Deep clone return type to ensure caller doesn't modify callee's metadata
            node->type_info = type_clone(sig->ret_type);
        }

        // Apply Lifetime Elision
        if (sig->elide_from_idx != -1 && node->type_info->kind == TYPE_POINTER)
        {
            int target_depth = 0; // Default to escaping if not found
            if (node->call.callee && node->call.callee->type == NODE_EXPR_MEMBER &&
                sig->elide_from_idx == 0)
            {
                if (node->call.callee->member.target->type_info)
                {
                    target_depth = node->call.callee->member.target->type_info->lifetime_depth;
                }
            }
            else
            {
                int current_idx =
                    (node->call.callee && node->call.callee->type == NODE_EXPR_MEMBER) ? 1 : 0;
                ASTNode *a = node->call.args;
                while (a)
                {
                    if (current_idx == sig->elide_from_idx)
                    {
                        if (a->type_info)
                        {
                            target_depth = a->type_info->lifetime_depth;
                        }
                        break;
                    }
                    current_idx++;
                    a = a->next;
                }
            }
            node->type_info->lifetime_depth = target_depth;
        }
        else
        {
            // Function results always have depth 0 (static/heap/escaping) by default
            node->type_info->lifetime_depth = 0;
        }
    }
    else if (!node->type_info && node->call.callee && node->call.callee->type_info)
    {
        Type *callee_t = get_inner_type(node->call.callee->type_info);
        if (callee_t->kind == TYPE_FUNCTION && callee_t->inner)
        {
            node->type_info = callee_t->inner;
        }
    }

    // Rule 17.7: Unused return values
    if (g_config.misra_mode && tc->is_stmt_context && node->type_info)
    {
        misra_check_function_return_usage(tc, node);
    }

    // Rule 13.2: Side effect collision detection in arguments
    ASTNode *receiver = (node->call.callee && node->call.callee->type == NODE_EXPR_MEMBER)
                            ? node->call.callee->member.target
                            : NULL;
    check_all_args_side_effects(tc, receiver, node->call.args, node->token);
}

static void check_block(TypeChecker *tc, ASTNode *block, int depth)
{
    tc_enter_scope(tc);
    ASTNode *stmt = block->block.statements;
    int seen_terminator = 0;
    Token terminator_token = {0};

    while (stmt)
    {
        // Warn if we see code after a terminating statement
        if (seen_terminator && stmt->type != NODE_LABEL)
        {
            const char *rule = g_config.misra_mode ? "MISRA Rule 2.1: " : "";
            const char *hints[] = {"Remove unreachable code or restructure control flow", NULL};
            char msg[256];
            snprintf(msg, sizeof(msg), "%sUnreachable code detected", rule);
            tc_error_with_hints(tc, stmt->token, msg, hints);
            seen_terminator = 0; // Only warn once per block
        }

        if (g_config.misra_mode)
        {
            // Rule 2.2: There shall be no dead code (expressions with no effect)
            // We ignore call expressions here as they are handled by Rule 17.7
            if (stmt->type >= NODE_EXPR_BINARY && stmt->type <= NODE_EXPR_SLICE &&
                stmt->type != NODE_EXPR_CALL)
            {
                if (!tc_expr_has_side_effects(stmt))
                {
                    tc_error(tc, stmt->token, "MISRA Rule 2.2: expression statement has no effect");
                }
            }
        }

        int old_stmt_ctx = tc->is_stmt_context;
        tc->is_stmt_context = 1;
        check_node(tc, stmt, depth + 1);
        tc->is_stmt_context = old_stmt_ctx;

        // Track terminating statements
        if (stmt->type == NODE_RETURN || stmt->type == NODE_BREAK || stmt->type == NODE_CONTINUE ||
            stmt->type == NODE_GOTO)
        {
            seen_terminator = 1;
            terminator_token = stmt->token;
        }

        stmt = stmt->next;
    }
    (void)terminator_token; // May be used for enhanced diagnostics later
    tc_exit_scope(tc);
}

static void extract_base_name(const char *full_name, char *base_buf, size_t max_len)
{
    if (!full_name)
    {
        base_buf[0] = '\0';
        return;
    }
    size_t i = 0;
    while (full_name[i] && full_name[i] != '<' && full_name[i] != '_' && i < max_len - 1)
    {
        base_buf[i] = full_name[i];
        i++;
    }
    base_buf[i] = '\0';
}

static int is_struct_base_match(Type *base, Type *instantiated)
{
    if (!base || !base->name || !instantiated || !instantiated->name)
    {
        return 0;
    }
    if (strcmp(base->name, instantiated->name) == 0)
    {
        return 1;
    }

    char base_str[MAX_TYPE_NAME_LEN];
    char inst_str[MAX_TYPE_NAME_LEN];
    extract_base_name(base->name, base_str, sizeof(base_str));
    extract_base_name(instantiated->name, inst_str, sizeof(inst_str));

    if (base_str[0] != '\0' && strcmp(base_str, inst_str) == 0)
    {
        return 1;
    }
    return 0;
}

static void apply_implicit_struct_pointer_conversions(TypeChecker *tc, ASTNode **expr_ptr,
                                                      Type *expected_type)
{
    if (!expr_ptr || !*expr_ptr || !expected_type)
    {
        return;
    }
    ASTNode *expr = *expr_ptr;
    Type *actual_type = expr->type_info;
    if (!actual_type)
    {
        return;
    }

    Type *e_res = get_inner_type(expected_type);
    Type *a_res = get_inner_type(actual_type);

    // T* (actual) -> T (expected) => Implicit Dereference *
    // This allows `return self` to return the struct value when self is a pointer.
    // We only do this if the type is Copy, otherwise it would trigger a
    // move-from-borrowed-reference error.
    if (a_res->kind == TYPE_POINTER && a_res->inner &&
        (a_res->inner->kind == TYPE_STRUCT || a_res->inner->kind == TYPE_ENUM) &&
        (type_eq(a_res->inner, e_res) || is_struct_base_match(a_res->inner, e_res)) &&
        is_type_copy(tc->pctx, a_res->inner))

    {
        ASTNode *deref = ast_create(NODE_EXPR_UNARY);
        deref->unary.op = xstrdup("*");
        deref->unary.operand = expr;
        deref->type_info = a_res->inner;
        deref->token = expr->token;
        deref->next = expr->next;
        expr->next = NULL;
        *expr_ptr = deref;
    }
    // T (actual) -> T* (expected) => Implicit Address-Of &
    else if (e_res->kind == TYPE_POINTER && e_res->inner &&
             (a_res->kind == TYPE_STRUCT || a_res->kind == TYPE_ENUM) &&
             (type_eq(a_res, e_res->inner) || is_struct_base_match(a_res, e_res->inner)))
    {
        ASTNode *addr = ast_create(NODE_EXPR_UNARY);
        int is_rvalue = (expr->type == NODE_EXPR_CALL || expr->type == NODE_EXPR_BINARY ||
                         expr->type == NODE_MATCH);
        addr->unary.op = is_rvalue ? xstrdup("&_rval") : xstrdup("&");
        addr->unary.operand = expr;
        addr->type_info = e_res;
        addr->token = expr->token;
        addr->next = expr->next;
        expr->next = NULL;
        *expr_ptr = addr;
    }
}

static int check_type_compatibility(TypeChecker *tc, Type *target, Type *value, Token t,
                                    ASTNode *value_node, int is_call_arg)
{
    if (!target || !value)
    {
        return 1; // Can't check incomplete types
    }

    Type *resolved_target = resolve_alias(target);
    Type *resolved_value = resolve_alias(value);

    // MISRA Pointer & Constant Checks (Rules 11.5, 11.9, etc.)
    if (g_config.misra_mode && resolved_target->kind == TYPE_POINTER)
    {
        misra_check_null_pointer_constant(tc, value_node, t);
        misra_check_void_ptr_cast(tc, target, value, t);
        misra_check_pointer_conversion(tc, target, value, t);
    }

    // Resolution of Integer compatibility (Rule 10.3)
    // This MUST happen before type_eq fast-path because type_eq is lax for integers.
    if (is_integer_type(resolved_target) && is_integer_type(resolved_value))
    {
        int target_width = integer_type_width(resolved_target);
        int value_width = integer_type_width(resolved_value);

        if (g_config.misra_mode)
        {
            misra_check_implicit_conversion(tc, target, value, value_node, t);
        }
        else
        {
            // Warn on narrowing conversions in non-MISRA mode
            if (target_width > 0 && value_width > 0 && target_width < value_width)
            {
                char *t_str = type_to_string(target);
                char *v_str = type_to_string(value);
                char msg[MAX_SHORT_MSG_LEN];
                snprintf(msg, sizeof(msg),
                         "Implicit narrowing conversion from '%s' (%d-bit) to '%s' (%d-bit)", v_str,
                         value_width, t_str, target_width);
                zwarn_at(t, "%s", msg);
                if (tc)
                {
                    tc->warning_count++;
                }
                zfree(t_str);
                zfree(v_str);
            }
        }
        return 1; // All integer pairs compatible (modulo MISRA/Warning checks above)
    }

    if (!is_call_arg && resolved_target->kind == TYPE_POINTER &&
        resolved_value->kind == TYPE_POINTER)
    {
        // Higher depth = shorter scope.
        if (resolved_target->lifetime_depth < resolved_value->lifetime_depth)
        {
            char msg[MAX_ERROR_MSG_LEN];
            snprintf(msg, sizeof(msg),
                     "Escape analysis error: pointer assigned to a location that outlives it");
            const char *hints[] = {"The source pointer belongs to a child scope and will become "
                                   "invalid when that scope ends.",
                                   "Consider copying the value instead of taking a reference.",
                                   NULL};
            tc_error_with_hints(tc, t, msg, hints);
            return 0;
        }
    }

    // Fast path: exact match
    if (type_eq(target, value))
    {
        // MISRA Rule 11.8: Check const qualification during pointer assignment
        if (g_config.misra_mode && resolved_target->kind == TYPE_POINTER &&
            resolved_value->kind == TYPE_POINTER)
        {
            misra_check_pointer_conversion(tc, target, value, t);
        }
        return 1;
    }

    if (g_config.misra_mode)
    {
        // Remaining pointer nesting and array param size checks are modularized
        // in their respective node visitors.
    }

    // Resolve type aliases (str -> string, etc.) for non-integer types
    // (Integer aliases handled by resolve_alias above)
    if (target->kind == TYPE_ALIAS && target->name)
    {
        const char *alias = find_type_alias(tc->pctx, target->name);
        if (alias)
        {
            // Check if resolved names match
            if (value->name && strcmp(alias, value->name) == 0)
            {
                return 1;
            }
        }
    }
    if (value->kind == TYPE_ALIAS && value->name)
    {
        const char *alias = find_type_alias(tc->pctx, value->name);
        if (alias)
        {
            if (target->name && strcmp(alias, target->name) == 0)
            {
                return 1;
            }
        }
    }

    // String types: str, string, *char are compatible
    if ((target->kind == TYPE_STRING || (target->name && strcmp(target->name, "str") == 0)) &&
        (value->kind == TYPE_STRING || (value->name && strcmp(value->name, "string") == 0)))
    {
        return 1;
    }

    // void* is generic pointer
    if (resolved_target->kind == TYPE_POINTER && resolved_target->inner &&
        resolved_target->inner->kind == TYPE_VOID)
    {
        return 1;
    }
    if (resolved_value->kind == TYPE_POINTER && resolved_value->inner &&
        resolve_alias(resolved_value->inner)->kind == TYPE_VOID)
    {
        return 1;
    }

    // Array decay: Array[T] -> T*
    // This allows passing a fixed-size array where a pointer is expected.
    if (resolved_target->kind == TYPE_POINTER && resolved_value->kind == TYPE_ARRAY)
    {
        if (resolved_target->inner && resolved_value->inner)
        {
            // Recursive check for inner types (e.g. char* <- char[10])
            if (type_eq(resolved_target->inner, resolved_value->inner))
            {
                return 1;
            }
            // Allow char* <- char[N] explicitly if type_eq is too strict
            if (is_char_type(resolved_target->inner) && is_char_type(resolved_value->inner))
            {
                return 1;
            }
        }
    }

    if (is_integer_type(resolved_target) && is_integer_type(resolved_value))
    {
        return 1;
    }

    // Float compatibility
    if (is_float_type(resolved_target) && is_float_type(resolved_value))
    {
        return 1;
    }

    // Trait object compatibility: Struct -> Trait
    if (resolved_target->name && is_trait(resolved_target->name) &&
        resolved_value->kind == TYPE_STRUCT && resolved_value->name)
    {
        if (check_impl(tc->pctx, resolved_target->name, resolved_value->name))
        {
            return 1;
        }
    }

    // Type mismatch - report error
    char *t_str = type_to_string(target);
    char *v_str = type_to_string(value);

    char msg[MAX_ERROR_MSG_LEN];
    snprintf(msg, sizeof(msg), "Type mismatch: expected '%s', but found '%s'", t_str, v_str);

    const char *hints[] = {
        "Check if you need an explicit cast",
        "Ensure the types match exactly (no implicit conversions for strict types)", NULL};

    tc_error_with_hints(tc, t, msg, hints);
    zfree(t_str);
    zfree(v_str);
    return 0;
}

static void check_var_decl(TypeChecker *tc, ASTNode *node, int depth)
{
    if (node->var_decl.type_info)
    {
        misra_check_pointer_nesting(tc, node->var_decl.type_info, node->token);
    }

    // MISRA: Mark type as used
    mark_type_as_used(tc, node->var_decl.type_info);

    // Initialize lifetime depth for escape analysis if this is a local variable
    if (node->var_decl.type_info && tc->current_func)
    {
        node->var_decl.type_info->lifetime_depth = tc->current_depth;
    }

    if (node->var_decl.init_expr)
    {
        if (node->type_info)
        {
            node->var_decl.init_expr->type_info = node->type_info;
        }
        int old_stmt_ctx = tc->is_stmt_context;
        tc->is_stmt_context = 0;
        check_node(tc, node->var_decl.init_expr, depth + 1);
        tc->is_stmt_context = old_stmt_ctx;

        Type *decl_type = node->var_decl.type_info;
        Type *init_type = node->var_decl.init_expr->type_info;

        if (decl_type && init_type)
        {
            apply_implicit_struct_pointer_conversions(tc, &node->var_decl.init_expr, decl_type);
            init_type = node->var_decl.init_expr->type_info;

            // The declared type retains its original lifetime metadata (from init or explicit decl)

            // If initialization exists, check compatibility
            if (node->var_decl.type_info)
            {
                check_type_compatibility(tc, node->var_decl.type_info, init_type, node->token,
                                         node->var_decl.init_expr, 0);
            }

            if (g_config.misra_mode)
            {
                misra_check_pointer_conversion(tc, decl_type, init_type, node->token);

                // Rule 9.3: Arrays shall not be partially initialized.
                if (decl_type->kind == TYPE_ARRAY && init_type->kind == TYPE_ARRAY)
                {
                    if (node->var_decl.init_expr->type == NODE_EXPR_ARRAY_LITERAL)
                    {
                        if (decl_type->array_size != init_type->array_size)
                        {
                            tc_error(tc, node->token, "MISRA Rule 9.3");
                        }
                    }
                }
            }
        }

        // Move Analysis: Check if the initializer moves a non-copy value.
        check_move_for_rvalue(tc, node->var_decl.init_expr);
    }

    if (node->type_info)
    {
        misra_check_pointer_nesting(tc, node->var_decl.type_info, node->token);
    }

    // If type is not explicit, we should ideally infer it from init_expr.
    Type *t = node->type_info;
    if (!t && node->var_decl.init_expr)
    {
        t = type_clone(node->var_decl.init_expr->type_info);
        // Ensure inferred local variables get the correct scope depth for escape analysis
        if (t && tc->current_func)
        {
            t->lifetime_depth = tc->current_depth;
        }
        node->type_info = t;
        node->var_decl.type_info = t;
    }

    misra_check_reserved_identifier(tc, node->var_decl.name, node->token);
    tc_add_symbol(tc, node->var_decl.name, t, node->token, 0);
    ZenSymbol *new_sym = tc_lookup(tc, node->var_decl.name);
    if (new_sym)
    {
        new_sym->is_static = node->var_decl.is_static;
        mark_symbol_valid(tc->pctx, new_sym, node);
    }

    if (g_config.misra_mode && t && t->kind == TYPE_ARRAY)
    {
        // Rule 8.11: Array with external linkage shall have explicit size
        ZenSymbol *existing = tc_lookup(tc, node->var_decl.name);
        int is_static = (existing && existing->is_static) || (node->var_decl.is_static);
        int is_local = (existing && existing->is_local) || (tc->current_func != NULL);

        misra_check_external_array_size(tc, t, node->token, is_static, is_local);

        // Rule 18.8: No variable length arrays
        // In Zen C, all [T; N] arrays have constant size N, so Rule 18.8 is satisfied.
        // We only report if Zen somehow allowed non-constant sizes (which it doesn't in fixed-size
        // arrays).
    }
}

static int block_always_returns(ASTNode *block);

static int stmt_always_returns(ASTNode *stmt)
{
    if (!stmt)
    {
        return 0;
    }

    switch (stmt->type)
    {
    case NODE_RETURN:
        return 1;

    case NODE_BLOCK:
        return block_always_returns(stmt);

    case NODE_IF:
        // Both branches must return for if to always return
        if (stmt->if_stmt.then_body && stmt->if_stmt.else_body)
        {
            return stmt_always_returns(stmt->if_stmt.then_body) &&
                   stmt_always_returns(stmt->if_stmt.else_body);
        }
        return 0;

    case NODE_MATCH:
    {
        if (!stmt->match_stmt.cases)
        {
            return 0;
        }

        int has_default = 0;
        ASTNode *case_node = stmt->match_stmt.cases;
        while (case_node)
        {
            if (case_node->type == NODE_MATCH_CASE)
            {
                if (!stmt_always_returns(case_node->match_case.body))
                {
                    return 0;
                }
                if (case_node->match_case.is_default)
                {
                    has_default = 1;
                }
            }
            case_node = case_node->next;
        }

        return has_default;
    }

    case NODE_LOOP:
        return 0;

    default:
        return 0;
    }
}

static int block_always_returns(ASTNode *block)
{
    if (!block || block->type != NODE_BLOCK)
    {
        return 0;
    }

    ASTNode *stmt = block->block.statements;
    while (stmt)
    {
        if (stmt_always_returns(stmt))
        {
            return 1;
        }
        stmt = stmt->next;
    }
    return 0;
}

static void check_function(TypeChecker *tc, ASTNode *node, int depth)
{
    if (!node)
    {
        return;
    }
    misra_check_param_nesting(tc, node);
    // Mark arg types as used
    for (int i = 0; i < node->func.arg_count; i++)
    {
        mark_type_as_used(tc, node->func.arg_types[i]);
    }

    // Mark return type as used
    mark_type_as_used(tc, node->func.ret_type_info);

    // Rule Zen 1.4: Reserved identifiers
    misra_check_reserved_identifier(tc, node->func.name, node->token);

    tc->current_func = node;
    tc_enter_scope(tc);

    int prev_unreachable = tc->is_unreachable;
    tc->is_unreachable = 0;
    tc->func_return_count = 0;

    MoveState *prev_move_state = tc->pctx->move_state;
    tc->pctx->move_state = move_state_create(NULL);

    for (int i = 0; i < node->func.arg_count; i++)
    {
        if (node->func.param_names && node->func.param_names[i])
        {
            Type *param_type =
                (node->func.arg_types && node->func.arg_types[i]) ? node->func.arg_types[i] : NULL;

            misra_check_tuple_size(tc, param_type, node->token);
            misra_check_pointer_nesting(tc, param_type, node->token);
            misra_check_reserved_identifier(tc, node->func.param_names[i], node->token);
            tc_add_symbol(tc, node->func.param_names[i], param_type, node->token,
                          g_config.misra_mode);
        }
    }

    if (node->func.ret_type_info)
    {
        misra_check_tuple_size(tc, node->func.ret_type_info, node->token);
        misra_check_pointer_nesting(tc, node->func.ret_type_info, node->token);

        // Lifetime Elision result was already computed in pre-pass for named functions.
        // For lambdas or if it somehow missed the pre-pass, ensure it's set.
        if (node->func.elide_from_idx == -1)
        {
            // Simple re-run if needed
        }
    }

    check_node(tc, node->func.body, depth + 1);

    // Control flow analysis: Check if non-void function always returns
    const char *ret_type = node->func.ret_type;
    int is_void = !ret_type || strcmp(ret_type, "void") == 0;

    // Special case: 'main' is allowed to fall off the end (C99 implicit return 0)
    int is_main = node->func.name && strcmp(node->func.name, "main") == 0;

    if (is_main && is_void)
    {
        warn_void_main(node->token);
    }

    if (!is_void && !is_main && node->func.body)
    {
        if (!block_always_returns(node->func.body))
        {
            char msg[MAX_SHORT_MSG_LEN];
            snprintf(msg, sizeof(msg), "Function '%s' may not return a value on all code paths",
                     node->func.name);

            const char *hints[] = {"Ensure all execution paths return a value",
                                   "Consider adding a default return at the end of the function",
                                   NULL};
            tc_error_with_hints(tc, node->token, msg, hints);
        }
    }

    move_state_free(tc->pctx->move_state);
    tc->pctx->move_state = prev_move_state;

    // MISRA audits before leaving function scope
    if (g_config.misra_mode && tc->pctx->current_scope)
    {
        for (int i = 0; i < node->func.arg_count; i++)
        {
            if (node->func.param_names && node->func.param_names[i])
            {
                ZenSymbol *psym =
                    symbol_lookup_local(tc->pctx->current_scope, node->func.param_names[i]);
                if (psym)
                {
                    // Rule 2.7: Unused parameter
                    if (!psym->is_used)
                    {
                        misra_check_unused_param(tc, psym->name, psym->decl_token);
                    }
                    // Rule 8.13: Pointer to const
                    if (psym->type_info && psym->type_info->kind == TYPE_POINTER)
                    {
                        Type *inner = resolve_alias(psym->type_info->inner);
                        if (inner && !inner->is_const && !psym->type_info->is_const &&
                            !psym->is_written_to)
                        {
                            misra_check_const_ptr_param(tc, psym->name, psym->decl_token);
                        }
                    }
                }
            }
        }
    }

    // MISRA Rule 15.5: A function shall have a single point of exit at the end.
    if (g_config.misra_mode && tc->func_return_count > 1)
    {
        char msg[MAX_ERROR_MSG_LEN];
        snprintf(msg, sizeof(msg),
                 "MISRA Rule 15.5: function '%s' has %d return points (must have 1)",
                 node->func.name ? node->func.name : "anonymous", tc->func_return_count);
        tc_error(tc, node->token, msg);
    }

    tc->is_unreachable = prev_unreachable;
    tc_exit_scope(tc);
    tc->current_func = NULL;
}

static void check_expr_var(TypeChecker *tc, ASTNode *node)
{
    // Check if it's an enum variant FIRST, prioritizing value over function constructor if no
    // payload
    EnumVariantReg *ev = find_enum_variant(tc->pctx, node->var_ref.name);
    if (ev)
    {
        FuncSig *sig = find_func(tc->pctx, node->var_ref.name);
        // If it's a no-payload variant, treat it as an enum value (TYPE_ENUM)
        // If it has payloads, it's a constructor function (TYPE_FUNCTION)
        if (!sig || sig->total_args == 0)
        {
            Type *et = type_new(TYPE_ENUM);
            et->name = xstrdup(ev->enum_name);
            node->type_info = et;
            return;
        }
    }

    ZenSymbol *sym = tc_lookup(tc, node->var_ref.name);
    node->var_ref.symbol = sym; // Store for MISRA audits

    if (sym && sym->type_info)
    {
        // Clone the type to keep metadata (like lifetime_depth) isolated per usage site
        node->type_info = type_clone(sym->type_info);

        // Rule 8.9 tracking: Identify globals used in only one function
        if (!sym->is_local && sym->kind == SYM_VARIABLE && tc->current_func)
        {
            ZenSymbol *orig = sym->original ? sym->original : sym;
            if (orig->first_using_func == NULL)
            {
                orig->first_using_func = tc->current_func;
            }
            else if (orig->first_using_func != tc->current_func)
            {
                orig->multi_func_use = 1;
            }
        }
    }
    else
    {
        // Fallback: Check if it's a mangled function name (e.g. from :: operator or enum
        // constructor)
        FuncSig *sig = find_func(tc->pctx, node->var_ref.name);
        if (sig)
        {
            Type *fn_type = type_new(TYPE_FUNCTION);
            fn_type->is_raw = 1;
            fn_type->inner = sig->ret_type ? sig->ret_type : type_new(TYPE_VOID);
            fn_type->arg_count = sig->total_args;
            if (sig->total_args > 0)
            {
                fn_type->args = xmalloc(sizeof(Type *) * sig->total_args);
                for (int i = 0; i < sig->total_args; i++)
                {
                    fn_type->args[i] = sig->arg_types[i];
                }
            }
            node->type_info = fn_type;
        }
        else if (g_config.misra_mode)
        {
            char msg[MAX_SHORT_MSG_LEN];
            snprintf(msg, sizeof(msg), "Undefined variable '%s' (MISRA Rule 17.3)",
                     node->var_ref.name);
            tc_error(tc, node->token, msg);
        }
    }

    if (!tc->is_assign_lhs)
    {
        check_use_validity(tc, node);
    }
}

static void tc_check_trait(TypeChecker *tc, ASTNode *node, int depth)
{
    ASTNode *method = node->trait.methods;
    while (method)
    {
        check_node(tc, method, depth + 1);
        method = method->next;
    }
}

static void tc_check_impl(TypeChecker *tc, ASTNode *node, int depth)
{
    // Skip templates
    if (node->impl.struct_name && strchr(node->impl.struct_name, '<'))
    {
        return;
    }

    ASTNode *method = node->impl.methods;
    while (method)
    {
        check_node(tc, method, depth + 1);
        method = method->next;
    }
}

static void tc_check_impl_trait(TypeChecker *tc, ASTNode *node, int depth)
{
    // Skip templates
    if (node->impl_trait.target_type && strchr(node->impl_trait.target_type, '<'))
    {
        return;
    }

    ASTNode *method = node->impl_trait.methods;
    while (method)
    {
        check_node(tc, method, depth + 1);
        method = method->next;
    }
}

static void check_expr_literal(TypeChecker *tc, ASTNode *node)
{
    (void)tc;
    if (node->type_info && node->type_info->kind != TYPE_UNKNOWN)
    {
        return;
    }

    switch (node->literal.type_kind)
    {
    case LITERAL_INT:
        node->type_info = type_new(TYPE_I32); // Default to i32
        break;
    case LITERAL_FLOAT:
        node->type_info = type_new(TYPE_F64); // Default to f64
        break;
    case LITERAL_STRING:
    case LITERAL_RAW_STRING:
        node->type_info = type_new(TYPE_STRING);
        break;
    case LITERAL_CHAR:
        node->type_info = type_new(TYPE_CHAR);
        break;
    }
}

static void check_struct_init(TypeChecker *tc, ASTNode *node, int depth)
{
    if (!node)
    {
        return;
    }
    RECURSION_GUARD_TOKEN(tc->pctx, node->token, );

    // MISRA: Mark struct type as used
    mark_type_as_used(tc, node->type_info);
    if (g_config.misra_mode)
    {
        ZenSymbol *struct_sym =
            symbol_lookup_kind(tc->pctx->global_scope, node->struct_init.struct_name, SYM_STRUCT);
        if (struct_sym)
        {
            struct_sym->is_dereferenced = 1;
        }
    }

    // Find struct definition
    ASTNode *def = find_struct_def(tc->pctx, node->struct_init.struct_name);
    if (!def)
    {
        char msg[MAX_SHORT_MSG_LEN];
        snprintf(msg, sizeof(msg), "Unknown struct '%s'", node->struct_init.struct_name);
        tc_error(tc, node->token, msg);
        RECURSION_EXIT(tc->pctx);
        return;
    }

    // Iterate provided fields
    ASTNode *field_init = node->struct_init.fields;
    while (field_init)
    {
        // Rule 9.4: Double initialization check
        if (g_config.misra_mode)
        {
            ASTNode *prev = node->struct_init.fields;
            while (prev != field_init)
            {
                if (strcmp(prev->var_decl.name, field_init->var_decl.name) == 0)
                {
                    misra_check_double_initialization(tc, field_init->var_decl.name,
                                                      field_init->token);
                    break;
                }
                prev = prev->next;
            }
        }

        // Find corresponding field in definition
        ASTNode *def_field = def->strct.fields;
        Type *expected_type = NULL;
        int found = 0;

        while (def_field)
        {
            if (def_field->type == NODE_FIELD &&
                strcmp(def_field->field.name, field_init->var_decl.name) == 0)
            {
                found = 1;
                expected_type = def_field->type_info;
                break;
            }
            def_field = def_field->next;
        }

        if (found)
        {
            field_init->type_info = expected_type;
        }

        if (found && expected_type && field_init->var_decl.init_expr->type == NODE_LAMBDA)
        {
            field_init->var_decl.init_expr->type_info = expected_type;
        }

        // Check the initialization expression
        check_node(tc, field_init->var_decl.init_expr, depth + 1);

        if (!found)
        {
            char msg[MAX_SHORT_MSG_LEN];
            snprintf(msg, sizeof(msg), "Struct '%s' has no field named '%s'",
                     node->struct_init.struct_name, field_init->var_decl.name);
            tc_error(tc, field_init->token, msg);
        }
        else if (expected_type && field_init->var_decl.init_expr->type_info)
        {
            // Localize expected field type depth for initialization check.
            // A local struct's fields expect lifetimes compatible with the struct's own scope.
            Type *localized_expected = type_clone(expected_type);
            if (localized_expected && tc->current_func)
            {
                localized_expected->lifetime_depth = tc->current_depth;
            }
            check_type_compatibility(tc, localized_expected,
                                     field_init->var_decl.init_expr->type_info, field_init->token,
                                     NULL, 0);
        }

        // Move Analysis: Check if the initializer moves a non-copy value.
        check_move_for_rvalue(tc, field_init->var_decl.init_expr);

        field_init = field_init->next;
    }

    // Check for missing required fields
    ASTNode *def_field = def->strct.fields;
    while (def_field)
    {
        if (def_field->type == NODE_FIELD && def_field->field.name)
        {
            int provided = 0;
            ASTNode *fi = node->struct_init.fields;
            while (fi)
            {
                if (fi->var_decl.name && strcmp(fi->var_decl.name, def_field->field.name) == 0)
                {
                    provided = 1;
                    break;
                }
                fi = fi->next;
            }
            if (!provided)
            {
                char msg[MAX_SHORT_MSG_LEN];
                snprintf(msg, sizeof(msg), "Missing field '%s' in initializer for struct '%s'",
                         def_field->field.name, node->struct_init.struct_name);
                const char *hints[] = {"All struct fields must be initialized", NULL};
                tc_error_with_hints(tc, node->token, msg, hints);
            }
        }
        def_field = def_field->next;
    }

    node->type_info = def->type_info;
    RECURSION_EXIT(tc->pctx);
}

static void check_loop_passes(TypeChecker *tc, ASTNode *node, int depth)
{
    if (!node)
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

    MoveState *prev_break = tc->loop_break_state;
    MoveState *prev_cont = tc->loop_continue_state;
    tc->loop_break_state = NULL;
    tc->loop_continue_state = NULL;

    MoveState *initial_state = tc->pctx->move_state;
    MoveState *loop_start = initial_state ? move_state_clone(initial_state) : NULL;
    MoveState *outer_start_state = tc->loop_start_state;
    tc->loop_start_state = loop_start;

    int outer_in_pass2 = tc->in_loop_pass2;
    tc->in_loop_pass2 = 0;

    int outer_break_count = tc->loop_break_count;
    tc->loop_break_count = 0;

    int initial_unreachable = tc->is_unreachable;

    // Pass 1: standard typecheck and move check
    tc->is_unreachable = 0; // The loop start is assumed reachable if we got here

    switch (node->type)
    {
    case NODE_WHILE:
    {
        misra_check_compound_body(tc, node->while_stmt.body, "while");
        int old_stmt_ctx = tc->is_stmt_context;
        tc->is_stmt_context = 0;
        check_node(tc, node->while_stmt.condition, depth + 1);
        tc->is_stmt_context = old_stmt_ctx;

        if (node->while_stmt.condition && node->while_stmt.condition->type_info)
        {
            Type *cond_type = resolve_alias(node->while_stmt.condition->type_info);
            if (g_config.misra_mode)
            {
                if (cond_type->kind != TYPE_BOOL)
                {
                    misra_check_condition_boolean(tc, node->while_stmt.condition->type_info,
                                                  node->while_stmt.condition->token);
                }
                int inv;
                if (is_expression_invariant(tc, node->while_stmt.condition, &inv))
                {
                    misra_check_invariant_condition(tc, node->while_stmt.condition->token);
                }
            }
            else if (cond_type->kind != TYPE_BOOL && !is_integer_type(cond_type) &&
                     cond_type->kind != TYPE_POINTER && cond_type->kind != TYPE_STRING)
            {
                const char *hints[] = {"While conditions must be boolean, integer, or pointer",
                                       NULL};
                tc_error_with_hints(tc, node->while_stmt.condition->token,
                                    "Condition must be a truthy type", hints);
            }
        }
        check_node(tc, node->while_stmt.body, depth + 1);
        break;
    }

    case NODE_FOR:
        misra_check_compound_body(tc, node->for_stmt.body, "for");
        tc_enter_scope(tc); // For loop init variable is scoped
        check_node(tc, node->for_stmt.init, depth + 1);

        // Loop start is conceptually here for FOR
        if (loop_start)
        {
            move_state_free(loop_start);
        }
        loop_start = tc->pctx->move_state ? move_state_clone(tc->pctx->move_state) : NULL;
        tc->loop_start_state = loop_start;

        check_node(tc, node->for_stmt.condition, depth + 1);
        if (node->for_stmt.condition && node->for_stmt.condition->type_info)
        {
            Type *cond_type = resolve_alias(node->for_stmt.condition->type_info);
            if (g_config.misra_mode)
            {
                if (cond_type->kind != TYPE_BOOL)
                {
                    misra_check_condition_boolean(tc, node->for_stmt.condition->type_info,
                                                  node->for_stmt.condition->token);
                }
                int inv;
                if (is_expression_invariant(tc, node->for_stmt.condition, &inv))
                {
                    misra_check_invariant_condition(tc, node->for_stmt.condition->token);
                }
            }
            else if (cond_type->kind != TYPE_BOOL && !is_integer_type(cond_type) &&
                     cond_type->kind != TYPE_POINTER && cond_type->kind != TYPE_STRING)
            {
                const char *hints[] = {"For conditions must be boolean, integer, or pointer", NULL};
                tc_error_with_hints(tc, node->for_stmt.condition->token,
                                    "Condition must be a truthy type", hints);
            }
        }
        check_node(tc, node->for_stmt.body, depth + 1);
        check_node(tc, node->for_stmt.step, depth + 1); // step happens after body

        if (g_config.misra_mode && node->for_stmt.step)
        {
            if (node->for_stmt.step->type == NODE_EXPR_BINARY)
            {
                const char *step_op = node->for_stmt.step->binary.op;
                if (strstr(step_op, "=") && node->for_stmt.step->binary.left->type_info)
                {
                    misra_check_loop_counter_float(tc, node->for_stmt.step->binary.left->type_info,
                                                   node->for_stmt.step->token);
                }
            }
        }
        break;

    case NODE_FOR_RANGE:
        check_node(tc, node->for_range.start, depth + 1);
        check_node(tc, node->for_range.end, depth + 1);

        // Loop start conceptually here
        if (loop_start)
        {
            move_state_free(loop_start);
        }
        loop_start = tc->pctx->move_state ? move_state_clone(tc->pctx->move_state) : NULL;
        tc->loop_start_state = loop_start;

        check_node(tc, node->for_range.body, depth + 1);
        break;

    case NODE_LOOP:
        check_node(tc, node->loop_stmt.body, depth + 1);
        break;

    case NODE_REPEAT:
        check_node(tc, node->repeat_stmt.body, depth + 1);
        break;

    case NODE_DO_WHILE:
    {
        misra_check_compound_body(tc, node->do_while_stmt.body, "do-while");
        int old_stmt_ctx = tc->is_stmt_context;
        tc->is_stmt_context = 0;
        check_node(tc, node->do_while_stmt.body, depth + 1);
        check_node(tc, node->do_while_stmt.condition, depth + 1);
        tc->is_stmt_context = old_stmt_ctx;

        if (node->do_while_stmt.condition && node->do_while_stmt.condition->type_info)
        {
            Type *cond_type = resolve_alias(node->do_while_stmt.condition->type_info);
            if (g_config.misra_mode)
            {
                if (cond_type->kind != TYPE_BOOL)
                {
                    misra_check_condition_boolean(tc, node->do_while_stmt.condition->type_info,
                                                  node->do_while_stmt.condition->token);
                }
                int inv;
                if (is_expression_invariant(tc, node->do_while_stmt.condition, &inv))
                {
                    misra_check_invariant_condition(tc, node->do_while_stmt.condition->token);
                }
            }
            else if (cond_type->kind != TYPE_BOOL && !is_integer_type(cond_type) &&
                     cond_type->kind != TYPE_POINTER && cond_type->kind != TYPE_STRING)
            {
                const char *hints[] = {"Do-while conditions must be boolean, integer, or pointer",
                                       NULL};
                tc_error_with_hints(tc, node->do_while_stmt.condition->token,
                                    "Condition must be a truthy type", hints);
            }
        }
    default:
        break;
    }
    }

    tc->loop_break_count = 0; // Reset for pass 2 (avoid double counts)

    // Determine next iter state based on continue and fallthrough
    MoveState *fallthrough_state = tc->pctx->move_state;
    int fallthrough_unreachable = tc->is_unreachable;

    MoveState *next_iter_state = NULL;
    if (!fallthrough_unreachable && fallthrough_state)
    {
        move_state_merge_into(&next_iter_state, fallthrough_state);
    }
    if (tc->loop_continue_state)
    {
        move_state_merge_into(&next_iter_state, tc->loop_continue_state);
    }

    // Pass 2: Re-run with next_iter_state to catch use-after-move across iterations
    if (next_iter_state)
    {
        int prev_move_checks_only = tc->move_checks_only;
        tc->move_checks_only = 1; // suppress type errors
        tc->in_loop_pass2 = 1;

        tc->pctx->move_state = move_state_clone(next_iter_state);
        tc->is_unreachable = 0;

        tc->loop_break_state = NULL;
        tc->loop_continue_state = NULL;

        // Re-run appropriate parts
        switch (node->type)
        {
        case NODE_WHILE:
            check_node(tc, node->while_stmt.condition, depth + 1);
            check_node(tc, node->while_stmt.body, depth + 1);
            break;
        case NODE_FOR:
            check_node(tc, node->for_stmt.condition, depth + 1);
            check_node(tc, node->for_stmt.body, depth + 1);
            check_node(tc, node->for_stmt.step, depth + 1);
            break;
        case NODE_FOR_RANGE:
            check_node(tc, node->for_range.body, depth + 1);
            break;
        case NODE_LOOP:
            check_node(tc, node->loop_stmt.body, depth + 1);
            break;
        case NODE_REPEAT:
            check_node(tc, node->repeat_stmt.body, depth + 1);
            break;
        case NODE_DO_WHILE:
            misra_check_compound_body(tc, node->do_while_stmt.body, "do-while");
            check_node(tc, node->do_while_stmt.body, depth + 1);
            check_node(tc, node->do_while_stmt.condition, depth + 1);
            break;
        default:
            break;
        }

        if (tc->pctx->move_state)
        {
            move_state_free(tc->pctx->move_state);
        }
        if (tc->loop_break_state)
        {
            move_state_free(tc->loop_break_state);
        }
        if (tc->loop_continue_state)
        {
            move_state_free(tc->loop_continue_state);
        }

        tc->move_checks_only = prev_move_checks_only;
    }

    // Compute final move state exiting the loop
    MoveState *final_state = NULL;
    // Loops can exit via condition falsification (next_iter_state) or breaks
    if (next_iter_state)
    {
        // Assume infinite loops (like NODE_LOOP) don't exit naturally unless broken,
        // but for safety we'll merge next_iter_state for all, treating condition as maybe false.
        if (node->type != NODE_LOOP)
        {
            move_state_merge_into(&final_state, next_iter_state);
        }
    }
    if (tc->loop_break_state)
    {
        move_state_merge_into(&final_state, tc->loop_break_state);
    }

    // Cleanup Pass 1 states
    if (tc->loop_break_state)
    {
        move_state_free(tc->loop_break_state);
    }
    if (tc->loop_continue_state)
    {
        move_state_free(tc->loop_continue_state);
    }

    tc->loop_break_count = outer_break_count;
    if (next_iter_state)
    {
        move_state_free(next_iter_state);
    }
    if (loop_start)
    {
        move_state_free(loop_start);
    }

    // Restore outer context
    if (node->type == NODE_FOR)
    {
        tc_exit_scope(tc);
    }

    // If the loop is an infinite loop and has no breaks, it is unconditionally unreachable after.
    if ((node->type == NODE_LOOP || node->type == NODE_REPEAT) && !final_state)
    {
        tc->is_unreachable = 1;
    }
    else if (final_state)
    {
        tc->is_unreachable = 0;
    }
    else
    {
        tc->is_unreachable = initial_unreachable;
    }

    if (tc->pctx->move_state)
    {
        move_state_free(tc->pctx->move_state);
    }
    tc->pctx->move_state = final_state ? final_state : initial_state;

    tc->loop_break_state = prev_break;
    tc->loop_continue_state = prev_cont;
    tc->loop_start_state = outer_start_state;
    tc->in_loop_pass2 = outer_in_pass2;
    RECURSION_EXIT(tc->pctx);
}

static void check_node(TypeChecker *tc, ASTNode *node, int depth)
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
                const char *rule = g_config.misra_mode ? "MISRA Rule 2.1: " : "";
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
            if (g_config.misra_mode)
            {
                misra_check_condition_boolean(tc, node->if_stmt.condition->type_info,
                                              node->if_stmt.condition->token);
                int inv;
                if (is_expression_invariant(tc, node->if_stmt.condition, &inv))
                {
                    misra_check_invariant_condition(tc, node->if_stmt.condition->token);
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

        misra_check_compound_body(tc, node->if_stmt.then_body, "if");
        if (node->if_stmt.else_body)
        {
            if (node->if_stmt.else_body->type == NODE_IF)
            {
                misra_check_terminal_else(tc, node);
            }
            else
            {
                misra_check_compound_body(tc, node->if_stmt.else_body, "else");
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
        misra_check_match_stmt(tc, node);
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
                    if (g_config.misra_mode && !tc->is_unreachable && mcase->match_case.body &&
                        mcase->match_case.body->type == NODE_BLOCK &&
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

                if (!g_config.misra_mode)
                {
                    const char *hints[] = {"Add a default '_' case to handle all possibilities",
                                           NULL};
                    tc_error_with_hints(tc, node->token,
                                        "Match may not be exhaustive (no default case)", hints);
                }

                misra_check_match_stmt(tc, node);
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
            misra_check_reserved_identifier(tc, node->strct.name, node->token);
            misra_check_struct_decl(tc, node);
            if (node->strct.is_union)
            {
                misra_check_union(tc, node->token);
            }
        }
        else if (node->type == NODE_ENUM)
        {
            misra_check_reserved_identifier(tc, node->enm.name, node->token);
        }
        else if (node->type == NODE_TYPE_ALIAS)
        {
            misra_check_reserved_identifier(tc, node->type_alias.alias, node->token);
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
                if (g_config.misra_mode)
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
            if (g_config.misra_mode)
            {
                misra_check_condition_boolean(tc, node->guard_stmt.condition->type_info,
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
            if (g_config.misra_mode)
            {
                misra_check_condition_boolean(tc, node->unless_stmt.condition->type_info,
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
            if (g_config.misra_mode)
            {
                misra_check_condition_boolean(tc, node->assert_stmt.condition->type_info,
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

            if (g_config.misra_mode && target_type)
            {
                misra_check_cast(tc, target_type, source_type, node->token,
                                 is_composite_expression(node->cast.expr));
                misra_check_pointer_conversion(tc, target_type, source_type, node->token);
                misra_check_void_ptr_cast(tc, target_type, source_type, node->token);
                if (target_type->kind == TYPE_POINTER)
                {
                    misra_check_null_pointer_constant(tc, node, node->token);
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
        misra_check_initializer_side_effects(tc, node);
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
        misra_check_initializer_side_effects(tc, node);
        ASTNode *elem = node->tuple_literal.elements;
        while (elem)
        {
            check_node(tc, elem, depth + 1);
            elem = elem->next;
        }
    }
    break;
    case NODE_EXPR_STRUCT_INIT:
        misra_check_initializer_side_effects(tc, node);
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
            if (g_config.misra_mode)
            {
                misra_check_condition_boolean(tc, node->ternary.cond->type_info,
                                              node->ternary.cond->token);
                int inv;
                if (is_expression_invariant(tc, node->ternary.cond, &inv))
                {
                    misra_check_invariant_condition(tc, node->ternary.cond->token);
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
                if (g_config.misra_mode)
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
                if (g_config.misra_mode)
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

            if (g_config.misra_mode && tc_expr_has_side_effects(node->size_of.expr))
            {
                misra_check_side_effects_sizeof(tc, node->size_of.expr);
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
            misra_check_iteration_termination(tc, node->token);
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
        if (g_config.misra_mode)
        {
            ZenSymbol *lbl = tc_lookup(tc, node->goto_stmt.label_name);
            if (lbl && lbl->decl_token.line != 0)
            {
                misra_check_goto_constraint(tc, node->token, lbl->decl_token);
            }
        }
        misra_check_goto(tc, node->token);
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
        misra_check_stdarg(tc, node->token);
        break;

    case NODE_RAW_STMT:
        misra_check_raw_block(tc, node->token);
        break;
    case NODE_PREPROC_DIRECTIVE:
        // Rule Zen 1.4 is already handled by parser_audit_preprocessor
        break;
    case NODE_PLUGIN:
        misra_check_plugin_block(tc, node->token);
        break;
    case NODE_LABEL:
        if (g_config.misra_mode)
        {
            ZenSymbol *lbl =
                symbol_add(tc->pctx->current_scope, node->label_stmt.label_name, SYM_LABEL);
            if (lbl)
            {
                lbl->decl_token = node->token;
            }
        }
        break;
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

static void check_expr_lambda(TypeChecker *tc, ASTNode *node, int depth)
{
    Type *expected = get_inner_type(node->type_info);
    if (expected && expected->kind == TYPE_FUNCTION && expected->is_raw)
    {
        if (node->lambda.num_captures == 0)
        {
            node->lambda.is_bare = 1;
        }
        else
        {
            const char *hints[] = {
                "Only non-capturing lambdas can be converted to raw function pointers", NULL};
            tc_error_with_hints(tc, node->token,
                                "Cannot convert capturing lambda to raw function pointer", hints);
        }
    }

    if (node->lambda.captured_vars)
    {
        for (int i = 0; i < node->lambda.num_captures; i++)
        {
            char *var_name = node->lambda.captured_vars[i];
            int mode = node->lambda.capture_modes ? node->lambda.capture_modes[i]
                                                  : node->lambda.default_capture_mode;

            ZenSymbol *sym = tc_lookup(tc, var_name);
            if (!sym)
            {
                continue;
            }

            check_path_validity(tc, var_name, node->token);

            if (mode == 0)
            {
                Type *t = sym->type_info;
                if (!is_type_copy(tc->pctx, t))
                {
                    mark_symbol_moved(tc->pctx, sym, node);
                }
            }
        }
    }

    tc_enter_scope(tc);

    for (int i = 0; i < node->lambda.num_params; i++)
    {
        char *pname = node->lambda.param_names[i];
        Type *ptype = NULL;
        Type *node_ti = get_inner_type(node->type_info);
        if (node_ti && node_ti->kind == TYPE_FUNCTION && node_ti->args)
        {
            if (i < node_ti->arg_count)
            {
                ptype = node_ti->args[i];
            }
        }
        tc_add_symbol(tc, pname, ptype, node->token, 0);
    }

    // Add captured variables to the scope to ensure visibility and immutability inside the lambda.
    if (node->lambda.captured_vars)
    {
        int saved_silent = tc->pctx->silent_warnings;
        tc->pctx->silent_warnings = 1;
        for (int i = 0; i < node->lambda.num_captures; i++)
        {
            char *var_name = node->lambda.captured_vars[i];
            int mode = node->lambda.capture_modes ? node->lambda.capture_modes[i]
                                                  : node->lambda.default_capture_mode;

            // Lookup original symbol to get its type
            ZenSymbol *orig_sym = tc_lookup(tc, var_name);
            if (orig_sym)
            {
                // Shadow the original variable in the lambda scope
                // Value captures are immutable.
                tc_add_symbol(tc, var_name, orig_sym->type_info, node->token, (mode == 0));
            }
        }
        tc->pctx->silent_warnings = saved_silent;
    }

    MoveState *prev_move_state = tc->pctx->move_state;
    tc->pctx->move_state = move_state_create(NULL);

    int prev_unreachable = tc->is_unreachable;
    tc->is_unreachable = 0;

    if (node->lambda.body)
    {
        if (node->lambda.body->type == NODE_BLOCK)
        {
            check_block(tc, node->lambda.body, depth + 1);
        }
        else
        {
            check_node(tc, node->lambda.body, depth + 1);
        }
    }

    move_state_free(tc->pctx->move_state);
    tc->pctx->move_state = prev_move_state;

    tc->is_unreachable = prev_unreachable;
    tc_exit_scope(tc);
}

static void infer_node_lifetime(TypeChecker *tc, ASTNode *node)
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

static void check_program_prepass(TypeChecker *tc, ASTNode *root, int depth)
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

    if (g_config.misra_mode)
    {
        misra_audit_unused_symbols(&tc);
        misra_audit_block_scope(&tc);
        misra_audit_identifier_uniqueness(&tc);
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
