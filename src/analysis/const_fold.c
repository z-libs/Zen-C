#include "analysis/const_fold.h"
#include <string.h>
#include <stdio.h>

int eval_const_int_expr(ASTNode *node, ParserContext *ctx, long long *out_val)
{
    if (!node)
    {
        return 0;
    }

    switch (node->type)
    {
    case NODE_EXPR_LITERAL:
        if (node->literal.type_kind == LITERAL_INT)
        {
            *out_val = node->literal.int_val;
            return 1;
        }
        return 0;

    case NODE_EXPR_VAR:
    {
        ZenSymbol *sym = find_symbol_entry(ctx, node->var_ref.name);
        if (sym && sym->is_const_value)
        {
            *out_val = sym->const_int_val;
            return 1;
        }
        // Check for enum variants
        EnumVariantReg *ev = find_enum_variant(ctx, node->var_ref.name);
        if (ev)
        {
            *out_val = (long long)ev->tag_id;
            return 1;
        }
        break;
    }

    case NODE_EXPR_BINARY:
    {
        long long left, right;
        if (!eval_const_int_expr(node->binary.left, ctx, &left))
        {
            return 0;
        }
        if (!eval_const_int_expr(node->binary.right, ctx, &right))
        {
            return 0;
        }

        if (strcmp(node->binary.op, "+") == 0)
        {
            *out_val = left + right;
        }
        else if (strcmp(node->binary.op, "-") == 0)
        {
            *out_val = left - right;
        }
        else if (strcmp(node->binary.op, "*") == 0)
        {
            *out_val = left * right;
        }
        else if (strcmp(node->binary.op, "/") == 0)
        {
            if (right == 0)
            {
                return 0; // Division by zero
            }
            *out_val = left / right;
        }
        else if (strcmp(node->binary.op, "%") == 0)
        {
            if (right == 0)
            {
                return 0;
            }
            *out_val = left % right;
        }
        else if (strcmp(node->binary.op, "<<") == 0)
        {
            *out_val = left << right;
        }
        else if (strcmp(node->binary.op, ">>") == 0)
        {
            *out_val = left >> right;
        }
        else if (strcmp(node->binary.op, "&") == 0)
        {
            *out_val = left & right;
        }
        else if (strcmp(node->binary.op, "|") == 0)
        {
            *out_val = left | right;
        }
        else if (strcmp(node->binary.op, "^") == 0)
        {
            *out_val = left ^ right;
        }
        else
        {
            return 0;
        }

        return 1;
    }

    case NODE_EXPR_UNARY:
    {
        long long operand;
        if (!eval_const_int_expr(node->unary.operand, ctx, &operand))
        {
            return 0;
        }

        if (strcmp(node->unary.op, "-") == 0)
        {
            *out_val = -operand;
        }
        else if (strcmp(node->unary.op, "+") == 0)
        {
            *out_val = +operand;
        }
        else if (strcmp(node->unary.op, "~") == 0)
        {
            *out_val = ~operand;
        }
        else
        {
            return 0;
        }

        return 1;
    }

    default:
        return 0;
    }
    return 0; // For warning.
}
