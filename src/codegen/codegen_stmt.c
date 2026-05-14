// SPDX-License-Identifier: MIT
#include "../parser/parser.h"

#include "codegen.h"
#include "zprep.h"
#include "../constants.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../plugins/plugin_manager.h"
#include "ast.h"
#include "zprep_plugin.h"

// Helper: emit a single pattern condition (either a value, or a range)
static void emit_single_pattern_cond(ParserContext *ctx, const char *pat, int id, int is_ptr)
{
    // Check for range pattern: "start..end" or "start..=end"
    char *range_incl = strstr(pat, "..=");
    char *range_excl = strstr(pat, "..");

    if (range_incl)
    {
        // Inclusive range: start..=end -> _m_id >= start && _m_id <= end
        int start_len = (int)(range_incl - pat);
        char *start = xmalloc(start_len + 1);
        strncpy(start, pat, start_len);
        start[start_len] = 0;
        char *end = xstrdup(range_incl + 3);
        if (is_ptr)
        {
            EMIT(ctx, "(*_m_%d >= %s && *_m_%d <= %s)", id, start, id, end);
        }
        else
        {
            EMIT(ctx, "(_m_%d >= %s && _m_%d <= %s)", id, start, id, end);
        }
        zfree(start);
        zfree(end);
    }
    else if (range_excl)
    {
        // Exclusive range: start..end -> _m_id >= start && _m_id < end
        int start_len = (int)(range_excl - pat);
        char *start = xmalloc(start_len + 1);
        strncpy(start, pat, start_len);
        start[start_len] = 0;
        char *end = xstrdup(range_excl + 2);
        if (is_ptr)
        {
            EMIT(ctx, "(*_m_%d >= %s && *_m_%d < %s)", id, start, id, end);
        }
        else
        {
            EMIT(ctx, "(_m_%d >= %s && _m_%d < %s)", id, start, id, end);
        }
        zfree(start);
        zfree(end);
    }
    else if (pat[0] == '"')
    {
        // String pattern - string comparison, _m is char* or similar
        if (is_ptr)
        {
            EMIT(ctx, "strcmp(*_m_%d, %s) == 0", id, pat);
        }
        else
        {
            EMIT(ctx, "strcmp(_m_%d, %s) == 0", id, pat);
        }
    }
    else
    {
        // Numeric, Char literal, or simple pattern
        if (is_ptr)
        {
            EMIT(ctx, "*_m_%d == %s", id, pat);
        }
        else
        {
            EMIT(ctx, "_m_%d == %s", id, pat);
        }
    }
}

// Helper: emit condition for a pattern (may contain OR patterns with '|')
static void emit_pattern_condition(ParserContext *ctx, const char *pattern, int id, int is_ptr)
{
    // Check if pattern contains '|' for OR patterns
    if (strchr(pattern, '|'))
    {
        // Split by '|' and emit OR conditions
        char *pattern_copy = xstrdup(pattern);
        char *saveptr;
        char *part = strtok_r(pattern_copy, "|", &saveptr);
        int first = 1;
        EMIT(ctx, "(");
        while (part)
        {
            if (!first)
            {
                EMIT(ctx, " || ");
            }

            // Check if part is an enum variant
            EnumVariantReg *reg = find_enum_variant(ctx, part);
            if (reg)
            {
                int simple = is_simple_enum(ctx, reg->enum_name);
                if (simple)
                {
                    if (is_ptr)
                    {
                        EMIT(ctx, "*_m_%d == %d", id, reg->tag_id);
                    }
                    else
                    {
                        EMIT(ctx, "_m_%d == %d", id, reg->tag_id);
                    }
                }
                else
                {
                    if (is_ptr)
                    {
                        EMIT(ctx, "_m_%d->tag == %d", id, reg->tag_id);
                    }
                    else
                    {
                        EMIT(ctx, "_m_%d.tag == %d", id, reg->tag_id);
                    }
                }
            }
            else
            {
                emit_single_pattern_cond(ctx, part, id, is_ptr);
            }
            first = 0;
            part = strtok_r(NULL, "|", &saveptr);
        }
        EMIT(ctx, ")");
        zfree(pattern_copy);
    }
    else
    {
        // Single pattern (may be a range)
        EnumVariantReg *reg = find_enum_variant(ctx, pattern);
        if (reg)
        {
            int simple = is_simple_enum(ctx, reg->enum_name);
            if (simple)
            {
                if (is_ptr)
                {
                    EMIT(ctx, "*_m_%d == %d", id, reg->tag_id);
                }
                else
                {
                    EMIT(ctx, "_m_%d == %d", id, reg->tag_id);
                }
            }
            else
            {
                if (is_ptr)
                {
                    EMIT(ctx, "_m_%d->tag == %d", id, reg->tag_id);
                }
                else
                {
                    EMIT(ctx, "_m_%d.tag == %d", id, reg->tag_id);
                }
            }
        }
        else
        {
            emit_single_pattern_cond(ctx, pattern, id, is_ptr);
        }
    }
}

// Helper
static bool is_int_type(TypeKind k)
{
    static const bool is_int[] = {
        [TYPE_CHAR] = true,    [TYPE_I8] = true,         [TYPE_U8] = true,
        [TYPE_I16] = true,     [TYPE_U16] = true,        [TYPE_I32] = true,
        [TYPE_U32] = true,     [TYPE_I64] = true,        [TYPE_U64] = true,
        [TYPE_I128] = true,    [TYPE_U128] = true,       [TYPE_INT] = true,
        [TYPE_UINT] = true,    [TYPE_USIZE] = true,      [TYPE_ISIZE] = true,
        [TYPE_BYTE] = true,    [TYPE_RUNE] = true,       [TYPE_ENUM] = true,
        [TYPE_C_INT] = true,   [TYPE_C_UINT] = true,     [TYPE_C_LONG] = true,
        [TYPE_C_ULONG] = true, [TYPE_C_LONGLONG] = true, [TYPE_C_ULONGLONG] = true,
        [TYPE_C_SHORT] = true, [TYPE_C_USHORT] = true,   [TYPE_C_CHAR] = true,
        [TYPE_C_UCHAR] = true, [TYPE_BITINT] = true,     [TYPE_UBITINT] = true};
    if (k >= 0 && k < (int)(sizeof(is_int) / sizeof(is_int[0])))
    {
        return is_int[k];
    }
    return false;
}

void codegen_match_internal(ParserContext *ctx, ASTNode *node, int use_result)
{
    int id = ctx->cg.tmp_counter++;
    int is_self = (node->match_stmt.expr->type == NODE_EXPR_VAR &&
                   strcmp(node->match_stmt.expr->var_ref.name, "self") == 0);

    char *ret_type = infer_type(ctx, node);
    int is_expr = (use_result && ret_type && strcmp(ret_type, "void") != 0);

    if (is_expr)
    {
        EMIT(ctx, "({ ");
    }
    else
    {
        EMIT(ctx, "{ ");
    }

    // Check if any case uses ref binding - only take address if needed
    int has_ref_binding = 0;
    ASTNode *ref_check = node->match_stmt.cases;
    while (ref_check)
    {
        if (ref_check->match_case.binding_refs)
        {
            for (int i = 0; i < ref_check->match_case.binding_count; i++)
            {
                if (ref_check->match_case.binding_refs[i])
                {
                    has_ref_binding = 1;
                    break;
                }
            }
        }
        if (has_ref_binding)
        {
            break;
        }
        ref_check = ref_check->next;
    }

    int is_lvalue_opt = (node->match_stmt.expr->type == NODE_EXPR_VAR ||
                         node->match_stmt.expr->type == NODE_EXPR_MEMBER ||
                         node->match_stmt.expr->type == NODE_EXPR_INDEX);

    emit_source_mapping(ctx, node); // Step through match statements elegantly

    if (is_self)
    {
        emit_auto_type(ctx, node->match_stmt.expr, node->token);
        EMIT(ctx, " _m_%d = ", id);
        codegen_expression(ctx, node->match_stmt.expr);
        EMIT(ctx, "; ");
    }
    else if (has_ref_binding && is_lvalue_opt)
    {
        // Take address for ref bindings
        EMIT(ctx, "ZC_AUTO_INIT(_m_%d, &", id);
        codegen_expression(ctx, node->match_stmt.expr);
        EMIT(ctx, "); ");
    }
    else if (has_ref_binding)
    {
        // Non-lvalue with ref binding: create temporary
        emit_auto_type(ctx, node->match_stmt.expr, node->token);
        EMIT(ctx, " _temp_%d = ", id);
        codegen_expression(ctx, node->match_stmt.expr);
        EMIT(ctx, "; ");

        EMIT(ctx, "ZC_AUTO_INIT(_m_%d, &_temp_%d); ", id, id);
    }
    else
    {
        // No ref bindings: store value directly (not pointer)
        emit_auto_type(ctx, node->match_stmt.expr, node->token);
        EMIT(ctx, " _m_%d = ", id);
        codegen_expression(ctx, node->match_stmt.expr);
        EMIT(ctx, "; ");
    }

    if (is_expr)
    {
        EMIT(ctx, "%s _r_%d; ", ret_type, id);
    }

    char *expr_type = infer_type(ctx, node->match_stmt.expr);
    int is_option = IS_OPTION_TYPE(expr_type);
    int is_result = IS_RESULT_TYPE(expr_type);

    char *enum_name = NULL;
    ASTNode *chk = node->match_stmt.cases;
    int has_wildcard = 0;
    while (chk)
    {
        if (strcmp(chk->match_case.pattern, "_") == 0)
        {
            has_wildcard = 1;
        }
        else if (!enum_name)
        {
            EnumVariantReg *reg = find_enum_variant(ctx, chk->match_case.pattern);
            if (reg)
            {
                enum_name = reg->enum_name;
            }
        }
        chk = chk->next;
    }

    if (enum_name && !has_wildcard)
    {
        // Iterate through all registered variants for this enum
        EnumVariantReg *v = ctx->enum_variants;
        while (v)
        {
            if (v->enum_name && strcmp(v->enum_name, enum_name) == 0)
            {
                int covered = 0;
                ASTNode *c2 = node->match_stmt.cases;
                while (c2)
                {
                    char mangled_v[512];
                    snprintf(mangled_v, sizeof(mangled_v), "%s__%s", v->enum_name, v->variant_name);

                    if (strcmp(c2->match_case.pattern, v->variant_name) == 0 ||
                        strcmp(c2->match_case.pattern, mangled_v) == 0)
                    {
                        covered = 1;
                        break;
                    }
                    c2 = c2->next;
                }
                if (!covered)
                {
                    zwarn_at(node->token, "Non-exhaustive match: Missing variant '%s'",
                             v->variant_name);
                }
            }
            v = v->next;
        }
    }

    ASTNode *c = node->match_stmt.cases;
    int first = 1;
    while (c)
    {
        int is_wildcard = (strcmp(c->match_case.pattern, "_") == 0);
        int is_final_wildcard = (is_wildcard && c->next == NULL);

        if (!first)
        {
            EMIT(ctx, " else ");
        }

        emit_source_mapping(ctx, c); // Step through match cases elegantly

        if (!is_final_wildcard || first)
        {
            EMIT(ctx, "if (");
            if (is_wildcard)
            {
                EMIT(ctx, "1");
            }
            else if (is_option)
            {
                int m_is_ptr = has_ref_binding || (expr_type && strchr(expr_type, '*'));
                const char *acc = m_is_ptr ? "->" : ".";

                if (c->match_case.pattern && strcmp(c->match_case.pattern, "Some") == 0)
                {
                    EMIT(ctx, "_m_%d%sis_some", id, acc);
                }
                else if (c->match_case.pattern && strcmp(c->match_case.pattern, "None") == 0)
                {
                    EMIT(ctx, "!_m_%d%sis_some", id, acc);
                }
                else
                {
                    EMIT(ctx, "1");
                }
            }
            else if (is_result)
            {
                int m_is_ptr = has_ref_binding || (expr_type && strchr(expr_type, '*'));
                const char *acc = m_is_ptr ? "->" : ".";

                if (c->match_case.pattern && strcmp(c->match_case.pattern, "Ok") == 0)
                {
                    EMIT(ctx, "_m_%d%sis_ok", id, acc);
                }
                else if (c->match_case.pattern && strcmp(c->match_case.pattern, "Err") == 0)
                {
                    EMIT(ctx, "!_m_%d%sis_ok", id, acc);
                }
                else
                {
                    EMIT(ctx, "1");
                }
            }
            else
            {
                // Use helper for OR patterns, range patterns, and simple patterns
                if (c->match_case.pattern)
                {
                    emit_pattern_condition(ctx, c->match_case.pattern, id, has_ref_binding);
                }
                else
                {
                    EMIT(ctx, "1");
                }
            }
        }

        if (!is_final_wildcard || first)
        {
            EMIT(ctx, ") ");
        }
        EMIT(ctx, "{ ");
        if (c->match_case.binding_count > 0)
        {
            for (int i = 0; i < c->match_case.binding_count; i++)
            {
                char *bname = c->match_case.binding_names[i];
                if (!bname)
                {
                    continue;
                }
                int is_r = c->match_case.binding_refs ? c->match_case.binding_refs[i] : 0;

                if (is_option)
                {
                    if (is_r)
                    {
                        EMIT(ctx, "ZC_AUTO_INIT(%s, &_m_%d->val); ", bname, id);
                    }
                    else if (has_ref_binding)
                    {
                        EMIT(ctx, "ZC_AUTO_INIT(%s, _m_%d->val); ", bname, id);
                    }
                    else
                    {
                        EMIT(ctx, "ZC_AUTO_INIT(%s, _m_%d.val); ", bname, id);
                    }
                }
                else if (is_result)
                {
                    char *field = "val";
                    if (strcmp(c->match_case.pattern, "Err") == 0)
                    {
                        field = "err";
                    }

                    if (is_r)
                    {
                        EMIT(ctx, "ZC_AUTO_INIT(%s, &_m_%d->%s); ", bname, id, field);
                    }
                    else if (has_ref_binding)
                    {
                        EMIT(ctx, "ZC_AUTO_INIT(%s, _m_%d->%s); ", bname, id, field);
                    }
                    else
                    {
                        EMIT(ctx, "ZC_AUTO_INIT(%s, _m_%d.%s); ", bname, id, field);
                    }
                }
                else
                {
                    char *v = strstr(c->match_case.pattern, "::");
                    if (v)
                    {
                        v += 2;
                    }
                    else
                    {
                        v = strrchr(c->match_case.pattern, '_');
                        if (v)
                        {
                            v++;
                        }
                        else
                        {
                            v = (char *)c->match_case.pattern;
                        }
                    }

                    if (c->match_case.binding_count > 1)
                    {
                        // Tuple destructuring: data.Variant.vI
                        if (is_r)
                        {
                            EMIT(ctx, "ZC_AUTO_INIT(%s, &_m_%d->data.%s.v%d); ", bname, id, v, i);
                        }
                        else if (has_ref_binding)
                        {
                            EMIT(ctx, "ZC_AUTO_INIT(%s, _m_%d->data.%s.v%d); ", bname, id, v, i);
                        }
                        else
                        {
                            EMIT(ctx, "ZC_AUTO_INIT(%s, _m_%d.data.%s.v%d); ", bname, id, v, i);
                        }
                    }
                    else
                    {
                        // Single destructuring: data.Variant
                        if (is_r)
                        {
                            EMIT(ctx, "ZC_AUTO_INIT(%s, &_m_%d->data.%s); ", bname, id, v);
                        }
                        else if (has_ref_binding)
                        {
                            EMIT(ctx, "ZC_AUTO_INIT(%s, _m_%d->data.%s); ", bname, id, v);
                        }
                        else
                        {
                            EMIT(ctx, "ZC_AUTO_INIT(%s, _m_%d.data.%s); ", bname, id, v);
                        }
                    }
                }
            }
        }

        // Check if body is a string literal (should auto-print).
        ASTNode *body = c->match_case.body;
        int is_string_literal =
            (body->type == NODE_EXPR_LITERAL && body->literal.type_kind == LITERAL_STRING);

        if (is_expr)
        {
            EMIT(ctx, "_r_%d = ", id);
            if (is_string_literal)
            {
                codegen_node_single(ctx, body);
            }
            else
            {
                if (body->type == NODE_BLOCK)
                {
                    int saved = ctx->cg.defer_count;
                    EMIT(ctx, "({ ");
                    ASTNode *stmt = body->block.statements;
                    while (stmt)
                    {
                        emit_source_mapping(ctx, stmt);
                        codegen_node_single(ctx, stmt);
                        stmt = stmt->next;
                    }
                    for (int i = ctx->cg.defer_count - 1; i >= saved; i--)
                    {
                        emit_source_mapping_duplicate(ctx, ctx->cg.defer_stack[i]);
                        codegen_node_single(ctx, ctx->cg.defer_stack[i]);
                    }
                    ctx->cg.defer_count = saved;
                    EMIT(ctx, " })");
                }
                else
                {
                    codegen_node_single(ctx, body);
                }
            }
            EMIT(ctx, ";");
        }
        else
        {
            if (is_string_literal)
            {
                char *inner = body->literal.string_val;
                char *code =
                    process_printf_sugar(ctx, body->token, inner, 1, "stdout", NULL, NULL, 0, 0, 0);

                EMIT(ctx, "%s;", code);
                zfree(code);
            }
            else
            {
                codegen_node_single(ctx, body);
            }
        }

        EMIT(ctx, " }");
        first = 0;
        c = c->next;
    }

    if (is_expr)
    {
        if (ctx->config->misra_mode && !has_wildcard)
        {
            EMIT(ctx, " else { } /* MISRA 15.7 */ ");
        }
        EMIT(ctx, " _r_%d; })", id);
    }
    else
    {
        if (ctx->config->misra_mode && !has_wildcard)
        {
            EMIT(ctx, " else { } /* MISRA 15.7 */ ");
        }
        EMIT(ctx, " }");
    }
}
typedef void (*CodegenHandler)(ParserContext *ctx, ASTNode *node);

static void handle_node_ast_comment(ParserContext *ctx, ASTNode *node)
{
    (void)ctx;
    EMIT(ctx, "%s\n", node->comment.content);
}

static void handle_node_match(ParserContext *ctx, ASTNode *node)
{
    codegen_match_internal(ctx, node, 0);
    EMIT(ctx, ";\n");
}

static void handle_node_assert(ParserContext *ctx, ASTNode *node)
{
    EMIT(ctx, "__zenc_assert(");
    codegen_expression(ctx, node->assert_stmt.condition);
    if (node->assert_stmt.message)
    {
        if (node->assert_stmt.message_is_literal)
        {
            EMIT(ctx, ", %s", node->assert_stmt.message);
        }
        else
        {
            EMIT(ctx, ", \"%%s\", %s", node->assert_stmt.message);
        }
    }
    else
    {
        EMIT(ctx, ", \"Assertion failed\"");
    }
    EMIT(ctx, ");\n");
}

static void handle_node_expect(ParserContext *ctx, ASTNode *node)
{
    EMIT(ctx, "__zenc_expect(");
    codegen_expression(ctx, node->assert_stmt.condition);
    if (node->assert_stmt.message)
    {
        if (node->assert_stmt.message_is_literal)
        {
            EMIT(ctx, ", %s", node->assert_stmt.message);
        }
        else
        {
            EMIT(ctx, ", \"%%s\", %s", node->assert_stmt.message);
        }
    }
    else
    {
        EMIT(ctx, ", \"Expectation failed\"");
    }
    EMIT(ctx, ");\n");
}

static void handle_node_defer(ParserContext *ctx, ASTNode *node)
{
    (void)ctx;
    if (ctx->cg.defer_count < MAX_DEFER)
    {
        ctx->cg.defer_stack[ctx->cg.defer_count++] = node->defer_stmt.stmt;
    }
}

static void handle_node_comptime(ParserContext *ctx, ASTNode *node)
{
    if (node->comptime.generated)
    {
        codegen_walker(ctx, node->comptime.generated);
    }
}

static void handle_node_block(ParserContext *ctx, ASTNode *node)
{
    int saved = ctx->cg.defer_count;
    EMIT(ctx, "{\n");
    emitter_indent(&ctx->cg.emitter);
    codegen_walker(ctx, node->block.statements);
    for (int i = ctx->cg.defer_count - 1; i >= saved; i--)
    {
        emit_source_mapping_duplicate(ctx, ctx->cg.defer_stack[i]);
        codegen_node_single(ctx, ctx->cg.defer_stack[i]);
    }
    ctx->cg.defer_count = saved;
    emitter_dedent(&ctx->cg.emitter);
    EMIT(ctx, "}\n");
}

static void handle_node_impl(ParserContext *ctx, ASTNode *node)
{
    char *sname = node->impl.struct_name;
    TypeAlias *ta = find_type_alias_node(ctx, sname);
    const char *resolved = (ta && !ta->is_opaque) ? ta->original_type : NULL;

    if (resolved)
    {
        int slen = strlen(sname);
        ASTNode *m = node->impl.methods;
        while (m)
        {
            if (m->type == NODE_FUNCTION && m->func.name &&
                strncmp(m->func.name, sname, slen) == 0 && m->func.name[slen] == '_' &&
                m->func.name[slen + 1] == '_')
            {
                const char *method_part = m->func.name + slen;
                size_t new_name_sz = strlen(resolved) + strlen(method_part) + 1;
                char *new_name = xmalloc(new_name_sz);
                snprintf(new_name, new_name_sz, "%s%s", resolved, method_part);
                zfree(m->func.name);
                m->func.name = new_name;
            }
            m = m->next;
        }
        ctx->cg.current_impl_type = (char *)resolved;
    }
    else
    {
        ctx->cg.current_impl_type = sname;
    }

    codegen_walker(ctx, node->impl.methods);
    ctx->cg.current_impl_type = NULL;
}

static void handle_node_function(ParserContext *ctx, ASTNode *node)
{
    if (!node->func.body || node->func.generic_params)
    {
        return;
    }
    if (node->cfg_condition)
    {
        EMIT(ctx, "#if %s\n", node->cfg_condition);
    }

    emit_source_mapping(ctx, node);

    if (node->func.is_async)
    {
        const char *final_name = (node->link_name) ? node->link_name : node->func.name;
        int has_ret = node->func.ret_type && strcmp(node->func.ret_type, "void") != 0;

        // Parse args
        char *args_copy = xstrdup(node->func.args);
        char *token = strtok(args_copy, ",");
        int arg_count = 0;
        char **arg_names = xmalloc(32 * sizeof(char *));
        char **arg_types = xmalloc(32 * sizeof(char *));
        while (token)
        {
            while (*token == ' ')
            {
                token++;
            }
            char *last_space = strrchr(token, ' ');
            if (last_space)
            {
                *last_space = 0;
                arg_types[arg_count] = xstrdup(token);
                arg_names[arg_count] = xstrdup(last_space + 1);
                arg_count++;
            }
            token = strtok(NULL, ",");
        }
        zfree(args_copy);

        // 1. Init function (struct definition emitted in protos)
        EMIT(ctx, "void %s_init(struct %s_Future *f", final_name, final_name);
        for (int i = 0; i < arg_count; i++)
        {
            EMIT(ctx, ", %s %s", arg_types[i], arg_names[i]);
        }
        EMIT(ctx, ")\n{\n");
        emitter_indent(&ctx->cg.emitter);
        EMIT(ctx, "f->_state = 0;\n");
        for (int i = 0; i < arg_count; i++)
        {
            EMIT(ctx, "f->%s = %s;\n", arg_names[i], arg_names[i]);
        }
        emitter_dedent(&ctx->cg.emitter);
        EMIT(ctx, "}\n");

        // 3. Emit the actual function body as _impl_%s (regular C function with normal returns)
        EMIT(ctx, "%s _impl_%s(", has_ret ? node->func.ret_type : "void", final_name);
        for (int i = 0; i < arg_count; i++)
        {
            EMIT(ctx, "%s%s %s", i > 0 ? ", " : "", arg_types[i], arg_names[i]);
        }
        EMIT(ctx, ")\n{\n");
        emitter_indent(&ctx->cg.emitter);
        ctx->cg.defer_count = 0;

        // Set up drop flags for parameters with destructors (e.g. String, Vec)
        for (int ai = 0; ai < arg_count && ai < node->func.arg_count; ai++)
        {
            Type *arg_type = node->func.arg_types[ai];
            if (!arg_type)
            {
                continue;
            }
            int has_drop = 0;
            char *drop_type_name = NULL;
            if (arg_type->kind == TYPE_STRUCT && arg_type->name)
            {
                ASTNode *def = find_struct_def(ctx, arg_type->name);
                if (def && def->type == NODE_STRUCT && def->type_info &&
                    def->type_info->traits.has_drop)
                {
                    has_drop = 1;
                    drop_type_name = arg_type->name;
                }
            }
            if (has_drop && ai < 32 && arg_names[ai])
            {
                EMIT(ctx, "int __z_drop_flag_%s = 1;\n", arg_names[ai]);
                ASTNode *defer_node = xmalloc(sizeof(ASTNode));
                defer_node->token = node->token;
                defer_node->type = NODE_RAW_STMT;
                size_t stmt_sz = 256 + strlen(arg_names[ai]) * 2 + strlen(drop_type_name);
                char *stmt_str = xmalloc(stmt_sz);
                if (strcmp(arg_names[ai], "self") == 0)
                {
                    snprintf(stmt_str, stmt_sz, "if (__z_drop_flag_%s) %s__Drop__glue(%s);",
                             arg_names[ai], drop_type_name, arg_names[ai]);
                }
                else
                {
                    snprintf(stmt_str, stmt_sz, "if (__z_drop_flag_%s) %s__Drop__glue(&%s);",
                             arg_names[ai], drop_type_name, arg_names[ai]);
                }
                defer_node->raw_stmt.content = stmt_str;
                defer_node->line = node->line;
                if (ctx->cg.defer_count < MAX_DEFER)
                {
                    ctx->cg.defer_stack[ctx->cg.defer_count++] = defer_node;
                }
            }
        }

        char *prev_ret = ctx->cg.current_func_ret_type;
        Type *prev_ret_info = ctx->cg.current_func_ret_type_info;
        ctx->cg.current_func_ret_type = node->func.ret_type;
        ctx->cg.current_func_ret_type_info = node->func.ret_type_info;

        codegen_walker(ctx, node->func.body);

        ctx->cg.current_func_ret_type = prev_ret;
        ctx->cg.current_func_ret_type_info = prev_ret_info;
        for (int i = ctx->cg.defer_count - 1; i >= 0; i--)
        {
            emit_source_mapping_duplicate(ctx, ctx->cg.defer_stack[i]);
            codegen_node_single(ctx, ctx->cg.defer_stack[i]);
        }
        emitter_dedent(&ctx->cg.emitter);
        EMIT(ctx, "}\n");

        // 4. Poll function — calls _impl_, stores result, zeroes moved params
        EMIT(ctx, "int %s_poll(void *ctx)\n", final_name);
        EMIT(ctx, "{\n");
        emitter_indent(&ctx->cg.emitter);
        EMIT(ctx, "struct %s_Future *f = ctx;\n", final_name);
        EMIT(ctx, "if (f->_state > 0) return 1;\n");
        EMIT(ctx, "f->_state = 1;\n");
        if (has_ret)
        {
            EMIT(ctx, "f->_result = _impl_%s(", final_name);
            for (int i = 0; i < arg_count; i++)
            {
                EMIT(ctx, "%sf->%s", i > 0 ? ", " : "", arg_names[i]);
            }
            EMIT(ctx, ");\n");
        }
        else
        {
            EMIT(ctx, "_impl_%s(", final_name);
            for (int i = 0; i < arg_count; i++)
            {
                EMIT(ctx, "%sf->%s", i > 0 ? ", " : "", arg_names[i]);
            }
            EMIT(ctx, ");\n");
            // Also drop the params that were passed by value
        }
        // Zero out future fields for types with destructors (ownership moved to _impl_)
        for (int ai = 0; ai < arg_count && ai < node->func.arg_count; ai++)
        {
            Type *arg_type = node->func.arg_types[ai];
            if (!arg_type)
            {
                continue;
            }
            if (arg_type->kind == TYPE_STRUCT && arg_type->name)
            {
                ASTNode *def = find_struct_def(ctx, arg_type->name);
                if (def && def->type == NODE_STRUCT && def->type_info &&
                    def->type_info->traits.has_drop && ai < 32 && arg_names[ai])
                {
                    EMIT(ctx, "memset(&f->%s, 0, sizeof(f->%s));\n", arg_names[ai], arg_names[ai]);
                }
            }
        }
        EMIT(ctx, "return 1;\n");
        emitter_dedent(&ctx->cg.emitter);
        EMIT(ctx, "}\n");

        // 5. Get function
        if (has_ret)
        {
            EMIT(ctx, "%s %s_get(struct %s_Future *f) { return f->_result; }\n",
                 node->func.ret_type, final_name, final_name);
        }

        for (int i = 0; i < arg_count; i++)
        {
            zfree(arg_names[i]);
            zfree(arg_types[i]);
        }
        zfree(arg_names);
        zfree(arg_types);

        if (node->cfg_condition)
        {
            EMIT(ctx, "#endif\n");
        }
        return;
    }

    ctx->cg.defer_count = 0;
    EMIT(ctx, "\n");

    // Emit GCC attributes before function
    {
        int has_attrs = node->func.constructor || node->func.destructor || node->func.noinline ||
                        node->func.unused || node->func.weak || node->func.cold || node->func.hot ||
                        node->func.noreturn || node->func.pure || node->func.section ||
                        node->func.is_export;
        if (has_attrs)
        {
            EMIT(ctx, "__attribute__((");
            int first = 1;
#define EMIT_ATTR(cond, name)                                                                      \
    if (cond)                                                                                      \
    {                                                                                              \
        if (!first)                                                                                \
            EMIT(ctx, ", ");                                                                       \
        EMIT(ctx, name);                                                                           \
        first = 0;                                                                                 \
    }
            EMIT_ATTR(node->func.constructor, "constructor");
            EMIT_ATTR(node->func.destructor, "destructor");
            EMIT_ATTR(node->func.noinline, "noinline");
            EMIT_ATTR(node->func.unused, "unused");
            EMIT_ATTR(node->func.weak, "weak");
            EMIT_ATTR(node->func.cold, "cold");
            EMIT_ATTR(node->func.hot, "hot");
            EMIT_ATTR(node->func.noreturn, "noreturn");
            EMIT_ATTR(node->func.pure, "pure");
            EMIT_ATTR(node->func.is_export, "visibility(\"default\")");
            if (node->func.section)
            {
                if (!first)
                {
                    EMIT(ctx, ", ");
                }
                EMIT(ctx, "section(\"%s\")", node->func.section);
            }

            Attribute *custom = node->func.attributes;
            while (custom)
            {
                if (!first)
                {
                    EMIT(ctx, ", ");
                }
                EMIT(ctx, "%s", custom->name);
                if (custom->arg_count > 0)
                {
                    EMIT(ctx, "(");
                    for (int i = 0; i < custom->arg_count; i++)
                    {
                        if (i > 0)
                        {
                            EMIT(ctx, ", ");
                        }
                        EMIT(ctx, "%s", custom->args[i]);
                    }
                    EMIT(ctx, ")");
                }
                first = 0;
                custom = custom->next;
            }

#undef EMIT_ATTR
            EMIT(ctx, ")) ");
        }
        else if (node->func.attributes)
        {
            // Handle case where specific attributes are missing but custom ones exist
            EMIT(ctx, "__attribute__((");
            int first = 1;
            Attribute *custom = node->func.attributes;
            while (custom)
            {
                if (!first)
                {
                    EMIT(ctx, ", ");
                }
                EMIT(ctx, "%s", custom->name);
                if (custom->arg_count > 0)
                {
                    EMIT(ctx, "(");
                    for (int i = 0; i < custom->arg_count; i++)
                    {
                        if (i > 0)
                        {
                            EMIT(ctx, ", ");
                        }
                        EMIT(ctx, "%s", custom->args[i]);
                    }
                    EMIT(ctx, ")");
                }
                first = 0;
                custom = custom->next;
            }
            EMIT(ctx, ")) ");
        }
    }

    if (node->func.is_inline)
    {
        EMIT(ctx, "inline ");
    }
    emit_func_signature(ctx, node, NULL);
    EMIT(ctx, "\n{\n");
    emitter_indent(&ctx->cg.emitter);
    if (ctx->config->misra_mode && node->func.ret_type && strcmp(node->func.ret_type, "void") != 0)
    {
        char *safe_ret_type = type_to_c_string(node->func.ret_type_info);
        EMIT(ctx, "%s _misra_ret = 0;\n", safe_ret_type);
        zfree(safe_ret_type);
    }
    char *prev_ret = ctx->cg.current_func_ret_type;
    Type *prev_ret_info = ctx->cg.current_func_ret_type_info;
    ctx->cg.current_func_ret_type = node->func.ret_type;
    ctx->cg.current_func_ret_type_info = node->func.ret_type_info;

    // Set self_is_pointer flag for codegen of the body
    int prev_self_is_ptr = ctx->self_is_pointer;
    ctx->self_is_pointer = 0;
    if (node->func.arg_count > 0 && node->func.param_names && node->func.param_names[0] &&
        strcmp(node->func.param_names[0], "self") == 0)
    {
        ctx->self_is_pointer = 1;
    }

    // Initialize drop flags for arguments that implement Drop
    for (int i = 0; i < node->func.arg_count; i++)
    {
        Type *arg_type = node->func.arg_types[i];
        char *arg_name = node->func.param_names ? node->func.param_names[i] : NULL;
        if (arg_type && arg_name)
        {
            // Check if type implements Drop
            int has_drop = 0;
            char *drop_type_name = NULL;

            if (arg_type->kind == TYPE_FUNCTION ||
                (arg_type->kind == TYPE_STRUCT && arg_type->name))
            {
                if (arg_type->kind == TYPE_FUNCTION)
                {
                    if (arg_type->traits.has_drop)
                    {
                        has_drop = 1;
                    }
                }
                else
                {
                    ASTNode *def = find_struct_def(ctx, arg_type->name);
                    if (def && def->type == NODE_STRUCT && def->type_info &&
                        def->type_info->traits.has_drop)
                    {
                        has_drop = 1;
                        drop_type_name = arg_type->name;
                    }
                }
            }

            if (has_drop)
            {
                emit_source_mapping_duplicate(ctx, node);
                EMIT(ctx, "int __z_drop_flag_%s = 1;\n", arg_name);

                ASTNode *defer_node = xmalloc(sizeof(ASTNode));
                defer_node->token = node->token;
                defer_node->type = NODE_RAW_STMT;
                char *stmt_str = NULL;
                if (arg_type->kind == TYPE_FUNCTION)
                {
                    size_t stmt_sz = 256 + strlen(arg_name) * 4;
                    stmt_str = xmalloc(stmt_sz);
                    snprintf(stmt_str, stmt_sz, "if (__z_drop_flag_%s && %s.drop) %s.drop(%s.ctx);",
                             arg_name, arg_name, arg_name, arg_name);
                }
                else
                {
                    size_t stmt_sz = 256 + strlen(arg_name) * 2 + strlen(drop_type_name);
                    stmt_str = xmalloc(stmt_sz);
                    // If it's self, it's already a pointer in C
                    if (strcmp(arg_name, "self") == 0)
                    {
                        snprintf(stmt_str, stmt_sz, "if (__z_drop_flag_%s) %s__Drop__glue(%s);",
                                 arg_name, drop_type_name, arg_name);
                    }
                    else
                    {
                        snprintf(stmt_str, stmt_sz, "if (__z_drop_flag_%s) %s__Drop__glue(&%s);",
                                 arg_name, drop_type_name, arg_name);
                    }
                }
                defer_node->raw_stmt.content = stmt_str;

                if (ctx->cg.defer_count < MAX_DEFER)
                {
                    ctx->cg.defer_stack[ctx->cg.defer_count++] = defer_node;
                }
            }
        }
    }

    codegen_walker(ctx, node->func.body);
    for (int i = ctx->cg.defer_count - 1; i >= 0; i--)
    {
        emit_source_mapping_duplicate(ctx, ctx->cg.defer_stack[i]);
        codegen_node_single(ctx, ctx->cg.defer_stack[i]);
    }
    ctx->cg.current_func_ret_type = prev_ret;
    ctx->cg.current_func_ret_type_info = prev_ret_info;
    ctx->self_is_pointer = prev_self_is_ptr;

    if (ctx->config->misra_mode)
    {
        EMIT(ctx, "goto _misra_end_of_func;\n");
        EMIT(ctx, "_misra_end_of_func:\n");
        if (node->func.ret_type && strcmp(node->func.ret_type, "void") != 0)
        {
            EMIT(ctx, "return _misra_ret;\n");
        }
        else
        {
            EMIT(ctx, "return;\n");
        }
    }
    emitter_dedent(&ctx->cg.emitter);
    EMIT(ctx, "}\n");
    if (node->cfg_condition)
    {
        EMIT(ctx, "#endif\n");
    }
}

static void handle_node_impl_trait(ParserContext *ctx, ASTNode *node)
{
    char *sname = node->impl_trait.target_type;
    TypeAlias *ta = find_type_alias_node(ctx, sname);
    const char *resolved = (ta && !ta->is_opaque) ? ta->original_type : NULL;

    if (resolved)
    {
        int slen = strlen(sname);
        ASTNode *m = node->impl_trait.methods;
        while (m)
        {
            if (m->type == NODE_FUNCTION && m->func.name &&
                strncmp(m->func.name, sname, slen) == 0 && m->func.name[slen] == '_' &&
                m->func.name[slen + 1] == '_')
            {
                const char *method_part = m->func.name + slen;
                size_t new_name_sz = strlen(resolved) + strlen(method_part) + 1;
                char *new_name = xmalloc(new_name_sz);
                snprintf(new_name, new_name_sz, "%s%s", resolved, method_part);
                zfree(m->func.name);
                m->func.name = new_name;
            }
            m = m->next;
        }
        ctx->cg.current_impl_type = (char *)resolved;
    }
    else
    {
        ctx->cg.current_impl_type = sname;
    }

    codegen_walker(ctx, node->impl_trait.methods);
    ctx->cg.current_impl_type = NULL;
}

static void handle_node_destruct_var(ParserContext *ctx, ASTNode *node)
{
    int id = ctx->cg.tmp_counter++;
    emit_auto_type(ctx, node->destruct.init_expr, node->token);
    EMIT(ctx, " _tmp_%d = ", id);
    codegen_expression(ctx, node->destruct.init_expr);
    EMIT(ctx, ";\n");

    if (node->destruct.is_guard)
    {
        char *variant = node->destruct.guard_variant;
        char *check = "val";

        emit_source_mapping_duplicate(ctx, node);
        if (strcmp(variant, "Some") == 0)
        {
            EMIT(ctx, "if (!_tmp_%d.is_some) {\n", id);
            emitter_indent(&ctx->cg.emitter);
        }
        else if (strcmp(variant, "Ok") == 0)
        {
            EMIT(ctx, "if (!_tmp_%d.is_ok) {\n", id);
            emitter_indent(&ctx->cg.emitter);
        }
        else if (strcmp(variant, "Err") == 0)
        {
            EMIT(ctx, "if (_tmp_%d.is_ok) {\n", id);
            emitter_indent(&ctx->cg.emitter);
            check = "err";
        }
        else
        {
            EMIT(ctx, "if (!_tmp_%d.is_%s) {\n", id, variant);
            emitter_indent(&ctx->cg.emitter);
        }

        codegen_walker(ctx, node->destruct.else_block->block.statements);
        emitter_dedent(&ctx->cg.emitter);
        EMIT(ctx, "}\n");

        emit_source_mapping_duplicate(ctx, node);
        if (z_path_match_compiler(ctx->config->cc, "tcc"))
        {
            EMIT(ctx, "__typeof__(_tmp_%d.%s) %s = _tmp_%d.%s;\n", id, check,
                 node->destruct.names[0], id, check);
        }
        else
        {
            EMIT(ctx, "ZC_AUTO %s = _tmp_%d.%s;\n", node->destruct.names[0], id, check);
        }
    }
    else
    {
        for (int i = 0; i < node->destruct.count; i++)
        {
            emit_source_mapping_duplicate(ctx, node);
            if (node->destruct.is_struct_destruct)
            {
                char *field = node->destruct.field_names ? node->destruct.field_names[i]
                                                         : node->destruct.names[i];
                if (z_path_match_compiler(ctx->config->cc, "tcc"))
                {
                    EMIT(ctx, "__typeof__(_tmp_%d.%s) %s = _tmp_%d.%s;\n", id, field,
                         node->destruct.names[i], id, field);
                }
                else
                {
                    EMIT(ctx, "ZC_AUTO %s = _tmp_%d.%s;\n", node->destruct.names[i], id, field);
                }
            }
            else
            {
                char *explicit_type = node->destruct.types ? node->destruct.types[i] : NULL;
                if (explicit_type)
                {
                    EMIT(ctx, "%s %s = _tmp_%d.v%d;\n", explicit_type, node->destruct.names[i], id,
                         i);
                }
                else if (z_path_match_compiler(ctx->config->cc, "tcc"))
                {
                    EMIT(ctx, "__typeof__(_tmp_%d.v%d) %s = _tmp_%d.v%d;\n", id, i,
                         node->destruct.names[i], id, i);
                }
                else
                {
                    EMIT(ctx, "ZC_AUTO %s = _tmp_%d.v%d;\n", node->destruct.names[i], id, i);
                }
            }
        }
    }
}

static void handle_node_var_decl(ParserContext *ctx, ASTNode *node)
{
    int saved_closure_frees = ctx->cg.pending_closure_free_count;

    if (strcmp(node->var_decl.name, "_") == 0 && node->var_decl.init_expr)
    {
        int is_void = 0;
        if (node->type_info && node->type_info->kind == TYPE_VOID)
        {
            is_void = 1;
        }
        else if (node->var_decl.type_str && strcmp(node->var_decl.type_str, "void") == 0)
        {
            is_void = 1;
        }
        else if (!node->type_info && !node->var_decl.type_str)
        {
            char *ret_type = infer_type(ctx, node->var_decl.init_expr);
            if (!ret_type || strcmp(ret_type, "void") == 0)
            {
                is_void = 1;
            }
        }
        if (is_void)
        {
            codegen_expression(ctx, node->var_decl.init_expr);
            EMIT(ctx, ";\n");
            return;
        }
    }
    if (node->var_decl.is_thread_local)
    {
        EMIT(ctx, "_Thread_local ");
    }
    if (node->var_decl.is_static)
    {
        EMIT(ctx, "static ");
    }
    if (node->var_decl.is_autofree)
    {
        EMIT(ctx, "__attribute__((cleanup(_z_autofree_impl))) ");
    }
    {
        char *tname = NULL;
        if (node->type_info &&
            (!node->var_decl.init_expr || node->var_decl.init_expr->type != NODE_AWAIT))
        {
            tname = type_to_c_string(node->type_info);
            // Async functions now return Async*; correct the type name
            // so the emitted C uses "Async*" instead of "Async" or "Async<...>".
            if (tname && strncmp(tname, "Async", 5) == 0 && node->var_decl.init_expr &&
                node->var_decl.init_expr->resolved_type &&
                strncmp(node->var_decl.init_expr->resolved_type, "Async", 5) == 0)
            {
                zfree(tname);
                tname = xstrdup("Async*");
            }
        }
        else if (node->var_decl.type_str && strcmp(node->var_decl.type_str, "__auto_type") != 0)
        {
            tname = node->var_decl.type_str;
        }

        if (tname && strcmp(tname, "void*") != 0 && strcmp(tname, "unknown") != 0)
        {
            char *clean_type = tname;
            if (strncmp(clean_type, "struct ", 7) == 0)
            {
                clean_type += 7;
            }

            ASTNode *def = find_struct_def(ctx, clean_type);
            int has_drop = (def && def->type_info && def->type_info->traits.has_drop);

            if (has_drop)
            {
                EMIT(ctx, "int __z_drop_flag_%s = 1; ", node->var_decl.name);

                ASTNode *defer_node = xmalloc(sizeof(ASTNode));
                defer_node->type = NODE_RAW_STMT;
                defer_node->token = node->token;
                size_t stmt_sz = 256 + strlen(node->var_decl.name) * 2 + strlen(clean_type);
                char *stmt_str = xmalloc(stmt_sz);
                snprintf(stmt_str, stmt_sz, "if (__z_drop_flag_%s) %s__Drop__glue(&%s);",
                         node->var_decl.name, clean_type, node->var_decl.name);
                defer_node->raw_stmt.content = stmt_str;
                defer_node->line = node->line;

                if (ctx->cg.defer_count < MAX_DEFER)
                {
                    ctx->cg.defer_stack[ctx->cg.defer_count++] = defer_node;
                }
            }

            emit_var_decl_type(ctx, tname, node->var_decl.name);
            add_symbol(ctx, node->var_decl.name, tname, node->type_info, 0);

            if (node->var_decl.init_expr)
            {
                EMIT(ctx, " = ");
                if (ctx->config->use_cpp && node->type_info &&
                    (node->type_info->kind == TYPE_POINTER || node->type_info->kind == TYPE_ENUM))
                {
                    char *ct = type_to_c_string(node->type_info);
                    EMIT(ctx, "(%s)(", ct);
                    codegen_expression(ctx, node->var_decl.init_expr);
                    EMIT(ctx, ")");
                    zfree(ct);
                }
                else
                {
                    codegen_expression(ctx, node->var_decl.init_expr);
                }
            }
            else if (node->type_info)
            {
                TypeKind k = node->type_info->kind;
                if (k == TYPE_ARRAY || k == TYPE_STRUCT)
                {
                    EMIT(ctx, " = %s", ctx->config->use_cpp ? "{}" : "{0}");
                }
                else if (is_int_type(k))
                {
                    EMIT(ctx, " = 0");
                }
                else if (k == TYPE_F32 || k == TYPE_FLOAT)
                {
                    EMIT(ctx, " = 0.0f");
                }
                else if (k == TYPE_F64)
                {
                    EMIT(ctx, " = 0.0");
                }
                else if (k == TYPE_BOOL)
                {
                    EMIT(ctx, " = false");
                }
            }
            EMIT(ctx, ";\n");
            if (node->var_decl.init_expr && emit_move_invalidation(ctx, node->var_decl.init_expr))
            {
                EMIT(ctx, ";\n");
            }

            if (node->type_info)
            {
                zfree(tname);
            }
        }
        else
        {
            char *inferred = NULL;
            if (node->var_decl.init_expr)
            {
                inferred = infer_type(ctx, node->var_decl.init_expr);
            }

            if (inferred && strcmp(inferred, "__auto_type") != 0)
            {
                char *clean_type = inferred;
                if (strncmp(clean_type, "struct ", 7) == 0)
                {
                    clean_type += 7;
                }

                ASTNode *def = find_struct_def(ctx, clean_type);
                int has_drop = (def && def->type_info && def->type_info->traits.has_drop);

                if (has_drop)
                {
                    EMIT(ctx, "int __z_drop_flag_%s = 1; ", node->var_decl.name);

                    ASTNode *defer_node = xmalloc(sizeof(ASTNode));
                    defer_node->type = NODE_RAW_STMT;
                    defer_node->token = node->token;
                    char *stmt_str = NULL;
                    if (node->var_decl.init_expr && node->var_decl.init_expr->type_info &&
                        node->var_decl.init_expr->type_info->kind == TYPE_FUNCTION)
                    {
                        size_t stmt_sz = 256 + strlen(node->var_decl.name) * 4;
                        stmt_str = xmalloc(stmt_sz);
                        snprintf(stmt_str, stmt_sz,
                                 "if (__z_drop_flag_%s && %s.drop) %s.drop(%s.ctx);",
                                 node->var_decl.name, node->var_decl.name, node->var_decl.name,
                                 node->var_decl.name);
                    }
                    else
                    {
                        size_t stmt_sz = 256 + strlen(node->var_decl.name) * 2 + strlen(clean_type);
                        stmt_str = xmalloc(stmt_sz);
                        snprintf(stmt_str, stmt_sz, "if (__z_drop_flag_%s) %s__Drop__glue(&%s);",
                                 node->var_decl.name, clean_type, node->var_decl.name);
                    }
                    defer_node->raw_stmt.content = stmt_str;
                    defer_node->line = node->line;

                    if (ctx->cg.defer_count < MAX_DEFER)
                    {
                        ctx->cg.defer_stack[ctx->cg.defer_count++] = defer_node;
                    }
                }

                emit_var_decl_type(ctx, inferred, node->var_decl.name);
                add_symbol(ctx, node->var_decl.name, inferred, NULL, 0);
                EMIT(ctx, " = ");
                if (ctx->config->use_cpp && inferred &&
                    (strchr(inferred, '*') || is_enum_type_name(ctx, inferred)))
                {
                    EMIT(ctx, "(%s)(", inferred);
                    codegen_expression(ctx, node->var_decl.init_expr);
                    EMIT(ctx, ")");
                }
                else
                {
                    codegen_expression(ctx, node->var_decl.init_expr);
                }
                EMIT(ctx, ";\n");

                if (node->var_decl.init_expr &&
                    emit_move_invalidation(ctx, node->var_decl.init_expr))
                {
                    EMIT(ctx, ";\n");
                }
            }
            else
            {
                emit_auto_type(ctx, node->var_decl.init_expr, node->token);
                EMIT(ctx, " %s", node->var_decl.name);

                if (inferred)
                {
                    add_symbol(ctx, node->var_decl.name, inferred, NULL, 0);
                }

                EMIT(ctx, " = ");
                codegen_expression(ctx, node->var_decl.init_expr);
                EMIT(ctx, ";\n");
                if (node->var_decl.init_expr &&
                    emit_move_invalidation(ctx, node->var_decl.init_expr))
                {
                    EMIT(ctx, ";\n");
                }
            }
        }
    }

    ctx->cg.pending_closure_free_count = saved_closure_frees;
}

static void handle_node_const(ParserContext *ctx, ASTNode *node)
{
    EMIT(ctx, "const ");
    if (node->var_decl.type_str)
    {
        EMIT(ctx, "%s %s", node->var_decl.type_str, node->var_decl.name);
    }
    else
    {
        emit_auto_type(ctx, node->var_decl.init_expr, node->token);
        EMIT(ctx, " %s", node->var_decl.name);
    }
    EMIT(ctx, " = ");
    codegen_expression(ctx, node->var_decl.init_expr);
    EMIT(ctx, ";\n");
}

static void handle_node_field(ParserContext *ctx, ASTNode *node)
{
    if (node->field.bit_width > 0)
    {
        EMIT(ctx, "%s %s : %d;\n", node->field.type, node->field.name, node->field.bit_width);
    }
    else
    {
        emit_var_decl_type(ctx, node->field.type, node->field.name);
        EMIT(ctx, ";\n");
    }
}

static void handle_node_if(ParserContext *ctx, ASTNode *node)
{
    EMIT(ctx, "if (");
    codegen_expression(ctx, node->if_stmt.condition);
    EMIT(ctx, ") ");
    if (ctx->config->misra_mode && node->if_stmt.then_body->type != NODE_BLOCK)
    {
        EMIT(ctx, "{\n");
        emitter_indent(&ctx->cg.emitter);
    }
    codegen_node_single(ctx, node->if_stmt.then_body);
    if (ctx->config->misra_mode && node->if_stmt.then_body->type != NODE_BLOCK)
    {
        emitter_dedent(&ctx->cg.emitter);
        EMIT(ctx, "\n}");
    }
    if (node->if_stmt.else_body)
    {
        emit_source_mapping(ctx, node->if_stmt.else_body);
        EMIT(ctx, " else ");
        if (ctx->config->misra_mode && node->if_stmt.else_body->type != NODE_BLOCK)
        {
            EMIT(ctx, "{\n");
            emitter_indent(&ctx->cg.emitter);
        }
        codegen_node_single(ctx, node->if_stmt.else_body);
        if (ctx->config->misra_mode && node->if_stmt.else_body->type != NODE_BLOCK)
        {
            emitter_dedent(&ctx->cg.emitter);
            EMIT(ctx, "\n}");
        }
    }
    else if (ctx->config->misra_mode)
    {
        EMIT(ctx, " else { } /* MISRA */ ");
    }
}

static void handle_node_unless(ParserContext *ctx, ASTNode *node)
{
    EMIT(ctx, "if (!(");
    codegen_expression(ctx, node->unless_stmt.condition);
    EMIT(ctx, ")) ");
    if (ctx->config->misra_mode && node->unless_stmt.body->type != NODE_BLOCK)
    {
        EMIT(ctx, "{\n");
        emitter_indent(&ctx->cg.emitter);
    }
    codegen_node_single(ctx, node->unless_stmt.body);
    if (ctx->config->misra_mode && node->unless_stmt.body->type != NODE_BLOCK)
    {
        emitter_dedent(&ctx->cg.emitter);
        EMIT(ctx, "\n}");
    }
}

static void handle_node_guard(ParserContext *ctx, ASTNode *node)
{
    EMIT(ctx, "if (!(");
    codegen_expression(ctx, node->guard_stmt.condition);
    EMIT(ctx, ")) ");
    if (ctx->config->misra_mode && node->guard_stmt.body->type != NODE_BLOCK)
    {
        EMIT(ctx, "{\n");
        emitter_indent(&ctx->cg.emitter);
    }
    codegen_node_single(ctx, node->guard_stmt.body);
    if (ctx->config->misra_mode && node->guard_stmt.body->type != NODE_BLOCK)
    {
        emitter_dedent(&ctx->cg.emitter);
        EMIT(ctx, "\n}");
    }
}

static void handle_node_while(ParserContext *ctx, ASTNode *node)
{
    if (node->while_stmt.loop_label)
    {
        EMIT(ctx, "%s:;\n", node->while_stmt.loop_label);
    }
    if (ctx->cg.loop_depth < 64)
    {
        ctx->cg.loop_defer_boundary[ctx->cg.loop_depth] = ctx->cg.defer_count;
    }
    ctx->cg.loop_depth++;
    EMIT(ctx, "while (");
    codegen_expression(ctx, node->while_stmt.condition);
    EMIT(ctx, ") ");
    if (node->while_stmt.loop_label)
    {
        EMIT(ctx, "{\n");
        emitter_indent(&ctx->cg.emitter);
        codegen_node_single(ctx, node->while_stmt.body);
        emitter_dedent(&ctx->cg.emitter);
        EMIT(ctx, "__continue_%s:;\n", node->while_stmt.loop_label);
        EMIT(ctx, "}\n");
    }
    else
    {
        if (ctx->config->misra_mode && node->while_stmt.body->type != NODE_BLOCK)
        {
            EMIT(ctx, "{\n");
            emitter_indent(&ctx->cg.emitter);
        }
        codegen_node_single(ctx, node->while_stmt.body);
        if (ctx->config->misra_mode && node->while_stmt.body->type != NODE_BLOCK)
        {
            emitter_dedent(&ctx->cg.emitter);
            EMIT(ctx, "\n}");
        }
    }
    ctx->cg.loop_depth--;
    if (node->while_stmt.loop_label)
    {
        EMIT(ctx, "__break_%s:;\n", node->while_stmt.loop_label);
    }
}

void codegen_node_single(ParserContext *ctx, ASTNode *node)
{
    if (!node)
    {
        return;
    }

    static const CodegenHandler handlers[256] = {
        [NODE_AST_COMMENT] = handle_node_ast_comment,
        [NODE_MATCH] = handle_node_match,
        [NODE_ASSERT] = handle_node_assert,
        [NODE_EXPECT] = handle_node_expect,
        [NODE_DEFER] = handle_node_defer,
        [NODE_BLOCK] = handle_node_block,
        [NODE_IMPL] = handle_node_impl,
        [NODE_FUNCTION] = handle_node_function,
        [NODE_IMPL_TRAIT] = handle_node_impl_trait,
        [NODE_DESTRUCT_VAR] = handle_node_destruct_var,
        [NODE_VAR_DECL] = handle_node_var_decl,
        [NODE_CONST] = handle_node_const,
        [NODE_FIELD] = handle_node_field,
        [NODE_IF] = handle_node_if,
        [NODE_UNLESS] = handle_node_unless,
        [NODE_GUARD] = handle_node_guard,
        [NODE_WHILE] = handle_node_while,
        [NODE_COMPTIME] = handle_node_comptime,
    };

    if (node->type >= 0 && node->type < 256 && handlers[node->type])
    {
        handlers[node->type](ctx, node);
        return;
    }

    switch (node->type)
    {
    case NODE_FOR:
    {
        if (node->for_stmt.loop_label)
        {
            EMIT(ctx, "%s:;\n", node->for_stmt.loop_label);
        }
        if (ctx->cg.loop_depth < 64)
        {
            ctx->cg.loop_defer_boundary[ctx->cg.loop_depth] = ctx->cg.defer_count;
        }
        ctx->cg.loop_depth++;
        EMIT(ctx, "for (");
        if (node->for_stmt.init)
        {
            if (node->for_stmt.init->type == NODE_VAR_DECL)
            {
                ASTNode *v = node->for_stmt.init;
                if (v->var_decl.type_str && strcmp(v->var_decl.type_str, "__auto_type") != 0)
                {
                    EMIT(ctx, "%s %s = (%s)(", v->var_decl.type_str, v->var_decl.name,
                         v->var_decl.type_str);
                    codegen_expression(ctx, v->var_decl.init_expr);
                    EMIT(ctx, ")");
                }
                else
                {
                    emit_auto_type(ctx, v->var_decl.init_expr, v->token);
                    EMIT(ctx, " %s = ", v->var_decl.name);
                    codegen_expression(ctx, v->var_decl.init_expr);
                }
            }
            else
            {
                codegen_expression(ctx, node->for_stmt.init);
            }
        }
        EMIT(ctx, "; ");
        if (node->for_stmt.condition)
        {
            codegen_expression_bare(ctx, node->for_stmt.condition);
        }
        EMIT(ctx, "; ");
        if (node->for_stmt.step)
        {
            codegen_expression_bare(ctx, node->for_stmt.step);
        }
        EMIT(ctx, ") ");
        if (node->for_stmt.loop_label)
        {
            EMIT(ctx, "{\n");
            emitter_indent(&ctx->cg.emitter);
            codegen_node_single(ctx, node->for_stmt.body);
            emitter_dedent(&ctx->cg.emitter);
            EMIT(ctx, "__continue_%s:;\n", node->for_stmt.loop_label);
            EMIT(ctx, "}\n");
        }
        else
        {
            if (ctx->config->misra_mode && node->for_stmt.body->type != NODE_BLOCK)
            {
                EMIT(ctx, "{\n");
                emitter_indent(&ctx->cg.emitter);
            }
            codegen_node_single(ctx, node->for_stmt.body);
            if (ctx->config->misra_mode && node->for_stmt.body->type != NODE_BLOCK)
            {
                emitter_dedent(&ctx->cg.emitter);
                EMIT(ctx, "\n}");
            }
        }
        ctx->cg.loop_depth--;
        if (node->for_stmt.loop_label)
        {
            EMIT(ctx, "__break_%s:;\n", node->for_stmt.loop_label);
        }
        break;
    }
    case NODE_BREAK:
        // Run defers from current scope down to loop boundary before breaking
        if (ctx->cg.loop_depth > 0)
        {
            int boundary = ctx->cg.loop_defer_boundary[ctx->cg.loop_depth - 1];
            for (int i = ctx->cg.defer_count - 1; i >= boundary; i--)
            {
                emit_source_mapping_duplicate(ctx, ctx->cg.defer_stack[i]);
                codegen_node_single(ctx, ctx->cg.defer_stack[i]);
            }
        }
        if (node->break_stmt.target_label)
        {
            EMIT(ctx, "goto __break_%s;\n", node->break_stmt.target_label);
        }
        else
        {
            EMIT(ctx, "break;\n");
        }
        break;
    case NODE_CONTINUE:
        // Run defers from current scope down to loop boundary before continuing
        if (ctx->cg.loop_depth > 0)
        {
            int boundary = ctx->cg.loop_defer_boundary[ctx->cg.loop_depth - 1];
            for (int i = ctx->cg.defer_count - 1; i >= boundary; i--)
            {
                emit_source_mapping_duplicate(ctx, ctx->cg.defer_stack[i]);
                codegen_node_single(ctx, ctx->cg.defer_stack[i]);
            }
        }
        if (node->continue_stmt.target_label)
        {
            EMIT(ctx, "goto __continue_%s;\n", node->continue_stmt.target_label);
        }
        else
        {
            EMIT(ctx, "continue;\n");
        }
        break;
    case NODE_GOTO:
        if (node->goto_stmt.goto_expr)
        {
            // Computed goto: goto *expr;
            EMIT(ctx, "goto *(");
            codegen_expression(ctx, node->goto_stmt.goto_expr);
            EMIT(ctx, ");\n");
        }
        else
        {
            EMIT(ctx, "goto %s;\n", node->goto_stmt.label_name);
        }
        break;
    case NODE_LABEL:
        EMIT(ctx, "%s:;\n", node->label_stmt.label_name);
        break;
    case NODE_DO_WHILE:
    {
        if (node->do_while_stmt.loop_label)
        {
            EMIT(ctx, "%s:;\n", node->do_while_stmt.loop_label);
        }
        if (ctx->cg.loop_depth < 64)
        {
            ctx->cg.loop_defer_boundary[ctx->cg.loop_depth] = ctx->cg.defer_count;
        }
        ctx->cg.loop_depth++;
        EMIT(ctx, "do ");
        if (node->do_while_stmt.loop_label)
        {
            EMIT(ctx, "{\n");
            emitter_indent(&ctx->cg.emitter);
            codegen_node_single(ctx, node->do_while_stmt.body);
            emitter_dedent(&ctx->cg.emitter);
            EMIT(ctx, "__continue_%s:;\n", node->do_while_stmt.loop_label);
            EMIT(ctx, "}\n");
        }
        else
        {
            codegen_node_single(ctx, node->do_while_stmt.body);
        }
        EMIT(ctx, " while (");
        codegen_expression(ctx, node->do_while_stmt.condition);
        EMIT(ctx, ");\n");
        ctx->cg.loop_depth--;
        if (node->do_while_stmt.loop_label)
        {
            EMIT(ctx, "__break_%s:;\n", node->do_while_stmt.loop_label);
        }
        break;
    }
    // Loop constructs: loop, repeat, for-in
    case NODE_LOOP:
    {
        // loop { ... } => while (1) { ... }
        if (node->loop_stmt.loop_label)
        {
            EMIT(ctx, "%s:;\n", node->loop_stmt.loop_label);
        }
        if (ctx->cg.loop_depth < 64)
        {
            ctx->cg.loop_defer_boundary[ctx->cg.loop_depth] = ctx->cg.defer_count;
        }
        ctx->cg.loop_depth++;
        EMIT(ctx, "while (1) ");
        if (node->loop_stmt.loop_label)
        {
            EMIT(ctx, "{\n");
            emitter_indent(&ctx->cg.emitter);
            codegen_node_single(ctx, node->loop_stmt.body);
            emitter_dedent(&ctx->cg.emitter);
            EMIT(ctx, "__continue_%s:;\n", node->loop_stmt.loop_label);
            EMIT(ctx, "}\n");
        }
        else
        {
            codegen_node_single(ctx, node->loop_stmt.body);
        }
        ctx->cg.loop_depth--;
        if (node->loop_stmt.loop_label)
        {
            EMIT(ctx, "__break_%s:;\n", node->loop_stmt.loop_label);
        }
        break;
    }
    case NODE_REPEAT:
    {
        if (node->repeat_stmt.loop_label)
        {
            EMIT(ctx, "%s:;\n", node->repeat_stmt.loop_label);
        }
        if (ctx->cg.loop_depth < 64)
        {
            ctx->cg.loop_defer_boundary[ctx->cg.loop_depth] = ctx->cg.defer_count;
        }
        ctx->cg.loop_depth++;
        EMIT(ctx, "for (int _rpt_i = 0; _rpt_i < (%s); _rpt_i++) ", node->repeat_stmt.count);
        if (node->repeat_stmt.loop_label)
        {
            EMIT(ctx, "{\n");
            emitter_indent(&ctx->cg.emitter);
            codegen_node_single(ctx, node->repeat_stmt.body);
            emitter_dedent(&ctx->cg.emitter);
            EMIT(ctx, "__continue_%s:;\n", node->repeat_stmt.loop_label);
            EMIT(ctx, "}\n");
        }
        else
        {
            codegen_node_single(ctx, node->repeat_stmt.body);
        }
        ctx->cg.loop_depth--;
        if (node->repeat_stmt.loop_label)
        {
            EMIT(ctx, "__break_%s:;\n", node->repeat_stmt.loop_label);
        }
        break;
    }
    case NODE_FOR_RANGE:
    {
        if (node->for_range.loop_label)
        {
            EMIT(ctx, "%s:;\n", node->for_range.loop_label);
        }
        // Track loop entry for defer boundary
        if (ctx->cg.loop_depth < 64)
        {
            ctx->cg.loop_defer_boundary[ctx->cg.loop_depth] = ctx->cg.defer_count;
        }
        ctx->cg.loop_depth++;

        EMIT(ctx, "for (");
        if (z_path_match_compiler(ctx->config->cc, "tcc"))
        {
            EMIT(ctx, "__typeof__((");
            codegen_expression(ctx, node->for_range.start);
            EMIT(ctx, ")) %s = ", node->for_range.var_name);
        }
        else
        {
            EMIT(ctx, "ZC_AUTO %s = ", node->for_range.var_name);
        }
        codegen_expression(ctx, node->for_range.start);
        if (node->for_range.step && node->for_range.step[0] == '-')
        {
            if (node->for_range.is_inclusive)
            {
                EMIT(ctx, "; %s >= ", node->for_range.var_name);
            }
            else
            {
                EMIT(ctx, "; %s > ", node->for_range.var_name);
            }
        }
        else
        {
            if (node->for_range.is_inclusive)
            {
                EMIT(ctx, "; %s <= ", node->for_range.var_name);
            }
            else
            {
                EMIT(ctx, "; %s < ", node->for_range.var_name);
            }
        }
        codegen_expression(ctx, node->for_range.end);
        EMIT(ctx, "; %s", node->for_range.var_name);
        if (node->for_range.step)
        {
            EMIT(ctx, " += %s) ", node->for_range.step);
        }
        else
        {
            EMIT(ctx, "++) ");
        }
        if (node->for_range.loop_label)
        {
            EMIT(ctx, "{\n");
            emitter_indent(&ctx->cg.emitter);
            codegen_node_single(ctx, node->for_range.body);
            emitter_dedent(&ctx->cg.emitter);
            EMIT(ctx, "__continue_%s:;\n", node->for_range.loop_label);
            EMIT(ctx, "}\n");
        }
        else
        {
            codegen_node_single(ctx, node->for_range.body);
        }

        ctx->cg.loop_depth--;
        if (node->for_range.loop_label)
        {
            EMIT(ctx, "__break_%s:;\n", node->for_range.loop_label);
        }
        break;
    }

    case NODE_ASM:
    {
        int is_extended = (node->asm_stmt.num_outputs > 0 || node->asm_stmt.num_inputs > 0 ||
                           node->asm_stmt.num_clobbers > 0);

        if (node->asm_stmt.is_volatile)
        {
            EMIT(ctx, "__asm__ __volatile__(");
        }
        else
        {
            EMIT(ctx, "__asm__(");
        }

        char *code = node->asm_stmt.code;
        size_t transformed_sz = strlen(code) * 3 + 128; // Extra buffer for expansions
        char *transformed = xmalloc(transformed_sz);
        char *dst = transformed;

        for (char *p = code; *p; p++)
        {
            if (*p == '{')
            {
                // Find matching }
                char *end = strchr(p + 1, '}');
                if (end)
                {
                    // Extract variable name
                    int var_len = end - p - 1;
                    char var_name[64];
                    strncpy(var_name, p + 1, var_len);
                    var_name[var_len] = 0;

                    // Find variable index
                    int idx = -1;

                    // Check outputs first
                    for (int i = 0; i < node->asm_stmt.num_outputs; i++)
                    {
                        if (strcmp(node->asm_stmt.outputs[i], var_name) == 0)
                        {
                            idx = i;
                            break;
                        }
                    }

                    // Then check inputs
                    if (idx == -1)
                    {
                        for (int i = 0; i < node->asm_stmt.num_inputs; i++)
                        {
                            if (strcmp(node->asm_stmt.inputs[i], var_name) == 0)
                            {
                                idx = node->asm_stmt.num_outputs + i;
                                break;
                            }
                        }
                    }

                    size_t rem = transformed_sz - (dst - transformed);
                    if (idx >= 0)
                    {
                        // Replace with %N
#if defined(ZC_ARCH_ARM64)
                        // Use most optimal register size on arm architectures
                        if (node->asm_stmt.register_size <= 32)
                        {
                            int _n = snprintf(dst, rem, "%%w%d", idx);
                            dst += (_n > 0 && (size_t)_n < rem) ? (size_t)_n
                                   : rem > 1                    ? rem - 1
                                                                : 0;
                        }
                        else
#endif
                        {
                            int _n = snprintf(dst, rem, "%%%d", idx);
                            dst += (_n > 0 && (size_t)_n < rem) ? (size_t)_n
                                   : rem > 1                    ? rem - 1
                                                                : 0;
                        }
                    }
                    else
                    {
                        // Variable not found - error or keep as-is?
                        int _n = snprintf(dst, rem, "{%s}", var_name);
                        dst += (_n > 0 && (size_t)_n < rem) ? (size_t)_n : rem > 1 ? rem - 1 : 0;
                    }

                    p = end; // Skip past }
                }
                else
                {
                    *dst++ = *p;
                }
            }
            else if (*p == '%')
            {
                if (is_extended)
                {
                    *dst++ = '%';
                    *dst++ = '%';
                }
                else
                {
                    *dst++ = '%';
                }
            }
            else
            {
                *dst++ = *p;
            }
        }
        *dst = 0;

        EMIT(ctx, "\"");
        for (char *p = transformed; *p; p++)
        {
            if (*p == '\n')
            {
                EMIT(ctx, "\\n\"\n        \"");
            }
            else if (*p == '"')
            {
                EMIT(ctx, "\\\"");
            }
            else if (*p == '\\')
            {
                EMIT(ctx, "\\\\");
            }
            else
            {
                EMIT(ctx, "%c", *p);
            }
        }
        EMIT(ctx, "\\n\"");

        if (node->asm_stmt.num_outputs > 0)
        {
            EMIT(ctx, "\n        : ");
            for (int i = 0; i < node->asm_stmt.num_outputs; i++)
            {
                if (i > 0)
                {
                    EMIT(ctx, ", ");
                }

                // Determine constraint
                char *mode = node->asm_stmt.output_modes[i];
                if (strcmp(mode, "out") == 0)
                {
                    EMIT(ctx, "\"=r\"(%s)", node->asm_stmt.outputs[i]);
                }
                else if (strcmp(mode, "inout") == 0)
                {
                    EMIT(ctx, "\"+r\"(%s)", node->asm_stmt.outputs[i]);
                }
                else
                {
                    EMIT(ctx, "\"=r\"(%s)", node->asm_stmt.outputs[i]);
                }
            }
        }

        if (node->asm_stmt.num_inputs > 0)
        {
            EMIT(ctx, "\n        : ");
            for (int i = 0; i < node->asm_stmt.num_inputs; i++)
            {
                if (i > 0)
                {
                    EMIT(ctx, ", ");
                }
                EMIT(ctx, "\"r\"(%s)", node->asm_stmt.inputs[i]);
            }
        }
        else if (node->asm_stmt.num_outputs > 0)
        {
            EMIT(ctx, "\n        : ");
        }

        if (node->asm_stmt.num_clobbers > 0)
        {
            EMIT(ctx, "\n        : ");
            for (int i = 0; i < node->asm_stmt.num_clobbers; i++)
            {
                if (i > 0)
                {
                    EMIT(ctx, ", ");
                }
                EMIT(ctx, "\"%s\"", node->asm_stmt.clobbers[i]);
            }
        }

        EMIT(ctx, ");\n");
        break;
    }
    case NODE_RETURN:
    {
        // Suppress pending closure frees — returned closures escape the scope
        ctx->cg.pending_closure_free_count = 0;
        int has_defers = (ctx->cg.defer_count > ctx->cg.func_defer_boundary);
        int handled = 0;

        if (node->ret.value && node->ret.value->type == NODE_EXPR_ARRAY_LITERAL &&
            ctx->cg.current_func_ret_type &&
            strncmp(ctx->cg.current_func_ret_type, "Slice__", 7) == 0)
        {
            // Heap allocation for slice literals to prevent use-after-return
            ASTNode *arr = node->ret.value;
            int count = arr->array_literal.count;
            char *elem_type = "void*"; // fallback

            // Prioritize the function return type (Slice_T) to determine the pointer type
            // This prevents "incompatible pointer type" errors in C when returning literals of
            // different types
            if (ctx->cg.current_func_ret_type &&
                strncmp(ctx->cg.current_func_ret_type, "Slice__", 7) == 0)
            {
                elem_type = xstrdup(ctx->cg.current_func_ret_type + 7);
            }
            else if (arr->array_literal.elements && arr->array_literal.elements->type_info)
            {
                elem_type = type_to_c_string(arr->array_literal.elements->type_info);
            }
            else if (arr->type_info && arr->type_info->inner)
            {
                elem_type = type_to_c_string(arr->type_info->inner);
            }
            else
            {
                elem_type = xstrdup("void*");
            }

            EMIT(ctx, "{ %s *_tmp_arr = (%s*)malloc(%d * sizeof(%s));\n", elem_type, elem_type,
                 count, elem_type);
            emitter_indent(&ctx->cg.emitter);

            ASTNode *elem = arr->array_literal.elements;
            int idx = 0;
            while (elem)
            {
                EMIT(ctx, "_tmp_arr[%d] = ", idx++);
                codegen_expression(ctx, elem);
                EMIT(ctx, ";\n");
                elem = elem->next;
            }

            if (ctx->config->use_cpp)
            {
                // Traditional initializer: (Slice){data, len, cap}
                EMIT(ctx, "return (%s){_tmp_arr, %d, %d};\n", ctx->cg.current_func_ret_type, count,
                     count);
            }
            else
            {
                EMIT(ctx, "return (%s){.data = _tmp_arr, .len = %d, .cap = %d};\n",
                     ctx->cg.current_func_ret_type, count, count);
            }
            emitter_dedent(&ctx->cg.emitter);
            EMIT(ctx, "}\n");
            handled = 1;
        }

        if (node->ret.value && node->ret.value->type == NODE_EXPR_VAR)
        {
            char *tname = infer_type(ctx, node->ret.value);
            if (tname)
            {
                char *clean = tname;
                if (strncmp(clean, "struct ", 7) == 0)
                {
                    clean += 7;
                }

                ASTNode *def = find_struct_def(ctx, clean);
                if (def && def->type_info && def->type_info->traits.has_drop)
                {
                    if (ctx->config->misra_mode)
                    {
                        EMIT(ctx, "_misra_ret = ({ ");
                    }
                    else
                    {
                        EMIT(ctx, "return ({ ");
                    }

                    if (z_path_match_compiler(ctx->config->cc, "tcc"))
                    {
                        EMIT(ctx, "__typeof__(");
                        codegen_expression(ctx, node->ret.value);
                        EMIT(ctx, ")");
                    }
                    else
                    {
                        EMIT(ctx, "ZC_AUTO");
                    }
                    EMIT(ctx, " _z_ret_mv = ");
                    if (ctx->self_is_pointer && strcmp(node->ret.value->var_ref.name, "self") == 0)
                    {
                        EMIT(ctx, "*");
                    }
                    codegen_expression(ctx, node->ret.value);
                    EMIT(ctx, "; memset(&");
                    if (ctx->self_is_pointer && strcmp(node->ret.value->var_ref.name, "self") == 0)
                    {
                        EMIT(ctx, "*");
                    }
                    codegen_expression(ctx, node->ret.value);
                    EMIT(ctx, ", 0, sizeof(_z_ret_mv)); ");
                    if (strcmp(node->ret.value->var_ref.name, "self") != 0)
                    {
                        EMIT(ctx, "__z_drop_flag_%s = 0; ", node->ret.value->var_ref.name);
                    }
                    for (int i = ctx->cg.defer_count - 1; i >= ctx->cg.func_defer_boundary; i--)
                    {
                        emit_source_mapping_duplicate(ctx, ctx->cg.defer_stack[i]);
                        codegen_node_single(ctx, ctx->cg.defer_stack[i]);
                    }
                    EMIT(ctx, "_z_ret_mv; });\n");

                    if (ctx->config->misra_mode)
                    {
                        EMIT(ctx, "goto _misra_end_of_func;\n");
                    }
                    handled = 1;
                }
                zfree(tname);
            }
        }

        if (!handled)
        {
            if (has_defers && node->ret.value)
            {
                EMIT(ctx, "{ ");
                emitter_indent(&ctx->cg.emitter);
                if (ctx->cg.current_func_ret_type_info)
                {
                    char *tstr = type_to_c_string(ctx->cg.current_func_ret_type_info);
                    EMIT(ctx, "%s", tstr);
                    zfree(tstr);
                }
                else if (ctx->cg.current_func_ret_type &&
                         strcmp(ctx->cg.current_func_ret_type, "void") != 0 &&
                         strcmp(ctx->cg.current_func_ret_type, "unknown") != 0)
                {
                    EMIT(ctx, "%s", ctx->cg.current_func_ret_type);
                }
                else
                {
                    emit_auto_type(ctx, node->ret.value, node->token);
                }
                EMIT(ctx, " _z_ret = ");
                if (node->ret.value->type == NODE_EXPR_VAR && ctx->self_is_pointer &&
                    strcmp(node->ret.value->var_ref.name, "self") == 0)
                {
                    EMIT(ctx, "*");
                }
                codegen_expression(ctx, node->ret.value);
                EMIT(ctx, "; ");
                for (int i = ctx->cg.defer_count - 1; i >= ctx->cg.func_defer_boundary; i--)
                {
                    emit_source_mapping_duplicate(ctx, ctx->cg.defer_stack[i]);
                    codegen_node_single(ctx, ctx->cg.defer_stack[i]);
                }
                if (ctx->config->misra_mode)
                {
                    emitter_dedent(&ctx->cg.emitter);
                    EMIT(ctx, "_misra_ret = _z_ret; goto _misra_end_of_func; }\n");
                }
                else
                {
                    emitter_dedent(&ctx->cg.emitter);
                    EMIT(ctx, "return _z_ret; }\n");
                }
            }
            else if (has_defers)
            {
                for (int i = ctx->cg.defer_count - 1; i >= ctx->cg.func_defer_boundary; i--)
                {
                    emit_source_mapping_duplicate(ctx, ctx->cg.defer_stack[i]);
                    codegen_node_single(ctx, ctx->cg.defer_stack[i]);
                }
                if (ctx->config->misra_mode)
                {
                    EMIT(ctx, "goto _misra_end_of_func;\n");
                }
                else
                {
                    EMIT(ctx, "return;\n");
                }
            }
            else
            {
                if (ctx->config->misra_mode)
                {
                    if (ctx->cg.current_func_ret_type &&
                        strcmp(ctx->cg.current_func_ret_type, "void") != 0)
                    {
                        EMIT(ctx, "_misra_ret = ");
                    }
                    else
                    {
                    }
                }
                else
                {
                    EMIT(ctx, "return ");
                }

                if (node->ret.value && node->ret.value->type == NODE_EXPR_VAR &&
                    ctx->self_is_pointer && strcmp(node->ret.value->var_ref.name, "self") == 0)
                {
                    // return self; -> return *self; (if returns by value)
                    EMIT(ctx, "*");
                }
                else if (node->ret.value && node->ret.value->type == NODE_EXPR_UNARY &&
                         strcmp(node->ret.value->unary.op, "&") == 0 &&
                         node->ret.value->unary.operand->type == NODE_EXPR_VAR &&
                         strcmp(node->ret.value->unary.operand->var_ref.name, "self") == 0)
                {
                    // return &self; -> return self; (since self is already a pointer in C)
                    codegen_expression(ctx, node->ret.value->unary.operand);
                    EMIT(ctx, ";\n");
                    if (ctx->config->misra_mode)
                    {
                        EMIT(ctx, "goto _misra_end_of_func;\n");
                    }
                    break;
                }
                if (node->ret.value)
                {
                    codegen_expression(ctx, node->ret.value);
                }
                EMIT(ctx, ";\n");
                if (ctx->config->misra_mode)
                {
                    EMIT(ctx, "goto _misra_end_of_func;\n");
                }
            }
        }
        break;
    }
    case NODE_EXPR_MEMBER:
    {
        codegen_expression(ctx, node->member.target);
        char *lt = infer_type(ctx, node->member.target);
        if (lt && (lt[strlen(lt) - 1] == '*' || strstr(lt, "*")))
        {
            EMIT(ctx, "->%s", node->member.field);
        }
        else
        {
            EMIT(ctx, ".%s", node->member.field);
        }
        if (lt)
        {
            zfree(lt);
        }
        break;
    }
    case NODE_REPL_PRINT:
    {
        EMIT(ctx, "{ ");
        emit_auto_type(ctx, node->repl_print.expr, node->token);
        EMIT(ctx, " _zval = (");
        codegen_expression(ctx, node->repl_print.expr);
        EMIT(ctx,
             "); fprintf(stdout, _z_str(_zval), _z_arg(_zval)); fprintf(stdout, \"\\n\"); }\n");
        break;
    }
    case NODE_AWAIT:
    {
        handle_node_await_internal(ctx, node);
        EMIT(ctx, ";\n");
        break;
    }
    case NODE_EXPR_LITERAL:
        // String literal statement should auto-print
        if (node->literal.type_kind == LITERAL_STRING)
        {
            EMIT(ctx, "printf(\"%%s\\n\", ");
            codegen_expression(ctx, node);
            EMIT(ctx, ");\n");
        }
        else
        {
            // Non-string literals as statements - just evaluate
            codegen_expression(ctx, node);
            EMIT(ctx, ";\n");
        }
        break;
    case NODE_CUDA_LAUNCH:
    {
        // Emit CUDA kernel launch: kernel<<<grid, block, shared, stream>>>(args);
        ASTNode *call = node->cuda_launch.call;

        // Get kernel name from callee
        if (call->call.callee->type == NODE_EXPR_VAR)
        {
            EMIT(ctx, "%s<<<", call->call.callee->var_ref.name);
        }
        else
        {
            codegen_expression(ctx, call->call.callee);
            EMIT(ctx, "<<<");
        }

        // Grid dimension
        codegen_expression(ctx, node->cuda_launch.grid);
        EMIT(ctx, ", ");

        // Block dimension
        codegen_expression(ctx, node->cuda_launch.block);

        // Optional shared memory size
        if (node->cuda_launch.shared_mem || node->cuda_launch.stream)
        {
            EMIT(ctx, ", ");
            if (node->cuda_launch.shared_mem)
            {
                codegen_expression(ctx, node->cuda_launch.shared_mem);
            }
            else
            {
                EMIT(ctx, "0");
            }
        }

        // Optional CUDA stream
        if (node->cuda_launch.stream)
        {
            EMIT(ctx, ", ");
            codegen_expression(ctx, node->cuda_launch.stream);
        }

        EMIT(ctx, ">>>(");

        // Arguments
        ASTNode *arg = call->call.args;
        int first = 1;
        while (arg)
        {
            if (!first)
            {
                EMIT(ctx, ", ");
            }
            codegen_expression(ctx, arg);
            first = 0;
            arg = arg->next;
        }

        EMIT(ctx, ");\n");
        break;
    }
    case NODE_PREPROC_DIRECTIVE:
    {
        EMIT(ctx, "%s\n", node->raw_stmt.content);
        break;
    }
    case NODE_RAW_STMT:
    {
        if (ctx->cg.current_lambda)
        {
            Lexer l;
            lexer_init(&l, node->raw_stmt.content, ctx->config);
            Token t;
            int last_pos = 0;
            while ((t = lexer_next(&l)).type != TOK_EOF)
            {
                int current_tok_start = (int)(t.start - node->raw_stmt.content);
                for (int i = last_pos; i < current_tok_start; i++)
                {
                    EMIT(ctx, "%c", node->raw_stmt.content[i]);
                }

                if (t.type == TOK_IDENT)
                {
                    char *name = token_strdup(t);
                    int captured = -1;
                    if (ctx->cg.current_lambda->lambda.captured_vars)
                    {
                        for (int i = 0; i < ctx->cg.current_lambda->lambda.num_captures; i++)
                        {
                            if (strcmp(name, ctx->cg.current_lambda->lambda.captured_vars[i]) == 0)
                            {
                                captured = i;
                                break;
                            }
                        }
                    }

                    if (captured != -1)
                    {
                        if (ctx->cg.current_lambda->lambda.capture_modes &&
                            ctx->cg.current_lambda->lambda.capture_modes[captured] == 1)
                        {
                            EMIT(ctx, "(*ctx->%s)", name);
                        }
                        else
                        {
                            EMIT(ctx, "ctx->%s", name);
                        }
                    }
                    else
                    {
                        EMIT(ctx, "%.*s", t.len, t.start);
                    }
                    zfree(name);
                }
                else
                {
                    EMIT(ctx, "%.*s", t.len, t.start);
                }
                last_pos = current_tok_start + t.len;
            }
            EMIT(ctx, "%s\n", node->raw_stmt.content + last_pos);
        }
        else
        {
            EMIT(ctx, "%s\n", node->raw_stmt.content);
        }
        break;
    }
    default:
        codegen_expression(ctx, node);
        EMIT(ctx, ";\n");
        if (node->type == NODE_EXPR_CALL && node->call.callee &&
            ctx->cg.pending_closure_free_count > 0)
        {
            int is_thread_spawn = 0;
            if (node->call.callee->type == NODE_EXPR_VAR && node->call.callee->var_ref.name &&
                strstr(node->call.callee->var_ref.name, "Thread::spawn"))
            {
                is_thread_spawn = 1;
            }
            else if (node->call.callee->type == NODE_EXPR_MEMBER &&
                     node->call.callee->member.target &&
                     node->call.callee->member.target->type == NODE_EXPR_VAR &&
                     strcmp(node->call.callee->member.target->var_ref.name, "Thread") == 0 &&
                     strcmp(node->call.callee->member.field, "spawn") == 0)
            {
                is_thread_spawn = 1;
            }
            if (is_thread_spawn)
            {
                ctx->cg.pending_closure_free_count = 0;
            }
        }
        emit_pending_closure_frees(ctx);
    }
}

// Walks AST nodes and generates code.
void codegen_walker(ParserContext *ctx, ASTNode *node)
{
    while (node)
    {
        emit_source_mapping(ctx, node); // Step to this expression
        codegen_node_single(ctx, node);
        node = node->next;
    }
}
