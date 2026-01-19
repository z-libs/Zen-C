
#include "../ast/ast.h"
#include "../zprep.h"
#include "codegen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Helper: Check if a struct depends on another struct/enum by-value.
static int struct_depends_on(ASTNode *s1, const char *target_name)
{
    if (!s1)
    {
        return 0;
    }

    // Only structs have dependencies that matter for ordering.
    if (s1->type != NODE_STRUCT)
    {
        return 0;
    }

    ASTNode *field = s1->strct.fields;
    while (field)
    {
        if (field->type == NODE_FIELD && field->field.type)
        {
            char *type_str = field->field.type;
            // Skip pointers - they don't create ordering dependency.
            if (strchr(type_str, '*'))
            {
                field = field->next;
                continue;
            }

            // Check if this field's type matches target (struct or enum).
            if (strcmp(type_str, target_name) == 0)
            {
                return 1;
            }
        }
        field = field->next;
    }
    return 0;
}

// Topologically sort a list of struct/enum nodes.
static ASTNode *topo_sort_structs(ASTNode *head)
{
    if (!head)
    {
        return NULL;
    }

    // Count all nodes (structs + enums + traits).
    int count = 0;
    ASTNode *n = head;
    while (n)
    {
        if (n->type == NODE_STRUCT || n->type == NODE_ENUM || n->type == NODE_TRAIT)
        {
            count++;
        }
        n = n->next;
    }
    if (count == 0)
    {
        return head;
    }

    // Build array of all nodes.
    ASTNode **nodes = malloc(count * sizeof(ASTNode *));
    int *emitted = calloc(count, sizeof(int));
    n = head;
    int idx = 0;
    while (n)
    {
        if (n->type == NODE_STRUCT || n->type == NODE_ENUM || n->type == NODE_TRAIT)
        {
            nodes[idx++] = n;
        }
        n = n->next;
    }

    // Build order array (indices in emission order).
    int *order = malloc(count * sizeof(int));
    int order_idx = 0;

    int changed = 1;
    int max_iterations = count * count;
    int iterations = 0;

    while (changed && iterations < max_iterations)
    {
        changed = 0;
        iterations++;

        for (int i = 0; i < count; i++)
        {
            if (emitted[i])
            {
                continue;
            }

            // Enums and traits have no dependencies, emit first.
            if (nodes[i]->type == NODE_ENUM || nodes[i]->type == NODE_TRAIT)
            {
                order[order_idx++] = i;
                emitted[i] = 1;
                changed = 1;
                continue;
            }

            // For structs, check if all dependencies are emitted.
            int can_emit = 1;
            for (int j = 0; j < count; j++)
            {
                if (i == j || emitted[j])
                {
                    continue;
                }

                // Get the name of the potential dependency.
                const char *dep_name = NULL;
                if (nodes[j]->type == NODE_STRUCT)
                {
                    dep_name = nodes[j]->strct.name;
                }
                else if (nodes[j]->type == NODE_ENUM)
                {
                    dep_name = nodes[j]->enm.name;
                }

                if (dep_name && struct_depends_on(nodes[i], dep_name))
                {
                    can_emit = 0;
                    break;
                }
            }

            if (can_emit)
            {
                order[order_idx++] = i;
                emitted[i] = 1;
                changed = 1;
            }
        }
    }

    // Add any remaining nodes (cycles).
    for (int i = 0; i < count; i++)
    {
        if (!emitted[i])
        {
            order[order_idx++] = i;
        }
    }

    // Now build the linked list in the correct order.
    ASTNode *result = NULL;
    ASTNode *result_tail = NULL;

    for (int i = 0; i < order_idx; i++)
    {
        ASTNode *node = nodes[order[i]];
        if (!result)
        {
            result = node;
            result_tail = node;
        }
        else
        {
            result_tail->next = node;
            result_tail = node;
        }
    }
    if (result_tail)
    {
        result_tail->next = NULL;
    }

    free(nodes);
    free(emitted);
    free(order);
    return result;
}

// Main entry point for code generation.
void codegen_node(ParserContext *ctx, ASTNode *node, FILE *out)
{
    if (node->type == NODE_ROOT)
    {
        ASTNode *kids = node->root.children;
        // Recursive Unwrap of Nested Roots (if accidentally wrapped multiple
        // times).
        while (kids && kids->type == NODE_ROOT)
        {
            kids = kids->root.children;
        }

        global_user_structs = kids;

        if (!ctx->skip_preamble)
        {
            emit_preamble(ctx, out);
        }
        emit_includes_and_aliases(kids, out);

        // Emit Hoisted Code (from plugins)
        if (ctx->hoist_out)
        {
            long pos = ftell(ctx->hoist_out);
            rewind(ctx->hoist_out);
            char buf[4096];
            size_t n;
            while ((n = fread(buf, 1, sizeof(buf), ctx->hoist_out)) > 0)
            {
                fwrite(buf, 1, n, out);
            }
            fseek(ctx->hoist_out, pos, SEEK_SET);
        }

        ASTNode *merged = NULL;
        ASTNode *merged_tail = NULL;

        ASTNode *s = ctx->instantiated_structs;
        while (s)
        {
            ASTNode *copy = xmalloc(sizeof(ASTNode));
            *copy = *s;
            copy->next = NULL;
            if (!merged)
            {
                merged = copy;
                merged_tail = copy;
            }
            else
            {
                merged_tail->next = copy;
                merged_tail = copy;
            }
            s = s->next;
        }

        StructRef *sr = ctx->parsed_structs_list;
        while (sr)
        {
            if (sr->node)
            {
                ASTNode *copy = xmalloc(sizeof(ASTNode));
                *copy = *sr->node;
                copy->next = NULL;
                if (!merged)
                {
                    merged = copy;
                    merged_tail = copy;
                }
                else
                {
                    merged_tail->next = copy;
                    merged_tail = copy;
                }
            }
            sr = sr->next;
        }

        StructRef *er = ctx->parsed_enums_list;
        while (er)
        {
            if (er->node)
            {
                ASTNode *copy = xmalloc(sizeof(ASTNode));
                *copy = *er->node;
                copy->next = NULL;
                if (!merged)
                {
                    merged = copy;
                    merged_tail = copy;
                }
                else
                {
                    merged_tail->next = copy;
                    merged_tail = copy;
                }
            }
            er = er->next;
        }

        ASTNode *k = kids;
        while (k)
        {
            if (k->type == NODE_STRUCT || k->type == NODE_ENUM)
            {
                int found = 0;
                ASTNode *chk = merged;
                while (chk)
                {
                    if (chk->type == k->type)
                    {
                        const char *n1 = (k->type == NODE_STRUCT) ? k->strct.name : k->enm.name;
                        const char *n2 =
                            (chk->type == NODE_STRUCT) ? chk->strct.name : chk->enm.name;
                        if (n1 && n2 && strcmp(n1, n2) == 0)
                        {
                            found = 1;
                            break;
                        }
                    }
                    chk = chk->next;
                }

                if (!found)
                {
                    ASTNode *copy = xmalloc(sizeof(ASTNode));
                    *copy = *k;
                    copy->next = NULL;
                    if (!merged)
                    {
                        merged = copy;
                        merged_tail = copy;
                    }
                    else
                    {
                        merged_tail->next = copy;
                        merged_tail = copy;
                    }
                }
            }
            k = k->next;
        }

        // Topologically sort.
        ASTNode *sorted = topo_sort_structs(merged);

        print_type_defs(ctx, out, sorted);
        emit_enum_protos(sorted, out);
        emit_global_aliases(ctx, out); // Emit ALL aliases (including imports)
        emit_type_aliases(kids, out);  // Emit local aliases (redundant but safe)
        emit_trait_defs(kids, out);

        // First pass: emit ONLY preprocessor directives before struct defs
        // so that macros like `panic` are available in function bodies
        ASTNode *raw_iter = kids;
        while (raw_iter)
        {
            if (raw_iter->type == NODE_RAW_STMT && raw_iter->raw_stmt.content)
            {
                const char *content = raw_iter->raw_stmt.content;
                // Skip leading whitespace
                while (*content == ' ' || *content == '\t' || *content == '\n')
                {
                    content++;
                }
                // Emit only if it's a preprocessor directive
                if (*content == '#')
                {
                    fprintf(out, "%s\n", raw_iter->raw_stmt.content);
                }
            }
            raw_iter = raw_iter->next;
        }

        if (sorted)
        {
            emit_struct_defs(ctx, sorted, out);
        }

        // Second pass: emit non-preprocessor raw statements after struct defs
        raw_iter = kids;
        while (raw_iter)
        {
            if (raw_iter->type == NODE_RAW_STMT && raw_iter->raw_stmt.content)
            {
                const char *content = raw_iter->raw_stmt.content;
                while (*content == ' ' || *content == '\t' || *content == '\n')
                {
                    content++;
                }
                if (*content != '#')
                {
                    fprintf(out, "%s\n", raw_iter->raw_stmt.content);
                }
            }
            raw_iter = raw_iter->next;
        }

        // Emit type aliases was here (moved up)

        ASTNode *merged_globals = NULL; // Head

        if (ctx->parsed_globals_list)
        {
            StructRef *s = ctx->parsed_globals_list;
            while (s)
            {
                ASTNode *copy = xmalloc(sizeof(ASTNode));
                *copy = *s->node;
                copy->next = merged_globals;
                merged_globals = copy;

                s = s->next;
            }
        }

        emit_globals(ctx, merged_globals, out);

        ASTNode *merged_funcs = NULL;
        ASTNode *merged_funcs_tail = NULL;

        if (ctx->instantiated_funcs)
        {
            ASTNode *s = ctx->instantiated_funcs;
            while (s)
            {
                ASTNode *copy = xmalloc(sizeof(ASTNode));
                *copy = *s;
                copy->next = NULL;
                if (!merged_funcs)
                {
                    merged_funcs = copy;
                    merged_funcs_tail = copy;
                }
                else
                {
                    merged_funcs_tail->next = copy;
                    merged_funcs_tail = copy;
                }
                s = s->next;
            }
        }

        if (ctx->parsed_funcs_list)
        {
            StructRef *s = ctx->parsed_funcs_list;
            while (s)
            {
                ASTNode *copy = xmalloc(sizeof(ASTNode));
                *copy = *s->node;
                copy->next = NULL;
                if (!merged_funcs)
                {
                    merged_funcs = copy;
                    merged_funcs_tail = copy;
                }
                else
                {
                    merged_funcs_tail->next = copy;
                    merged_funcs_tail = copy;
                }
                s = s->next;
            }
        }

        if (ctx->parsed_impls_list)
        {
            StructRef *s = ctx->parsed_impls_list;
            while (s)
            {
                ASTNode *copy = xmalloc(sizeof(ASTNode));
                *copy = *s->node;
                copy->next = NULL;
                if (!merged_funcs)
                {
                    merged_funcs = copy;
                    merged_funcs_tail = copy;
                }
                else
                {
                    merged_funcs_tail->next = copy;
                    merged_funcs_tail = copy;
                }
                s = s->next;
            }
        }

        emit_protos(merged_funcs, out);

        emit_impl_vtables(ctx, out);

        emit_lambda_defs(ctx, out);

        emit_tests_and_runner(ctx, kids, out);

        ASTNode *iter = merged_funcs;
        while (iter)
        {
            if (iter->type == NODE_IMPL)
            {
                char *sname = iter->impl.struct_name;
                if (!sname)
                {
                    iter = iter->next;
                    continue;
                }

                char *mangled = replace_string_type(sname);
                ASTNode *def = find_struct_def_codegen(ctx, mangled);
                int skip = 0;
                if (def)
                {
                    if (def->type == NODE_STRUCT && def->strct.is_template)
                    {
                        skip = 1;
                    }
                    else if (def->type == NODE_ENUM && def->enm.is_template)
                    {
                        skip = 1;
                    }
                }
                else
                {
                    char *lt = strchr(sname, '<');
                    if (lt)
                    {
                        int len = lt - sname;
                        char *buf = xmalloc(len + 1);
                        strncpy(buf, sname, len);
                        buf[len] = 0;
                        def = find_struct_def_codegen(ctx, buf);
                        if (def && def->strct.is_template)
                        {
                            skip = 1;
                        }
                        free(buf);
                    }
                }
                if (mangled)
                {
                    free(mangled);
                }
                if (skip)
                {
                    iter = iter->next;
                    continue;
                }
            }
            if (iter->type == NODE_IMPL_TRAIT)
            {
                char *sname = iter->impl_trait.target_type;
                if (!sname)
                {
                    iter = iter->next;
                    continue;
                }

                char *mangled = replace_string_type(sname);
                ASTNode *def = find_struct_def_codegen(ctx, mangled);
                int skip = 0;
                if (def)
                {
                    if (def->strct.is_template)
                    {
                        skip = 1;
                    }
                }
                else
                {
                    char *lt = strchr(sname, '<');
                    if (lt)
                    {
                        int len = lt - sname;
                        char *buf = xmalloc(len + 1);
                        strncpy(buf, sname, len);
                        buf[len] = 0;
                        def = find_struct_def_codegen(ctx, buf);
                        if (def && def->strct.is_template)
                        {
                            skip = 1;
                        }
                        free(buf);
                    }
                }
                if (mangled)
                {
                    free(mangled);
                }
                if (skip)
                {
                    iter = iter->next;
                    continue;
                }
            }
            codegen_node_single(ctx, iter, out);
            iter = iter->next;
        }

        int has_user_main = 0;
        ASTNode *chk = merged_funcs;
        while (chk)
        {
            if (chk->type == NODE_FUNCTION && strcmp(chk->func.name, "main") == 0)
            {
                has_user_main = 1;
                break;
            }
            chk = chk->next;
        }

        if (!has_user_main)
        {
            fprintf(out, "\nint main() { _z_run_tests(); return 0; }\n");
        }
    }
}
