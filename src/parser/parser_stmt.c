
#include "parser.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <unistd.h>
#else
#include <io.h>
#define access _access
#ifndef PATH_MAX
#define PATH_MAX _MAX_PATH
#endif
#define realpath(N,R) _fullpath((R),(N), PATH_MAX)
#ifndef R_OK
#define	R_OK 0x04
#endif
#endif
#include "../ast/ast.h"
#include "../plugins/plugin_manager.h"
#include "../zen/zen_facts.h"
#include "zprep_plugin.h"
#include "../codegen/codegen.h"

char *curr_func_ret = NULL;
char *run_comptime_block(ParserContext *ctx, Lexer *l);
extern char *g_current_filename;

/**
 * @brief Auto-imports std/slice.zc if not already imported.
 *
 * This is called when array iteration is detected in for-in loops,
 * to ensure the Slice<T>, SliceIter<T>, and Option<T> templates are available.
 */
static void auto_import_std_slice(ParserContext *ctx)
{
    // Check if already imported via templates
    GenericTemplate *t = ctx->templates;
    while (t)
    {
        if (strcmp(t->name, "Slice") == 0)
        {
            return; // Already have the Slice template
        }
        t = t->next;
    }

    // Try to find and import std/slice.zc
    static const char *std_paths[] = {"std/slice.zc", "./std/slice.zc", NULL};
    static const char *system_paths[] = {"/usr/local/share/zenc", "/usr/share/zenc", NULL};

    char resolved_path[1024];
    int found = 0;

    // First, try relative to current file
    if (g_current_filename)
    {
        char *current_dir = xstrdup(g_current_filename);
        char *last_slash = strrchr(current_dir, '/');
        if (last_slash)
        {
            *last_slash = 0;
            snprintf(resolved_path, sizeof(resolved_path), "%s/std/slice.zc", current_dir);
            if (access(resolved_path, R_OK) == 0)
            {
                found = 1;
            }
        }
        free(current_dir);
    }

    // Try relative paths
    if (!found)
    {
        for (int i = 0; std_paths[i] && !found; i++)
        {
            if (access(std_paths[i], R_OK) == 0)
            {
                strncpy(resolved_path, std_paths[i], sizeof(resolved_path) - 1);
                resolved_path[sizeof(resolved_path) - 1] = '\0';
                found = 1;
            }
        }
    }

    // Try system paths
    if (!found)
    {
        for (int i = 0; system_paths[i] && !found; i++)
        {
            snprintf(resolved_path, sizeof(resolved_path), "%s/std/slice.zc", system_paths[i]);
            if (access(resolved_path, R_OK) == 0)
            {
                found = 1;
            }
        }
    }

    if (!found)
    {
        return; // Could not find std/slice.zc, instantiate_generic will error
    }

    // Canonicalize path
    char *real_fn = realpath(resolved_path, NULL);
    if (real_fn)
    {
        strncpy(resolved_path, real_fn, sizeof(resolved_path) - 1);
        resolved_path[sizeof(resolved_path) - 1] = '\0';
        free(real_fn);
    }

    // Check if already imported
    if (is_file_imported(ctx, resolved_path))
    {
        return;
    }
    mark_file_imported(ctx, resolved_path);

    // Load and parse the file
    char *src = load_file(resolved_path);
    if (!src)
    {
        return; // Could not load file
    }

    Lexer i;
    lexer_init(&i, src);

    // Save and restore filename context
    char *saved_fn = g_current_filename;
    g_current_filename = resolved_path;

    // Parse the slice module contents
    parse_program_nodes(ctx, &i);

    g_current_filename = saved_fn;
}

static void check_assignment_condition(ASTNode *cond)
{
    if (!cond)
    {
        return;
    }
    if (cond->type == NODE_EXPR_BINARY)
    {
        if (cond->binary.op && strcmp(cond->binary.op, "=") == 0)
        {
            zwarn_at(cond->token, "Assignment in condition");
            fprintf(stderr, COLOR_CYAN "   = note: " COLOR_RESET "Did you mean '=='?\n");
        }
    }
}

ASTNode *parse_match(ParserContext *ctx, Lexer *l)
{
    init_builtins();
    Token start_token = lexer_peek(l);
    lexer_next(l); // eat 'match'
    ASTNode *expr = parse_expression(ctx, l);

    Token t_brace = lexer_next(l);
    if (t_brace.type != TOK_LBRACE)
    {
        zpanic_at(t_brace, "Expected { in match");
    }

    ASTNode *h = 0, *tl = 0;
    while (lexer_peek(l).type != TOK_RBRACE)
    {
        skip_comments(l);
        if (lexer_peek(l).type == TOK_RBRACE)
        {
            break;
        }
        if (lexer_peek(l).type == TOK_COMMA)
        {
            lexer_next(l);
        }
        skip_comments(l);
        if (lexer_peek(l).type == TOK_RBRACE)
        {
            break;
        }

        // Parse Patterns (with OR and range support)
        // Patterns can be:
        //   - Single value: 1
        //   - OR patterns: 1 || 2 or 1 or 2 or 1, 2
        //   - Range patterns: 1..5 or 1..=5 or 1..<5
        char patterns_buf[1024];
        patterns_buf[0] = 0;
        int pattern_count = 0;

        while (1)
        {
            Token p = lexer_next(l);
            char *p_str = token_strdup(p);

            while (lexer_peek(l).type == TOK_DCOLON)
            {
                lexer_next(l); // eat ::
                Token suffix = lexer_next(l);
                char *tmp = xmalloc(strlen(p_str) + suffix.len + 2);
                // Join with underscore: Result::Ok -> Result_Ok
                sprintf(tmp, "%s_%.*s", p_str, suffix.len, suffix.start);
                free(p_str);
                p_str = tmp;
            }

            // Check for range pattern: value..end, value..<end or value..=end
            if (lexer_peek(l).type == TOK_DOTDOT || lexer_peek(l).type == TOK_DOTDOT_EQ ||
                lexer_peek(l).type == TOK_DOTDOT_LT)
            {
                int is_inclusive = (lexer_peek(l).type == TOK_DOTDOT_EQ);
                lexer_next(l); // eat operator
                Token end_tok = lexer_next(l);
                char *end_str = token_strdup(end_tok);

                // Build range pattern: "start..end" or "start..=end"
                char *range_str = xmalloc(strlen(p_str) + strlen(end_str) + 4);
                sprintf(range_str, "%s%s%s", p_str, is_inclusive ? "..=" : "..", end_str);
                free(p_str);
                free(end_str);
                p_str = range_str;
            }

            if (pattern_count > 0)
            {
                strcat(patterns_buf, "|");
            }
            strcat(patterns_buf, p_str);
            free(p_str);
            pattern_count++;

            // Check for OR continuation: ||, 'or', or comma (legacy)
            Token next = lexer_peek(l);
            skip_comments(l);
            int is_or = (next.type == TOK_OR) ||
                        (next.type == TOK_OP && next.len == 2 && next.start[0] == '|' &&
                         next.start[1] == '|') ||
                        (next.type == TOK_COMMA); // Legacy comma support
            if (is_or)
            {
                lexer_next(l); // eat ||, 'or', or comma
                skip_comments(l);
                continue;
            }
            else
            {
                break;
            }
        }

        char *pattern = xstrdup(patterns_buf);
        int is_default = (strcmp(pattern, "_") == 0);
        int is_destructure = 0;

        // Handle Destructuring: Ok(v) or Rect(w, h)
        char **bindings = NULL;
        int *binding_refs = NULL;
        int binding_count = 0;

        if (!is_default && pattern_count == 1 && lexer_peek(l).type == TOK_LPAREN)
        {
            lexer_next(l); // eat (

            bindings = xmalloc(sizeof(char *) * 8); // hardcap at 8 for now or realloc
            binding_refs = xmalloc(sizeof(int) * 8);

            while (1)
            {
                int is_r = 0;
                // Check for 'ref' keyword
                if (lexer_peek(l).type == TOK_IDENT && lexer_peek(l).len == 3 &&
                    strncmp(lexer_peek(l).start, "ref", 3) == 0)
                {
                    lexer_next(l); // eat 'ref'
                    is_r = 1;
                }

                Token b = lexer_next(l);
                if (b.type != TOK_IDENT)
                {
                    zpanic_at(b, "Expected variable name in pattern");
                }
                bindings[binding_count] = token_strdup(b);
                binding_refs[binding_count] = is_r;
                binding_count++;

                if (lexer_peek(l).type == TOK_COMMA)
                {
                    lexer_next(l);
                    continue;
                }
                break;
            }

            if (lexer_next(l).type != TOK_RPAREN)
            {
                zpanic_at(lexer_peek(l), "Expected )");
            }
            is_destructure = 1;
        }

        // Parse Guard (if condition)
        ASTNode *guard = NULL;
        if (lexer_peek(l).type == TOK_IDENT && strncmp(lexer_peek(l).start, "if", 2) == 0)
        {
            lexer_next(l);
            guard = parse_expression(ctx, l);
            check_assignment_condition(guard);
        }

        if (lexer_next(l).type != TOK_ARROW)
        {
            zpanic_at(lexer_peek(l), "Expected =>");
        }

        // Create scope for the case to hold the binding
        enter_scope(ctx);
        if (binding_count > 0)
        {
            // Try to infer binding type from enum variant payload
            // Look up the enum variant to get its payload type
            EnumVariantReg *vreg = find_enum_variant(ctx, pattern);

            ASTNode *payload_node_field = NULL;
            int is_tuple_payload = 0;
            Type *payload_type = NULL;
            ASTNode *enum_def = NULL;

            if (vreg)
            {
                // Find the enum definition
                enum_def = find_struct_def(ctx, vreg->enum_name);
                if (enum_def && enum_def->type == NODE_ENUM)
                {
                    // Find the specific variant
                    ASTNode *v = enum_def->enm.variants;
                    while (v)
                    {
                        // Match by variant name (pattern suffix after last _)
                        char *v_full =
                            xmalloc(strlen(vreg->enum_name) + strlen(v->variant.name) + 2);
                        sprintf(v_full, "%s_%s", vreg->enum_name, v->variant.name);
                        if (strcmp(v_full, pattern) == 0 && v->variant.payload)
                        {
                            // Found the variant, extract payload type
                            payload_type = v->variant.payload;
                            if (payload_type && payload_type->kind == TYPE_STRUCT &&
                                strncmp(payload_type->name, "Tuple_", 6) == 0)
                            {
                                is_tuple_payload = 1;
                                ASTNode *tuple_def = find_struct_def(ctx, payload_type->name);
                                if (tuple_def)
                                {
                                    payload_node_field = tuple_def->strct.fields;
                                }
                            }
                            free(v_full);
                            break;
                        }
                        v = v->next;
                    }
                }
            }

            for (int i = 0; i < binding_count; i++)
            {
                char *binding = bindings[i];
                int is_ref = binding_refs[i];
                char *binding_type = is_ref ? "void*" : "unknown";
                Type *binding_type_info = NULL; // Default unknown

                if (payload_type)
                {
                    if (binding_count == 1 && !is_tuple_payload)
                    {
                        binding_type = type_to_string(payload_type);
                        binding_type_info = payload_type;
                    }
                    else if (binding_count == 1 && is_tuple_payload)
                    {
                        binding_type = type_to_string(payload_type);
                        binding_type_info = payload_type;
                    }
                    else if (binding_count > 1 && is_tuple_payload)
                    {
                        if (payload_node_field)
                        {
                            Lexer tmp;
                            lexer_init(&tmp, payload_node_field->field.type);
                            binding_type_info = parse_type_formal(ctx, &tmp);
                            binding_type = type_to_string(binding_type_info);
                            payload_node_field = payload_node_field->next;
                        }
                    }
                }

                if (is_ref && binding_type_info)
                {
                    Type *ptr = type_new(TYPE_POINTER);
                    ptr->inner = binding_type_info;
                    binding_type_info = ptr;

                    char *ptr_s = xmalloc(strlen(binding_type) + 2);
                    sprintf(ptr_s, "%s*", binding_type);
                    binding_type = ptr_s;
                }

                int is_generic_unresolved = 0;

                if (enum_def)
                {
                    if (enum_def->enm.generic_param)
                    {
                        char *param = enum_def->enm.generic_param;
                        if (strstr(binding_type, param))
                        {
                            is_generic_unresolved = 1;
                        }
                    }
                }

                if (!is_generic_unresolved &&
                    (strcmp(binding_type, "T") == 0 || strcmp(binding_type, "T*") == 0))
                {
                    is_generic_unresolved = 1;
                }

                if (is_generic_unresolved)
                {
                    if (is_ref)
                    {
                        binding_type = "unknown*";
                        Type *u = type_new(TYPE_UNKNOWN);
                        Type *p = type_new(TYPE_POINTER);
                        p->inner = u;
                        binding_type_info = p;
                    }
                    else
                    {
                        binding_type = "unknown";
                        binding_type_info = type_new(TYPE_UNKNOWN);
                    }
                }

                add_symbol(ctx, binding, binding_type, binding_type_info);
            }
        }

        ASTNode *body;
        Token pk = lexer_peek(l);
        if (pk.type == TOK_LBRACE)
        {
            body = parse_block(ctx, l);
        }
        else if (pk.type == TOK_ASSERT ||
                 (pk.type == TOK_IDENT && strncmp(pk.start, "assert", 6) == 0))
        {
            body = parse_assert(ctx, l);
        }
        else if (pk.type == TOK_IDENT && strncmp(pk.start, "return", 6) == 0)
        {
            body = parse_return(ctx, l);
        }
        else
        {
            body = parse_expression(ctx, l);
        }

        exit_scope(ctx);

        ASTNode *c = ast_create(NODE_MATCH_CASE);
        c->match_case.pattern = pattern;
        c->match_case.binding_names = bindings;
        c->match_case.binding_count = binding_count;
        c->match_case.binding_refs = binding_refs;
        c->match_case.is_destructuring = is_destructure;
        c->match_case.guard = guard;
        c->match_case.body = body;
        c->match_case.is_default = is_default;

        if (!h)
        {
            h = c;
        }
        else
        {
            tl->next = c;
        }
        tl = c;
    }
    lexer_next(l); // eat }

    ASTNode *n = ast_create(NODE_MATCH);
    n->line = start_token.line;
    n->token = start_token; // Capture token for rich warning
    n->match_stmt.expr = expr;
    n->match_stmt.cases = h;
    return n;
}

ASTNode *parse_loop(ParserContext *ctx, Lexer *l)
{
    lexer_next(l);
    ASTNode *b = parse_block(ctx, l);
    ASTNode *n = ast_create(NODE_LOOP);
    n->loop_stmt.body = b;
    return n;
}

ASTNode *parse_repeat(ParserContext *ctx, Lexer *l)
{
    Token t = lexer_next(l);
    zwarn_at(t, "repeat is deprecated. Use 'for _ in 0..N' instead.");
    char *c = rewrite_expr_methods(ctx, parse_condition_raw(ctx, l));
    ASTNode *b = parse_block(ctx, l);
    ASTNode *n = ast_create(NODE_REPEAT);
    n->repeat_stmt.count = c;
    n->repeat_stmt.body = b;
    return n;
}

ASTNode *parse_unless(ParserContext *ctx, Lexer *l)
{
    lexer_next(l);
    ASTNode *cond = parse_expression(ctx, l);
    ASTNode *body = parse_block(ctx, l);
    ASTNode *n = ast_create(NODE_UNLESS);
    n->unless_stmt.condition = cond;
    n->unless_stmt.body = body;
    return n;
}

ASTNode *parse_guard(ParserContext *ctx, Lexer *l)
{
    lexer_next(l); // consume 'guard'

    // Parse the condition as an AST
    ASTNode *cond = parse_expression(ctx, l);

    // Check for 'else'
    Token t = lexer_peek(l);
    if (t.type != TOK_IDENT || strncmp(t.start, "else", 4) != 0)
    {
        zpanic_at(t, "Expected 'else' after guard condition");
    }
    lexer_next(l); // consume 'else'

    // Parse the body - either a block or a single statement
    ASTNode *body;
    if (lexer_peek(l).type == TOK_LBRACE)
    {
        body = parse_block(ctx, l);
    }
    else
    {
        // Single statement (e.g., guard x != NULL else return;)
        body = parse_statement(ctx, l);
    }

    // Create the node
    ASTNode *n = ast_create(NODE_GUARD);
    n->guard_stmt.condition = cond;
    n->guard_stmt.body = body;
    return n;
}

ASTNode *parse_defer(ParserContext *ctx, Lexer *l)
{
    Token defer_token = lexer_next(l); // defer

    // Track that we're parsing inside a defer block
    int prev_in_defer = ctx->in_defer_block;
    ctx->in_defer_block = 1;

    ASTNode *s;
    if (lexer_peek(l).type == TOK_LBRACE)
    {
        s = parse_block(ctx, l);
    }
    else
    {
        s = parse_statement(ctx, l);
    }

    ctx->in_defer_block = prev_in_defer;

    ASTNode *n = ast_create(NODE_DEFER);
    n->token = defer_token;
    n->defer_stmt.stmt = s;
    return n;
}

ASTNode *parse_asm(ParserContext *ctx, Lexer *l)
{
    (void)ctx; // suppress unused parameter warning
    Token t = lexer_peek(l);
    zen_trigger_at(TRIGGER_ASM, t);
    lexer_next(l); // eat 'asm'

    // Check for 'volatile'
    int is_volatile = 0;
    if (lexer_peek(l).type == TOK_VOLATILE)
    {
        is_volatile = 1;
        lexer_next(l);
    }

    // Expect {
    if (lexer_peek(l).type != TOK_LBRACE)
    {
        zpanic_at(lexer_peek(l), "Expected { after asm");
    }
    lexer_next(l);

    // Parse assembly template strings
    char *code = xmalloc(4096); // Buffer for assembly code
    code[0] = 0;

    while (1)
    {
        Token t = lexer_peek(l);

        // Check for end of asm block or start of operands
        if (t.type == TOK_RBRACE)
        {
            break;
        }
        if (t.type == TOK_COLON)
        {
            break;
        }

        // Support string literals for assembly instructions
        if (t.type == TOK_STRING)
        {
            lexer_next(l);
            // Extract string content (strip quotes)
            int str_len = t.len - 2;
            if (strlen(code) > 0)
            {
                strcat(code, "\n");
            }
            strncat(code, t.start + 1, str_len);
        }
        // Also support bare identifiers for simple instructions like 'nop', 'pause'
        else if (t.type == TOK_IDENT)
        {
            lexer_next(l);
            if (strlen(code) > 0)
            {
                strcat(code, "\n");
            }
            strncat(code, t.start, t.len);

            // Check for instruction arguments
            while (lexer_peek(l).type != TOK_RBRACE && lexer_peek(l).type != TOK_COLON)
            {
                Token arg = lexer_peek(l);

                if (arg.type == TOK_SEMICOLON)
                {
                    lexer_next(l);
                    break;
                }

                // Handle substitution {var}
                if (arg.type == TOK_LBRACE)
                {
                    lexer_next(l);
                    strcat(code, "{");
                    // Consume until }
                    while (lexer_peek(l).type != TOK_RBRACE && lexer_peek(l).type != TOK_EOF)
                    {
                        Token sub = lexer_next(l);
                        strncat(code, sub.start, sub.len);
                    }
                    if (lexer_peek(l).type == TOK_RBRACE)
                    {
                        lexer_next(l);
                        strcat(code, "}");
                    }
                    continue;
                }

                if (arg.type == TOK_IDENT)
                {
                    // Check prev char for % or $
                    char last_char = 0;
                    size_t clen = strlen(code);
                    if (clen > 0)
                    {
                        if (code[clen - 1] == ' ' && clen > 1)
                        {
                            last_char = code[clen - 2];
                        }
                        else
                        {
                            last_char = code[clen - 1];
                        }
                    }
                    if (last_char != '%' && last_char != '$' && last_char != ',')
                    {
                        break;
                    }
                }

                lexer_next(l);

                // No space logic
                int no_space = 0;
                size_t clen = strlen(code);
                if (clen > 0)
                {
                    char lc = code[clen - 1];
                    if (lc == '%' || lc == '$')
                    {
                        no_space = 1;
                    }
                }

                if (!no_space)
                {
                    strcat(code, " ");
                }
                strncat(code, arg.start, arg.len);
            }
        }
        else
        {
            zpanic_at(t, "Expected assembly string, instruction, or ':' in asm block");
        }
    }

    // Parse outputs (: out(x), inout(y))
    char **outputs = NULL;
    char **output_modes = NULL;
    int num_outputs = 0;

    if (lexer_peek(l).type == TOK_COLON)
    {
        lexer_next(l); // eat :

        outputs = xmalloc(sizeof(char *) * 16);
        output_modes = xmalloc(sizeof(char *) * 16);

        while (1)
        {
            Token t = lexer_peek(l);
            if (t.type == TOK_COLON || t.type == TOK_RBRACE)
            {
                break;
            }
            if (t.type == TOK_COMMA)
            {
                lexer_next(l);
                continue;
            }

            // Parse out(var) or inout(var)
            if (t.type == TOK_IDENT)
            {
                char *mode = token_strdup(t);
                lexer_next(l);

                if (lexer_peek(l).type != TOK_LPAREN)
                {
                    zpanic_at(lexer_peek(l), "Expected ( after output mode");
                }
                lexer_next(l);

                Token var = lexer_next(l);
                if (var.type != TOK_IDENT)
                {
                    zpanic_at(var, "Expected variable name");
                }

                if (lexer_peek(l).type != TOK_RPAREN)
                {
                    zpanic_at(lexer_peek(l), "Expected ) after variable");
                }
                lexer_next(l);

                outputs[num_outputs] = token_strdup(var);
                output_modes[num_outputs] = mode;
                num_outputs++;
            }
            else
            {
                break;
            }
        }
    }

    // Parse inputs (: in(a), in(b))
    char **inputs = NULL;
    int num_inputs = 0;

    if (lexer_peek(l).type == TOK_COLON)
    {
        lexer_next(l); // eat :

        inputs = xmalloc(sizeof(char *) * 16);

        while (1)
        {
            Token t = lexer_peek(l);
            if (t.type == TOK_COLON || t.type == TOK_RBRACE)
            {
                break;
            }
            if (t.type == TOK_COMMA)
            {
                lexer_next(l);
                continue;
            }

            // Parse in(var)
            if (t.type == TOK_IDENT && strncmp(t.start, "in", 2) == 0)
            {
                lexer_next(l);

                if (lexer_peek(l).type != TOK_LPAREN)
                {
                    zpanic_at(lexer_peek(l), "Expected ( after in");
                }
                lexer_next(l);

                Token var = lexer_next(l);
                if (var.type != TOK_IDENT)
                {
                    zpanic_at(var, "Expected variable name");
                }

                if (lexer_peek(l).type != TOK_RPAREN)
                {
                    zpanic_at(lexer_peek(l), "Expected ) after variable");
                }
                lexer_next(l);

                inputs[num_inputs] = token_strdup(var);
                num_inputs++;
            }
            else
            {
                break;
            }
        }
    }

    // Parse clobbers (: "eax", "memory" OR : clobber("eax"), clobber("memory"))
    char **clobbers = NULL;
    int num_clobbers = 0;

    if (lexer_peek(l).type == TOK_COLON)
    {
        lexer_next(l); // eat :

        clobbers = xmalloc(sizeof(char *) * 16);

        while (1)
        {
            Token t = lexer_peek(l);
            if (t.type == TOK_RBRACE)
            {
                break;
            }
            if (t.type == TOK_COMMA)
            {
                lexer_next(l);
                continue;
            }

            // check for clobber("...")
            if (t.type == TOK_IDENT && strncmp(t.start, "clobber", 7) == 0)
            {
                lexer_next(l); // eat clobber
                if (lexer_peek(l).type != TOK_LPAREN)
                {
                    zpanic_at(lexer_peek(l), "Expected ( after clobber");
                }
                lexer_next(l); // eat (

                Token clob = lexer_next(l);
                if (clob.type != TOK_STRING)
                {
                    zpanic_at(clob, "Expected string literal for clobber");
                }

                if (lexer_peek(l).type != TOK_RPAREN)
                {
                    zpanic_at(lexer_peek(l), "Expected ) after clobber string");
                }
                lexer_next(l); // eat )

                char *c = xmalloc(clob.len);
                strncpy(c, clob.start + 1, clob.len - 2);
                c[clob.len - 2] = 0;
                clobbers[num_clobbers++] = c;
            }
            else
            {
                zpanic_at(t, "Expected 'clobber(\"...\")' in clobber list");
                break;
            }
        }
    }

    // Expect closing }
    if (lexer_peek(l).type != TOK_RBRACE)
    {
        zpanic_at(lexer_peek(l), "Expected } at end of asm block");
    }
    lexer_next(l);

    // Create AST node
    ASTNode *n = ast_create(NODE_ASM);
    n->asm_stmt.code = code;
    n->asm_stmt.is_volatile = is_volatile;
    n->asm_stmt.outputs = outputs;
    n->asm_stmt.output_modes = output_modes;
    n->asm_stmt.inputs = inputs;
    n->asm_stmt.clobbers = clobbers;
    n->asm_stmt.num_outputs = num_outputs;
    n->asm_stmt.num_inputs = num_inputs;
    n->asm_stmt.num_clobbers = num_clobbers;

    return n;
}

ASTNode *parse_test(ParserContext *ctx, Lexer *l)
{
    lexer_next(l); // eat 'test'
    Token t = lexer_next(l);
    if (t.type != TOK_STRING)
    {
        zpanic_at(t, "Test name must be a string literal");
    }

    // Strip quotes for AST storage
    char *name = xmalloc(t.len);
    strncpy(name, t.start + 1, t.len - 2);
    name[t.len - 2] = 0;

    ASTNode *body = parse_block(ctx, l);

    ASTNode *n = ast_create(NODE_TEST);
    n->test_stmt.name = name;
    n->test_stmt.body = body;
    return n;
}

ASTNode *parse_assert(ParserContext *ctx, Lexer *l)
{
    lexer_next(l); // assert
    if (lexer_peek(l).type == TOK_LPAREN)
    {
        lexer_next(l); // optional paren? usually yes
    }

    ASTNode *cond = parse_expression(ctx, l);

    char *msg = NULL;
    if (lexer_peek(l).type == TOK_COMMA)
    {
        lexer_next(l);
        Token st = lexer_next(l);
        if (st.type != TOK_STRING)
        {
            zpanic_at(st, "Expected message string");
        }
        msg = xmalloc(st.len + 1);
        strncpy(msg, st.start, st.len);
        msg[st.len] = 0;
    }

    if (lexer_peek(l).type == TOK_RPAREN)
    {
        lexer_next(l);
    }
    if (lexer_peek(l).type == TOK_SEMICOLON)
    {
        lexer_next(l);
    }

    ASTNode *n = ast_create(NODE_ASSERT);
    n->assert_stmt.condition = cond;
    n->assert_stmt.message = msg;
    return n;
}

ASTNode *parse_return(ParserContext *ctx, Lexer *l)
{
    Token return_token = lexer_next(l); // eat 'return'

    // Error if return is used inside a defer block
    if (ctx->in_defer_block)
    {
        zpanic_at(return_token, "'return' is not allowed inside a 'defer' block");
    }

    ASTNode *n = ast_create(NODE_RETURN);
    n->token = return_token;

    int handled = 0;

    // 1. Check for Tuple Literal Return: return (a, b);
    // Condition: Function returns Tuple_..., starts with '(', and contains ',' at
    // top level
    if (curr_func_ret && strncmp(curr_func_ret, "Tuple_", 6) == 0 &&
        lexer_peek(l).type == TOK_LPAREN)
    {

        // Peek ahead to distinguish "(expr)" from "(a, b)"
        int is_tuple_lit = 0;
        int depth = 0;

        // Just scan tokens manually using a temp lexer to be safe
        Lexer temp_l = *l;

        while (1)
        {
            Token t = lexer_next(&temp_l);
            if (t.type == TOK_EOF)
            {
                break;
            }
            if (t.type == TOK_SEMICOLON)
            {
                break; // Safety break
            }

            if (t.type == TOK_LPAREN)
            {
                depth++;
            }
            if (t.type == TOK_RPAREN)
            {
                depth--;
                if (depth == 0)
                {
                    break; // End of potential tuple
                }
            }

            // If we find a comma at depth 1 (inside the first parens), it's a tuple
            // literal!
            if (depth == 1 && t.type == TOK_COMMA)
            {
                is_tuple_lit = 1;
                break;
            }
        }

        if (is_tuple_lit)
        {
            char *code = parse_tuple_literal(ctx, l, curr_func_ret);
            ASTNode *raw = ast_create(NODE_RAW_STMT);
            raw->raw_stmt.content = code;
            n->ret.value = raw;
            handled = 1;
        }
    }
    // 2. Check for Array Literal Return: return [a, b];
    else if (curr_func_ret && strncmp(curr_func_ret, "Slice_", 6) == 0 &&
             lexer_peek(l).type == TOK_LBRACKET)
    {
        char *code = parse_array_literal(ctx, l, curr_func_ret);
        ASTNode *raw = ast_create(NODE_RAW_STMT);
        raw->raw_stmt.content = code;
        n->ret.value = raw;
        handled = 1;
    }

    // 3. Standard Expression Return
    if (!handled)
    {
        if (lexer_peek(l).type == TOK_SEMICOLON)
        {
            n->ret.value = NULL;
        }
        else
        {
            n->ret.value = parse_expression(ctx, l);
            check_move_usage(ctx, n->ret.value, n->ret.value ? n->ret.value->token : lexer_peek(l));

            // Note: Returning a non-Copy variable effectively moves it out.
            // We could mark it as moved, but scope ends anyway.
            // The critical part is checking we aren't returning an ALREADY moved value.
        }
    }

    if (lexer_peek(l).type == TOK_SEMICOLON)
    {
        lexer_next(l);
    }
    return n;
}

ASTNode *parse_if(ParserContext *ctx, Lexer *l)
{
    lexer_next(l); // eat if
    ASTNode *cond = parse_expression(ctx, l);
    check_assignment_condition(cond);

    ASTNode *then_b = NULL;
    if (lexer_peek(l).type == TOK_LBRACE)
    {
        then_b = parse_block(ctx, l);
    }
    else
    {
        // Single statement: Wrap in scope + block
        enter_scope(ctx);
        ASTNode *s = parse_statement(ctx, l);
        exit_scope(ctx);
        then_b = ast_create(NODE_BLOCK);
        then_b->block.statements = s;
    }

    ASTNode *else_b = NULL;
    skip_comments(l);
    if (lexer_peek(l).type == TOK_IDENT && strncmp(lexer_peek(l).start, "else", 4) == 0)
    {
        lexer_next(l);
        if (lexer_peek(l).type == TOK_IDENT && strncmp(lexer_peek(l).start, "if", 2) == 0)
        {
            else_b = parse_if(ctx, l);
        }
        else if (lexer_peek(l).type == TOK_LBRACE)
        {
            else_b = parse_block(ctx, l);
        }
        else
        {
            // Single statement else
            enter_scope(ctx);
            ASTNode *s = parse_statement(ctx, l);
            exit_scope(ctx);
            else_b = ast_create(NODE_BLOCK);
            else_b->block.statements = s;
        }
    }
    ASTNode *n = ast_create(NODE_IF);
    n->if_stmt.condition = cond;
    n->if_stmt.then_body = then_b;
    n->if_stmt.else_body = else_b;
    return n;
}

ASTNode *parse_while(ParserContext *ctx, Lexer *l)
{
    lexer_next(l);
    ASTNode *cond = parse_expression(ctx, l);
    check_assignment_condition(cond);

    // Zen: While(true)
    if ((cond->type == NODE_EXPR_LITERAL && cond->literal.type_kind == LITERAL_INT &&
         cond->literal.int_val == 1) ||
        (cond->type == NODE_EXPR_VAR && strcmp(cond->var_ref.name, "true") == 0))
    {
        zen_trigger_at(TRIGGER_WHILE_TRUE, cond->token);
    }
    ASTNode *body;
    if (lexer_peek(l).type == TOK_LBRACE)
    {
        body = parse_block(ctx, l);
    }
    else
    {
        body = parse_statement(ctx, l);
    }
    ASTNode *n = ast_create(NODE_WHILE);
    n->while_stmt.condition = cond;
    n->while_stmt.body = body;
    return n;
}

ASTNode *parse_for(ParserContext *ctx, Lexer *l)
{
    lexer_next(l);

    // Range Loop: for i in 0..10
    if (lexer_peek(l).type == TOK_IDENT)
    {
        int saved_pos = l->pos;
        Token var = lexer_next(l);
        Token in_tok = lexer_next(l);

        if (in_tok.type == TOK_IDENT && strncmp(in_tok.start, "in", 2) == 0)
        {
            ASTNode *start_expr = parse_expression(ctx, l);
            // Check for Range Loop (.. or ..= or ..<)
            TokenType next_tok = lexer_peek(l).type;
            if (next_tok == TOK_DOTDOT || next_tok == TOK_DOTDOT_LT || next_tok == TOK_DOTDOT_EQ)
            {
                int is_inclusive = 0;
                if (next_tok == TOK_DOTDOT || next_tok == TOK_DOTDOT_LT)
                {
                    lexer_next(l); // consume .. or ..<
                }
                else if (next_tok == TOK_DOTDOT_EQ)
                {
                    is_inclusive = 1;
                    lexer_next(l); // consume ..=
                }

                if (1) // Block to keep scope for variables
                {
                    ASTNode *end_expr = parse_expression(ctx, l);

                    ASTNode *n = ast_create(NODE_FOR_RANGE);
                    n->for_range.var_name = xmalloc(var.len + 1);
                    strncpy(n->for_range.var_name, var.start, var.len);
                    n->for_range.var_name[var.len] = 0;
                    n->for_range.start = start_expr;
                    n->for_range.end = end_expr;
                    n->for_range.is_inclusive = is_inclusive;

                    if (lexer_peek(l).type == TOK_IDENT &&
                        strncmp(lexer_peek(l).start, "step", 4) == 0)
                    {
                        lexer_next(l);
                        Token s_tok = lexer_next(l);
                        char *sval = xmalloc(s_tok.len + 1);
                        strncpy(sval, s_tok.start, s_tok.len);
                        sval[s_tok.len] = 0;
                        n->for_range.step = sval;
                    }
                    else
                    {
                        n->for_range.step = NULL;
                    }

                    enter_scope(ctx);
                    add_symbol(ctx, n->for_range.var_name, "int", type_new(TYPE_INT));

                    if (lexer_peek(l).type == TOK_LBRACE)
                    {
                        n->for_range.body = parse_block(ctx, l);
                    }
                    else
                    {
                        n->for_range.body = parse_statement(ctx, l);
                    }
                    exit_scope(ctx);

                    return n;
                }
            }
            else
            {
                // Iterator Loop: for x in obj
                // Desugar to:
                /*
                   {
                       var __it = obj.iterator();
                       while (true) {
                           var __opt = __it.next();
                           if (__opt.is_none()) break;
                           var x = __opt.unwrap();
                           <body...>
                       }
                   }
                */

                char *var_name = xmalloc(var.len + 1);
                strncpy(var_name, var.start, var.len);
                var_name[var.len] = 0;

                ASTNode *obj_expr = start_expr;
                char *iter_method = "iterator";
                ASTNode *slice_decl = NULL; // Track if we need to add a slice declaration

                // Check for reference iteration: for x in &vec
                if (obj_expr->type == NODE_EXPR_UNARY && obj_expr->unary.op &&
                    strcmp(obj_expr->unary.op, "&") == 0)
                {
                    obj_expr = obj_expr->unary.operand;
                    iter_method = "iter_ref";
                }

                // Check for array iteration: wrap with Slice<T>::from_array
                if (obj_expr->type_info && obj_expr->type_info->kind == TYPE_ARRAY &&
                    obj_expr->type_info->array_size > 0)
                {
                    // Create a var decl for the slice
                    slice_decl = ast_create(NODE_VAR_DECL);
                    slice_decl->var_decl.name = xstrdup("__zc_arr_slice");

                    // Build type string: Slice<elem_type>
                    char *elem_type_str = type_to_string(obj_expr->type_info->inner);
                    char slice_type[256];
                    sprintf(slice_type, "Slice<%s>", elem_type_str);
                    slice_decl->var_decl.type_str = xstrdup(slice_type);

                    ASTNode *from_array_call = ast_create(NODE_EXPR_CALL);
                    ASTNode *static_method = ast_create(NODE_EXPR_VAR);

                    // The function name for static methods is Type::method format
                    char func_name[512];
                    snprintf(func_name, 511, "%s::from_array", slice_type);
                    static_method->var_ref.name = xstrdup(func_name);

                    from_array_call->call.callee = static_method;

                    // Create arguments
                    ASTNode *arr_addr = ast_create(NODE_EXPR_UNARY);
                    arr_addr->unary.op = xstrdup("&");
                    arr_addr->unary.operand = obj_expr;

                    ASTNode *arr_cast = ast_create(NODE_EXPR_CAST);
                    char cast_type[256];
                    sprintf(cast_type, "%s*", elem_type_str);
                    arr_cast->cast.target_type = xstrdup(cast_type);
                    arr_cast->cast.expr = arr_addr;

                    ASTNode *size_arg = ast_create(NODE_EXPR_LITERAL);
                    size_arg->literal.type_kind = LITERAL_INT;
                    size_arg->literal.int_val = obj_expr->type_info->array_size;
                    char size_buf[32];
                    sprintf(size_buf, "%d", obj_expr->type_info->array_size);
                    size_arg->literal.string_val = xstrdup(size_buf);

                    arr_cast->next = size_arg;
                    from_array_call->call.args = arr_cast;
                    from_array_call->call.arg_count = 2;

                    slice_decl->var_decl.init_expr = from_array_call;

                    // Manually trigger generic instantiation for Slice<T>
                    // This ensures that Slice_int, Slice_float, etc. structures are generated
                    // First, ensure std/slice.zc is imported (auto-import if needed)
                    auto_import_std_slice(ctx);
                    Token dummy_tok = {0};
                    instantiate_generic(ctx, "Slice", elem_type_str, elem_type_str, dummy_tok);

                    // Instantiate SliceIter and Option too for the loop logic
                    char iter_type[256];
                    sprintf(iter_type, "SliceIter<%s>", elem_type_str);
                    instantiate_generic(ctx, "SliceIter", elem_type_str, elem_type_str, dummy_tok);

                    char option_type[256];
                    sprintf(option_type, "Option<%s>", elem_type_str);
                    instantiate_generic(ctx, "Option", elem_type_str, elem_type_str, dummy_tok);

                    // Replace obj_expr with a reference to the slice variable
                    ASTNode *slice_ref = ast_create(NODE_EXPR_VAR);
                    slice_ref->var_ref.name = xstrdup("__zc_arr_slice");
                    slice_ref->resolved_type =
                        xstrdup(slice_type); // Explicitly set type for codegen
                    obj_expr = slice_ref;

                    free(elem_type_str);
                }

                // var __it = obj.iterator();
                ASTNode *it_decl = ast_create(NODE_VAR_DECL);
                it_decl->var_decl.name = xstrdup("__it");
                it_decl->var_decl.type_str = NULL; // inferred

                // obj.iterator() or obj.iter_ref()
                ASTNode *call_iter = ast_create(NODE_EXPR_CALL);
                ASTNode *memb_iter = ast_create(NODE_EXPR_MEMBER);
                memb_iter->member.target = obj_expr;
                memb_iter->member.field = xstrdup(iter_method);
                call_iter->call.callee = memb_iter;
                call_iter->call.args = NULL;
                call_iter->call.arg_count = 0;

                it_decl->var_decl.init_expr = call_iter;

                // while(true)
                ASTNode *while_loop = ast_create(NODE_WHILE);
                ASTNode *true_lit = ast_create(NODE_EXPR_LITERAL);
                true_lit->literal.type_kind = LITERAL_INT; // Treated as bool in conditions
                true_lit->literal.int_val = 1;
                true_lit->literal.string_val = xstrdup("1");
                while_loop->while_stmt.condition = true_lit;

                ASTNode *loop_body = ast_create(NODE_BLOCK);
                ASTNode *stmts_head = NULL;
                ASTNode *stmts_tail = NULL;

#define APPEND_STMT(node)                                                                          \
    if (!stmts_head)                                                                               \
    {                                                                                              \
        stmts_head = node;                                                                         \
        stmts_tail = node;                                                                         \
    }                                                                                              \
    else                                                                                           \
    {                                                                                              \
        stmts_tail->next = node;                                                                   \
        stmts_tail = node;                                                                         \
    }

                char *iter_type_ptr = NULL;
                char *option_type_ptr = NULL;

                if (slice_decl)
                {
                    char *slice_t = slice_decl->var_decl.type_str;
                    char *start = strchr(slice_t, '<');
                    if (start)
                    {
                        char *end = strrchr(slice_t, '>');
                        if (end)
                        {
                            int len = end - start - 1;
                            char *elem = xmalloc(len + 1);
                            strncpy(elem, start + 1, len);
                            elem[len] = 0;

                            iter_type_ptr = xmalloc(256);
                            sprintf(iter_type_ptr, "SliceIter<%s>", elem);

                            option_type_ptr = xmalloc(256);
                            sprintf(option_type_ptr, "Option<%s>", elem);

                            free(elem);
                        }
                    }
                }

                // var __opt = __it.next();
                ASTNode *opt_decl = ast_create(NODE_VAR_DECL);
                opt_decl->var_decl.name = xstrdup("__opt");
                opt_decl->var_decl.type_str = NULL;

                // __it.next()
                ASTNode *call_next = ast_create(NODE_EXPR_CALL);
                ASTNode *memb_next = ast_create(NODE_EXPR_MEMBER);
                ASTNode *it_ref = ast_create(NODE_EXPR_VAR);
                it_ref->var_ref.name = xstrdup("__it");
                if (iter_type_ptr)
                {
                    it_ref->resolved_type = xstrdup(iter_type_ptr);
                }
                memb_next->member.target = it_ref;
                memb_next->member.field = xstrdup("next");
                call_next->call.callee = memb_next;

                opt_decl->var_decl.init_expr = call_next;
                APPEND_STMT(opt_decl);

                // __opt.is_none()
                ASTNode *call_is_none = ast_create(NODE_EXPR_CALL);
                ASTNode *memb_is_none = ast_create(NODE_EXPR_MEMBER);
                ASTNode *opt_ref1 = ast_create(NODE_EXPR_VAR);
                opt_ref1->var_ref.name = xstrdup("__opt");
                if (option_type_ptr)
                {
                    opt_ref1->resolved_type = xstrdup(option_type_ptr);
                }
                memb_is_none->member.target = opt_ref1;
                memb_is_none->member.field = xstrdup("is_none");
                call_is_none->call.callee = memb_is_none;
                call_is_none->call.args = NULL;
                call_is_none->call.arg_count = 0;

                // if (__opt.is_none()) break;
                ASTNode *if_break = ast_create(NODE_IF);
                if_break->if_stmt.condition = call_is_none;
                ASTNode *break_stmt = ast_create(NODE_BREAK);
                if_break->if_stmt.then_body = break_stmt;
                if_break->if_stmt.else_body = NULL;
                APPEND_STMT(if_break);

                // var <user_var> = __opt.unwrap();
                ASTNode *user_var_decl = ast_create(NODE_VAR_DECL);
                user_var_decl->var_decl.name = var_name;
                user_var_decl->var_decl.type_str = NULL;

                // __opt.unwrap()
                ASTNode *call_unwrap = ast_create(NODE_EXPR_CALL);
                ASTNode *memb_unwrap = ast_create(NODE_EXPR_MEMBER);
                ASTNode *opt_ref2 = ast_create(NODE_EXPR_VAR);
                opt_ref2->var_ref.name = xstrdup("__opt");
                if (option_type_ptr)
                {
                    opt_ref2->resolved_type = xstrdup(option_type_ptr);
                }
                memb_unwrap->member.target = opt_ref2;
                memb_unwrap->member.field = xstrdup("unwrap");
                call_unwrap->call.callee = memb_unwrap;
                call_unwrap->call.args = NULL;
                call_unwrap->call.arg_count = 0;

                user_var_decl->var_decl.init_expr = call_unwrap;
                APPEND_STMT(user_var_decl);

                // User body statements
                enter_scope(ctx);
                add_symbol(ctx, var_name, NULL, NULL);

                // Body block
                ASTNode *stmt = parse_statement(ctx, l);
                ASTNode *user_body_node = stmt;
                if (stmt && stmt->type != NODE_BLOCK)
                {
                    ASTNode *blk = ast_create(NODE_BLOCK);
                    blk->block.statements = stmt;
                    user_body_node = blk;
                }
                exit_scope(ctx);

                // Append user body statements to our loop body
                APPEND_STMT(user_body_node);

                loop_body->block.statements = stmts_head;
                while_loop->while_stmt.body = loop_body;

                // Wrap entire thing in a block to scope __it (and __zc_arr_slice if present)
                ASTNode *outer_block = ast_create(NODE_BLOCK);
                if (slice_decl)
                {
                    // Chain: slice_decl -> it_decl -> while_loop
                    slice_decl->next = it_decl;
                    it_decl->next = while_loop;
                    outer_block->block.statements = slice_decl;
                }
                else
                {
                    // Chain: it_decl -> while_loop
                    it_decl->next = while_loop;
                    outer_block->block.statements = it_decl;
                }

                return outer_block;
            }
        }
        l->pos = saved_pos; // Restore
    }

    // C-Style For Loop
    enter_scope(ctx);
    if (lexer_peek(l).type == TOK_LPAREN)
    {
        lexer_next(l);
    }

    ASTNode *init = NULL;
    if (lexer_peek(l).type != TOK_SEMICOLON)
    {
        if (lexer_peek(l).type == TOK_IDENT && strncmp(lexer_peek(l).start, "let", 3) == 0)
        {
            init = parse_var_decl(ctx, l);
        }
        else if (lexer_peek(l).type == TOK_IDENT && strncmp(lexer_peek(l).start, "var", 3) == 0)
        {
            zpanic_at(lexer_peek(l), "'var' is deprecated. Use 'let' instead.");
        }
        else
        {
            init = parse_expression(ctx, l);
            if (lexer_peek(l).type == TOK_SEMICOLON)
            {
                lexer_next(l);
            }
        }
    }
    else
    {
        lexer_next(l);
    }

    ASTNode *cond = NULL;
    if (lexer_peek(l).type != TOK_SEMICOLON)
    {
        cond = parse_expression(ctx, l);
    }
    else
    {
        // Empty condition = true
        ASTNode *true_lit = ast_create(NODE_EXPR_LITERAL);
        true_lit->literal.type_kind = LITERAL_INT;
        true_lit->literal.int_val = 1;
        cond = true_lit;
    }
    if (lexer_peek(l).type == TOK_SEMICOLON)
    {
        lexer_next(l);
    }

    ASTNode *step = NULL;
    if (lexer_peek(l).type != TOK_RPAREN && lexer_peek(l).type != TOK_LBRACE)
    {
        step = parse_expression(ctx, l);
    }

    if (lexer_peek(l).type == TOK_RPAREN)
    {
        lexer_next(l);
    }

    ASTNode *body;
    if (lexer_peek(l).type == TOK_LBRACE)
    {
        body = parse_block(ctx, l);
    }
    else
    {
        body = parse_statement(ctx, l);
    }
    exit_scope(ctx);

    ASTNode *n = ast_create(NODE_FOR);
    n->for_stmt.init = init;
    n->for_stmt.condition = cond;
    n->for_stmt.step = step;
    n->for_stmt.body = body;
    return n;
}

char *process_printf_sugar(ParserContext *ctx, const char *content, int newline, const char *target,
                           char ***used_syms, int *count, int check_symbols)
{
    int saved_silent = ctx->silent_warnings;
    ctx->silent_warnings = !check_symbols;
    char *gen = xmalloc(8192);
    strcpy(gen, "");

    char *s = xstrdup(content);
    char *cur = s;

    while (*cur)
    {
        // 1. Find text before the next '{'
        char *brace = cur;
        while (*brace && *brace != '{')
        {
            brace++;
        }

        if (brace > cur)
        {
            // Append text literal
            char buf[256];
            sprintf(buf, "fprintf(%s, \"%%s\", \"", target);
            strcat(gen, buf);
            strncat(gen, cur, brace - cur);
            strcat(gen, "\"); ");
        }

        if (*brace == 0)
        {
            break;
        }

        // 2. Handle {expression}
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
                if (*(p + 1) == ':')
                {
                    p++;
                }
                else
                {
                    colon = p;
                }
            }
            if (depth == 0)
            {
                break;
            }
            p++;
        }

        *p = 0; // Terminate expression
        char *expr = brace + 1;

        // Unescape \" to " in the expression code to ensure correct parsing
        char *read = expr;
        char *write = expr;
        while (*read)
        {
            if (*read == '\\' && *(read + 1) == '"')
            {
                *write = '"';
                read += 2;
                write++;
            }
            else
            {
                *write = *read;
                read++;
                write++;
            }
        }
        *write = 0;
        char *fmt = NULL;
        if (colon)
        {
            *colon = 0;
            fmt = colon + 1;
        }

        char *clean_expr = expr;
        while (*clean_expr == ' ')
        {
            clean_expr++; // Skip leading spaces
        }

        // Analyze usage & Type Check for to_string()
        char *final_expr = xstrdup(clean_expr);

        if (check_symbols)
        {
            Lexer lex;
            lexer_init(&lex, clean_expr); // Scan original for symbols
            Token t;
            while ((t = lexer_next(&lex)).type != TOK_EOF)
            {
                if (t.type == TOK_IDENT)
                {
                    char *name = token_strdup(t);
                    ZenSymbol *sym = find_symbol_entry(ctx, name);
                    if (sym)
                    {
                        sym->is_used = 1;
                    }

                    if (used_syms && count)
                    {
                        *used_syms = xrealloc(*used_syms, sizeof(char *) * (*count + 1));
                        (*used_syms)[*count] = name;
                        (*count)++;
                    }
                    else
                    {
                        free(name);
                    }
                }
            }
        }

        expr = final_expr;
        clean_expr = final_expr;

        // Parse expression fully
        Lexer lex;
        lexer_init(&lex, clean_expr);

        ASTNode *expr_node = parse_expression(ctx, &lex);

        char *rw_expr = NULL;
        int used_codegen = 0;

        if (expr_node)
        {
            // Check for to_string conversion on struct types
            if (expr_node->type_info)
            {
                Type *t = expr_node->type_info;
                char *struct_name = NULL;
                int is_ptr = 0;

                if (t->kind == TYPE_STRUCT)
                {
                    struct_name = t->name;
                }
                else if (t->kind == TYPE_POINTER && t->inner && t->inner->kind == TYPE_STRUCT)
                {
                    struct_name = t->inner->name;
                    is_ptr = 1;
                }

                if (struct_name)
                {
                    char mangled[256];
                    sprintf(mangled, "%s__to_string", struct_name);
                    if (find_func(ctx, mangled))
                    {
                        char *inner_c = NULL;
                        size_t len = 0;
                        FILE *ms = tmpfile();
                        if (ms)
                        {
                            codegen_expression(ctx, expr_node, ms);
                            len = ftell(ms);
                            fseek(ms, 0, SEEK_SET);
                            inner_c = xmalloc(len + 1);
                            fread(inner_c, 1, len, ms);
                            inner_c[len] = 0;
                            fclose(ms);
                        }

                        if (inner_c)
                        {
                            char *new_expr = xmalloc(strlen(inner_c) + strlen(mangled) + 64);
                            if (is_ptr)
                            {
                                sprintf(new_expr, "%s(%s)", mangled, inner_c);
                            }
                            else
                            {
                                sprintf(new_expr, "%s(({ %s _z_tmp = (%s); &_z_tmp; }))", mangled,
                                        struct_name, inner_c);
                            }
                            rw_expr = new_expr;
                            free(inner_c);
                        }
                    }
                }
            }

            if (!rw_expr)
            {
                char *buf = NULL;
                size_t len = 0;
                FILE *ms = tmpfile();
                if (ms)
                {
                    codegen_expression(ctx, expr_node, ms);
                    len = ftell(ms);
                    fseek(ms, 0, SEEK_SET);
                    buf = xmalloc(len + 1);
                    fread(buf, 1, len, ms);
                    buf[len] = 0;
                    fclose(ms);
                    rw_expr = buf;
                    used_codegen = 1;
                }
            }
        }

        if (!rw_expr)
        {
            rw_expr = xstrdup(expr); // Fallback
        }

        if (fmt)
        {
            // Explicit format: {x:%.2f}
            char buf[128];
            sprintf(buf, "fprintf(%s, \"%%", target);
            strcat(gen, buf);
            strcat(gen, fmt);
            strcat(gen, "\", ");
            strcat(gen, rw_expr); // Use rewritten expr
            strcat(gen, "); ");
        }
        else
        {
            const char *format_spec = NULL;
            Type *t = expr_node ? expr_node->type_info : NULL;
            char *inferred_type = t ? type_to_string(t) : find_symbol_type(ctx, clean_expr);

            int is_bool = 0;
            if (inferred_type)
            {
                if (strcmp(inferred_type, "bool") == 0)
                {
                    format_spec = "%s";
                    is_bool = 1;
                }
                else if (strcmp(inferred_type, "int") == 0 || strcmp(inferred_type, "i32") == 0)
                {
                    format_spec = "%d";
                }
                else if (strcmp(inferred_type, "long") == 0 || strcmp(inferred_type, "i64") == 0 ||
                         strcmp(inferred_type, "isize") == 0)
                {
                    format_spec = "%ld";
                }
                else if (strcmp(inferred_type, "usize") == 0 || strcmp(inferred_type, "u64") == 0)
                {
                    format_spec = "%lu";
                }
                else if (strcmp(inferred_type, "float") == 0 || strcmp(inferred_type, "f32") == 0 ||
                         strcmp(inferred_type, "double") == 0)
                {
                    format_spec = "%f";
                }
                else if (strcmp(inferred_type, "char") == 0 || strcmp(inferred_type, "byte") == 0)
                {
                    format_spec = "%c";
                }
                else if (strcmp(inferred_type, "string") == 0 ||
                         strcmp(inferred_type, "str") == 0 ||
                         (inferred_type[strlen(inferred_type) - 1] == '*' &&
                          strstr(inferred_type, "char")))
                {
                    format_spec = "%s";
                }
                else if (strstr(inferred_type, "*"))
                {
                    format_spec = "%p"; // Pointer
                }
                if (t)
                {
                    free(inferred_type);
                }
            }

            // Check for Literals if variable lookup failed
            if (!format_spec)
            {
                if (isdigit(clean_expr[0]) || clean_expr[0] == '-')
                {
                    format_spec = "%d"; // Naive integer guess (could be float)
                }
                else if (clean_expr[0] == '"')
                {
                    format_spec = "%s";
                }
                else if (clean_expr[0] == '\'')
                {
                    format_spec = "%c";
                }
            }

            if (format_spec)
            {
                char buf[128];
                sprintf(buf, "fprintf(%s, \"", target);
                strcat(gen, buf);
                strcat(gen, format_spec);
                strcat(gen, "\", ");
                if (is_bool)
                {
                    strcat(gen, "_z_bool_str(");
                    strcat(gen, rw_expr);
                    strcat(gen, ")");
                }
                else
                {
                    strcat(gen, rw_expr);
                }
                strcat(gen, "); ");
            }
            else
            {
                // Fallback to runtime macro
                char buf[128];
                sprintf(buf, "fprintf(%s, _z_str(", target);
                strcat(gen, buf);
                strcat(gen, rw_expr);
                strcat(gen, "), _z_arg(");
                strcat(gen, rw_expr);
                strcat(gen, ")); ");
            }
        }

        if (rw_expr && used_codegen)
        {
            free(rw_expr);
        }
        else if (rw_expr && !used_codegen)
        {
            free(rw_expr);
        }

        cur = p + 1;
    }

    if (newline)
    {
        char buf[128];
        sprintf(buf, "fprintf(%s, \"\\n\"); ", target);
        strcat(gen, buf);
    }
    else
    {
        strcat(gen, "fflush(stdout); ");
    }

    strcat(gen, "");

    free(s);
    ctx->silent_warnings = saved_silent;
    return gen;
}

ASTNode *parse_macro_call(ParserContext *ctx, Lexer *l, char *macro_name)
{
    Token start_tok = lexer_peek(l);
    if (lexer_peek(l).type != TOK_OP || lexer_peek(l).start[0] != '!')
    {
        return NULL;
    }
    lexer_next(l); // consume !

    // Expect {
    if (lexer_peek(l).type != TOK_LBRACE)
    {
        zpanic_at(lexer_peek(l), "Expected { after macro invocation");
    }
    lexer_next(l); // consume {

    // Collect body until }
    char *body = xmalloc(8192);
    body[0] = '\0';
    int body_len = 0;
    int depth = 1;
    int last_line = start_tok.line;

    while (depth > 0)
    {
        Token t = lexer_peek(l);
        if (t.type == TOK_EOF)
        {
            zpanic_at(t, "Unexpected EOF in macro block");
        }

        if (t.type == TOK_LBRACE)
        {
            depth++;
        }
        if (t.type == TOK_RBRACE)
        {
            depth--;
        }

        if (depth > 0)
        {
            if (body_len + t.len + 2 < 8192)
            {
                // Preserve newlines
                if (t.line > last_line)
                {
                    body[body_len] = '\n';
                    body[body_len + 1] = 0;
                    body_len++;
                }
                else
                {
                    body[body_len] = ' ';
                    body[body_len + 1] = 0;
                    body_len++;
                }

                strncat(body, t.start, t.len);
                body_len += t.len;
            }
        }

        last_line = t.line;
        lexer_next(l);
    }

    // Resolve plugin name
    const char *plugin_name = resolve_plugin(ctx, macro_name);
    if (!plugin_name)
    {
        char err[256];
        snprintf(err, sizeof(err), "Unknown plugin: %s (did you forget 'import plugin \"%s\"'?)",
                 macro_name, macro_name);
        zpanic_at(start_tok, err);
    }

    // Find Plugin Definition
    // Verify plugin exists
    ZPlugin *found = zptr_find_plugin(plugin_name);

    if (!found)
    {
        char err[256];
        snprintf(err, sizeof(err), "Plugin implementation not found: %s", plugin_name);
        zpanic_at(start_tok, err);
    }

    // Execute Plugin Immediately (Expansion)
    FILE *capture = tmpfile();
    if (!capture)
    {
        zpanic_at(start_tok, "Failed to create capture buffer for plugin expansion");
    }

    ZApi api = {.filename = g_current_filename ? g_current_filename : "input.zc",
                .current_line = start_tok.line,
                .out = capture,
                .hoist_out = ctx->hoist_out};

    found->fn(body, &api);

    // Read captured output
    long len = ftell(capture);
    rewind(capture);
    char *expanded_code = xmalloc(len + 1);
    fread(expanded_code, 1, len, capture);
    expanded_code[len] = 0;
    fclose(capture);
    free(body);

    // Create Raw Statement/Expression Node
    ASTNode *n = ast_create(NODE_RAW_STMT);
    n->line = start_tok.line;
    n->raw_stmt.content = expanded_code;

    return n;
}

ASTNode *parse_statement(ParserContext *ctx, Lexer *l)
{
    Token tk = lexer_peek(l);

    ASTNode *s = NULL;

    if (tk.type == TOK_SEMICOLON)
    {
        lexer_next(l);
        ASTNode *nop = ast_create(NODE_BLOCK); // Empty block as NOP
        nop->block.statements = NULL;
        return nop;
    }

    if (tk.type == TOK_PREPROC)
    {
        lexer_next(l); // consume token
        char *content = xmalloc(tk.len + 2);
        strncpy(content, tk.start, tk.len);
        content[tk.len] = '\n'; // Ensure newline
        content[tk.len + 1] = 0;
        ASTNode *s = ast_create(NODE_RAW_STMT);
        s->raw_stmt.content = content;
        return s;
    }

    if (tk.type == TOK_STRING || tk.type == TOK_FSTRING)
    {
        Lexer lookahead = *l;
        lexer_next(&lookahead);
        TokenType next_type = lexer_peek(&lookahead).type;

        if (next_type == TOK_SEMICOLON || next_type == TOK_DOTDOT || next_type == TOK_RBRACE)
        {
            Token t = lexer_next(l); // consume string

            char *inner = xmalloc(t.len);
            // Strip quotes
            if (t.type == TOK_FSTRING)
            {
                strncpy(inner, t.start + 2, t.len - 3);
                inner[t.len - 3] = 0;
            }
            else
            {
                strncpy(inner, t.start + 1, t.len - 2);
                inner[t.len - 2] = 0;
            }

            int is_ln = (next_type == TOK_SEMICOLON || next_type == TOK_RBRACE);
            char **used_syms = NULL;
            int used_count = 0;
            char *code =
                process_printf_sugar(ctx, inner, is_ln, "stdout", &used_syms, &used_count, 1);

            if (next_type == TOK_SEMICOLON)
            {
                lexer_next(l); // consume ;
            }
            else if (next_type == TOK_DOTDOT)
            {
                lexer_next(l); // consume ..
                if (lexer_peek(l).type == TOK_SEMICOLON)
                {
                    lexer_next(l); // consume optional ;
                }
            }
            // If TOK_RBRACE, do not consume it, so parse_block can see it and terminate loop.

            ASTNode *n = ast_create(NODE_RAW_STMT);
            // Append semicolon to Statement Expression to make it a valid statement
            char *stmt_code = xmalloc(strlen(code) + 2);
            sprintf(stmt_code, "%s;", code);
            free(code);
            n->raw_stmt.content = stmt_code;
            n->raw_stmt.used_symbols = used_syms;
            n->raw_stmt.used_symbol_count = used_count;
            free(inner);
            return n;
        }
    }

    // Block
    if (tk.type == TOK_LBRACE)
    {
        return parse_block(ctx, l);
    }

    // Keywords / Special
    if (tk.type == TOK_TRAIT)
    {
        return parse_trait(ctx, l);
    }
    if (tk.type == TOK_IMPL)
    {
        return parse_impl(ctx, l);
    }
    if (tk.type == TOK_AUTOFREE)
    {
        lexer_next(l);
        if (lexer_peek(l).type != TOK_IDENT || strncmp(lexer_peek(l).start, "let", 3) != 0)
        {
            zpanic_at(lexer_peek(l), "Expected 'let' after autofree");
        }
        s = parse_var_decl(ctx, l);
        s->var_decl.is_autofree = 1;
        // Mark symbol as autofree to suppress unused variable warning
        ZenSymbol *sym = find_symbol_entry(ctx, s->var_decl.name);
        if (sym)
        {
            sym->is_autofree = 1;
        }
        return s;
    }
    if (tk.type == TOK_TEST)
    {
        return parse_test(ctx, l);
    }
    if (tk.type == TOK_COMPTIME)
    {
        char *src = run_comptime_block(ctx, l);
        Lexer new_l;
        lexer_init(&new_l, src);
        ASTNode *head = NULL, *tail = NULL;

        while (lexer_peek(&new_l).type != TOK_EOF)
        {
            ASTNode *s = parse_statement(ctx, &new_l);
            if (!s)
            {
                break;
            }
            if (!head)
            {
                head = s;
            }
            else
            {
                tail->next = s;
            }
            tail = s;
            while (tail->next)
            {
                tail = tail->next;
            }
        }

        if (head && !head->next)
        {
            return head;
        }

        ASTNode *b = ast_create(NODE_BLOCK);
        b->block.statements = head;
        return b;
    }
    if (tk.type == TOK_ASSERT)
    {
        return parse_assert(ctx, l);
    }
    if (tk.type == TOK_DEFER)
    {
        return parse_defer(ctx, l);
    }
    if (tk.type == TOK_ASM)
    {
        return parse_asm(ctx, l);
    }
    if (tk.type == TOK_DEF)
    {
        return parse_def(ctx, l);
    }

    // Identifiers (Keywords or Expressions)
    if (tk.type == TOK_IDENT)
    {
        // Check for macro invocation: identifier! { code }
        Lexer lookahead = *l;
        lexer_next(&lookahead);
        Token exclaim = lexer_peek(&lookahead);
        lexer_next(&lookahead);
        Token lbrace = lexer_peek(&lookahead);
        if (exclaim.type == TOK_OP && exclaim.len == 1 && exclaim.start[0] == '!' &&
            lbrace.type == TOK_LBRACE)
        {
            // This is a macro invocation
            char *macro_name = token_strdup(tk);
            lexer_next(l); // consume identifier

            ASTNode *n = parse_macro_call(ctx, l, macro_name);
            free(macro_name);
            return n;
        }

        // Check for raw blocks
        if (strncmp(tk.start, "raw", 3) == 0 && tk.len == 3)
        {
            lexer_next(l); // eat raw
            if (lexer_peek(l).type != TOK_LBRACE)
            {
                zpanic_at(lexer_peek(l), "Expected { after raw");
            }
            lexer_next(l); // eat {

            const char *start = l->src + l->pos;
            int depth = 1;
            while (depth > 0)
            {
                Token t = lexer_next(l);
                if (t.type == TOK_EOF)
                {
                    zpanic_at(t, "Unexpected EOF in raw block");
                }
                if (t.type == TOK_LBRACE)
                {
                    depth++;
                }
                if (t.type == TOK_RBRACE)
                {
                    depth--;
                }
            }
            const char *end = l->src + l->pos - 1;
            size_t len = end - start;

            char *content = xmalloc(len + 1);
            memcpy(content, start, len);
            content[len] = 0;

            ASTNode *s = ast_create(NODE_RAW_STMT);
            s->raw_stmt.content = content;
            return s;
        }

        // Check for plugin blocks
        if (strncmp(tk.start, "plugin", 6) == 0 && tk.len == 6)
        {
            lexer_next(l); // consume 'plugin'
            return parse_plugin(ctx, l);
        }

        if (strncmp(tk.start, "let", 3) == 0 && tk.len == 3)
        {
            return parse_var_decl(ctx, l);
        }

        if (strncmp(tk.start, "var", 3) == 0 && tk.len == 3)
        {
            zpanic_at(tk, "'var' is deprecated. Use 'let' instead.");
        }

        // Static local variable: static let x = 0;
        if (strncmp(tk.start, "static", 6) == 0 && tk.len == 6)
        {
            lexer_next(l); // eat 'static'
            Token next = lexer_peek(l);
            if (strncmp(next.start, "let", 3) == 0 && next.len == 3)
            {
                ASTNode *v = parse_var_decl(ctx, l);
                v->var_decl.is_static = 1;
                return v;
            }
            zpanic_at(next, "Expected 'let' after 'static'");
        }

        if (strncmp(tk.start, "const", 5) == 0 && tk.len == 5)
        {
            zpanic_at(tk, "'const' for declarations is deprecated. Use 'def' for constants or 'let "
                          "x: const T' for read-only variables.");
        }
        if (strncmp(tk.start, "return", 6) == 0 && tk.len == 6)
        {
            return parse_return(ctx, l);
        }
        if (strncmp(tk.start, "if", 2) == 0 && tk.len == 2)
        {
            return parse_if(ctx, l);
        }
        if (strncmp(tk.start, "while", 5) == 0 && tk.len == 5)
        {
            return parse_while(ctx, l);
        }
        if (strncmp(tk.start, "for", 3) == 0 && tk.len == 3)
        {
            return parse_for(ctx, l);
        }
        if (strncmp(tk.start, "match", 5) == 0 && tk.len == 5)
        {
            return parse_match(ctx, l);
        }

        // Break with optional label: break; or break 'outer;
        if (strncmp(tk.start, "break", 5) == 0 && tk.len == 5)
        {
            Token break_token = lexer_next(l);

            // Error if break is used inside a defer block
            if (ctx->in_defer_block)
            {
                zpanic_at(break_token, "'break' is not allowed inside a 'defer' block");
            }

            ASTNode *n = ast_create(NODE_BREAK);
            n->token = break_token;
            n->break_stmt.target_label = NULL;
            // Check for 'label
            if (lexer_peek(l).type == TOK_CHAR)
            {
                Token label_tok = lexer_next(l);
                // Extract label name (strip quotes)
                char *label = xmalloc(label_tok.len);
                strncpy(label, label_tok.start + 1, label_tok.len - 2);
                label[label_tok.len - 2] = 0;
                n->break_stmt.target_label = label;
            }
            if (lexer_peek(l).type == TOK_SEMICOLON)
            {
                lexer_next(l);
            }
            return n;
        }

        // Continue with optional label
        if (strncmp(tk.start, "continue", 8) == 0 && tk.len == 8)
        {
            Token continue_token = lexer_next(l);

            // Error if continue is used inside a defer block
            if (ctx->in_defer_block)
            {
                zpanic_at(continue_token, "'continue' is not allowed inside a 'defer' block");
            }

            ASTNode *n = ast_create(NODE_CONTINUE);
            n->token = continue_token;
            n->continue_stmt.target_label = NULL;
            if (lexer_peek(l).type == TOK_CHAR)
            {
                Token label_tok = lexer_next(l);
                char *label = xmalloc(label_tok.len);
                strncpy(label, label_tok.start + 1, label_tok.len - 2);
                label[label_tok.len - 2] = 0;
                n->continue_stmt.target_label = label;
            }
            if (lexer_peek(l).type == TOK_SEMICOLON)
            {
                lexer_next(l);
            }
            return n;
        }

        if (strncmp(tk.start, "loop", 4) == 0 && tk.len == 4)
        {
            return parse_loop(ctx, l);
        }
        if (strncmp(tk.start, "repeat", 6) == 0 && tk.len == 6)
        {
            return parse_repeat(ctx, l);
        }
        if (strncmp(tk.start, "unless", 6) == 0 && tk.len == 6)
        {
            return parse_unless(ctx, l);
        }
        if (strncmp(tk.start, "guard", 5) == 0 && tk.len == 5)
        {
            return parse_guard(ctx, l);
        }

        // CUDA launch: launch kernel(args) with { grid: X, block: Y };
        if (strncmp(tk.start, "launch", 6) == 0 && tk.len == 6)
        {
            Token launch_tok = lexer_next(l); // eat 'launch'

            // Parse the kernel call expression
            ASTNode *call = parse_expression(ctx, l);
            if (!call || call->type != NODE_EXPR_CALL)
            {
                zpanic_at(launch_tok, "Expected kernel call after 'launch'");
            }

            // Expect 'with'
            Token with_tok = lexer_peek(l);
            if (with_tok.type != TOK_IDENT || strncmp(with_tok.start, "with", 4) != 0 ||
                with_tok.len != 4)
            {
                zpanic_at(with_tok, "Expected 'with' after kernel call in launch statement");
            }
            lexer_next(l); // eat 'with'

            // Expect '{' for configuration block
            if (lexer_peek(l).type != TOK_LBRACE)
            {
                zpanic_at(lexer_peek(l), "Expected '{' after 'with' in launch statement");
            }
            lexer_next(l); // eat '{'

            ASTNode *grid = NULL;
            ASTNode *block = NULL;
            ASTNode *shared_mem = NULL;
            ASTNode *stream = NULL;

            // Parse configuration fields
            while (lexer_peek(l).type != TOK_RBRACE && lexer_peek(l).type != TOK_EOF)
            {
                Token field_name = lexer_next(l);
                if (field_name.type != TOK_IDENT)
                {
                    zpanic_at(field_name, "Expected field name in launch configuration");
                }

                // Expect ':'
                if (lexer_peek(l).type != TOK_COLON)
                {
                    zpanic_at(lexer_peek(l), "Expected ':' after field name");
                }
                lexer_next(l); // eat ':'

                // Parse value expression
                ASTNode *value = parse_expression(ctx, l);

                // Assign to appropriate field
                if (strncmp(field_name.start, "grid", 4) == 0 && field_name.len == 4)
                {
                    grid = value;
                }
                else if (strncmp(field_name.start, "block", 5) == 0 && field_name.len == 5)
                {
                    block = value;
                }
                else if (strncmp(field_name.start, "shared_mem", 10) == 0 && field_name.len == 10)
                {
                    shared_mem = value;
                }
                else if (strncmp(field_name.start, "stream", 6) == 0 && field_name.len == 6)
                {
                    stream = value;
                }
                else
                {
                    zpanic_at(field_name, "Unknown launch configuration field (expected: grid, "
                                          "block, shared_mem, stream)");
                }

                // Optional comma
                if (lexer_peek(l).type == TOK_COMMA)
                {
                    lexer_next(l);
                }
            }

            // Expect '}'
            if (lexer_peek(l).type != TOK_RBRACE)
            {
                zpanic_at(lexer_peek(l), "Expected '}' to close launch configuration");
            }
            lexer_next(l); // eat '}'

            // Expect ';'
            if (lexer_peek(l).type == TOK_SEMICOLON)
            {
                lexer_next(l);
            }

            // Require at least grid and block
            if (!grid || !block)
            {
                zpanic_at(launch_tok, "Launch configuration requires at least 'grid' and 'block'");
            }

            ASTNode *n = ast_create(NODE_CUDA_LAUNCH);
            n->cuda_launch.call = call;
            n->cuda_launch.grid = grid;
            n->cuda_launch.block = block;
            n->cuda_launch.shared_mem = shared_mem;
            n->cuda_launch.stream = stream;
            n->token = launch_tok;
            return n;
        }

        // Do-while loop: do { body } while condition;
        if (strncmp(tk.start, "do", 2) == 0 && tk.len == 2)
        {
            lexer_next(l); // eat 'do'
            ASTNode *body = parse_block(ctx, l);

            // Expect 'while'
            Token while_tok = lexer_peek(l);
            if (while_tok.type != TOK_IDENT || strncmp(while_tok.start, "while", 5) != 0 ||
                while_tok.len != 5)
            {
                zpanic_at(while_tok, "Expected 'while' after do block");
            }
            lexer_next(l); // eat 'while'

            ASTNode *cond = parse_expression(ctx, l);
            if (lexer_peek(l).type == TOK_SEMICOLON)
            {
                lexer_next(l);
            }

            ASTNode *n = ast_create(NODE_DO_WHILE);
            n->do_while_stmt.body = body;
            n->do_while_stmt.condition = cond;
            n->do_while_stmt.loop_label = NULL;
            return n;
        }

        if (strncmp(tk.start, "defer", 5) == 0 && tk.len == 5)
        {
            return parse_defer(ctx, l);
        }

        // Goto statement: goto label_name; OR goto *expr; (computed goto)
        if (strncmp(tk.start, "goto", 4) == 0 && tk.len == 4)
        {
            Token goto_tok = lexer_next(l); // eat 'goto'

            // Error if goto is used inside a defer block
            if (ctx->in_defer_block)
            {
                zpanic_at(goto_tok, "'goto' is not allowed inside a 'defer' block");
            }

            Token next = lexer_peek(l);

            // Computed goto: goto *ptr;
            if (next.type == TOK_OP && next.start[0] == '*')
            {
                lexer_next(l); // eat '*'
                ASTNode *target = parse_expression(ctx, l);
                if (lexer_peek(l).type == TOK_SEMICOLON)
                {
                    lexer_next(l);
                }

                ASTNode *n = ast_create(NODE_GOTO);
                n->goto_stmt.label_name = NULL;
                n->goto_stmt.goto_expr = target;
                n->token = goto_tok;
                return n;
            }

            // Regular goto
            Token label = lexer_next(l);
            if (label.type != TOK_IDENT)
            {
                zpanic_at(label, "Expected label name after goto");
            }
            if (lexer_peek(l).type == TOK_SEMICOLON)
            {
                lexer_next(l);
            }
            ASTNode *n = ast_create(NODE_GOTO);
            n->goto_stmt.label_name = token_strdup(label);
            n->token = goto_tok;
            zen_trigger_at(TRIGGER_GOTO, goto_tok);
            return n;
        }

        // Label detection: identifier followed by : (but not ::)
        {
            Lexer lookahead = *l;
            Token ident = lexer_next(&lookahead);
            Token maybe_colon = lexer_peek(&lookahead);
            if (maybe_colon.type == TOK_COLON)
            {
                // Check it's not :: (double colon for namespaces)
                lexer_next(&lookahead);
                Token after_colon = lexer_peek(&lookahead);
                if (after_colon.type != TOK_COLON)
                {
                    // This is a label!
                    lexer_next(l); // eat identifier
                    lexer_next(l); // eat :
                    ASTNode *n = ast_create(NODE_LABEL);
                    n->label_stmt.label_name = token_strdup(ident);
                    n->token = ident;
                    return n;
                }
            }
        }

        if ((strncmp(tk.start, "print", 5) == 0 && tk.len == 5) ||
            (strncmp(tk.start, "println", 7) == 0 && tk.len == 7) ||
            (strncmp(tk.start, "eprint", 6) == 0 && tk.len == 6) ||
            (strncmp(tk.start, "eprintln", 8) == 0 && tk.len == 8))
        {

            // Revert: User requested print without newline
            int is_ln = (tk.len == 7 || tk.len == 8);
            // int is_ln = (tk.len == 7 || tk.len == 8);
            int is_err = (tk.start[0] == 'e');
            char *target = is_err ? "stderr" : "stdout";

            lexer_next(l); // eat keyword

            Token t = lexer_next(l);
            if (t.type != TOK_STRING && t.type != TOK_FSTRING)
            {
                zpanic_at(t, "Expected string literal after print/eprint");
            }

            char *inner = xmalloc(t.len);
            if (t.type == TOK_FSTRING)
            {
                strncpy(inner, t.start + 2, t.len - 3);
                inner[t.len - 3] = 0;
            }
            else
            {
                strncpy(inner, t.start + 1, t.len - 2);
                inner[t.len - 2] = 0;
            }

            char **used_syms = NULL;
            int used_count = 0;
            char *code =
                process_printf_sugar(ctx, inner, is_ln, target, &used_syms, &used_count, 1);
            free(inner);

            if (lexer_peek(l).type == TOK_SEMICOLON)
            {
                lexer_next(l);
            }

            ASTNode *n = ast_create(NODE_RAW_STMT);
            // Append semicolon to Statement Expression to make it a valid statement
            char *stmt_code = xmalloc(strlen(code) + 2);
            sprintf(stmt_code, "%s;", code);
            free(code);
            n->raw_stmt.content = stmt_code;
            n->raw_stmt.used_symbols = used_syms;
            n->raw_stmt.used_symbol_count = used_count;
            return n;
        }
    }

    // Default: Expression Statement
    s = parse_expression(ctx, l);

    int has_semi = 0;
    if (lexer_peek(l).type == TOK_SEMICOLON)
    {
        lexer_next(l);
        has_semi = 1;
    }

    // Auto-print in REPL: If no semicolon (implicit expr at block end)
    // and not an assignment, print it.
    if (ctx->is_repl && s && !has_semi)
    {
        int is_assign = 0;
        if (s->type == NODE_EXPR_BINARY)
        {
            char *op = s->binary.op;
            if (strcmp(op, "=") == 0 ||
                (strlen(op) > 1 && op[strlen(op) - 1] == '=' && strcmp(op, "==") != 0 &&
                 strcmp(op, "!=") != 0 && strcmp(op, "<=") != 0 && strcmp(op, ">=") != 0))
            {
                is_assign = 1;
            }
        }

        if (!is_assign)
        {
            ASTNode *print_node = ast_create(NODE_REPL_PRINT);
            print_node->repl_print.expr = s;
            // Preserve line info
            print_node->line = s->line;
            print_node->token = s->token;
            return print_node;
        }
    }

    if (s)
    {
        s->line = tk.line;
    }

    // Check for discarded must_use result
    if (s && s->type == NODE_EXPR_CALL)
    {
        ASTNode *callee = s->call.callee;
        if (callee && callee->type == NODE_EXPR_VAR)
        {
            FuncSig *sig = find_func(ctx, callee->var_ref.name);
            if (sig && sig->must_use)
            {
                zwarn_at(tk, "Ignoring return value of function marked @must_use");
                fprintf(stderr, COLOR_CYAN "   = note: " COLOR_RESET
                                           "Use the result or explicitly discard with `_ = ...`\n");
            }
        }
    }

    return s;
}

ASTNode *parse_block(ParserContext *ctx, Lexer *l)
{
    lexer_next(l); // eat '{'
    enter_scope(ctx);
    ASTNode *head = 0, *tail = 0;

    int unreachable = 0;

    while (1)
    {
        skip_comments(l);
        Token tk = lexer_peek(l);
        if (tk.type == TOK_RBRACE)
        {
            lexer_next(l);
            break;
        }

        if (tk.type == TOK_EOF)
        {
            break;
        }

        if (unreachable == 1)
        {
            warn_unreachable_code(tk);
            unreachable = 2; // Warned once, don't spam
        }

        if (tk.type == TOK_COMPTIME)
        {
            // lexer_next(l); // don't eat here, run_comptime_block expects it
            char *src = run_comptime_block(ctx, l);
            Lexer new_l;
            lexer_init(&new_l, src);
            // Parse statements from the generated source
            while (lexer_peek(&new_l).type != TOK_EOF)
            {
                ASTNode *s = parse_statement(ctx, &new_l);
                if (!s)
                {
                    break; // EOF or error handling dependency
                }

                // Link
                if (!head)
                {
                    head = s;
                }
                else
                {
                    tail->next = s;
                }
                tail = s;
                while (tail->next)
                {
                    tail = tail->next;
                }
            }
            continue;
        }

        ASTNode *s = parse_statement(ctx, l);
        if (s)
        {
            if (!head)
            {
                head = s;
            }
            else
            {
                tail->next = s;
            }
            tail = s;
            while (tail->next)
            {
                tail = tail->next; // Handle chains (e.g. var decl + defer)
            }

            // Check for control flow interruption
            if (s->type == NODE_RETURN || s->type == NODE_BREAK || s->type == NODE_CONTINUE)
            {
                if (unreachable == 0)
                {
                    unreachable = 1;
                }
            }
        }
    }

    // Check for unused variables in this block scope
    if (ctx->current_scope && !ctx->is_repl)
    {
        ZenSymbol *sym = ctx->current_scope->symbols;
        while (sym)
        {
            // Skip special names and already warned
            if (!sym->is_used && sym->name[0] != '_' && strcmp(sym->name, "it") != 0 &&
                strcmp(sym->name, "self") != 0)
            {
                // Skip autofree variables (used implicitly for cleanup)
                if (sym->is_autofree)
                {
                    sym = sym->next;
                    continue;
                }

                // RAII: Don't warn if type implements Drop (it is used implicitly)
                int has_drop = (sym->type_info && sym->type_info->traits.has_drop);
                if (!has_drop && sym->type_info && sym->type_info->name)
                {
                    ASTNode *def = find_struct_def(ctx, sym->type_info->name);
                    if (def && def->type_info && def->type_info->traits.has_drop)
                    {
                        has_drop = 1;
                    }
                }

                if (!has_drop)
                {
                    warn_unused_variable(sym->decl_token, sym->name);
                }
            }
            sym = sym->next;
        }
    }

    exit_scope(ctx);
    ASTNode *b = ast_create(NODE_BLOCK);
    b->block.statements = head;
    return b;
}

ASTNode *parse_include(ParserContext *ctx, Lexer *l)
{
    lexer_next(l); // eat 'include'
    Token t = lexer_next(l);
    char *path = NULL;
    int is_system = 0;

    if (t.type == TOK_LANGLE)
    {
        // System include: include <raylib.h>
        is_system = 1;
        char buf[256];
        buf[0] = 0;
        while (1)
        {
            Token i = lexer_next(l);
            if (i.type == TOK_RANGLE)
            {
                break;
            }
            strncat(buf, i.start, i.len);
        }
        path = xstrdup(buf);

        // Mark that this file has external includes (suppress undefined warnings)
        ctx->has_external_includes = 1;
    }
    else
    {
        // Local include: include "file.h"
        is_system = 0;
        int len = t.len - 2;
        path = xmalloc(len + 1);
        strncpy(path, t.start + 1, len);
        path[len] = 0;
    }

    ASTNode *n = ast_create(NODE_INCLUDE);
    n->include.path = path;
    n->include.is_system = is_system;
    return n;
}
ASTNode *parse_import(ParserContext *ctx, Lexer *l)
{
    lexer_next(l); // eat 'import'

    // Check for 'plugin' keyword
    Token next = lexer_peek(l);
    if (next.type == TOK_IDENT && next.len == 6 && strncmp(next.start, "plugin", 6) == 0)
    {
        lexer_next(l); // consume "plugin"

        // Expect string literal with plugin name
        Token plugin_tok = lexer_next(l);
        if (plugin_tok.type != TOK_STRING)
        {
            zpanic_at(plugin_tok, "Expected string literal after 'import plugin'");
        }

        // Extract plugin name (strip quotes)
        int name_len = plugin_tok.len - 2;
        char *plugin_name = xmalloc(name_len + 1);
        strncpy(plugin_name, plugin_tok.start + 1, name_len);
        plugin_name[name_len] = '\0';

        if (plugin_name[0] == '.' &&
            (plugin_name[1] == '/' || (plugin_name[1] == '.' && plugin_name[2] == '/')))
        {
            char *current_dir = xstrdup(g_current_filename);
            char *last_slash = strrchr(current_dir, '/');
            if (last_slash)
            {
                *last_slash = 0;
                char resolved_path[1024];
                snprintf(resolved_path, sizeof(resolved_path), "%s/%s", current_dir, plugin_name);
                free(plugin_name);
                plugin_name = xstrdup(resolved_path);
            }
            free(current_dir);
        }

        // Check for optional "as alias"
        char *alias = NULL;
        Token as_tok = lexer_peek(l);
        if (as_tok.type == TOK_IDENT && as_tok.len == 2 && strncmp(as_tok.start, "as", 2) == 0)
        {
            lexer_next(l); // consume "as"
            Token alias_tok = lexer_next(l);
            if (alias_tok.type != TOK_IDENT)
            {
                zpanic_at(alias_tok, "Expected identifier after 'as'");
            }
            alias = token_strdup(alias_tok);
        }

        // Register the plugin
        register_plugin(ctx, plugin_name, alias);

        // Consume optional semicolon
        if (lexer_peek(l).type == TOK_SEMICOLON)
        {
            lexer_next(l);
        }

        // Return NULL - no AST node needed for imports
        return NULL;
    }

    // Regular module import handling follows...
    // Check if this is selective import: import { ... } from "file"
    int is_selective = 0;
    char *symbols[32]; // Max 32 selective imports
    char *aliases[32];
    int symbol_count = 0;

    if (lexer_peek(l).type == TOK_LBRACE)
    {
        is_selective = 1;
        lexer_next(l); // eat {

        // Parse symbol list
        while (lexer_peek(l).type != TOK_RBRACE)
        {
            if (symbol_count > 0 && lexer_peek(l).type == TOK_COMMA)
            {
                lexer_next(l); // eat comma
            }

            Token sym_tok = lexer_next(l);
            if (sym_tok.type != TOK_IDENT)
            {
                zpanic_at(sym_tok, "Expected identifier in selective import");
            }

            symbols[symbol_count] = xmalloc(sym_tok.len + 1);
            strncpy(symbols[symbol_count], sym_tok.start, sym_tok.len);
            symbols[symbol_count][sym_tok.len] = 0;

            // Check for 'as alias'
            Token next = lexer_peek(l);
            if (next.type == TOK_IDENT && next.len == 2 && strncmp(next.start, "as", 2) == 0)
            {
                lexer_next(l); // eat 'as'
                Token alias_tok = lexer_next(l);
                if (alias_tok.type != TOK_IDENT)
                {
                    zpanic_at(alias_tok, "Expected identifier after 'as'");
                }

                aliases[symbol_count] = xmalloc(alias_tok.len + 1);
                strncpy(aliases[symbol_count], alias_tok.start, alias_tok.len);
                aliases[symbol_count][alias_tok.len] = 0;
            }
            else
            {
                aliases[symbol_count] = NULL; // No alias
            }

            symbol_count++;
        }

        lexer_next(l); // eat }

        // Expect 'from'
        Token from_tok = lexer_next(l);
        if (from_tok.type != TOK_IDENT || from_tok.len != 4 ||
            strncmp(from_tok.start, "from", 4) != 0)
        {
            zpanic_at(from_tok, "Expected 'from' after selective import list, got type=%d",
                      from_tok.type);
        }
    }

    // Parse filename
    Token t = lexer_next(l);
    if (t.type != TOK_STRING)
    {
        zpanic_at(t,
                  "Expected string (filename) after 'from' in selective import, got "
                  "type %d",
                  t.type);
    }
    int ln = t.len - 2; // Remove quotes
    char *fn = xmalloc(ln + 1);
    strncpy(fn, t.start + 1, ln);
    fn[ln] = 0;

    // Resolve paths relative to current file
    char resolved_path[1024];
    int is_explicit_relative = (fn[0] == '.' && (fn[1] == '/' || (fn[1] == '.' && fn[2] == '/')));

    // Try to resolve relative to current file if not absolute
    if (fn[0] != '/')
    {
        char *current_dir = xstrdup(g_current_filename);
        char *last_slash = strrchr(current_dir, '/');
        if (last_slash)
        {
            *last_slash = 0; // Truncate to directory

            // Handle explicit relative differently?
            // Existing logic enforced it. Let's try to verify existence first.

            // Construct candidate path
            const char *leaf = fn;
            // Clean up ./ prefix for cleaner path construction if we want
            // but keeping it is fine too, /path/to/./file works.

            snprintf(resolved_path, sizeof(resolved_path), "%s/%s", current_dir, leaf);

            // If it's an explicit relative path, OR if the file exists at this relative location
            if (is_explicit_relative || access(resolved_path, R_OK) == 0)
            {
                free(fn);
                fn = xstrdup(resolved_path);
            }
        }
        free(current_dir);
    }

    // Check if file exists, if not try system-wide paths
    if (access(fn, R_OK) != 0)
    {
        // Try system-wide standard library location
        static const char *system_paths[] = {"/usr/local/share/zenc", "/usr/share/zenc", NULL};

        char system_path[1024];
        int found = 0;

        for (int i = 0; system_paths[i] && !found; i++)
        {
            snprintf(system_path, sizeof(system_path), "%s/%s", system_paths[i], fn);
            if (access(system_path, R_OK) == 0)
            {
                free(fn);
                fn = xstrdup(system_path);
                found = 1;
            }
        }

        if (!found)
        {
            // File not found anywhere - will error later when trying to open
        }
    }

    // Canonicalize path to avoid duplicates (for example: "./std/io.zc" vs "std/io.zc")
    char *real_fn = realpath(fn, NULL);
    if (real_fn)
    {
        free(fn);
        fn = real_fn;
    }

    // Check if file already imported
    if (is_file_imported(ctx, fn))
    {
        free(fn);
        return NULL;
    }
    mark_file_imported(ctx, fn);

    // For selective imports, register them BEFORE parsing the file
    char *module_base_name = NULL;
    if (is_selective)
    {
        module_base_name = extract_module_name(fn);
        for (int i = 0; i < symbol_count; i++)
        {
            register_selective_import(ctx, symbols[i], aliases[i], module_base_name);
        }
    }

    // Check for 'as alias' syntax (for namespaced imports)
    char *alias = NULL;
    if (!is_selective)
    {
        Token next_tok = lexer_peek(l);
        if (next_tok.type == TOK_IDENT && next_tok.len == 2 &&
            strncmp(next_tok.start, "as", 2) == 0)
        {
            lexer_next(l); // eat 'as'
            Token alias_tok = lexer_next(l);
            if (alias_tok.type != TOK_IDENT)
            {
                zpanic_at(alias_tok, "Expected identifier after 'as'");
            }

            alias = xmalloc(alias_tok.len + 1);
            strncpy(alias, alias_tok.start, alias_tok.len);
            alias[alias_tok.len] = 0;

            // Register the module

            // Check if C header
            int is_header = 0;
            if (strlen(fn) > 2 && strcmp(fn + strlen(fn) - 2, ".h") == 0)
            {
                is_header = 1;
            }

            // Register the module
            Module *m = xmalloc(sizeof(Module));
            m->alias = xstrdup(alias);
            m->path = xstrdup(fn);
            m->base_name = extract_module_name(fn);
            m->is_c_header = is_header;
            m->next = ctx->modules;
            ctx->modules = m;
        }
    }

    // C Header: Emit include and return (don't parse)
    if (strlen(fn) > 2 && strcmp(fn + strlen(fn) - 2, ".h") == 0)
    {
        ASTNode *n = ast_create(NODE_INCLUDE);
        n->include.path = xstrdup(fn); // Store exact path
        n->include.is_system = 0;      // Double quotes
        return n;
    }

    // Load and parse the file
    char *src = load_file(fn);
    if (!src)
    {
        zpanic_at(t, "Not found: %s", fn);
    }

    Lexer i;
    lexer_init(&i, src);

    // If this is a namespaced import or selective import, set the module prefix
    char *prev_module_prefix = ctx->current_module_prefix;
    char *temp_module_prefix = NULL;

    if (alias)
    { // For 'import "file" as alias'
        temp_module_prefix = extract_module_name(fn);
        ctx->current_module_prefix = temp_module_prefix;
    }
    else if (is_selective)
    { // For 'import {sym} from "file"'
        temp_module_prefix = extract_module_name(fn);
        ctx->current_module_prefix = temp_module_prefix;
    }

    // Update global filename context for relative imports inside the new file
    const char *saved_fn = g_current_filename;
    g_current_filename = fn;

    ASTNode *r = parse_program_nodes(ctx, &i);

    // Restore filename context
    g_current_filename = (char *)saved_fn;

    // Restore previous module context
    if (temp_module_prefix)
    {
        free(temp_module_prefix);
        ctx->current_module_prefix = prev_module_prefix;
    }

    // Free selective import symbols and aliases
    if (is_selective)
    {
        for (int k = 0; k < symbol_count; k++)
        {
            free(symbols[k]);
            if (aliases[k])
            {
                free(aliases[k]);
            }
        }
    }

    if (alias)
    {
        free(alias);
    }

    if (module_base_name)
    { // This was only used for selective import
      // registration, not for ctx->current_module_prefix
        free(module_base_name);
    }

    free(fn);
    return r;
}

// Helper: Execute comptime block and return generated source
char *run_comptime_block(ParserContext *ctx, Lexer *l)
{
    (void)ctx;
    expect(l, TOK_COMPTIME, "comptime");
    expect(l, TOK_LBRACE, "expected { after comptime");

    const char *start = l->src + l->pos;
    int depth = 1;
    while (depth > 0)
    {
        Token t = lexer_next(l);
        if (t.type == TOK_EOF)
        {
            zpanic_at(t, "Unexpected EOF in comptime block");
        }
        if (t.type == TOK_LBRACE)
        {
            depth++;
        }
        if (t.type == TOK_RBRACE)
        {
            depth--;
        }
    }
    // End is passed the closing brace, so pos points after it.
    // The code block is between start and (current pos - 1)
    int len = (l->src + l->pos - 1) - start;
    char *code = xmalloc(len + 1);
    strncpy(code, start, len);
    code[len] = 0;

    // Wrap in block to parse mixed statements/declarations
    int wrapped_len = len + 4; // "{ " + code + " }"
    char *wrapped_code = xmalloc(wrapped_len + 1);
    sprintf(wrapped_code, "{ %s }", code);

    Lexer cl;
    lexer_init(&cl, wrapped_code);
    ParserContext cctx;
    memset(&cctx, 0, sizeof(cctx));
    enter_scope(&cctx); // Global scope
    register_builtins(&cctx);

    ASTNode *block = parse_block(&cctx, &cl);
    ASTNode *nodes = block ? block->block.statements : NULL;

    free(wrapped_code);

    char filename[64];
    sprintf(filename, "_tmp_comptime_%d.c", rand());
    FILE *f = fopen(filename, "w");
    if (!f)
    {
        zpanic_at(lexer_peek(l), "Could not create temp file %s", filename);
    }

    emit_preamble(ctx, f);
    fprintf(
        f,
        "size_t _z_check_bounds(size_t index, size_t size) { if (index >= size) { fprintf(stderr, "
        "\"Index out of bounds: %%zu >= %%zu\\n\", index, size); exit(1); } return index; }\n");

    ASTNode *curr = nodes;
    ASTNode *stmts = NULL;
    ASTNode *stmts_tail = NULL;

    while (curr)
    {
        ASTNode *next = curr->next;
        curr->next = NULL;

        if (curr->type == NODE_INCLUDE)
        {
            emit_includes_and_aliases(curr, f);
        }
        else if (curr->type == NODE_STRUCT)
        {
            emit_struct_defs(&cctx, curr, f);
        }
        else if (curr->type == NODE_ENUM)
        {
            emit_enum_protos(curr, f);
        }
        else if (curr->type == NODE_CONST)
        {
            emit_globals(&cctx, curr, f);
        }
        else if (curr->type == NODE_FUNCTION)
        {
            codegen_node_single(&cctx, curr, f);
        }
        else if (curr->type == NODE_IMPL)
        {
            // Impl support pending
        }
        else
        {
            // Statement or expression -> main
            if (!stmts)
            {
                stmts = curr;
            }
            else
            {
                stmts_tail->next = curr;
            }
            stmts_tail = curr;
        }
        curr = next;
    }

    {
        StructRef *ref = ctx->parsed_funcs_list;
        while (ref)
        {
            ASTNode *fn = ref->node;
            if (fn && fn->type == NODE_FUNCTION && fn->func.is_comptime)
            {
                emit_func_signature(ctx, f, fn, NULL);
                fprintf(f, ";\n");
                codegen_node_single(ctx, fn, f);
            }
            ref = ref->next;
        }
    }

    fprintf(f, "int main() {\n");
    curr = stmts;
    while (curr)
    {
        if (curr->type >= NODE_EXPR_BINARY && curr->type <= NODE_EXPR_SLICE)
        {
            codegen_expression(&cctx, curr, f);
            fprintf(f, ";\n");
        }
        else
        {
            codegen_node_single(&cctx, curr, f);
        }
        curr = curr->next;
    }
    fprintf(f, "return 0;\n}\n");
    fclose(f);

    char cmd[4096];
    char bin[1024];
    if (z_is_windows())
    {
        sprintf(bin, "%s.exe", filename);
    }
    else
    {
        sprintf(bin, "%s.bin", filename);
    }
    sprintf(cmd, "%s %s -o %s", g_config.cc, filename, bin);
    if (!g_config.verbose)
    {
        strcat(cmd, " > /dev/null 2>&1");
    }
    int res = system(cmd);
    if (res != 0)
    {
        zpanic_at(lexer_peek(l), "Comptime compilation failed for:\n%s", code);
    }

    char out_file[1024];
    sprintf(out_file, "%s.out", filename);

    // Platform-neutral execution
    if (z_is_windows())
    {
        sprintf(cmd, "%s > %s", bin, out_file);
    }
    else
    {
        sprintf(cmd, "./%s > %s", bin, out_file);
    }

    if (system(cmd) != 0)
    {
        zpanic_at(lexer_peek(l), "Comptime execution failed");
    }

    char *output_src = load_file(out_file);
    if (!output_src)
    {
        output_src = xstrdup(""); // Empty output is valid
    }

    remove(filename);
    remove(bin);
    remove(out_file);
    free(code);

    return output_src;
}

ASTNode *parse_comptime(ParserContext *ctx, Lexer *l)
{
    char *output_src = run_comptime_block(ctx, l);

    Lexer new_l;
    lexer_init(&new_l, output_src);
    return parse_program_nodes(ctx, &new_l);
}

// Parse plugin block: plugin name ... end
ASTNode *parse_plugin(ParserContext *ctx, Lexer *l)
{
    (void)ctx;

    // Expect 'plugin' keyword (already consumed by caller)
    // Next should be plugin name
    Token tk = lexer_next(l);
    if (tk.type != TOK_IDENT)
    {
        zpanic_at(tk, "Expected plugin name after 'plugin' keyword");
    }

    // Extract plugin name
    char *plugin_name = xmalloc(tk.len + 1);
    strncpy(plugin_name, tk.start, tk.len);
    plugin_name[tk.len] = '\0';

    // Collect everything until 'end'
    char *body = xmalloc(8192);
    body[0] = '\0';
    int body_len = 0;

    while (1)
    {
        Token t = lexer_peek(l);
        if (t.type == TOK_EOF)
        {
            zpanic_at(t, "Unexpected EOF in plugin block, expected 'end'");
        }

        // Check for 'end'
        if (t.type == TOK_IDENT && t.len == 3 && strncmp(t.start, "end", 3) == 0)
        {
            lexer_next(l); // consume 'end'
            break;
        }

        // Append token to body
        if (body_len + t.len + 2 < 8192)
        {
            strncat(body, t.start, t.len);
            body[body_len + t.len] = ' ';
            body[body_len + t.len + 1] = '\0';
            body_len += t.len + 1;
        }

        lexer_next(l);
    }

    // Create plugin node
    ASTNode *n = ast_create(NODE_PLUGIN);
    n->plugin_stmt.plugin_name = plugin_name;
    n->plugin_stmt.body = body;

    if (lexer_peek(l).type == TOK_SEMICOLON)
    {
        lexer_next(l);
    }
    return n;
}
