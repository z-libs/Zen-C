
#include "parser.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../ast/ast.h"
#include "../compat/compat.h"
#include "../plugins/plugin_manager.h"
#include "../zen/zen_facts.h"
#include "zprep_plugin.h"
#include "../codegen/codegen.h"

static char *curr_func_ret = NULL;
char *run_comptime_block(ParserContext *ctx, Lexer *l);

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

ASTNode *parse_function(ParserContext *ctx, Lexer *l, int is_async)
{
    lexer_next(l); // eat 'fn'
    Token name_tok = lexer_next(l);
    char *name = token_strdup(name_tok);

    if (is_async)
    {
        ctx->has_async = 1;
    }

    // Check for C reserved word conflict
    if (is_c_reserved_word(name))
    {
        warn_c_reserved_word(name_tok, name);
    }

    char *gen_param = NULL;
    if (lexer_peek(l).type == TOK_LANGLE)
    {
        lexer_next(l);
        Token gt = lexer_next(l);
        gen_param = token_strdup(gt);
        if (lexer_next(l).type != TOK_RANGLE)
        {
            zpanic_at(lexer_peek(l), "Expected >");
        }
    }

    enter_scope(ctx);
    char **defaults;
    int count;
    Type **arg_types;
    char **param_names;
    int is_varargs = 0;

    char *args =
        parse_and_convert_args(ctx, l, &defaults, &count, &arg_types, &param_names, &is_varargs);

    char *ret = "void";
    Type *ret_type_obj = type_new(TYPE_VOID);

    if (strcmp(name, "main") == 0)
    {
        ret = "int";
        ret_type_obj = type_new(TYPE_INT);
    }

    if (lexer_peek(l).type == TOK_ARROW)
    {
        lexer_next(l);
        ret_type_obj = parse_type_formal(ctx, l);
        ret = type_to_string(ret_type_obj);
    }

    extern char *curr_func_ret;
    curr_func_ret = ret;

    // Auto-prefix function name if in module context
    // Don't prefix generic templates or functions inside impl blocks (already
    // mangled)
    if (ctx->current_module_prefix && !gen_param && !ctx->current_impl_struct)
    {
        char *prefixed_name = xmalloc(strlen(ctx->current_module_prefix) + strlen(name) + 2);
        sprintf(prefixed_name, "%s_%s", ctx->current_module_prefix, name);
        free(name);
        name = prefixed_name;
    }

    // Register if concrete (Global functions only)
    if (!gen_param && !ctx->current_impl_struct)
    {
        register_func(ctx, name, count, defaults, arg_types, ret_type_obj, is_varargs, is_async,
                      name_tok);
        // Note: must_use is set after return by caller (parser_core.c)
    }

    ASTNode *body = NULL;
    if (lexer_peek(l).type == TOK_SEMICOLON)
    {
        lexer_next(l); // consume ;
    }
    else
    {
        body = parse_block(ctx, l);
    }

    // Check for unused parameters
    // The current scope contains arguments (since parse_block creates a new child
    // scope for body) Only check if we parsed a body (not a prototype) function
    if (body && ctx->current_scope)
    {
        Symbol *sym = ctx->current_scope->symbols;
        while (sym)
        {
            // Check if unused and not prefixed with '_' (conventional ignore)
            // also ignore 'self' as it is often mandated by traits
            if (!sym->is_used && sym->name[0] != '_' && strcmp(sym->name, "self") != 0 &&
                strcmp(name, "main") != 0)
            {
                warn_unused_parameter(sym->decl_token, sym->name, name);
            }
            sym = sym->next;
        }
    }

    exit_scope(ctx);
    curr_func_ret = NULL;

    ASTNode *node = ast_create(NODE_FUNCTION);
    node->token = name_tok; // Save definition location
    node->func.name = name;
    node->func.args = args;
    node->func.ret_type = ret;
    node->func.body = body;

    node->func.arg_types = arg_types;
    node->func.param_names = param_names;
    node->func.arg_count = count;
    node->func.defaults = defaults;
    node->func.ret_type_info = ret_type_obj;
    node->func.is_varargs = is_varargs;

    if (gen_param)
    {
        register_func_template(ctx, name, gen_param, node);
        return NULL;
    }
    if (!ctx->current_impl_struct)
    {
        add_to_func_list(ctx, node);
    }
    return node;
}

char *patch_self_args(const char *args, const char *struct_name)
{
    if (!args)
    {
        return NULL;
    }
    char *new_args = xmalloc(strlen(args) + strlen(struct_name) + 10);

    // Check if it starts with "void* self"
    if (strncmp(args, "void* self", 10) == 0)
    {
        sprintf(new_args, "%s* self%s", struct_name, args + 10);
    }
    else
    {
        strcpy(new_args, args);
    }
    return new_args;
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

        // --- 1. Parse Comma-Separated Patterns ---
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

            if (pattern_count > 0)
            {
                strcat(patterns_buf, ",");
            }
            strcat(patterns_buf, p_str);
            free(p_str);
            pattern_count++;

            Lexer lookahead = *l;
            skip_comments(&lookahead);
            if (lexer_peek(&lookahead).type == TOK_COMMA)
            {
                lexer_next(l); // eat comma
                skip_comments(l);
            }
            else
            {
                break;
            }
        }

        char *pattern = xstrdup(patterns_buf);
        int is_default = (strcmp(pattern, "_") == 0);

        char *binding = NULL;
        int is_destructure = 0;

        // --- 2. Handle Destructuring: Ok(v) ---
        // (Only allowed if we matched a single pattern, e.g. "Result::Ok(val)")
        if (!is_default && pattern_count == 1 && lexer_peek(l).type == TOK_LPAREN)
        {
            lexer_next(l); // eat (
            Token b = lexer_next(l);
            if (b.type != TOK_IDENT)
            {
                zpanic_at(b, "Expected variable name in pattern");
            }
            binding = token_strdup(b);
            if (lexer_next(l).type != TOK_RPAREN)
            {
                zpanic_at(lexer_peek(l), "Expected )");
            }
            is_destructure = 1;
        }

        // --- 3. Parse Guard (if condition) ---
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

        ASTNode *c = ast_create(NODE_MATCH_CASE);
        c->match_case.pattern = pattern;
        c->match_case.binding_name = binding;
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
    lexer_next(l);
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
    lexer_next(l); // defer
    ASTNode *s;
    if (lexer_peek(l).type == TOK_LBRACE)
    {
        s = parse_block(ctx, l);
    }
    else
    {
        s = ast_create(NODE_RAW_STMT);
        s->raw_stmt.content = consume_and_rewrite(ctx, l);
    }
    ASTNode *n = ast_create(NODE_DEFER);
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

    // Parse clobbers (: "eax", "memory")
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

            if (t.type == TOK_STRING)
            {
                lexer_next(l);
                // Extract string content
                char *clob = xmalloc(t.len);
                strncpy(clob, t.start + 1, t.len - 2);
                clob[t.len - 2] = 0;
                clobbers[num_clobbers++] = clob;
            }
            else
            {
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

// Helper for Value-Returning Defer
static void replace_it_with_var(ASTNode *node, char *var_name)
{
    if (!node)
    {
        return;
    }
    if (node->type == NODE_EXPR_VAR)
    {
        if (strcmp(node->var_ref.name, "it") == 0)
        {
            // Replace 'it' with var_name
            node->var_ref.name = xstrdup(var_name);
        }
    }
    else if (node->type == NODE_EXPR_CALL)
    {
        replace_it_with_var(node->call.callee, var_name);
        ASTNode *arg = node->call.args;
        while (arg)
        {
            replace_it_with_var(arg, var_name);
            arg = arg->next;
        }
    }
    else if (node->type == NODE_EXPR_MEMBER)
    {
        replace_it_with_var(node->member.target, var_name);
    }
    else if (node->type == NODE_EXPR_BINARY)
    {
        replace_it_with_var(node->binary.left, var_name);
        replace_it_with_var(node->binary.right, var_name);
    }
    else if (node->type == NODE_EXPR_UNARY)
    {
        replace_it_with_var(node->unary.operand, var_name);
    }
    else if (node->type == NODE_BLOCK)
    {
        ASTNode *s = node->block.statements;
        while (s)
        {
            replace_it_with_var(s, var_name);
            s = s->next;
        }
    }
}

ASTNode *parse_var_decl(ParserContext *ctx, Lexer *l)
{
    lexer_next(l); // eat 'var'

    // Check for 'mut' keyword
    int is_mutable = 0;
    if (lexer_peek(l).type == TOK_MUT)
    {
        is_mutable = 1;
        lexer_next(l);
    }
    else
    {
        // Default mutability depends on directive
        is_mutable = !ctx->immutable_by_default;
    }

    // Destructuring: var {x, y} = ...
    if (lexer_peek(l).type == TOK_LBRACE || lexer_peek(l).type == TOK_LPAREN)
    {
        int is_struct = (lexer_peek(l).type == TOK_LBRACE);
        lexer_next(l);
        char **names = xmalloc(16 * sizeof(char *));
        int count = 0;
        while (1)
        {
            Token t = lexer_next(l);
            char *nm = token_strdup(t);
            // UPDATE: Pass NULL to add_symbol
            names[count++] = nm;
            add_symbol(ctx, nm, "unknown", NULL);
            // Register mutability for each destructured variable
            register_var_mutability(ctx, nm, is_mutable);
            Token next = lexer_next(l);
            if (next.type == (is_struct ? TOK_RBRACE : TOK_RPAREN))
            {
                break;
            }
            if (next.type != TOK_COMMA)
            {
                zpanic_at(next, "Expected comma");
            }
        }
        if (lexer_next(l).type != TOK_OP)
        {
            zpanic_at(lexer_peek(l), "Expected =");
        }
        ASTNode *init = parse_expression(ctx, l);
        if (lexer_peek(l).type == TOK_SEMICOLON)
        {
            lexer_next(l);
        }
        ASTNode *n = ast_create(NODE_DESTRUCT_VAR);
        n->destruct.names = names;
        n->destruct.count = count;
        n->destruct.init_expr = init;
        n->destruct.is_struct_destruct = is_struct;
        return n;
    }

    // Normal Declaration OR Named Struct Destructuring
    Token name_tok = lexer_next(l);
    char *name = token_strdup(name_tok);

    // Check for Struct Destructuring: var Point { x, y }
    if (lexer_peek(l).type == TOK_LBRACE)
    {
        lexer_next(l); // eat {
        char **names = xmalloc(16 * sizeof(char *));
        char **fields = xmalloc(16 * sizeof(char *));
        int count = 0;

        while (1)
        {
            // Parse field:name or just name
            Token t = lexer_next(l);
            char *ident = token_strdup(t);

            if (lexer_peek(l).type == TOK_COLON)
            {
                // field: var_name
                lexer_next(l); // eat :
                Token v = lexer_next(l);
                fields[count] = ident;
                names[count] = token_strdup(v);
            }
            else
            {
                // Shorthand: field (implies var name = field)
                fields[count] = ident;
                names[count] = ident; // Share pointer or duplicate? duplicate safer if we free
            }
            // Register symbol for variable
            add_symbol(ctx, names[count], "unknown", NULL);
            register_var_mutability(ctx, names[count], is_mutable);

            count++;

            Token next = lexer_next(l);
            if (next.type == TOK_RBRACE)
            {
                break;
            }
            if (next.type != TOK_COMMA)
            {
                zpanic_at(next, "Expected comma in struct pattern");
            }
        }

        if (lexer_next(l).type != TOK_OP)
        {
            zpanic_at(lexer_peek(l), "Expected =");
        }
        ASTNode *init = parse_expression(ctx, l);
        if (lexer_peek(l).type == TOK_SEMICOLON)
        {
            lexer_next(l);
        }

        ASTNode *n = ast_create(NODE_DESTRUCT_VAR);
        n->destruct.names = names;
        n->destruct.field_names = fields;
        n->destruct.count = count;
        n->destruct.init_expr = init;
        n->destruct.is_struct_destruct = 1;
        n->destruct.struct_name = name; // "Point"
        return n;
    }

    // Check for Guard Pattern: var Some(val) = opt else { ... }
    if (lexer_peek(l).type == TOK_LPAREN)
    {
        lexer_next(l); // eat (
        Token val_tok = lexer_next(l);
        char *val_name = token_strdup(val_tok);

        if (lexer_next(l).type != TOK_RPAREN)
        {
            zpanic_at(lexer_peek(l), "Expected ')' in guard pattern");
        }

        if (lexer_next(l).type != TOK_OP)
        {
            zpanic_at(lexer_peek(l), "Expected '=' after guard pattern");
        }

        ASTNode *init = parse_expression(ctx, l);

        Token t = lexer_next(l);
        if (t.type != TOK_IDENT || strncmp(t.start, "else", 4) != 0)
        {
            zpanic_at(t, "Expected 'else' in guard statement");
        }

        ASTNode *else_blk;
        if (lexer_peek(l).type == TOK_LBRACE)
        {
            else_blk = parse_block(ctx, l);
        }
        else
        {
            else_blk = ast_create(NODE_BLOCK);
            else_blk->block.statements = parse_statement(ctx, l);
        }

        if (lexer_peek(l).type == TOK_SEMICOLON)
        {
            lexer_next(l);
        }

        ASTNode *n = ast_create(NODE_DESTRUCT_VAR);
        n->destruct.names = xmalloc(sizeof(char *));
        n->destruct.names[0] = val_name;
        n->destruct.count = 1;
        n->destruct.init_expr = init;
        n->destruct.is_guard = 1;
        n->destruct.guard_variant = name;
        n->destruct.else_block = else_blk;

        add_symbol(ctx, val_name, "unknown", NULL);
        register_var_mutability(ctx, val_name, is_mutable);

        return n;
    }

    char *type = NULL;
    Type *type_obj = NULL; // --- NEW: Formal Type Object ---

    if (lexer_peek(l).type == TOK_COLON)
    {
        lexer_next(l);
        // Hybrid Parse: Get Object AND String
        type_obj = parse_type_formal(ctx, l);
        type = type_to_string(type_obj);
    }

    ASTNode *init = NULL;
    if (lexer_peek(l).type == TOK_OP && is_token(lexer_peek(l), "="))
    {
        lexer_next(l);

        // Peek for special initializers
        Token next = lexer_peek(l);
        if (next.type == TOK_IDENT && strncmp(next.start, "embed", 5) == 0)
        {
            char *e = parse_embed(ctx, l);
            init = ast_create(NODE_RAW_STMT);
            init->raw_stmt.content = e;
            if (!type)
            {
                register_slice(ctx, "char");
                type = xstrdup("Slice_char");
            }
            if (lexer_peek(l).type == TOK_SEMICOLON)
            {
                lexer_next(l);
            }
        }
        else if (next.type == TOK_LBRACKET && type && strncmp(type, "Slice_", 6) == 0)
        {
            char *code = parse_array_literal(ctx, l, type);
            init = ast_create(NODE_RAW_STMT);
            init->raw_stmt.content = code;
            if (lexer_peek(l).type == TOK_SEMICOLON)
            {
                lexer_next(l);
            }
        }
        else if (next.type == TOK_LPAREN && type && strncmp(type, "Tuple_", 6) == 0)
        {
            char *code = parse_tuple_literal(ctx, l, type);
            init = ast_create(NODE_RAW_STMT);
            init->raw_stmt.content = code;
            if (lexer_peek(l).type == TOK_SEMICOLON)
            {
                lexer_next(l);
            }
        }
        else
        {
            init = parse_expression(ctx, l);
        }

        if (init && type)
        {
            char *rhs_type = init->resolved_type;
            if (!rhs_type && init->type_info)
            {
                rhs_type = type_to_string(init->type_info);
            }

            if (rhs_type && strchr(type, '*') && strchr(rhs_type, '*'))
            {
                // Strip stars to get struct names
                char target_struct[256];
                strcpy(target_struct, type);
                target_struct[strlen(target_struct) - 1] = 0;
                char source_struct[256];
                strcpy(source_struct, rhs_type);
                source_struct[strlen(source_struct) - 1] = 0;

                ASTNode *def = find_struct_def(ctx, source_struct);

                if (def && def->strct.parent && strcmp(def->strct.parent, target_struct) == 0)
                {
                    // Create Cast Node
                    ASTNode *cast = ast_create(NODE_EXPR_CAST);
                    cast->cast.target_type = xstrdup(type);
                    cast->cast.expr = init;
                    cast->type_info = type_obj; // Inherit formal type

                    init = cast; // Replace init with cast
                }
            }
        }

        // ** Type Inference Logic **
        if (!type && init)
        {
            if (init->type_info)
            {
                type_obj = init->type_info;
                type = type_to_string(type_obj);
            }
            else if (init->type == NODE_EXPR_SLICE)
            {
                zpanic_at(init->token, "Slice Node has NO Type Info!");
            }
            // Fallbacks for literals
            else if (init->type == NODE_EXPR_LITERAL)
            {
                if (init->literal.type_kind == 0)
                {
                    type = xstrdup("int");
                    type_obj = type_new(TYPE_INT);
                }
                else if (init->literal.type_kind == 1)
                {
                    type = xstrdup("float");
                    type_obj = type_new(TYPE_FLOAT);
                }
                else if (init->literal.type_kind == 2)
                {
                    type = xstrdup("string");
                    type_obj = type_new(TYPE_STRING);
                }
            }
            else if (init->type == NODE_EXPR_STRUCT_INIT)
            {
                type = xstrdup(init->struct_init.struct_name);
                type_obj = type_new(TYPE_STRUCT);
                type_obj->name = xstrdup(type);
            }
        }
    }

    if (!type && !init)
    {
        zpanic_at(name_tok, "Variable '%s' requires a type or initializer", name);
    }

    // Register in symbol table with actual token
    add_symbol_with_token(ctx, name, type, type_obj, name_tok);
    register_var_mutability(ctx, name, is_mutable);

    // NEW: Capture Const Integer Values
    if (!is_mutable && init && init->type == NODE_EXPR_LITERAL && init->literal.type_kind == 0)
    {
        Symbol *s = find_symbol_entry(ctx, name); // Helper to find the struct
        if (s)
        {
            s->is_const_value = 1;
            s->const_int_val = init->literal.int_val;
        }
    }

    if (lexer_peek(l).type == TOK_SEMICOLON)
    {
        lexer_next(l);
    }

    ASTNode *n = ast_create(NODE_VAR_DECL);
    n->token = name_tok; // Save location
    n->var_decl.name = name;
    n->var_decl.type_str = type;
    n->var_decl.is_mutable = is_mutable;
    n->type_info = type_obj;

    // Auto-construct Trait Object
    if (type && is_trait(type) && init && init->type == NODE_EXPR_UNARY &&
        strcmp(init->unary.op, "&") == 0 && init->unary.operand->type == NODE_EXPR_VAR)
    {
        char *var_ref_name = init->unary.operand->var_ref.name;
        char *struct_type = find_symbol_type(ctx, var_ref_name);
        if (struct_type)
        {
            char *code = xmalloc(512);
            sprintf(code, "(%s){.self=&%s, .vtable=&%s_%s_VTable}", type, var_ref_name, struct_type,
                    type);
            ASTNode *wrapper = ast_create(NODE_RAW_STMT);
            wrapper->raw_stmt.content = code;
            init = wrapper;
        }
    }

    n->var_decl.init_expr = init;

    // Global detection: Either no scope (yet) OR root scope (no parent)
    if (!ctx->current_scope || !ctx->current_scope->parent)
    {
        add_to_global_list(ctx, n);
    }

    // Check for 'defer' (Value-Returning Defer)
    // Only capture if it is NOT a block defer (defer { ... })
    // If it is a block defer, we leave it for the next parse_statement call.
    if (lexer_peek(l).type == TOK_DEFER)
    {
        Lexer lookahead = *l;
        lexer_next(&lookahead); // Eat defer
        if (lexer_peek(&lookahead).type != TOK_LBRACE)
        {
            // Proceed to consume
            lexer_next(l); // eat defer (real)

            // Parse the defer expression/statement
            // Usually defer close(it);
            // We parse expression.
            ASTNode *expr = parse_expression(ctx, l);

            // Handle "it" substitution
            replace_it_with_var(expr, name);

            if (lexer_peek(l).type == TOK_SEMICOLON)
            {
                lexer_next(l);
            }

            ASTNode *d = ast_create(NODE_DEFER);
            d->defer_stmt.stmt = expr;

            // Chain it: var_decl -> defer
            n->next = d;
        }
    }

    return n;
}

ASTNode *parse_const(ParserContext *ctx, Lexer *l)
{
    lexer_next(l); // eat const
    Token n = lexer_next(l);

    char *type_str = NULL;
    Type *type_obj = NULL;

    if (lexer_peek(l).type == TOK_COLON)
    {
        lexer_next(l);
        // Hybrid Parse
        type_obj = parse_type_formal(ctx, l);
        type_str = type_to_string(type_obj);
    }

    char *ns = token_strdup(n);
    if (!type_obj)
    {
        type_obj = type_new(TYPE_UNKNOWN); // Ensure we have an object
    }
    type_obj->is_const = 1;
    add_symbol(ctx, ns, type_str ? type_str : "unknown", type_obj);

    ASTNode *i = 0;
    if (lexer_peek(l).type == TOK_OP && is_token(lexer_peek(l), "="))
    {
        lexer_next(l);

        // Check for constant integer literal
        if (lexer_peek(l).type == TOK_INT)
        {
            Token val_tok = lexer_peek(l);
            int val = atoi(token_strdup(val_tok)); // quick check

            Symbol *s = find_symbol_entry(ctx, ns);
            if (s)
            {
                s->is_const_value = 1;
                s->const_int_val = val;

                if (!s->type_name || strcmp(s->type_name, "unknown") == 0)
                {
                    if (s->type_name)
                    {
                        free(s->type_name);
                    }
                    s->type_name = xstrdup("int");
                    if (s->type_info)
                    {
                        free(s->type_info);
                    }
                    s->type_info = type_new(TYPE_INT);
                }
            }
        }

        if (lexer_peek(l).type == TOK_LPAREN && type_str && strncmp(type_str, "Tuple_", 6) == 0)
        {
            char *code = parse_tuple_literal(ctx, l, type_str);
            i = ast_create(NODE_RAW_STMT);
            i->raw_stmt.content = code;
        }
        else
        {
            i = parse_expression(ctx, l);
        }
    }
    else
    {
        lexer_next(l);
    }

    if (lexer_peek(l).type == TOK_SEMICOLON)
    {
        lexer_next(l);
    }

    ASTNode *o = ast_create(NODE_CONST);
    o->var_decl.name = ns;
    o->var_decl.type_str = type_str;
    o->var_decl.init_expr = i;

    if (!ctx->current_scope || !ctx->current_scope->parent)
    {
        add_to_global_list(ctx, o);
    }

    return o;
}

ASTNode *parse_type_alias(ParserContext *ctx, Lexer *l)
{
    lexer_next(l); // consume 'type' or 'alias'
    Token n = lexer_next(l);
    if (n.type != TOK_IDENT)
    {
        zpanic_at(n, "Expected identifier for type alias");
    }

    lexer_next(l); // consume '='

    char *o = parse_type(ctx, l);
    // printf("DEBUG: parse_type returned '%s'\n", o);

    lexer_next(l); // consume ';' (parse_type doesn't consume it? parse_type calls parse_type_formal
                   // which doesn't consume ;?)
    // Note: parse_type_stmt usually expects ; but parse_type just parses type expression.
    // Check previous implementation: it had lexer_next(l) at end. This assumes ;.

    ASTNode *node = ast_create(NODE_TYPE_ALIAS);
    node->type_alias.alias = xmalloc(n.len + 1);
    strncpy(node->type_alias.alias, n.start, n.len);
    node->type_alias.alias[n.len] = 0;
    node->type_alias.original_type = o;

    register_type_alias(ctx, node->type_alias.alias, o);

    return node;
}

ASTNode *parse_return(ParserContext *ctx, Lexer *l)
{
    lexer_next(l); // eat 'return'
    ASTNode *n = ast_create(NODE_RETURN);

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
    if ((cond->type == NODE_EXPR_LITERAL && cond->literal.type_kind == TOK_INT &&
         strcmp(cond->literal.string_val, "1") == 0) ||
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
            if (lexer_peek(l).type == TOK_DOTDOT)
            {
                lexer_next(l); // consume ..
                ASTNode *end_expr = parse_expression(ctx, l);

                ASTNode *n = ast_create(NODE_FOR_RANGE);
                n->for_range.var_name = xmalloc(var.len + 1);
                strncpy(n->for_range.var_name, var.start, var.len);
                n->for_range.var_name[var.len] = 0;
                n->for_range.start = start_expr;
                n->for_range.end = end_expr;

                if (lexer_peek(l).type == TOK_IDENT && strncmp(lexer_peek(l).start, "step", 4) == 0)
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

                // Fix: Enter scope to register loop variable
                enter_scope(ctx);
                // Register loop variable so body can see it
                add_symbol(ctx, n->for_range.var_name, "int", type_new(TYPE_INT));

                // Handle body (brace or single stmt)
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
        if (lexer_peek(l).type == TOK_IDENT && strncmp(lexer_peek(l).start, "var", 3) == 0)
        {
            init = parse_var_decl(ctx, l);
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
        true_lit->literal.type_kind = 0;
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
                           char ***used_syms, int *count)
{
    char *gen = xmalloc(8192);
    strcpy(gen, "({ ");

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
                colon = p;
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

        // Use final_expr in usage analysis if needed, but mainly for symbol tracking
        {
            Lexer lex;
            lexer_init(&lex, clean_expr); // Scan original for symbols
            Token t;
            while ((t = lexer_next(&lex)).type != TOK_EOF)
            {
                if (t.type == TOK_IDENT)
                {
                    char *name = token_strdup(t);
                    Symbol *sym = find_symbol_entry(ctx, name);
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
        char *allocated_expr = NULL;
        clean_expr = final_expr;

        int skip_rewrite = 0;

        // Check if struct and has to_string (Robust Logic)
        {
            Lexer lex;
            lexer_init(&lex, clean_expr);
            // Parse using temporary lexer to check type
            ASTNode *expr_node = parse_expression(ctx, &lex);

            if (expr_node && expr_node->type_info)
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
                        char *inner_wrapped = xmalloc(strlen(clean_expr) + 5);
                        sprintf(inner_wrapped, "#{%s}", clean_expr);
                        char *inner_c = rewrite_expr_methods(ctx, inner_wrapped);
                        free(inner_wrapped);

                        // Now wrap in to_string call using C99 compound literal for safety
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

                        if (expr != s)
                        {
                            free(expr); // Free if explicitly allocated
                        }
                        expr = new_expr;
                        skip_rewrite = 1; // Don't rewrite again on the C99 syntax
                    }
                }
            }
        }

        // Rewrite the expression to handle pointer access (header_ptr.magic ->
        // header_ptr->magic)
        char *rw_expr;
        if (skip_rewrite)
        {
            rw_expr = xstrdup(expr);
        }
        else
        {
            char *wrapped_expr = xmalloc(strlen(expr) + 5);
            sprintf(wrapped_expr, "#{%s}", expr);
            rw_expr = rewrite_expr_methods(ctx, wrapped_expr);
            free(wrapped_expr);
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
            // Auto-detect format based on type if possible
            const char *format_spec = NULL;
            char *inferred_type = find_symbol_type(ctx, clean_expr); // Simple variable lookup

            // Basic Type Mappings
            if (inferred_type)
            {
                if (strcmp(inferred_type, "int") == 0 || strcmp(inferred_type, "i32") == 0 ||
                    strcmp(inferred_type, "bool") == 0)
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
                strcat(gen, rw_expr);
                strcat(gen, "); ");
            }
            else
            {
                // Fallback to runtime macro
                char buf[128];
                sprintf(buf, "fprintf(%s, _z_str(", target);
                strcat(gen, buf);
                strcat(gen, rw_expr);
                strcat(gen, "), ");
                strcat(gen, rw_expr);
                strcat(gen, "); ");
            }
        }

        free(rw_expr); // Don't forget to free!
        if (allocated_expr)
        {
            free(allocated_expr); // Don't forget to free the auto-generated call!
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

    strcat(gen, "0; })");

    free(s);
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
        ZTokenType next_type = lexer_peek(&lookahead).type;

        if (next_type == TOK_SEMICOLON || next_type == TOK_DOTDOT)
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

            // ; means println (end of line), .. means print (continuation)
            int is_ln = (next_type == TOK_SEMICOLON);
            char **used_syms = NULL;
            int used_count = 0;
            char *code = process_printf_sugar(ctx, inner, is_ln, "stdout", &used_syms, &used_count);

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

            ASTNode *n = ast_create(NODE_RAW_STMT);
            n->raw_stmt.content = code;
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
        if (lexer_peek(l).type != TOK_IDENT || strncmp(lexer_peek(l).start, "var", 3) != 0)
        {
            zpanic_at(lexer_peek(l), "Expected 'var' after autofree");
        }
        s = parse_var_decl(ctx, l);
        s->var_decl.is_autofree = 1;
        // Mark symbol as autofree to suppress unused variable warning
        Symbol *sym = find_symbol_entry(ctx, s->var_decl.name);
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

        if (strncmp(tk.start, "var", 3) == 0 && tk.len == 3)
        {
            return parse_var_decl(ctx, l);
        }

        // Static local variable: static var x = 0;
        if (strncmp(tk.start, "static", 6) == 0 && tk.len == 6)
        {
            lexer_next(l); // eat 'static'
            Token next = lexer_peek(l);
            if (strncmp(next.start, "var", 3) == 0 && next.len == 3)
            {
                ASTNode *v = parse_var_decl(ctx, l);
                v->var_decl.is_static = 1;
                return v;
            }
            zpanic_at(next, "Expected 'var' after 'static'");
        }
        if (strncmp(tk.start, "const", 5) == 0 && tk.len == 5)
        {
            return parse_const(ctx, l);
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
            lexer_next(l);
            ASTNode *n = ast_create(NODE_BREAK);
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
            lexer_next(l);
            ASTNode *n = ast_create(NODE_CONTINUE);
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
            char *code = process_printf_sugar(ctx, inner, is_ln, target, &used_syms, &used_count);
            free(inner);

            if (lexer_peek(l).type == TOK_SEMICOLON)
            {
                lexer_next(l);
            }

            ASTNode *n = ast_create(NODE_RAW_STMT);
            n->raw_stmt.content = code;
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
        Symbol *sym = ctx->current_scope->symbols;
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
                int has_drop = (sym->type_info && sym->type_info->has_drop);
                if (!has_drop && sym->type_info && sym->type_info->name)
                {
                    ASTNode *def = find_struct_def(ctx, sym->type_info->name);
                    if (def && def->type_info && def->type_info->has_drop)
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

// Trait Parsing
ASTNode *parse_trait(ParserContext *ctx, Lexer *l)
{
    lexer_next(l); // eat trait
    Token n = lexer_next(l);
    if (n.type != TOK_IDENT)
    {
        zpanic_at(n, "Expected trait name");
    }
    char *name = xmalloc(n.len + 1);
    strncpy(name, n.start, n.len);
    name[n.len] = 0;

    lexer_next(l); // eat {

    ASTNode *methods = NULL, *tail = NULL;
    while (1)
    {
        skip_comments(l);
        if (lexer_peek(l).type == TOK_RBRACE)
        {
            lexer_next(l);
            break;
        }

        // Parse method signature: fn name(args...) -> ret;
        // Re-use parse_function but stop at semicolon?
        // Actually trait methods might have default impls later, but for now just
        // signatures. Let's parse full function but body might be empty/null? Or
        // simpler: just parse signature manually.

        Token ft = lexer_next(l);
        if (ft.type != TOK_IDENT || strncmp(ft.start, "fn", 2) != 0)
        {
            zpanic_at(ft, "Expected fn in trait");
        }

        Token mn = lexer_next(l);
        char *mname = xmalloc(mn.len + 1);
        strncpy(mname, mn.start, mn.len);
        mname[mn.len] = 0;

        char **defaults = NULL;
        int arg_count = 0;
        Type **arg_types = NULL;
        char **param_names = NULL;
        int is_varargs = 0;
        char *args = parse_and_convert_args(ctx, l, &defaults, &arg_count, &arg_types, &param_names,
                                            &is_varargs);

        char *ret = xstrdup("void");
        if (lexer_peek(l).type == TOK_ARROW)
        {
            lexer_next(l);
            char *rt = parse_type(ctx, l);
            free(ret);
            ret = rt;
        }

        if (lexer_peek(l).type == TOK_SEMICOLON)
        {
            lexer_next(l);
            ASTNode *m = ast_create(NODE_FUNCTION);
            m->func.param_names = param_names;
            m->func.name = mname;
            m->func.args = args;
            m->func.ret_type = ret;
            m->func.body = NULL;
            if (!methods)
            {
                methods = m;
            }
            else
            {
                tail->next = m;
            }
            tail = m;
        }
        else
        {
            // Default implementation? Not supported yet.
            zpanic_at(lexer_peek(l), "Trait methods must end with ; for now");
        }
    }

    ASTNode *n_node = ast_create(NODE_TRAIT);
    n_node->trait.name = name;
    n_node->trait.methods = methods;
    register_trait(name);
    return n_node;
}

ASTNode *parse_impl(ParserContext *ctx, Lexer *l)
{

    lexer_next(l); // eat impl
    Token t1 = lexer_next(l);
    char *name1 = token_strdup(t1);

    char *gen_param = NULL;
    // Check for <T> on the struct name
    if (lexer_peek(l).type == TOK_LANGLE)
    {
        lexer_next(l); // eat <
        Token gt = lexer_next(l);
        gen_param = token_strdup(gt);
        if (lexer_next(l).type != TOK_RANGLE)
        {
            zpanic_at(lexer_peek(l), "Expected >");
        }
    }

    // Check for "for" (Trait impl)
    Token pk = lexer_peek(l);
    if (pk.type == TOK_FOR ||
        (pk.type == TOK_IDENT && strncmp(pk.start, "for", 3) == 0 && pk.len == 3))
    {
        if (pk.type != TOK_FOR)
        {
            lexer_next(l);
        }
        else
        {
            lexer_next(l); // eat for
        }
        Token t2 = lexer_next(l);
        char *name2 = token_strdup(t2);

        register_impl(ctx, name1, name2);

        // RAII: Check for "Drop" trait implementation
        if (strcmp(name1, "Drop") == 0)
        {
            Symbol *s = find_symbol_entry(ctx, name2);
            if (s && s->type_info)
            {
                s->type_info->has_drop = 1;
            }
            else
            {
                // Try finding struct definition
                ASTNode *def = find_struct_def(ctx, name2);
                if (def && def->type_info)
                {
                    def->type_info->has_drop = 1;
                }
            }
        }

        ctx->current_impl_struct = name2; // Set context to prevent duplicate emission and prefixing

        lexer_next(l); // eat {
        ASTNode *h = 0, *tl = 0;
        while (1)
        {
            skip_comments(l);
            if (lexer_peek(l).type == TOK_RBRACE)
            {
                lexer_next(l);
                break;
            }
            if (lexer_peek(l).type == TOK_IDENT && strncmp(lexer_peek(l).start, "fn", 2) == 0)
            {
                ASTNode *f = parse_function(ctx, l, 0);
                // Mangle: Type_Trait_Method
                char *mangled = xmalloc(strlen(name2) + strlen(name1) + strlen(f->func.name) + 4);
                sprintf(mangled, "%s__%s_%s", name2, name1, f->func.name);
                free(f->func.name);
                f->func.name = mangled;
                char *na = patch_self_args(f->func.args, name2);
                free(f->func.args);
                f->func.args = na;
                if (!h)
                {
                    h = f;
                }
                else
                {
                    tl->next = f;
                }
                tl = f;
            }
            else if (lexer_peek(l).type == TOK_ASYNC)
            {
                lexer_next(l); // eat async
                if (lexer_peek(l).type == TOK_IDENT && strncmp(lexer_peek(l).start, "fn", 2) == 0)
                {
                    ASTNode *f = parse_function(ctx, l, 1);
                    f->func.is_async = 1;
                    // Mangle: Type_Trait_Method
                    char *mangled =
                        xmalloc(strlen(name2) + strlen(name1) + strlen(f->func.name) + 5);
                    sprintf(mangled, "%s__%s_%s", name2, name1, f->func.name);
                    free(f->func.name);
                    f->func.name = mangled;
                    char *na = patch_self_args(f->func.args, name2);
                    free(f->func.args);
                    f->func.args = na;
                    if (!h)
                    {
                        h = f;
                    }
                    else
                    {
                        tl->next = f;
                    }
                    tl = f;
                }
                else
                {
                    zpanic_at(lexer_peek(l), "Expected 'fn' after 'async'");
                }
            }
            else
            {
                lexer_next(l);
            }
        }
        ctx->current_impl_struct = NULL; // Restore context
        ASTNode *n = ast_create(NODE_IMPL_TRAIT);
        n->impl_trait.trait_name = name1;
        n->impl_trait.target_type = name2;
        n->impl_trait.methods = h;
        add_to_impl_list(ctx, n);
        return n;
    }
    else
    {
        // Regular impl Struct (impl Box or impl Box<T>)

        // Auto-prefix struct name if in module context
        if (ctx->current_module_prefix && !gen_param)
        {
            char *prefixed_name = xmalloc(strlen(ctx->current_module_prefix) + strlen(name1) + 2);
            sprintf(prefixed_name, "%s_%s", ctx->current_module_prefix, name1);
            free(name1);
            name1 = prefixed_name;
        }

        ctx->current_impl_struct = name1; // For patch_self_args inside parse_function

        if (gen_param)
        {
            // GENERIC IMPL TEMPLATE: impl Box<T>
            if (lexer_next(l).type != TOK_LBRACE)
            {
                zpanic_at(lexer_peek(l), "Expected {");
            }
            ASTNode *h = 0, *tl = 0;
            while (1)
            {
                skip_comments(l);
                if (lexer_peek(l).type == TOK_RBRACE)
                {
                    lexer_next(l);
                    break;
                }
                if (lexer_peek(l).type == TOK_IDENT && strncmp(lexer_peek(l).start, "fn", 2) == 0)
                {
                    ASTNode *f = parse_function(ctx, l, 0);
                    // Standard Mangle for template: Box_method
                    char *mangled = xmalloc(strlen(name1) + strlen(f->func.name) + 3);
                    sprintf(mangled, "%s__%s", name1, f->func.name);
                    free(f->func.name);
                    f->func.name = mangled;

                    char *na = patch_self_args(f->func.args, name1);
                    free(f->func.args);
                    f->func.args = na;

                    if (!h)
                    {
                        h = f;
                    }
                    else
                    {
                        tl->next = f;
                    }
                    tl = f;
                }
                else if (lexer_peek(l).type == TOK_ASYNC)
                {
                    lexer_next(l); // eat async
                    if (lexer_peek(l).type == TOK_IDENT &&
                        strncmp(lexer_peek(l).start, "fn", 2) == 0)
                    {
                        ASTNode *f = parse_function(ctx, l, 1);
                        f->func.is_async = 1;
                        char *mangled = xmalloc(strlen(name1) + strlen(f->func.name) + 3);
                        sprintf(mangled, "%s__%s", name1, f->func.name);
                        free(f->func.name);
                        f->func.name = mangled;
                        char *na = patch_self_args(f->func.args, name1);
                        free(f->func.args);
                        f->func.args = na;
                        if (!h)
                        {
                            h = f;
                        }
                        else
                        {
                            tl->next = f;
                        }
                        tl = f;
                    }
                    else
                    {
                        zpanic_at(lexer_peek(l), "Expected 'fn' after 'async'");
                    }
                }
                else
                {
                    lexer_next(l);
                }
            }
            // Register Template
            ASTNode *n = ast_create(NODE_IMPL);
            n->impl.struct_name = name1;
            n->impl.methods = h;
            register_impl_template(ctx, name1, gen_param, n);
            ctx->current_impl_struct = NULL;
            return NULL; // Do not emit generic template
        }
        else
        {
            // REGULAR IMPL
            lexer_next(l); // eat {
            ASTNode *h = 0, *tl = 0;
            while (1)
            {
                skip_comments(l);
                if (lexer_peek(l).type == TOK_RBRACE)
                {
                    lexer_next(l);
                    break;
                }
                if (lexer_peek(l).type == TOK_IDENT && strncmp(lexer_peek(l).start, "fn", 2) == 0)
                {
                    ASTNode *f = parse_function(ctx, l, 0);

                    // Standard Mangle: Struct_method
                    char *mangled = xmalloc(strlen(name1) + strlen(f->func.name) + 3);
                    sprintf(mangled, "%s__%s", name1, f->func.name);
                    free(f->func.name);
                    f->func.name = mangled;

                    char *na = patch_self_args(f->func.args, name1);
                    free(f->func.args);
                    f->func.args = na;

                    register_func(ctx, mangled, f->func.arg_count, f->func.defaults,
                                  f->func.arg_types, f->func.ret_type_info, f->func.is_varargs, 0,
                                  f->token);

                    if (!h)
                    {
                        h = f;
                    }
                    else
                    {
                        tl->next = f;
                    }
                    tl = f;
                }
                else if (lexer_peek(l).type == TOK_ASYNC)
                {
                    lexer_next(l);
                    if (lexer_peek(l).type == TOK_IDENT &&
                        strncmp(lexer_peek(l).start, "fn", 2) == 0)
                    {
                        ASTNode *f = parse_function(ctx, l, 1);
                        f->func.is_async = 1;
                        char *mangled = xmalloc(strlen(name1) + strlen(f->func.name) + 3);
                        sprintf(mangled, "%s__%s", name1, f->func.name);
                        free(f->func.name);
                        f->func.name = mangled;
                        char *na = patch_self_args(f->func.args, name1);
                        free(f->func.args);
                        f->func.args = na;
                        register_func(ctx, mangled, f->func.arg_count, f->func.defaults,
                                      f->func.arg_types, f->func.ret_type_info, f->func.is_varargs,
                                      1, f->token);
                        if (!h)
                        {
                            h = f;
                        }
                        else
                        {
                            tl->next = f;
                        }
                        tl = f;
                    }
                    else
                    {
                        zpanic_at(lexer_peek(l), "Expected 'fn' after 'async'");
                    }
                }
                else
                {
                    lexer_next(l);
                }
            }
            ctx->current_impl_struct = NULL;
            ASTNode *n = ast_create(NODE_IMPL);
            n->impl.struct_name = name1;
            n->impl.methods = h;
            add_to_impl_list(ctx, n);
            return n;
        }
    }
}

ASTNode *parse_struct(ParserContext *ctx, Lexer *l, int is_union)
{

    lexer_next(l); // eat struct or union
    Token n = lexer_next(l);
    char *name = token_strdup(n);

    // Generic Params <T> or <K, V>
    char **gps = NULL;
    int gp_count = 0;
    if (lexer_peek(l).type == TOK_LANGLE)
    {
        lexer_next(l); // eat <
        while (1)
        {
            Token g = lexer_next(l);
            gps = realloc(gps, sizeof(char *) * (gp_count + 1));
            gps[gp_count++] = token_strdup(g);

            Token next = lexer_peek(l);
            if (next.type == TOK_COMMA)
            {
                lexer_next(l); // eat ,
            }
            else if (next.type == TOK_RANGLE)
            {
                lexer_next(l); // eat >
                break;
            }
            else
            {
                zpanic_at(next, "Expected ',' or '>' in generic parameter list");
            }
        }
        register_generic(ctx, name);
    }

    // Check for prototype (forward declaration)
    if (lexer_peek(l).type == TOK_SEMICOLON)
    {
        lexer_next(l);
        ASTNode *n = ast_create(NODE_STRUCT);
        n->strct.name = name;
        n->strct.is_template = (gp_count > 0);
        n->strct.generic_params = gps;
        n->strct.generic_param_count = gp_count;
        n->strct.is_union = is_union;
        n->strct.fields = NULL;
        n->strct.is_incomplete = 1;

        return n;
    }

    lexer_next(l); // eat {
    ASTNode *h = 0, *tl = 0;

    while (1)
    {
        skip_comments(l);
        Token t = lexer_peek(l);
        // printf("DEBUG: parse_struct loop seeing '%.*s'\n", t.len, t.start);
        if (t.type == TOK_RBRACE)
        {
            lexer_next(l);
            break;
        }
        if (t.type == TOK_SEMICOLON || t.type == TOK_COMMA)
        {
            lexer_next(l);
            continue;
        }

        // --- HANDLE 'use' (Struct Embedding) ---
        if (t.type == TOK_USE)
        {
            lexer_next(l); // eat use
            // Parse the type (e.g. Header<I32>)
            Type *use_type = parse_type_formal(ctx, l);
            char *use_name = type_to_string(use_type);

            expect(l, TOK_SEMICOLON, "Expected ; after use");

            // Find the definition and COPY fields
            ASTNode *def = find_struct_def(ctx, use_name);
            if (!def && is_known_generic(ctx, use_type->name))
            {
                // Try to force instantiation if not found?
                // For now, rely on parse_type having triggered instantiation.
                char *mangled =
                    type_to_string(use_type); // This works if type_to_string returns mangled name
                def = find_struct_def(ctx, mangled);
                free(mangled);
            }

            if (def && def->type == NODE_STRUCT)
            {
                ASTNode *f = def->strct.fields;
                while (f)
                {
                    ASTNode *nf = ast_create(NODE_FIELD);
                    nf->field.name = xstrdup(f->field.name);
                    nf->field.type = xstrdup(f->field.type);
                    if (!h)
                    {
                        h = nf;
                    }
                    else
                    {
                        tl->next = nf;
                    }
                    tl = nf;
                    f = f->next;
                }
            }
            else
            {
                // If definition not found (e.g. user struct defined later), we can't
                // embed fields yet. Compiler limitation: 'use' requires struct to be
                // defined before. Fallback: Emit a placeholder field so compilation
                // doesn't crash, but layout will be wrong. printf("Warning: Could not
                // find struct '%s' for embedding.\n", use_name);
            }
            free(use_name);
            continue;
        }
        // ---------------------------------------

        if (t.type == TOK_IDENT)
        {
            Token f_name = lexer_next(l);
            expect(l, TOK_COLON, "Expected :");
            char *f_type = parse_type(ctx, l);

            ASTNode *f = ast_create(NODE_FIELD);
            f->field.name = token_strdup(f_name);
            f->field.type = f_type;
            f->field.bit_width = 0;

            // Optional bit width: name: type : 3
            if (lexer_peek(l).type == TOK_COLON)
            {
                lexer_next(l); // eat :
                Token width_tok = lexer_next(l);
                if (width_tok.type != TOK_INT)
                {
                    zpanic_at(width_tok, "Expected bit width integer");
                }
                f->field.bit_width = atoi(token_strdup(width_tok));
            }

            if (!h)
            {
                h = f;
            }
            else
            {
                tl->next = f;
            }
            tl = f;

            if (lexer_peek(l).type == TOK_SEMICOLON || lexer_peek(l).type == TOK_COMMA)
            {
                lexer_next(l);
            }
        }
        else
        {
            lexer_next(l);
        }
    }

    ASTNode *node = ast_create(NODE_STRUCT);
    add_to_struct_list(ctx, node);

    // Auto-prefix struct name if in module context
    if (ctx->current_module_prefix && gp_count == 0)
    { // Don't prefix generic templates
        char *prefixed_name = xmalloc(strlen(ctx->current_module_prefix) + strlen(name) + 2);
        sprintf(prefixed_name, "%s_%s", ctx->current_module_prefix, name);
        free(name);
        name = prefixed_name;
    }

    node->strct.name = name;

    // Initialize Type Info so we can track traits (like Drop)
    node->type_info = type_new(TYPE_STRUCT);
    node->type_info->name = xstrdup(name);
    if (gp_count > 0)
    {
        node->type_info->kind = TYPE_GENERIC;
        // TODO: track generic params
    }

    node->strct.fields = h;
    node->strct.generic_params = gps;
    node->strct.generic_param_count = gp_count;
    node->strct.is_union = is_union;

    if (gp_count > 0)
    {
        node->strct.is_template = 1;
        register_template(ctx, name, node);
    }

    // Register definition for 'use' lookups and LSP
    if (gp_count == 0)
    {
        register_struct_def(ctx, name, node);
    }

    return node;
}

Type *parse_type_obj(ParserContext *ctx, Lexer *l)
{
    // Parse the base type (int, U32, MyStruct, etc.)
    Type *t = parse_type_base(ctx, l);

    // Handle Pointers
    while (lexer_peek(l).type == TOK_OP && lexer_peek(l).start[0] == '*')
    {
        lexer_next(l); // eat *
        // Wrap the current type in a Pointer type
        Type *ptr = type_new(TYPE_POINTER);
        ptr->inner = t;
        t = ptr;
    }

    return t;
}

ASTNode *parse_enum(ParserContext *ctx, Lexer *l)
{
    lexer_next(l);
    Token n = lexer_next(l);

    // 1. Check for Generic <T>
    char *gp = NULL;
    if (lexer_peek(l).type == TOK_LANGLE)
    {
        lexer_next(l); // eat <
        Token g = lexer_next(l);
        gp = token_strdup(g);
        lexer_next(l); // eat >
        register_generic(ctx, n.start ? token_strdup(n) : "anon");
    }

    lexer_next(l); // eat {

    ASTNode *h = 0, *tl = 0;
    int v = 0;
    char *ename = token_strdup(n); // Store enum name

    while (1)
    {
        skip_comments(l);
        Token t = lexer_peek(l);
        if (t.type == TOK_RBRACE)
        {
            lexer_next(l);
            break;
        }
        if (t.type == TOK_COMMA)
        {
            lexer_next(l);
            continue;
        }

        if (t.type == TOK_IDENT)
        {
            Token vt = lexer_next(l);
            char *vname = token_strdup(vt);

            // 2. Parse Payload Type (Ok(int))
            Type *payload = NULL;
            if (lexer_peek(l).type == TOK_LPAREN)
            {
                lexer_next(l);
                payload = parse_type_obj(ctx, l);
                if (lexer_next(l).type != TOK_RPAREN)
                {
                    zpanic_at(lexer_peek(l), "Expected )");
                }
            }

            ASTNode *va = ast_create(NODE_ENUM_VARIANT);
            va->variant.name = vname;
            va->variant.tag_id = v++;      // Use tag_id instead of value
            va->variant.payload = payload; // Store Type*

            // Register Variant (Mangled name to avoid collisions: Result_Ok)
            char mangled[256];
            sprintf(mangled, "%s_%s", ename, vname);
            register_enum_variant(ctx, ename, mangled, va->variant.tag_id);

            // Handle explicit assignment: Ok = 5
            if (lexer_peek(l).type == TOK_OP && *lexer_peek(l).start == '=')
            {
                lexer_next(l);
                va->variant.tag_id = atoi(lexer_next(l).start);
                v = va->variant.tag_id + 1;
            }

            if (!h)
            {
                h = va;
            }
            else
            {
                tl->next = va;
            }
            tl = va;
        }
        else
        {
            lexer_next(l);
        }
    }

    // Auto-prefix enum name if in module context
    if (ctx->current_module_prefix && !gp)
    { // Don't prefix generic templates
        char *prefixed_name = xmalloc(strlen(ctx->current_module_prefix) + strlen(ename) + 2);
        sprintf(prefixed_name, "%s_%s", ctx->current_module_prefix, ename);
        free(ename);
        ename = prefixed_name;
    }

    ASTNode *node = ast_create(NODE_ENUM);
    node->enm.name = ename;

    node->enm.variants = h;
    node->enm.generic_param = gp; // 3. Store generic param

    if (gp)
    {
        node->enm.is_template = 1;
        register_template(ctx, node->enm.name, node);
    }

    add_to_enum_list(ctx, node); // Register globally

    return node;
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

    // Resolve import path using unified search logic
    char *current_dir = NULL;
    char *last_slash = strrchr(g_current_filename, '/');
    if (last_slash)
    {
        size_t dir_len = last_slash - g_current_filename;
        current_dir = xmalloc(dir_len + 1);
        strncpy(current_dir, g_current_filename, dir_len);
        current_dir[dir_len] = 0;
    }

    char *resolved = zc_resolve_import_path(fn, current_dir);
    if (resolved)
    {
        free(fn);
        fn = resolved;
    }
    // If not resolved, keep original fn - will error later when trying to open

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
    sprintf(bin, "%s.bin", filename);
    // Suppress GCC output
    sprintf(cmd, "gcc -I./std %s -o %s > " ZC_NULL_DEVICE " 2>&1", filename, bin);
    int res = system(cmd);
    if (res != 0)
    {
        zpanic_at(lexer_peek(l), "Comptime compilation failed for:\n%s", code);
    }

    char out_file[1024];
    sprintf(out_file, "%s.out", filename);
    sprintf(cmd, "./%s > %s", bin, out_file);
    if (system(cmd) != 0)
    {
        zpanic_at(lexer_peek(l), "Comptime execution failed");
    }

    char *output_src = load_file(out_file);
    if (!output_src)
    {
        output_src = xstrdup(""); // Empty output is valid
    }

    // Cleanup
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
