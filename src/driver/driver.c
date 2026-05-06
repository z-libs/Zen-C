
#include "driver.h"
#include "../parser/parser.h"
#include "../codegen/codegen.h"
#include "../analysis/typecheck.h"
#include "../analysis/move_check.h"
#include "../analysis/const_fold.h"
#include "../zen/zen_facts.h"
#include "../zen/zen_doc.h"
#include "../utils/cmd.h"
#include "../plugins/plugin_manager.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Forward declarations of helpers used in main.c that we might need to move or export
extern char *load_file(const char *path);
extern void mark_file_imported(ParserContext *ctx, const char *path);
extern int is_file_imported(ParserContext *ctx, const char *path);
extern void scan_build_directives(ParserContext *ctx, const char *src);
extern void propagate_vector_inner_types(ParserContext *ctx);
extern void propagate_drop_traits(ParserContext *ctx);
extern int validate_types(ParserContext *ctx);
extern void build_compile_arg_list(ArgList *list, const char *outfile, const char *temp_source);
extern void init_builtins(void);
extern void load_all_configs(void);

int driver_run(ZenCompiler *compiler)
{
    // This will be the main entry point from main.c
    // It should handle the high-level flow

    // For now, let's just implement driver_compile and call it
    return driver_compile(compiler);
}

int driver_compile(ZenCompiler *compiler)
{
    ParserContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.compiler = compiler;
    g_parser_ctx = &ctx;

    // Load file
    char *src = load_file(compiler->config.input_file);
    if (!src)
    {
        fprintf(stderr, COLOR_BOLD COLOR_RED "error" COLOR_RESET ": could not read file '%s'\n",
                compiler->config.input_file);
        return 1;
    }

    // Scan build directives
    scan_build_directives(&ctx, src);

    Lexer l;
    lexer_init(&l, src);

    ctx.hoist_out = z_tmpfile();
    if (!ctx.hoist_out)
    {
        perror("tmpfile for hoisting");
        return 1;
    }

    if (!compiler->config.quiet)
    {
        printf(COLOR_BOLD COLOR_GREEN "   Compiling" COLOR_RESET " %s...\n",
               compiler->config.input_file);
        fflush(stdout);
    }

    ASTNode *root = parse_program(&ctx, &l);
    if (!root)
    {
        return 1;
    }

    // Extra files logic... (omitted for brevity in first draft, will add)

    if (!compiler->config.mode_doc || compiler->config.use_typecheck)
    {
        propagate_vector_inner_types(&ctx);
        propagate_drop_traits(&ctx);

        if (!validate_types(&ctx))
        {
            return 1;
        }

        if (!compiler->config.use_typecheck && !compiler->config.mode_check)
        {
            if (check_moves_only(&ctx, root) != 0)
            {
                return 1;
            }
        }
    }

    // ... rest of the logic from main.c ...
    // This is a complex migration. I'll do it in chunks.

    return 0;
}
