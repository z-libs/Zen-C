// SPDX-License-Identifier: MIT
#include "driver.h"
#include "../parser/parser.h"
#include "../codegen/codegen.h"
#include "../codegen/compat.h"
#include "../analysis/typecheck.h"
#include "../analysis/move_check.h"
#include "../analysis/const_fold.h"
#include "../zen/zen_facts.h"
#include "../zen/zen_doc.h"
#include "../utils/cmd.h"
#include "../plugins/plugin_manager.h"
#include "../utils/colors.h"
#include "../platform/os.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if !ZC_OS_WINDOWS
#include <sys/wait.h>
#endif

// Forward declarations for functions defined in other modules
extern char *load_file(const char *fn);
extern void init_builtins(void);
extern void load_all_configs(CompilerConfig *cfg);
extern void scan_build_directives(ParserContext *ctx, const char *src);
extern void propagate_vector_inner_types(ParserContext *ctx);
extern void propagate_drop_traits(ParserContext *ctx);
extern int validate_types(ParserContext *ctx);

int driver_run(ZenCompiler *compiler)
{
    // Backend detection for @cfg purposes
    if (z_path_match_compiler(compiler->config.cc, "tcc"))
    {
        zvec_push_Str(&compiler->config.cfg_defines, xstrdup("__TINYC__"));
    }
    else if (z_path_match_compiler(compiler->config.cc, "clang"))
    {
        zvec_push_Str(&compiler->config.cfg_defines, xstrdup("__clang__"));
    }
    else if (z_path_match_compiler(compiler->config.cc, "zig"))
    {
        zvec_push_Str(&compiler->config.cfg_defines, xstrdup("__ZIG__"));
    }
    else if (z_path_match_compiler(compiler->config.cc, "filcc"))
    {
        zvec_push_Str(&compiler->config.cfg_defines, xstrdup("__FILCC__"));
    }

    init_builtins();
    zen_init();

#ifndef ZC_NO_PLUGINS
    zptr_plugin_mgr_init();
#endif

    load_all_configs(&compiler->config);

    int result = driver_compile(compiler);

#ifndef ZC_NO_PLUGINS
    if (compiler->config.verbose)
    {
        printf(COLOR_BOLD COLOR_CYAN "   Cleaning up" COLOR_RESET " plugins...\n");
        fflush(stdout);
    }
    zptr_plugin_mgr_cleanup();
#endif

    if (compiler->config.verbose)
    {
        printf(COLOR_BOLD COLOR_CYAN "   Evaluating" COLOR_RESET " Zen facts...\n");
        fflush(stdout);
    }
    zen_trigger_global(&compiler->config);

    return result;
}

int driver_compile(ZenCompiler *compiler)
{
    ParserContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.compiler = compiler;
    ctx.config = &compiler->config;
    g_parser_ctx = &ctx;

    // Set runtime hooks based on configuration
    if (ctx.config->misra_mode)
    {
        extern void misra_check_identifier_collision(Token tok, const char *name1,
                                                     const char *name2, int limit);
        extern void misra_check_preprocessor_expression_parser(ParserContext * ctx, Token tok,
                                                               const char *expression);
        extern void misra_check_standard_macro_name(Token tok, const char *name);

        ctx.hook_check_identifier_collision = misra_check_identifier_collision;
        ctx.hook_check_preprocessor_expr = misra_check_preprocessor_expression_parser;
        ctx.hook_check_standard_macro_name = misra_check_standard_macro_name;
    }

    // Plugin hooks (always available)
    ctx.hook_find_plugin = (void *(*)(const char *))zptr_find_plugin;
    ctx.hook_plugin_init_api = (void (*)(void *, const char *, int, CompilerConfig *))zptr_init_api;

    // Zen hooks
    ctx.hook_zen_trigger = (int (*)(int, Token, CompilerConfig *))zen_trigger_at;

    char *src = load_file(compiler->config.input_file);
    if (!src)
    {
        fprintf(stderr, COLOR_BOLD COLOR_RED "error" COLOR_RESET ": could not read file '%s'\n",
                compiler->config.input_file);
        return 1;
    }

    scan_build_directives(&ctx, src);

    Lexer l;
    lexer_init(&l, src, ctx.config);

    ctx.cg.hoist_out = z_tmpfile();
    if (!ctx.cg.hoist_out)
    {
        perror("tmpfile for hoisting");
        return 1;
    }

    compiler->start_time = z_get_monotonic_time();

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

    // Handle extra files
    if (compiler->config.extra_files.length > 0)
    {
        char *primary_real = realpath(compiler->config.input_file, NULL);
        if (primary_real)
        {
            mark_file_imported(&ctx, primary_real);
            zfree(primary_real);
        }

        for (size_t ef = 0; ef < compiler->config.extra_files.length; ef++)
        {
            const char *extra_path = compiler->config.extra_files.data[ef];
            char *real_path = realpath(extra_path, NULL);
            const char *path = real_path ? real_path : extra_path;

            const char *ext = strrchr(path, '.');
            if (ext && ZC_IS_BACKEND_EXT(ext))
            {
                zvec_push_Str(&compiler->config.c_files, xstrdup(path));
                if (real_path)
                {
                    zfree(real_path);
                }
                continue;
            }

            if (is_file_imported(&ctx, path))
            {
                if (real_path)
                {
                    zfree(real_path);
                }
                continue;
            }
            mark_file_imported(&ctx, path);

            char *extra_src = load_file(path);
            if (!extra_src)
            {
                zerror_at((Token){0}, "could not read file '%s'", extra_path);
                if (real_path)
                {
                    zfree(real_path);
                }
                return 1;
            }

            if (compiler->config.verbose)
            {
                printf(COLOR_BOLD COLOR_GREEN "   Compiling" COLOR_RESET " %s\n", extra_path);
                fflush(stdout);
            }

            scan_build_directives(&ctx, extra_src);
            Lexer extra_l;
            lexer_init(&extra_l, extra_src, ctx.config);
            ASTNode *extra_root = parse_program_nodes(&ctx, &extra_l);

            if (extra_root)
            {
                ASTNode *tail = root->root.children;
                if (!tail)
                {
                    root->root.children = extra_root;
                }
                else
                {
                    while (tail->next)
                    {
                        tail = tail->next;
                    }
                    tail->next = extra_root;
                }
            }
            if (real_path)
            {
                zfree(real_path);
            }
        }
    }

    // Semantic Analysis & Validation
    if (!compiler->config.mode_doc || compiler->config.use_typecheck)
    {
        propagate_vector_inner_types(&ctx);
        propagate_drop_traits(&ctx);
        fix_type_refs_has_drop(&ctx);

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

    int tc_result = 0;
    if (compiler->config.use_typecheck || compiler->config.mode_check)
    {
        if (compiler->config.verbose)
        {
            printf(COLOR_BOLD COLOR_GREEN "   Analyzing" COLOR_RESET " %s\n",
                   compiler->config.input_file);
            fflush(stdout);
        }
        tc_result = check_program(&ctx, root);
        if (tc_result != 0 && !compiler->config.mode_check)
        {
            return 1;
        }
    }

    if (compiler->config.mode_doc)
    {
        generate_docs(&ctx, root);
        return 0;
    }

    if (compiler->config.mode_check)
    {
        if (tc_result != 0 || compiler->error_count > 0)
        {
            fprintf(stderr,
                    COLOR_BOLD COLOR_RED "       Check" COLOR_RESET " failed with %d errors\n",
                    compiler->error_count);
            return 1;
        }
        return 0;
    }

    // Determine output file extension from backend
    const CodegenBackend *backend = codegen_get_backend(compiler->config.backend_name);
    const char *ext_p = backend ? backend->extension : ".c";
    char temp_source_buf[1024];

    if (!compiler->config.output_file)
    {
        char *base = z_basename(compiler->config.input_file);
        char *stripped = z_strip_ext(base);
        zfree(base);
        if (compiler->config.mode_transpile)
        {
            char *with_ext = xmalloc(strlen(stripped) + strlen(ext_p) + 1);
            sprintf(with_ext, "%s%s", stripped, ext_p);
            compiler->config.output_file = with_ext;
        }
        else
        {
            compiler->config.output_file = stripped;
        }
    }

    if (compiler->config.output_file)
    {
        size_t out_len = strlen(compiler->config.output_file);
        size_t ext_len = strlen(ext_p);
        if (out_len >= ext_len &&
            strcmp(compiler->config.output_file + out_len - ext_len, ext_p) == 0)
        {
            snprintf(temp_source_buf, sizeof(temp_source_buf), "%s.tmp%s",
                     compiler->config.output_file, ext_p);
        }
        else
        {
            snprintf(temp_source_buf, sizeof(temp_source_buf), "%s%s", compiler->config.output_file,
                     ext_p);
        }
    }
    else
    {
        snprintf(temp_source_buf, sizeof(temp_source_buf), "out%s", ext_p);
    }

    FILE *out_f = fopen(temp_source_buf, "w");
    if (!out_f)
    {
        perror("fopen temp output");
        return 1;
    }
    emitter_init_file(&ctx.cg.emitter, out_f);
    codegen_node(&ctx, root);
    fclose(out_f);

    if (compiler->config.mode_transpile)
    {
        if (rename(temp_source_buf, compiler->config.output_file) != 0)
        {
            perror("rename output");
            return 1;
        }
        return 0;
    }

    // Skip C compilation for backends that don't need it (e.g. AST dump)
    if (backend && !backend->needs_cc)
    {
        if (compiler->config.output_file &&
            strcmp(temp_source_buf, compiler->config.output_file) != 0)
        {
            rename(temp_source_buf, compiler->config.output_file);
        }
        else if (!compiler->config.emit_c)
        {
            remove(temp_source_buf);
        }
        return 0;
    }

    // Compile C
    const char *outfile = compiler->config.output_file ? compiler->config.output_file
                                                       : (z_is_windows() ? "a.exe" : "a.out");
    ArgList compile_args;
    arg_list_init(&compile_args);
    build_compile_arg_list(&compile_args, outfile, temp_source_buf, &compiler->config);

    if (compiler->config.verbose)
    {
        printf(COLOR_BOLD COLOR_BLUE "     Command" COLOR_RESET);
        for (size_t k = 0; k < compile_args.count; k++)
        {
            printf(" %s", compile_args.args[k]);
        }
        printf("\n");
    }

    int ret = arg_run(&compile_args);
    arg_list_free(&compile_args);

    if (!compiler->config.emit_c)
    {
        remove(temp_source_buf);
    }
    if (ret != 0)
    {
        return 1;
    }

    // Run if needed
    if (compiler->config.mode_run)
    {
        ArgList run_args;
        arg_list_init(&run_args);
        char exe_path[1024];
        if (z_is_windows())
        {
            snprintf(exe_path, sizeof(exe_path), "%s", outfile);
            if (access(exe_path, F_OK) != 0)
            {
                snprintf(exe_path, sizeof(exe_path), "%s.exe", outfile);
            }
        }
        else
        {
            snprintf(exe_path, sizeof(exe_path), "./%s", outfile);
        }

        arg_list_add(&run_args, exe_path);
        if (!compiler->config.quiet)
        {
            printf(COLOR_BOLD COLOR_GREEN "     Running" COLOR_RESET " %s\n", exe_path);
        }

        int run_ret = arg_run(&run_args);
        arg_list_free(&run_args);
        remove(exe_path);

#if defined(WIFEXITED) && defined(WEXITSTATUS)
        return WIFEXITED(run_ret) ? WEXITSTATUS(run_ret) : run_ret;
#else
        return run_ret;
#endif
    }

    double end_time = z_get_monotonic_time();
    if (!compiler->config.quiet)
    {
        printf(COLOR_BOLD COLOR_GREEN "    Finished" COLOR_RESET
                                      " build in %.2fs with %d errors and %d warnings\n",
               end_time - compiler->start_time, compiler->error_count, compiler->warning_count);
    }

    return 0;
}
