#include "codegen/codegen.h"
#include "parser/parser.h"
#include "plugins/plugin_manager.h"
#include "repl/repl.h"
#include "zen/zen_facts.h"
#include "zprep.h"
#include "analysis/typecheck.h"
#include "codegen/compat.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "utils/cmd.h"

// Forward decl for LSP
int lsp_main(int argc, char **argv);

int main(int argc, char **argv)
{
    memset(&g_config, 0, sizeof(g_config));
    if (z_is_windows())
    {
        strcpy(g_config.cc, "gcc.exe");
    }
    else
    {
        strcpy(g_config.cc, "gcc");
    }

    if (argc < 2)
    {
        print_usage();
        return 1;
    }

    // Parse command
    char *command = argv[1];
    int arg_start = 2;

    if (strcmp(command, "lsp") == 0)
    {
        return lsp_main(argc, argv);
    }
    else if (strcmp(command, "repl") == 0)
    {
        run_repl(argv[0]); // Pass self path for recursive calls
        return 0;
    }
    else if (strcmp(command, "transpile") == 0)
    {
        g_config.mode_transpile = 1;
        g_config.emit_c = 1; // Transpile implies emitting C
    }
    else if (strcmp(command, "run") == 0)
    {
        g_config.mode_run = 1;
    }
    else if (strcmp(command, "check") == 0)
    {
        g_config.mode_check = 1;
    }
    else if (strcmp(command, "build") == 0)
    {
        // default mode
    }
    else if (strcmp(command, "--help") == 0 || strcmp(command, "-h") == 0)
    {
        print_usage();
        return 0;
    }
    else if (command[0] == '-')
    {
        // implicit build or run? assume build if starts with flag, but usually
        // command first If file provided directly: "zc file.zc" -> build
        if (strchr(command, '.'))
        {
            // treat as filename
            g_config.input_file = command;
            arg_start = 2; // already consumed
        }
        else
        {
            // Flags
            arg_start = 1;
        }
    }
    else
    {
        // Check if file
        if (strchr(command, '.'))
        {
            g_config.input_file = command;
            arg_start = 2;
        }
    }

    // Parse args
    for (int i = arg_start; i < argc; i++)
    {
        char *arg = argv[i];
        if (strcmp(arg, "--emit-c") == 0)
        {
            g_config.emit_c = 1;
        }
        else if (strcmp(arg, "--json") == 0)
        {
            g_config.json_output = 1;
        }
        else if (strcmp(arg, "--keep-comments") == 0)
        {
            g_config.keep_comments = 1;
        }
        else if (strcmp(arg, "--version") == 0 || strcmp(arg, "-V") == 0)
        {
            print_version();
            return 0;
        }
        else if (strcmp(arg, "--paths") == 0)
        {
            print_search_paths();
            return 0;
        }
        else if (strcmp(arg, "--verbose") == 0 || strcmp(arg, "-v") == 0)
        {
            g_config.verbose = 1;
        }
        else if (strcmp(arg, "--quiet") == 0 || strcmp(arg, "-q") == 0)
        {
            g_config.quiet = 1;
        }
        else if (strcmp(arg, "--no-zen") == 0)
        {
            g_config.no_zen = 1;
        }
        else if (strcmp(arg, "--typecheck") == 0)
        {
            g_config.use_typecheck = 1;
        }
        else if (strcmp(arg, "--freestanding") == 0)
        {
            g_config.is_freestanding = 1;
        }
        else if (strcmp(arg, "--cpp") == 0)
        {
            strcpy(g_config.cc, "g++");
            g_config.use_cpp = 1;
        }
        else if (strcmp(arg, "--cuda") == 0)
        {
            strcpy(g_config.cc, "nvcc");
            g_config.use_cuda = 1;
            g_config.use_cpp = 1; // CUDA implies C++ mode.
        }
        else if (strcmp(arg, "--objc") == 0 || strcmp(arg, "--objective-c") == 0)
        {
            g_config.use_objc = 1;
        }
        else if (strcmp(arg, "--check") == 0)
        {
            g_config.mode_check = 1;
        }
        else if (strcmp(arg, "--cc") == 0)
        {
            if (i + 1 < argc)
            {
                char *cc_arg = argv[++i];
                // Handle "zig" shorthand for "zig cc"
                if (strcmp(cc_arg, "zig") == 0)
                {
                    strcpy(g_config.cc, "zig cc");
                }
                else
                {
                    snprintf(g_config.cc, sizeof(g_config.cc), "%s", cc_arg);
                }
            }
        }
        else if (strncmp(arg, "-o", 2) == 0)
        {
            if (strlen(arg) > 2)
            {
                g_config.output_file = arg + 2;
            }
            else if (i + 1 < argc)
            {
                g_config.output_file = argv[++i];
            }
            else
            {
                fprintf(stderr, COLOR_BOLD COLOR_RED "error" COLOR_RESET
                                                     ": missing output filename after '-o'\n");
                return 1;
            }
        }
        else if (strncmp(arg, "-I", 2) == 0)
        {
            strcat(g_config.gcc_flags, " -I");
            if (strlen(arg) > 2)
            {
                strcat(g_config.gcc_flags, arg + 2);
            }
            else if (i + 1 < argc)
            {
                strcat(g_config.gcc_flags, argv[++i]);
            }
        }
        else if (strncmp(arg, "-L", 2) == 0 || strncmp(arg, "-l", 2) == 0)
        {
            char prefix[3] = {arg[0], arg[1], '\0'};
            if (strlen(g_link_flags) > 0)
            {
                strcat(g_link_flags, " ");
            }
            strcat(g_link_flags, prefix);

            if (strlen(arg) > 2)
            {
                strcat(g_link_flags, arg + 2);
            }
            else if (i + 1 < argc)
            {
                strcat(g_link_flags, argv[++i]);
            }
        }
        else if (strncmp(arg, "-O", 2) == 0)
        {
            // Add to CFLAGS
            size_t len = strlen(g_config.gcc_flags);
            snprintf(g_config.gcc_flags + len, sizeof(g_config.gcc_flags) - len, " -O");
            if (strlen(arg) > 2)
            {
                len = strlen(g_config.gcc_flags);
                snprintf(g_config.gcc_flags + len, sizeof(g_config.gcc_flags) - len, "%s", arg + 2);
            }
            else if (i + 1 < argc)
            {
                len = strlen(g_config.gcc_flags);
                snprintf(g_config.gcc_flags + len, sizeof(g_config.gcc_flags) - len, "%s",
                         argv[++i]);
            }
        }
        else if (strcmp(arg, "-g") == 0)
        {
            size_t len = strlen(g_config.gcc_flags);
            snprintf(g_config.gcc_flags + len, sizeof(g_config.gcc_flags) - len, " -g");
        }
        else if (arg[0] == '-')
        {
            // Unknown flag or C flag
            size_t len = strlen(g_config.gcc_flags);
            snprintf(g_config.gcc_flags + len, sizeof(g_config.gcc_flags) - len, " %s", arg);
        }
        else
        {
            if (!g_config.input_file)
            {
                g_config.input_file = arg;
            }
            else if (g_config.extra_file_count < 64)
            {
                g_config.extra_files[g_config.extra_file_count++] = arg;
            }
        }
    }

    if (!g_config.input_file)
    {
        fprintf(stderr, COLOR_BOLD COLOR_RED "error" COLOR_RESET ": no input file specified\n");
        return 1;
    }

    g_current_filename = g_config.input_file;

    // Load file
    char *src = load_file(g_config.input_file);
    if (!src)
    {
        fprintf(stderr, COLOR_BOLD COLOR_RED "error" COLOR_RESET ": could not read file '%s'\n",
                g_config.input_file);
        return 1;
    }

    init_builtins();
    zen_init();

    // Initialize Plugin Manager
    zptr_plugin_mgr_init();

    // Load all configurations (system, hidden project, visible project)
    load_all_configs();

    // Parse context init
    ParserContext ctx;
    memset(&ctx, 0, sizeof(ctx));

    // Scan for build directives (e.g. //> link: -lm)
    scan_build_directives(&ctx, src);

    Lexer l;
    lexer_init(&l, src);

    ctx.hoist_out = z_tmpfile();
    if (!ctx.hoist_out)
    {
        perror("tmpfile for hoisting");
        return 1;
    }
    g_parser_ctx = &ctx;

    z_setup_terminal();

    double start_time = z_get_monotonic_time();

    if (!g_config.quiet)
    {
        printf(COLOR_BOLD COLOR_GREEN "   Compiling" COLOR_RESET " %s\n", g_config.input_file);
        fflush(stdout);
    }

    ASTNode *root = parse_program(&ctx, &l);

    if (!root)
    {
        // Parse failed
        return 1;
    }

    // Parse extra input files and merge into AST
    if (g_config.extra_file_count > 0)
    {
        // Mark primary file as imported to prevent re-parsing
        char *primary_real = realpath(g_config.input_file, NULL);
        if (primary_real)
        {
            mark_file_imported(&ctx, primary_real);
            free(primary_real);
        }

        for (int ef = 0; ef < g_config.extra_file_count; ef++)
        {
            const char *extra_path = g_config.extra_files[ef];
            char *real_path = realpath(extra_path, NULL);
            const char *path = real_path ? real_path : extra_path;

            const char *ext = strrchr(path, '.');
            if (ext && ZC_IS_BACKEND_EXT(ext))
            {
                if (g_config.c_file_count < 64)
                {
                    g_config.c_files[g_config.c_file_count++] =
                        real_path ? strdup(real_path) : strdup(extra_path);
                }
                if (real_path)
                {
                    free(real_path);
                }
                continue;
            }

            if (is_file_imported(&ctx, path))
            {
                if (real_path)
                {
                    free(real_path);
                }
                continue;
            }
            mark_file_imported(&ctx, path);

            char *extra_src = load_file(path);
            if (!extra_src)
            {
                fprintf(stderr,
                        COLOR_BOLD COLOR_RED "error" COLOR_RESET ": could not read file '%s'\n",
                        extra_path);
                return 1;
            }

            if (!g_config.quiet)
            {
                printf(COLOR_BOLD COLOR_GREEN "   Compiling" COLOR_RESET " %s\n", extra_path);
                fflush(stdout);
            }

            const char *saved_fn = g_current_filename;
            g_current_filename = (char *)path;

            scan_build_directives(&ctx, extra_src);

            Lexer extra_l;
            lexer_init(&extra_l, extra_src);
            ASTNode *extra_root = parse_program_nodes(&ctx, &extra_l);
            g_current_filename = (char *)saved_fn;

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
                free(real_path);
            }
        }
    }

    propagate_drop_traits(&ctx);

    if (!validate_types(&ctx))
    {
        // Type validation failed
        return 1;
    }

    if (!g_config.use_typecheck && !g_config.mode_check)
    {
        int move_result = check_moves_only(&ctx, root);
        if (move_result != 0)
        {
            return 1;
        }
    }

    // Run Semantic Analysis (Type Checker) if enabled or in check mode
    int tc_result = 0;
    if (g_config.use_typecheck || g_config.mode_check)
    {
        tc_result = check_program(&ctx, root);
        if (tc_result != 0 && !g_config.mode_check)
        {
            return 1; // Stop if type errors found
        }
    }

    // In check mode, exit after type checking
    if (g_config.mode_check)
    {
        if (tc_result != 0)
        {
            return 1;
        }
        printf(COLOR_BOLD COLOR_GREEN "       Check" COLOR_RESET " passed\n");
        return 0;
    }

    // Determine temporary filename based on mode
    char temp_source_buf[1024];
    const char *ext = ".c";
    if (g_config.use_cuda)
    {
        ext = ".cu";
    }
    else if (g_config.use_cpp)
    {
        ext = ".cpp";
    }
    else if (g_config.use_objc)
    {
        ext = ".m";
    }

    if (g_config.output_file)
    {
        snprintf(temp_source_buf, sizeof(temp_source_buf), "%s%s", g_config.output_file, ext);
    }
    else
    {
        snprintf(temp_source_buf, sizeof(temp_source_buf), "out%s", ext);
    }
    const char *temp_source_file = temp_source_buf;

    // Codegen to C/C++/CUDA
    FILE *out = fopen(temp_source_file, "w");
    if (!out)
    {
        perror("fopen temp output");
        return 1;
    }

    codegen_node(&ctx, root, out);
    fclose(out);

    if (g_config.mode_transpile)
    {
        if (g_config.output_file)
        {
            // If user specified -o, rename temp file to that
            if (rename(temp_source_file, g_config.output_file) != 0)
            {
                perror("rename output");
                return 1;
            }
            if (!g_config.quiet)
            {
                printf(COLOR_BOLD COLOR_CYAN "  Transpiled" COLOR_RESET " to %s\n",
                       g_config.output_file);
            }
        }
        else
        {
            if (!g_config.quiet)
            {
                printf(COLOR_BOLD COLOR_CYAN "  Transpiled" COLOR_RESET " to %s\n",
                       temp_source_file);
            }
        }
        // Done, no C compilation
        return 0;
    }

    // Compile C
    char cmd[32768];
    char *outfile = g_config.output_file ? g_config.output_file : "a.out";

    char extra_c_sources[4096] = {0};
    for (int i = 0; i < g_config.c_file_count; i++)
    {
        size_t len = strlen(extra_c_sources);
        snprintf(extra_c_sources + len, sizeof(extra_c_sources) - len, " %s", g_config.c_files[i]);
    }

    // Build command
    build_compile_command(cmd, sizeof(cmd), outfile, temp_source_file, extra_c_sources);

    if (g_config.verbose)
    {
        printf(COLOR_BOLD COLOR_BLUE "     Command" COLOR_RESET " %s\n", cmd);
    }

    int ret = system(cmd);
    if (ret != 0)
    {
        fprintf(stderr, COLOR_BOLD COLOR_RED "error" COLOR_RESET ": C compilation failed\n");
        if (!g_config.emit_c)
        {
            remove(temp_source_file);
        }
        return 1;
    }

    if (!g_config.emit_c)
    {
        remove(temp_source_file);
    }

    if (g_config.mode_run)
    {
        char run_cmd[2048];
        int n;
        if (z_is_windows())
        {
            n = snprintf(run_cmd, sizeof(run_cmd), "%s", outfile);
        }
        else
        {
            n = snprintf(run_cmd, sizeof(run_cmd), "./%s", outfile);
        }

        if (n < 0 || n >= (int)sizeof(run_cmd))
        {
            fprintf(stderr, COLOR_BOLD COLOR_RED "error" COLOR_RESET ": run command too long\n");
            return 1;
        }
        if (!g_config.quiet)
        {
            printf(COLOR_BOLD COLOR_GREEN "     Running" COLOR_RESET " %s\n", outfile);
            fflush(stdout);
        }
        ret = system(run_cmd);
        remove(outfile);
        zptr_plugin_mgr_cleanup();
        zen_trigger_global();
#if defined(WIFEXITED) && defined(WEXITSTATUS)
        return WIFEXITED(ret) ? WEXITSTATUS(ret) : ret;
#else
        return ret;
#endif
    }

    zptr_plugin_mgr_cleanup();
    zen_trigger_global();

    double end_time = z_get_monotonic_time();
    double time_taken = end_time - start_time;

    if (!g_config.quiet && !g_config.mode_run && !g_config.mode_check)
    {
        if (g_warning_count > 0)
        {
            printf(COLOR_BOLD COLOR_GREEN "    Finished" COLOR_RESET
                                          " build in %.2fs with %d warning%s\n",
                   time_taken, g_warning_count, g_warning_count == 1 ? "" : "s");
        }
        else
        {
            printf(COLOR_BOLD COLOR_GREEN "    Finished" COLOR_RESET " build in %.2fs\n",
                   time_taken);
        }
        fflush(stdout);
    }

    return 0;
}
