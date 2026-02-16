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

// Forward decl for LSP
int lsp_main(int argc, char **argv);

void print_search_paths()
{
    printf("Search paths:\n");
    printf("  ./\n");
    printf("  ./std/\n");
    printf("  /usr/local/share/zenc\n");
    printf("  /usr/share/zenc\n");
}

void print_version()
{
    printf(COLOR_BOLD "zc" COLOR_RESET " %s\n", ZEN_VERSION);
}

void print_usage()
{
    printf(COLOR_BOLD "Zen C" COLOR_RESET " - The language of monks\n\n");
    printf(COLOR_BOLD "Usage:" COLOR_RESET
                      " zc [command] [options] <file.zc> [extra files...]\n\n");
    printf(COLOR_BOLD COLOR_YELLOW "Commands:" COLOR_RESET "\n");
    printf("  " COLOR_GREEN "run" COLOR_RESET "          Compile and run the program\n");
    printf("  " COLOR_GREEN "build" COLOR_RESET "        Compile to executable\n");
    printf("  " COLOR_GREEN "check" COLOR_RESET "        Check for errors only\n");
    printf("  " COLOR_GREEN "repl" COLOR_RESET "         Start Interactive REPL\n");
    printf("  " COLOR_GREEN "transpile" COLOR_RESET
           "    Transpile to C code only (no compilation)\n");
    printf("  " COLOR_GREEN "lsp" COLOR_RESET "          Start Language Server\n");
    printf("\n" COLOR_BOLD COLOR_YELLOW "Options:" COLOR_RESET "\n");
    printf("  " COLOR_CYAN "-o" COLOR_RESET " <file>       Output executable name\n");
    printf("  " COLOR_CYAN "-O" COLOR_RESET "<level>       Optimization level\n");
    printf("  " COLOR_CYAN "-g" COLOR_RESET "              Debug info\n");
    printf("  " COLOR_CYAN "-c" COLOR_RESET "              Compile only (produce .o)\n");
    printf("  " COLOR_CYAN "-v" COLOR_RESET ", " COLOR_CYAN "--verbose" COLOR_RESET
           "   Verbose output\n");
    printf("  " COLOR_CYAN "-q" COLOR_RESET ", " COLOR_CYAN "--quiet" COLOR_RESET
           "     Quiet output\n");
    printf("  " COLOR_CYAN "--emit-c" COLOR_RESET "        Keep generated C file (out.c)\n");
    printf("  " COLOR_CYAN "--keep-comments" COLOR_RESET " Preserve comments in output C\n");
    printf("  " COLOR_CYAN "--freestanding" COLOR_RESET "  Freestanding mode (no stdlib)\n");
    printf("  " COLOR_CYAN "--cc" COLOR_RESET
           " <compiler> C compiler to use (gcc, clang, tcc, zig)\n");
    printf("  " COLOR_CYAN "--typecheck" COLOR_RESET "     Enable semantic analysis\n");
    printf("  " COLOR_CYAN "--json" COLOR_RESET "          Emit diagnostics as JSON\n");
    printf("  " COLOR_CYAN "--no-zen" COLOR_RESET "        Disable Zen facts\n");
    printf("  " COLOR_CYAN "--cpp" COLOR_RESET "           Use C++ mode\n");
    printf("  " COLOR_CYAN "--objective-c" COLOR_RESET "   Use Objective-C mode\n");
    printf("  " COLOR_CYAN "--cuda" COLOR_RESET "          Use CUDA mode (requires nvcc)\n");
    printf("  " COLOR_CYAN "--help" COLOR_RESET "          Print this help message\n");
    printf("  " COLOR_CYAN "--version" COLOR_RESET "       Print version information\n");
}

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
                    strcpy(g_config.cc, cc_arg);
                }
            }
        }
        else if (strcmp(arg, "-o") == 0)
        {
            if (i + 1 < argc)
            {
                g_config.output_file = argv[++i];
            }
        }
        else if (strncmp(arg, "-O", 2) == 0)
        {
            // Add to CFLAGS
            strcat(g_config.gcc_flags, " ");
            strcat(g_config.gcc_flags, arg);
        }
        else if (strcmp(arg, "-g") == 0)
        {
            strcat(g_config.gcc_flags, " -g");
        }
        else if (arg[0] == '-')
        {
            // Unknown flag or C flag
            strcat(g_config.gcc_flags, " ");
            strcat(g_config.gcc_flags, arg);
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

    ctx.hoist_out = tmpfile(); // Temp file for plugin hoisting
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
                ASTNode *tail = root;
                while (tail->next)
                {
                    tail = tail->next;
                }
                tail->next = extra_root;
            }

            if (real_path)
            {
                free(real_path);
            }
        }
    }

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
    const char *temp_source_file = "out.c";
    if (g_config.use_cuda)
    {
        temp_source_file = "out.cu";
    }
    else if (g_config.use_cpp)
    {
        temp_source_file = "out.cpp";
    }
    else if (g_config.use_objc)
    {
        temp_source_file = "out.m";
    }

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
    char cmd[16384];
    char *outfile = g_config.output_file ? g_config.output_file : "a.out";

    const char *thread_flag = g_parser_ctx->has_async ? "-lpthread" : "";
    const char *math_flag = "-lm";

    if (z_is_windows())
    {
        // Windows might use different flags or none for math/threads
        math_flag = "";
        if (g_parser_ctx->has_async)
        {
            thread_flag = "";
        }
    }

    // If using cosmocc, it handles these usually, but keeping them is okay for Linux targets

    // Construct extra C sources string
    char extra_c_sources[4096] = {0};
    for (int i = 0; i < g_config.c_file_count; i++)
    {
        strcat(extra_c_sources, " ");
        strcat(extra_c_sources, g_config.c_files[i]);
    }

    // Construct linker flags
    char linker_flags[1024] = {0};
    strcpy(linker_flags, g_link_flags);
    if (z_is_windows())
    {
        strcat(linker_flags, " -lws2_32");
    }

    snprintf(cmd, sizeof(cmd), "%s %s %s %s %s -o %s %s %s %s %s -I./src %s", g_config.cc,
             g_config.gcc_flags, g_cflags, g_config.is_freestanding ? "-ffreestanding" : "",
             g_config.quiet ? "-w" : "", outfile, temp_source_file, extra_c_sources, math_flag,
             thread_flag, linker_flags);

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
        // remove("out.c"); // Keep it for debugging for now or follow flag
        remove(temp_source_file);
    }

    if (g_config.mode_run)
    {
        char run_cmd[2048];
        if (z_is_windows())
        {
            sprintf(run_cmd, "%s", outfile);
        }
        else
        {
            sprintf(run_cmd, "./%s", outfile);
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
