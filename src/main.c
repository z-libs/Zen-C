#include "codegen/codegen.h"
#include "parser/parser.h"
#include "plugins/plugin_manager.h"
#include "repl/repl.h"
#include "zen/zen_facts.h"
#include "zprep.h"
#include "analysis/typecheck.h"
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

#ifndef ZEN_VERSION
#define ZEN_VERSION "0.1.0"
#endif

void print_version()
{
    printf("Zen C version %s\n", ZEN_VERSION);
}

void print_usage()
{
    printf("Usage: zc [command] [options] <file.zc>\n");
    printf("Commands:\n");
    printf("  run     Compile and run the program\n");
    printf("  build   Compile to executable\n");
    printf("  check   Check for errors only\n");
    printf("  repl    Start Interactive REPL\n");
    printf("  transpile Transpile to C code only (no compilation)\n");
    printf("  lsp     Start Language Server\n");
    printf("Options:\n");
    printf("  --help          Print this help message\n");
    printf("  --version       Print version information\n");
    printf("  -o <file>       Output executable name\n");
    printf("  --emit-c        Keep generated C file (out.c)\n");
    printf("  --keep-comments Preserve comments in output C file\n");
    printf("  --freestanding  Freestanding mode (no stdlib)\n");
    printf("  --cc <compiler> C compiler to use (gcc, clang, tcc, zig)\n");
    printf("  -O<level>       Optimization level\n");
    printf("  -g              Debug info\n");
    printf("  -v, --verbose   Verbose output\n");
    printf("  -q, --quiet     Quiet output\n");
    printf("  --json          Emit diagnostics as JSON objects\n");
    printf("  --typecheck     Enable semantic analysis (Typecheck)\n");
    printf("  --no-zen        Disable Zen facts\n");
    printf("  -c              Compile only (produce .o)\n");
    printf("  --cpp           Use C++ mode.\n");
    printf("  --objective-c   Use Objective-C mode.\n");
    printf("  --cuda          Use CUDA mode (requires nvcc).\n");
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
            else
            {
                printf("Multiple input files not supported yet.\n");
                return 1;
            }
        }
    }

    if (!g_config.input_file)
    {
        printf("Error: No input file specified.\n");
        return 1;
    }

    g_current_filename = g_config.input_file;

    // Load file
    char *src = load_file(g_config.input_file);
    if (!src)
    {
        printf("Error: Could not read file %s\n", g_config.input_file);
        return 1;
    }

    init_builtins();
    zen_init();

    // Initialize Plugin Manager
    zptr_plugin_mgr_init();

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

    if (!validate_types(&ctx))
    {
        // Type validation failed
        return 1;
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
        printf("Check passed.\n");
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
    char cmd[8192];
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

    snprintf(cmd, sizeof(cmd), "%s %s %s %s %s -o %s %s %s %s -I./src %s", g_config.cc,
             g_config.gcc_flags, g_cflags, g_config.is_freestanding ? "-ffreestanding" : "",
             g_config.quiet ? "-w" : "", outfile, temp_source_file, math_flag, thread_flag,
             g_link_flags);

    if (g_config.verbose)
    {
        printf("[CMD] %s\n", cmd);
    }

    int ret = system(cmd);
    if (ret != 0)
    {
        printf("C compilation failed.\n");
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
