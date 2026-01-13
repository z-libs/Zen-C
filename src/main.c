#include "codegen/codegen.h"
#include "compat/compat.h"
#include "parser/parser.h"
#include "plugins/plugin_manager.h"
#include "repl/repl.h"
#include "zen/zen_facts.h"
#include "zprep.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define ZC_DEFAULT_OUTPUT "a.exe"
#define ZC_PTHREAD_FLAG ""
#define ZC_MATH_FLAG ""
#else
#define ZC_DEFAULT_OUTPUT "a.out"
#define ZC_PTHREAD_FLAG "-lpthread"
#define ZC_MATH_FLAG "-lm"
#endif

// Forward decl for LSP
int lsp_main(int argc, char **argv);

void print_search_paths()
{
    printf("Search paths:\n");
    printf("  ./\n");
    printf("  ./std/\n");
    printf("  /usr/local/lib/zen/\n");
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
    printf("  -o <file>       Output executable name\n");
    printf("  --emit-c        Keep generated C file (out.c)\n");
    printf("  --freestanding  Freestanding mode (no stdlib)\n");
    printf("  --cc <compiler> C compiler to use (gcc, clang, tcc, zig)\n");
    printf("  -O<level>       Optimization level\n");
    printf("  -g              Debug info\n");
    printf("  -v, --verbose   Verbose output\n");
    printf("  -q, --quiet     Quiet output\n");
    printf("  -c              Compile only (produce .o)\n");
}

int main(int argc, char **argv)
{
    // Defaults
    memset(&g_config, 0, sizeof(g_config));
    strcpy(g_config.cc, "gcc");

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
        else if (strcmp(arg, "--verbose") == 0 || strcmp(arg, "-v") == 0)
        {
            g_config.verbose = 1;
        }
        else if (strcmp(arg, "--quiet") == 0 || strcmp(arg, "-q") == 0)
        {
            g_config.quiet = 1;
        }
        else if (strcmp(arg, "--freestanding") == 0)
        {
            g_config.is_freestanding = 1;
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

    // Register built-in plugins
    extern ZPlugin brainfuck_plugin;
    extern ZPlugin befunge_plugin;
    extern ZPlugin lisp_plugin;
    extern ZPlugin forth_plugin;
    extern ZPlugin regex_plugin;
    extern ZPlugin sql_plugin;

    zptr_register_plugin(&brainfuck_plugin);
    zptr_register_plugin(&befunge_plugin);
    zptr_register_plugin(&lisp_plugin);
    zptr_register_plugin(&forth_plugin);
    zptr_register_plugin(&regex_plugin);
    zptr_register_plugin(&sql_plugin);

    Lexer l;
    lexer_init(&l, src);

    ctx.hoist_out = tmpfile(); // Temp file for plugin hoisting
    if (!ctx.hoist_out)
    {
        perror("tmpfile for hoisting");
        return 1;
    }
    g_parser_ctx = &ctx;

    if (!g_config.quiet)
    {
        printf("[zc] Compiling %s...\n", g_config.input_file);
    }

    ASTNode *root = parse_program(&ctx, &l);
    if (!root)
    {
        // Parse failed
        return 1;
    }

    if (g_config.mode_check)
    {
        // Just verify
        printf("Check passed.\n");
        return 0;
    }

    // Codegen to C
    FILE *out = fopen("out.c", "w");
    if (!out)
    {
        perror("fopen out.c");
        return 1;
    }

    codegen_node(&ctx, root, out);
    fclose(out);

    if (g_config.mode_transpile)
    {
        if (g_config.output_file)
        {
            // If user specified -o, rename out.c to that
            if (rename("out.c", g_config.output_file) != 0)
            {
                perror("rename out.c");
                return 1;
            }
            if (!g_config.quiet)
            {
                printf("[zc] Transpiled to %s\n", g_config.output_file);
            }
        }
        else
        {
            if (!g_config.quiet)
            {
                printf("[zc] Transpiled to out.c\n");
            }
        }
        // Done, no C compilation
        return 0;
    }

    // Compile C
    char cmd[8192];
    char *outfile = g_config.output_file ? g_config.output_file : ZC_DEFAULT_OUTPUT;

    snprintf(cmd, sizeof(cmd), "%s %s %s %s %s -o %s out.c " ZC_MATH_FLAG " " ZC_PTHREAD_FLAG " -I./src -I./std %s", g_config.cc,
             g_config.gcc_flags, g_cflags, g_config.is_freestanding ? "-ffreestanding" : "", "",
             outfile, g_link_flags);

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
            remove("out.c");
        }
        return 1;
    }

    if (!g_config.emit_c)
    {
        // remove("out.c"); // Keep it for debugging for now or follow flag
        remove("out.c");
    }

    if (g_config.mode_run)
    {
        char run_cmd[2048];
#ifdef _WIN32
        sprintf(run_cmd, "%s", outfile);
#else
        sprintf(run_cmd, "./%s", outfile);
#endif
        ret = system(run_cmd);
        remove(outfile);
        zptr_plugin_mgr_cleanup();
        zen_trigger_global();
        return ret;
    }

    zptr_plugin_mgr_cleanup();
    zen_trigger_global();
    return 0;
}
