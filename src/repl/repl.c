
#include "repl.h"
#include "ast.h"
#include "compat/compat.h"
#include "parser/parser.h"
#include "zprep.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

ASTNode *parse_program(ParserContext *ctx, Lexer *l);

static int is_header_line(const char *line)
{
    return (strncmp(line, "import ", 7) == 0 || strncmp(line, "include ", 8) == 0 ||
            strncmp(line, "#include", 8) == 0);
}

void run_repl(const char *self_path)
{
    printf("\033[1;36mZen C REPL (v0.1)\033[0m\n");
    printf("Type 'exit' or 'quit' to leave.\n");
    printf("Type :help for commands.\n");

    // Dynamic history.
    int history_cap = 64;
    int history_len = 0;
    char **history = xmalloc(history_cap * sizeof(char *));

    char history_path[512];
    const char *home = getenv("HOME");
#ifdef _WIN32
    if (!home || !home[0])
    {
        home = getenv("USERPROFILE");
    }
    if ((!home || !home[0]) && getenv("HOMEDRIVE") && getenv("HOMEPATH"))
    {
        static char home_path[512];
        snprintf(home_path, sizeof(home_path), "%s%s", getenv("HOMEDRIVE"), getenv("HOMEPATH"));
        home = home_path;
    }
#endif
    if (home)
    {
        snprintf(history_path, sizeof(history_path), "%s/.zprep_history", home);
        FILE *hf = fopen(history_path, "r");
        if (hf)
        {
            char buf[1024];
            while (fgets(buf, sizeof(buf), hf))
            {
                size_t l = strlen(buf);
                if (l > 0 && buf[l - 1] == '\n')
                {
                    buf[--l] = 0;
                }
                if (l == 0)
                {
                    continue;
                }
                if (history_len >= history_cap)
                {
                    history_cap *= 2;
                    history = realloc(history, history_cap * sizeof(char *));
                }
                history[history_len++] = strdup(buf);
            }
            fclose(hf);
            if (history_len > 0)
            {
                printf("Loaded %d entries from history.\n", history_len);
            }
        }
    }
    else
    {
        history_path[0] = 0;
    }

    // Watch list.
    char *watches[16];
    int watches_len = 0;
    for (int i = 0; i < 16; i++)
    {
        watches[i] = NULL;
    }

    // Load startup file (~/.zprep_init.zc) if exists
    if (home)
    {
        char init_path[512];
        snprintf(init_path, sizeof(init_path), "%s/.zprep_init.zc", home);
        FILE *init_f = fopen(init_path, "r");
        if (init_f)
        {
            char buf[1024];
            int init_count = 0;
            while (fgets(buf, sizeof(buf), init_f))
            {
                size_t l = strlen(buf);
                if (l > 0 && buf[l - 1] == '\n')
                {
                    buf[--l] = 0;
                }
                char *p = buf;
                while (*p == ' ' || *p == '\t')
                {
                    p++;
                }
                if (*p == 0 || *p == '/' || *p == '#')
                {
                    continue;
                }
                if (history_len >= history_cap)
                {
                    history_cap *= 2;
                    history = realloc(history, history_cap * sizeof(char *));
                }
                history[history_len++] = strdup(p);
                init_count++;
            }
            fclose(init_f);
            if (init_count > 0)
            {
                printf("Loaded %d lines from ~/.zprep_init.zc\n", init_count);
            }
        }
    }

    char line_buf[1024];

    char *input_buffer = NULL;
    size_t input_len = 0;
    int brace_depth = 0;
    int paren_depth = 0;

    while (1)
    {
        if (brace_depth > 0 || paren_depth > 0)
        {
            printf("... ");
        }
        else
        {
            printf("\033[1;32m>>>\033[0m ");
        }

        if (!fgets(line_buf, sizeof(line_buf), stdin))
        {
            break;
        }

        // Handle commands (only on fresh line).
        if (NULL == input_buffer)
        {
            size_t len = strlen(line_buf);
            char cmd_buf[1024];
            strcpy(cmd_buf, line_buf);
            if (len > 0 && cmd_buf[len - 1] == '\n')
            {
                cmd_buf[--len] = 0;
            }
            while (len > 0 && (cmd_buf[len - 1] == ' ' || cmd_buf[len - 1] == '\t'))
            {
                cmd_buf[--len] = 0;
            }

            if (0 == strcmp(cmd_buf, "exit") || 0 == strcmp(cmd_buf, "quit"))
            {
                break;
            }

            // Commands
            if (cmd_buf[0] == ':' || cmd_buf[0] == '!')
            {
                if (0 == strcmp(cmd_buf, ":help"))
                {
                    printf("REPL Commands:\n");
                    printf("  :help       Show this help\n");
                    printf("  :reset      Clear history\n");
                    printf("  :imports    Show active imports\n");
                    printf("  :vars       Show active variables\n");
                    printf("  :funcs      Show user functions\n");
                    printf("  :structs    Show user structs\n");
                    printf("  :history    Show command history\n");
                    printf("  :type <x>   Show type of expression\n");
                    printf("  :time <x>   Benchmark expression (1000 iters)\n");
                    printf("  :c <x>      Show generated C code\n");
                    printf("  :doc <x>    Show documentation for symbol\n");
                    printf("  :run        Execute full session\n");
                    printf("  :edit [n]   Edit command n (default: last) in $EDITOR\n");
                    printf("  :save <f>   Save session to file\n");
                    printf("  :load <f>   Load file into session\n");
                    printf("  :load <f>   Load file into session\n");
                    printf("  :watch <x>  Watch expression output\n");
                    printf("  :unwatch <n> Remove watch n\n");
                    printf("  :undo       Remove last command\n");
                    printf("  :delete <n> Remove command at index n\n");
                    printf("  :clear      Clear screen\n");
                    printf("  ! <cmd>     Run shell command\n");
                    printf("  :quit       Exit REPL\n");
                    continue;
                }
                else if (0 == strcmp(cmd_buf, ":reset"))
                {
                    for (int i = 0; i < history_len; i++)
                    {
                        free(history[i]);
                    }
                    history_len = 0;
                    printf("History cleared.\n");
                    continue;
                }
                else if (0 == strcmp(cmd_buf, ":quit"))
                {
                    break;
                }
                else if (0 == strcmp(cmd_buf, ":clear"))
                {
                    printf("\033[2J\033[H"); // ANSI clear screen
                    continue;
                }
                else if (0 == strcmp(cmd_buf, ":undo"))
                {
                    if (history_len > 0)
                    {
                        history_len = history_len - 1;
                        free(history[history_len]);
                        printf("Removed last entry.\n");
                    }
                    else
                    {
                        printf("History is empty.\n");
                    }
                    continue;
                }
                else if (0 == strncmp(cmd_buf, ":delete ", 8))
                {
                    int idx = atoi(cmd_buf + 8) - 1;
                    if (idx >= 0 && idx < history_len)
                    {
                        free(history[idx]);
                        for (int i = idx; i < history_len - 1; i++)
                        {
                            history[i] = history[i + 1];
                        }
                        history_len = history_len - 1;
                        printf("Deleted entry %d.\n", idx + 1);
                    }
                    else
                    {
                        printf("Invalid index. Use :history to see valid indices.\n");
                    }
                    continue;
                }
                else if (0 == strncmp(cmd_buf, ":edit", 5))
                {
                    int idx = history_len - 1;
                    if (strlen(cmd_buf) > 6)
                    {
                        idx = atoi(cmd_buf + 6) - 1;
                    }

                    if (history_len == 0)
                    {
                        printf("History is empty.\n");
                        continue;
                    }

                    if (idx < 0 || idx >= history_len)
                    {
                        printf("Invalid index.\n");
                        continue;
                    }

                    char edit_path[256];
                    sprintf(edit_path, "%szprep_edit_%d.zc", zc_get_temp_dir(), rand());
                    FILE *f = fopen(edit_path, "w");
                    if (f)
                    {
                        fprintf(f, "%s", history[idx]);
                        fclose(f);

                        const char *editor = getenv("EDITOR");
                        if (!editor)
                        {
                            editor = "nano"; // Default fallback,
                                             // 'cause I know some of you
                                             // don't know how to exit Vim.
                        }

                        char cmd[1024];
                        sprintf(cmd, "%s %s", editor, edit_path);
                        int status = system(cmd);

                        if (0 == status)
                        {
                            // Read back file.
                            FILE *fr = fopen(edit_path, "r");
                            if (fr)
                            {
                                fseek(fr, 0, SEEK_END);
                                long length = ftell(fr);
                                fseek(fr, 0, SEEK_SET);
                                char *buffer = malloc(length + 1);
                                if (buffer)
                                {
                                    fread(buffer, 1, length, fr);
                                    buffer[length] = 0;

                                    while (length > 0 && buffer[length - 1] == '\n')
                                    {
                                        buffer[--length] = 0;
                                    }

                                    if (strlen(buffer) > 0)
                                    {
                                        printf("Running: %s\n", buffer);
                                        if (history_len >= history_cap)
                                        {
                                            history_cap *= 2;
                                            history =
                                                realloc(history, history_cap * sizeof(char *));
                                        }
                                        history[history_len++] = strdup(buffer);
                                    }
                                    else
                                    {
                                        free(buffer);
                                    }
                                }
                                fclose(fr);
                            }
                        }
                    }
                    continue;
                }
                else if (0 == strncmp(cmd_buf, ":watch ", 7))
                {
                    char *expr = cmd_buf + 7;
                    while (*expr == ' ')
                    {
                        expr++;
                    }
                    size_t l = strlen(expr);
                    while (l > 0 && expr[l - 1] == ' ')
                    {
                        expr[--l] = 0;
                    }

                    if (l > 0)
                    {
                        if (watches_len < 16)
                        {
                            watches[watches_len++] = strdup(expr);
                            printf("Watching: %s\n", expr);
                        }
                        else
                        {
                            printf("Watch list full (max 16).\n");
                        }
                    }
                    else
                    {
                        if (watches_len == 0)
                        {
                            printf("No active watches.\n");
                        }
                        else
                        {
                            for (int i = 0; i < watches_len; i++)
                            {
                                printf("%d: %s\n", i + 1, watches[i]);
                            }
                        }
                    }
                    continue;
                }
                else if (0 == strncmp(cmd_buf, ":unwatch ", 9))
                {
                    // Remove watch.
                    int idx = atoi(cmd_buf + 9) - 1;
                    if (idx >= 0 && idx < watches_len)
                    {
                        free(watches[idx]);
                        for (int i = idx; i < watches_len - 1; i++)
                        {
                            watches[i] = watches[i + 1];
                        }

                        watches_len--;

                        printf("Removed watch %d.\n", idx + 1);
                    }
                    else
                    {
                        printf("Invalid index.\n");
                    }
                    continue;
                }
                else if (cmd_buf[0] == '!')
                {
                    // Shell escape.
                    system(cmd_buf + 1);
                    continue;
                }
                else if (0 == strncmp(cmd_buf, ":save ", 6))
                {
                    char *filename = cmd_buf + 6;
                    FILE *f = fopen(filename, "w");
                    if (f)
                    {
                        for (int i = 0; i < history_len; i++)
                        {
                            if (is_header_line(history[i]))
                            {
                                fprintf(f, "%s\n", history[i]);
                            }
                        }
                        // Write main function body.
                        fprintf(f, "\nfn main() {\n");
                        for (int i = 0; i < history_len; i++)
                        {
                            if (!is_header_line(history[i]))
                            {
                                fprintf(f, "    %s\n", history[i]);
                            }
                        }
                        fprintf(f, "}\n");
                        fclose(f);
                        printf("Session saved to %s\n", filename);
                    }
                    else
                    {
                        printf("Error: Cannot write to %s\n", filename);
                    }
                    continue;
                }
                else if (0 == strncmp(cmd_buf, ":load ", 6))
                {
                    char *filename = cmd_buf + 6;
                    FILE *f = fopen(filename, "r");
                    if (f)
                    {
                        char buf[1024];
                        int count = 0;
                        while (fgets(buf, sizeof(buf), f))
                        {
                            size_t l = strlen(buf);
                            if (l > 0 && buf[l - 1] == '\n')
                            {
                                buf[--l] = 0;
                            }
                            if (l == 0)
                            {
                                continue;
                            }
                            if (history_len >= history_cap)
                            {
                                history_cap *= 2;
                                history = realloc(history, history_cap * sizeof(char *));
                            }
                            history[history_len++] = strdup(buf);
                            count++;
                        }
                        fclose(f);
                        printf("Loaded %d lines from %s\n", count, filename);
                    }
                    else
                    {
                        printf("Error: Cannot read %s\n", filename);
                    }
                    continue;
                }
                else if (0 == strcmp(cmd_buf, ":imports"))
                {
                    printf("Active Imports:\n");
                    for (int i = 0; i < history_len; i++)
                    {
                        if (is_header_line(history[i]))
                        {
                            printf("  %s\n", history[i]);
                        }
                    }
                    continue;
                }
                else if (0 == strcmp(cmd_buf, ":history"))
                {
                    printf("Session History:\n");
                    for (int i = 0; i < history_len; i++)
                    {
                        printf("%4d  %s\n", i + 1, history[i]);
                    }
                    continue;
                }
                else if (0 == strcmp(cmd_buf, ":vars") || 0 == strcmp(cmd_buf, ":funcs") ||
                         0 == strcmp(cmd_buf, ":structs"))
                {
                    size_t code_size = 4096;
                    for (int i = 0; i < history_len; i++)
                    {
                        code_size += strlen(history[i]) + 2;
                    }
                    char *code = malloc(code_size + 128);
                    strcpy(code, "");

                    for (int i = 0; i < history_len; i++)
                    {
                        if (is_header_line(history[i]))
                        {
                            strcat(code, history[i]);
                            strcat(code, "\n");
                        }
                    }
                    strcat(code, "fn main() { ");
                    for (int i = 0; i < history_len; i++)
                    {
                        if (!is_header_line(history[i]))
                        {
                            strcat(code, history[i]);
                            strcat(code, " ");
                        }
                    }
                    strcat(code, " }");

                    ParserContext ctx = {0};
                    ctx.is_repl = 1;
                    ctx.skip_preamble = 1;

                    Lexer l;
                    lexer_init(&l, code);
                    ASTNode *nodes = parse_program(&ctx, &l);

                    ASTNode *search = nodes;
                    if (search && search->type == NODE_ROOT)
                    {
                        search = search->root.children;
                    }

                    if (0 == strcmp(cmd_buf, ":vars"))
                    {
                        ASTNode *main_func = NULL;
                        for (ASTNode *n = search; n; n = n->next)
                        {
                            if (n->type == NODE_FUNCTION && 0 == strcmp(n->func.name, "main"))
                            {
                                main_func = n;
                                break;
                            }
                        }
                        printf("Variables:\n");
                        if (main_func && main_func->func.body &&
                            main_func->func.body->type == NODE_BLOCK)
                        {
                            int found = 0;
                            for (ASTNode *s = main_func->func.body->block.statements; s;
                                 s = s->next)
                            {
                                if (s->type == NODE_VAR_DECL)
                                {
                                    char *t =
                                        s->var_decl.type_str ? s->var_decl.type_str : "Inferred";
                                    printf("  %s: %s\n", s->var_decl.name, t);
                                    found = 1;
                                }
                            }
                            if (!found)
                            {
                                printf("  (none)\n");
                            }
                        }
                        else
                        {
                            printf("  (none)\n");
                        }
                    }
                    else if (0 == strcmp(cmd_buf, ":funcs"))
                    {
                        printf("Functions:\n");
                        int found = 0;
                        for (ASTNode *n = search; n; n = n->next)
                        {
                            if (n->type == NODE_FUNCTION && 0 != strcmp(n->func.name, "main"))
                            {
                                printf("  fn %s()\n", n->func.name);
                                found = 1;
                            }
                        }
                        if (!found)
                        {
                            printf("  (none)\n");
                        }
                    }
                    else if (0 == strcmp(cmd_buf, ":structs"))
                    {
                        printf("Structs:\n");
                        int found = 0;
                        for (ASTNode *n = search; n; n = n->next)
                        {
                            if (n->type == NODE_STRUCT)
                            {
                                printf("  struct %s\n", n->strct.name);
                                found = 1;
                            }
                        }
                        if (!found)
                        {
                            printf("  (none)\n");
                        }
                    }

                    free(code);
                    continue;
                }
                else if (0 == strncmp(cmd_buf, ":type ", 6))
                {
                    char *expr = cmd_buf + 6;

                    size_t probe_size = 4096;
                    for (int i = 0; i < history_len; i++)
                    {
                        probe_size += strlen(history[i]) + 2;
                    }

                    char *probe_code = malloc(probe_size + strlen(expr) + 256);
                    strcpy(probe_code, "");

                    for (int i = 0; i < history_len; i++)
                    {
                        if (is_header_line(history[i]))
                        {
                            strcat(probe_code, history[i]);
                            strcat(probe_code, "\n");
                        }
                    }

                    strcat(probe_code, "fn main() { _z_suppress_stdout(); ");
                    for (int i = 0; i < history_len; i++)
                    {
                        if (!is_header_line(history[i]))
                        {
                            strcat(probe_code, history[i]);
                            strcat(probe_code, " ");
                        }
                    }

                    strcat(probe_code, " raw { typedef struct { int _u; } __REVEAL_TYPE__; } ");
                    strcat(probe_code, " var _z_type_probe: __REVEAL_TYPE__; _z_type_probe = (");
                    strcat(probe_code, expr);
                    strcat(probe_code, "); }");

                    char tmp_path[256];
                    sprintf(tmp_path, "%szprep_repl_type_%d.zc", zc_get_temp_dir(), rand());
                    FILE *f = fopen(tmp_path, "w");
                    if (f)
                    {
                        fprintf(f, "%s", probe_code);
                        fclose(f);

                        char cmd[2048];
                        sprintf(cmd, "%s run -q %s 2>&1", self_path, tmp_path);

                        FILE *p = popen(cmd, "r");
                        if (p)
                        {
                            char buf[1024];
                            int found = 0;
                            while (fgets(buf, sizeof(buf), p))
                            {
                                char *marker = "right operand has type '";
                                char *start = strstr(buf, marker);
                                if (start)
                                {
                                    start += strlen(marker);
                                    char *end = strchr(start, '\'');
                                    if (end)
                                    {
                                        *end = 0;
                                        printf("\033[1;36mType: %s\033[0m\n", start);
                                        found = 1;
                                        break;
                                    }
                                }
                            }
                            pclose(p);
                            if (!found)
                            {
                                printf("Type: <unknown>\n");
                            }
                        }
                    }
                    free(probe_code);
                    continue;
                }
                else if (0 == strncmp(cmd_buf, ":time ", 6))
                {
                    // Benchmark an expression.
                    char *expr = cmd_buf + 6;

                    size_t code_size = 4096;
                    for (int i = 0; i < history_len; i++)
                    {
                        code_size += strlen(history[i]) + 2;
                    }
                    char *code = malloc(code_size + strlen(expr) + 256);
                    strcpy(code, "");

                    for (int i = 0; i < history_len; i++)
                    {
                        if (is_header_line(history[i]))
                        {
                            strcat(code, history[i]);
                            strcat(code, "\n");
                        }
                    }
                    strcat(code, "include \"time.h\"\n");
                    strcat(code, "fn main() { _z_suppress_stdout();\n");
                    for (int i = 0; i < history_len; i++)
                    {
                        if (!is_header_line(history[i]))
                        {
                            strcat(code, history[i]);
                            strcat(code, " ");
                        }
                    }
                    strcat(code, "_z_restore_stdout();\n");
                    strcat(code, "raw { clock_t _start = clock(); }\n");
                    strcat(code, "for _i in 0..1000 { ");
                    strcat(code, expr);
                    strcat(code, "; }\n");
                    strcat(code, "raw { clock_t _end = clock(); double _elapsed = (double)(_end - "
                                 "_start) / CLOCKS_PER_SEC; printf(\"1000 iterations: %.4fs "
                                 "(%.6fs/iter)\\n\", _elapsed, _elapsed/1000); }\n");
                    strcat(code, "}");

                    char tmp_path[256];
                    sprintf(tmp_path, "%szprep_repl_time_%d.zc", zc_get_temp_dir(), rand());
                    FILE *f = fopen(tmp_path, "w");
                    if (f)
                    {
                        fprintf(f, "%s", code);
                        fclose(f);
                        char cmd[2048];
                        sprintf(cmd, "%s run -q %s", self_path, tmp_path);
                        system(cmd);
                    }
                    free(code);
                    continue;
                }
                else if (0 == strncmp(cmd_buf, ":c ", 3))
                {
                    char *expr_buf = malloc(8192);
                    strcpy(expr_buf, cmd_buf + 3);

                    int brace_depth = 0;
                    for (char *p = expr_buf; *p; p++)
                    {
                        if (*p == '{')
                        {
                            brace_depth++;
                        }
                        else if (*p == '}')
                        {
                            brace_depth--;
                        }
                    }

                    while (brace_depth > 0)
                    {
                        printf("... ");
                        char more[1024];
                        if (!fgets(more, sizeof(more), stdin))
                        {
                            break;
                        }
                        size_t mlen = strlen(more);
                        if (mlen > 0 && more[mlen - 1] == '\n')
                        {
                            more[--mlen] = 0;
                        }
                        strcat(expr_buf, "\n");
                        strcat(expr_buf, more);
                        for (char *p = more; *p; p++)
                        {
                            if (*p == '{')
                            {
                                brace_depth++;
                            }
                            else if (*p == '}')
                            {
                                brace_depth--;
                            }
                        }
                    }

                    size_t code_size = 4096 + strlen(expr_buf);
                    for (int i = 0; i < history_len; i++)
                    {
                        code_size += strlen(history[i]) + 2;
                    }
                    char *code = malloc(code_size + 128);
                    strcpy(code, "");

                    for (int i = 0; i < history_len; i++)
                    {
                        if (is_header_line(history[i]))
                        {
                            strcat(code, history[i]);
                            strcat(code, "\n");
                        }
                    }
                    strcat(code, "fn main() {\n");
                    for (int i = 0; i < history_len; i++)
                    {
                        if (!is_header_line(history[i]))
                        {
                            strcat(code, history[i]);
                            strcat(code, " ");
                        }
                    }
                    strcat(code, expr_buf);
                    strcat(code, "\n}");
                    free(expr_buf);

                    char tmp_path[256];
                    sprintf(tmp_path, "%szprep_repl_c_%d.zc", zc_get_temp_dir(), rand());
                    FILE *f = fopen(tmp_path, "w");
                    if (f)
                    {
                        fprintf(f, "%s", code);
                        fclose(f);

                        char out_path[256];
                        sprintf(out_path, "%szprep_repl_out", zc_get_temp_dir());
                        char cmd[2048];
#ifdef _WIN32
                        sprintf(cmd, "%s build -q --emit-c -o %s %s 2>" ZC_NULL_DEVICE,
                                self_path, out_path, tmp_path);
                        system(cmd);

                        char c_path[270];
                        sprintf(c_path, "%s.c", out_path);
                        char *body = extract_main_body(c_path);
                        if (body)
                        {
                            printf("%s", body);
                            free(body);
                        }
#else
                        sprintf(cmd,
                                "%s build -q --emit-c -o %s %s "
                                "2>" ZC_NULL_DEVICE "; sed "
                                "-n '/^int main() {$/,/^}$/p' %s.c "
                                "2>" ZC_NULL_DEVICE " | "
                                "tail -n +3 | head -n -2 | sed 's/^    //'",
                                self_path, out_path, tmp_path, out_path);
                        system(cmd);
#endif
                    }
                    free(code);
                    continue;
                }
                else if (0 == strcmp(cmd_buf, ":run"))
                {
                    size_t code_size = 4096;
                    for (int i = 0; i < history_len; i++)
                    {
                        code_size += strlen(history[i]) + 2;
                    }
                    char *code = malloc(code_size);
                    strcpy(code, "");

                    for (int i = 0; i < history_len; i++)
                    {
                        if (is_header_line(history[i]))
                        {
                            strcat(code, history[i]);
                            strcat(code, "\n");
                        }
                    }
                    strcat(code, "fn main() {\n");
                    for (int i = 0; i < history_len; i++)
                    {
                        if (!is_header_line(history[i]))
                        {
                            strcat(code, "    ");
                            strcat(code, history[i]);
                            strcat(code, "\n");
                        }
                    }
                    strcat(code, "}\n");

                    char tmp_path[256];
                    sprintf(tmp_path, "%szprep_repl_run_%d.zc", zc_get_temp_dir(), rand());
                    FILE *f = fopen(tmp_path, "w");
                    if (f)
                    {
                        fprintf(f, "%s", code);
                        fclose(f);
                        char cmd[2048];
                        sprintf(cmd, "%s run %s", self_path, tmp_path);
                        system(cmd);
                    }
                    free(code);
                    continue;
                }
                else if (0 == strncmp(cmd_buf, ":doc ", 5))
                {
                    char *sym = cmd_buf + 5;
                    while (*sym == ' ')
                    {
                        sym++;
                    }
                    size_t symlen = strlen(sym);
                    while (symlen > 0 && sym[symlen - 1] == ' ')
                    {
                        sym[--symlen] = 0;
                    }

                    // Documentation database

                    struct
                    {
                        const char *name;
                        const char *doc;
                    } docs[] = {
                        {"Vec", "Vec<T> - Dynamic array (generic)\n  Fields: data: T*, "
                                "len: usize, cap: "
                                "usize\n  Methods: new, push, pop, get, set, insert, "
                                "remove, contains, "
                                "clear, free, clone, reverse, first, last, length, "
                                "is_empty, eq"},
                        {"Vec.new", "fn Vec<T>::new() -> Vec<T>\n  Creates an empty vector."},
                        {"Vec.push", "fn push(self, item: T)\n  Appends item to the end. "
                                     "Auto-grows capacity."},
                        {"Vec.pop", "fn pop(self) -> T\n  Removes and returns the last element. "
                                    "Panics if empty."},
                        {"Vec.get", "fn get(self, idx: usize) -> T\n  Returns element at index. "
                                    "Panics if out of bounds."},
                        {"Vec.set", "fn set(self, idx: usize, item: T)\n  Sets element at index. "
                                    "Panics if out of bounds."},
                        {"Vec.insert", "fn insert(self, idx: usize, item: T)\n  Inserts item at "
                                       "index, shifting elements right."},
                        {"Vec.remove", "fn remove(self, idx: usize) -> T\n  Removes and returns "
                                       "element at index, shifting elements left."},
                        {"Vec.contains", "fn contains(self, item: T) -> bool\n  Returns true if "
                                         "item is in the vector."},
                        {"Vec.clear", "fn clear(self)\n  Removes all elements but keeps capacity."},
                        {"Vec.free", "fn free(self)\n  Frees memory. Sets data to null."},
                        {"Vec.clone", "fn clone(self) -> Vec<T>\n  Returns a shallow copy."},
                        {"Vec.reverse", "fn reverse(self)\n  Reverses elements in place."},
                        {"Vec.first", "fn first(self) -> T\n  Returns first element. "
                                      "Panics if empty."},
                        {"Vec.last",
                         "fn last(self) -> T\n  Returns last element. Panics if empty."},
                        {"Vec.length", "fn length(self) -> usize\n  Returns number of elements."},
                        {"Vec.is_empty",
                         "fn is_empty(self) -> bool\n  Returns true if length is 0."},
                        {"String", "String - Mutable string (alias for char*)\n  "
                                   "Methods: len, split, trim, "
                                   "contains, starts_with, ends_with, to_upper, "
                                   "to_lower, substring, find"},
                        {"String.len", "fn len(self) -> usize\n  Returns string length."},
                        {"String.contains", "fn contains(self, substr: string) -> bool\n  Returns "
                                            "true if string contains substr."},
                        {"String.starts_with", "fn starts_with(self, prefix: string) -> bool\n  "
                                               "Returns true if string starts with prefix."},
                        {"String.ends_with", "fn ends_with(self, suffix: string) -> bool\n  "
                                             "Returns true if string ends with suffix."},
                        {"String.substring", "fn substring(self, start: usize, len: usize) -> "
                                             "string\n  Returns a substring. Caller must free."},
                        {"String.find", "fn find(self, substr: string) -> int\n  Returns index of "
                                        "substr, or -1 if not found."},
                        {"println", "println \"format string {expr}\"\n  Prints to stdout with "
                                    "newline. Auto-formats {expr} values."},
                        {"print", "print \"format string {expr}\"\n  Prints to stdout "
                                  "without newline."},
                        {"eprintln",
                         "eprintln \"format string\"\n  Prints to stderr with newline."},
                        {"eprint", "eprint \"format string\"\n  Prints to stderr without newline."},
                        {"guard", "guard condition else action\n  Early exit pattern. "
                                  "Executes action if "
                                  "condition is false.\n  Example: guard ptr != NULL "
                                  "else return;"},
                        {"defer", "defer statement\n  Executes statement at end of scope.\n  "
                                  "Example: defer free(ptr);"},
                        {"sizeof", "sizeof(type) or sizeof(expr)\n  Returns size in bytes."},
                        {"typeof", "typeof(expr)\n  Returns the type of expression "
                                   "(compile-time)."},
                        {"malloc", "void *malloc(size_t size)\n  Allocates size bytes. Returns "
                                   "pointer or NULL. Free with free()."},
                        {"free", "void free(void *ptr)\n  Frees memory allocated by "
                                 "malloc/calloc/realloc."},
                        {"calloc", "void *calloc(size_t n, size_t size)\n  Allocates n*size bytes, "
                                   "zeroed. Returns pointer or NULL."},
                        {"realloc", "void *realloc(void *ptr, size_t size)\n  Resizes allocation "
                                    "to size bytes. May move memory."},
                        {"memcpy", "void *memcpy(void *dest, const void *src, size_t n)\n  Copies "
                                   "n bytes from src to dest. Returns dest. No overlap."},
                        {"memmove", "void *memmove(void *dest, const void *src, size_t n)\n  "
                                    "Copies n bytes, handles overlapping regions."},
                        {"memset", "void *memset(void *s, int c, size_t n)\n  Sets n "
                                   "bytes of s to value c."},
                        {"strlen", "size_t strlen(const char *s)\n  Returns length of string (not "
                                   "including null terminator)."},
                        {"strcpy", "char *strcpy(char *dest, const char *src)\n  Copies src to "
                                   "dest including null terminator. No bounds check."},
                        {"strncpy", "char *strncpy(char *dest, const char *src, size_t n)\n  "
                                    "Copies up to n chars. May not null-terminate."},
                        {"strcat", "char *strcat(char *dest, const char *src)\n  Appends "
                                   "src to dest."},
                        {"strcmp", "int strcmp(const char *s1, const char *s2)\n  Compares "
                                   "strings. Returns 0 if equal, <0 or >0 otherwise."},
                        {"strncmp", "int strncmp(const char *s1, const char *s2, size_t n)\n  "
                                    "Compares up to n characters."},
                        {"strstr", "char *strstr(const char *haystack, const char *needle)\n  "
                                   "Finds first occurrence of needle. Returns pointer or NULL."},
                        {"strchr", "char *strchr(const char *s, int c)\n  Finds first occurrence "
                                   "of char c. Returns pointer or NULL."},
                        {"strdup", "char *strdup(const char *s)\n  Duplicates string. Caller must "
                                   "free the result."},
                        {"printf", "int printf(const char *fmt, ...)\n  Prints formatted output to "
                                   "stdout. Returns chars written."},
                        {"sprintf", "int sprintf(char *str, const char *fmt, ...)\n  Prints "
                                    "formatted output to string buffer."},
                        {"snprintf", "int snprintf(char *str, size_t n, const char *fmt, ...)\n  "
                                     "Safe sprintf with size limit."},
                        {"fprintf", "int fprintf(FILE *f, const char *fmt, ...)\n  Prints "
                                    "formatted output to file stream."},
                        {"scanf", "int scanf(const char *fmt, ...)\n  Reads formatted "
                                  "input from stdin."},
                        {"fopen", "FILE *fopen(const char *path, const char *mode)\n  Opens file. "
                                  "Modes: "
                                  "\"r\", \"w\", \"a\", \"rb\", \"wb\". Returns NULL on error."},
                        {"fclose", "int fclose(FILE *f)\n  Closes file. Returns 0 on success."},
                        {"fread", "size_t fread(void *ptr, size_t size, size_t n, FILE *f)\n  "
                                  "Reads n items of size bytes. Returns items read."},
                        {"fwrite", "size_t fwrite(const void *ptr, size_t size, size_t n, FILE "
                                   "*f)\n  Writes n items of size bytes. Returns items written."},
                        {"fgets", "char *fgets(char *s, int n, FILE *f)\n  Reads line up to n-1 "
                                  "chars. Includes newline. Returns s or NULL."},
                        {"fputs", "int fputs(const char *s, FILE *f)\n  Writes string to file. "
                                  "Returns non-negative or EOF."},
                        {"exit", "void exit(int status)\n  Terminates program with "
                                 "status code. 0 "
                                 "= success."},
                        {"atoi", "int atoi(const char *s)\n  Converts string to int. "
                                 "Returns 0 on error."},
                        {"atof", "double atof(const char *s)\n  Converts string to double."},
                        {"abs", "int abs(int n)\n  Returns absolute value."},
                        {"rand", "int rand(void)\n  Returns pseudo-random int in [0, RAND_MAX]."},
                        {"srand", "void srand(unsigned seed)\n  Seeds the random number "
                                  "generator."},
                        {"qsort", "void qsort(void *base, size_t n, size_t size, int(*cmp)(const "
                                  "void*, const void*))\n  Quicksorts array in-place."},
                        {NULL, NULL}};

                    int found = 0;
                    for (int i = 0; docs[i].name != NULL; i++)
                    {
                        if (0 == strcmp(sym, docs[i].name))
                        {
                            printf("\033[1;36m%s\033[0m\n%s\n", docs[i].name, docs[i].doc);
                            found = 1;
                            break;
                        }
                    }
                    if (!found)
                    {
#ifndef _WIN32
                        // Fallback: try man pages, show only SYNOPSIS.
                        char man_cmd[256];
                        sprintf(man_cmd,
                                "man 3 %s 2>/dev/null | sed -n '/^SYNOPSIS/,/^[A-Z]/p' | "
                                "head -10",
                                sym);
                        FILE *mp = popen(man_cmd, "r");
                        if (mp)
                        {
                            char buf[256];
                            int lines = 0;
                            while (fgets(buf, sizeof(buf), mp) && lines < 8)
                            {
                                printf("%s", buf);
                                lines++;
                            }
                            int status = pclose(mp);
                            if (0 == status && lines > 0)
                            {
                                found = 1;
                                printf("\033[90m(man 3 %s)\033[0m\n", sym);
                            }
                        }
#endif
                        if (!found)
                        {
                            printf("No documentation for '%s'.\n", sym);
                        }
                    }
                    continue;
                }
                else
                {
                    printf("Unknown command: %s\n", cmd_buf);
                    continue;
                }
            }
        }

        int in_quote = 0;
        int escaped = 0;
        for (int i = 0; line_buf[i]; i++)
        {
            char c = line_buf[i];
            if (escaped)
            {
                escaped = 0;
                continue;
            }
            if (c == '\\')
            {
                escaped = 1;
                continue;
            }
            if (c == '"')
            {
                in_quote = !in_quote;
                continue;
            }

            if (!in_quote)
            {
                if (c == '{')
                {
                    brace_depth++;
                }
                if (c == '}')
                {
                    brace_depth--;
                }
                if (c == '(')
                {
                    paren_depth++;
                }
                if (c == ')')
                {
                    paren_depth--;
                }
            }
        }

        size_t len = strlen(line_buf);
        input_buffer = realloc(input_buffer, input_len + len + 1);
        strcpy(input_buffer + input_len, line_buf);
        input_len += len;

        if (brace_depth > 0 || paren_depth > 0)
        {
            continue;
        }

        if (input_len > 0 && input_buffer[input_len - 1] == '\n')
        {
            input_buffer[--input_len] = 0;
        }

        if (input_len == 0)
        {
            free(input_buffer);
            input_buffer = NULL;
            input_len = 0;
            brace_depth = 0;
            paren_depth = 0;
            continue;
        }

        // Add to history.
        if (history_len >= history_cap)
        {
            history_cap *= 2;
            history = realloc(history, history_cap * sizeof(char *));
        }
        history[history_len++] = strdup(input_buffer);

        free(input_buffer);
        input_buffer = NULL;
        input_len = 0;
        brace_depth = 0;
        paren_depth = 0;

        size_t total_size = 4096;
        for (int i = 0; i < history_len; i++)
        {
            total_size += strlen(history[i]) + 2;
        }
        if (watches_len > 0)
        {
            total_size += 16 * 1024; // Plenty of space for watches. Yeah static ik.
        }

        char *full_code = malloc(total_size);
        strcpy(full_code, "");

        // Hoisting pass.
        for (int i = 0; i < history_len; i++)
        {
            if (is_header_line(history[i]))
            {
                strcat(full_code, history[i]);
                strcat(full_code, "\n");
            }
        }

        strcat(full_code, "fn main() { _z_suppress_stdout(); ");

        for (int i = 0; i < history_len - 1; i++)
        {
            if (is_header_line(history[i]))
            {
                continue;
            }
            strcat(full_code, history[i]);
            strcat(full_code, " ");
        }

        strcat(full_code, "_z_restore_stdout(); ");

        if (history_len > 0 && !is_header_line(history[history_len - 1]))
        {
            char *last_line = history[history_len - 1];

            char *check_buf = malloc(strlen(last_line) + 2);
            strcpy(check_buf, last_line);
            strcat(check_buf, ";");

            ParserContext ctx = {0};
            ctx.is_repl = 1;
            ctx.skip_preamble = 1;
            Lexer l;
            lexer_init(&l, check_buf);
            ASTNode *node = parse_statement(&ctx, &l);
            free(check_buf);

            int is_expr = 0;
            if (node)
            {
                ASTNode *child = node;
                if (child->type == NODE_EXPR_BINARY || child->type == NODE_EXPR_UNARY ||
                    child->type == NODE_EXPR_LITERAL || child->type == NODE_EXPR_VAR ||
                    child->type == NODE_EXPR_CALL || child->type == NODE_EXPR_MEMBER ||
                    child->type == NODE_EXPR_INDEX || child->type == NODE_EXPR_CAST ||
                    child->type == NODE_EXPR_SIZEOF || child->type == NODE_EXPR_STRUCT_INIT ||
                    child->type == NODE_EXPR_ARRAY_LITERAL || child->type == NODE_EXPR_SLICE ||
                    child->type == NODE_TERNARY || child->type == NODE_MATCH)
                {
                    is_expr = 1;
                }
            }

            if (is_expr)
            {
                size_t probesz = 4096;
                for (int i = 0; i < history_len - 1; i++)
                {
                    probesz += strlen(history[i]) + 2;
                }
                char *probe_code = malloc(probesz + strlen(last_line) + 512);
                strcpy(probe_code, "");

                for (int i = 0; i < history_len - 1; i++)
                {
                    if (is_header_line(history[i]))
                    {
                        strcat(probe_code, history[i]);
                        strcat(probe_code, "\n");
                    }
                }

                strcat(probe_code, "fn main() { _z_suppress_stdout(); ");

                for (int i = 0; i < history_len - 1; i++)
                {
                    if (!is_header_line(history[i]))
                    {
                        strcat(probe_code, history[i]);
                        strcat(probe_code, " ");
                    }
                }

                strcat(probe_code, " raw { typedef struct { int _u; } __REVEAL_TYPE__; } ");
                strcat(probe_code, " var _z_type_probe: __REVEAL_TYPE__; _z_type_probe = (");
                strcat(probe_code, last_line);
                strcat(probe_code, "); }");

                const char *temp_dir = zc_get_temp_dir();
                char p_path[256];
                snprintf(p_path, sizeof(p_path), "%szprep_repl_probe_%d.zc", temp_dir, rand());
                FILE *pf = fopen(p_path, "w");
                if (pf)
                {
                    fprintf(pf, "%s", probe_code);
                    fclose(pf);

                    char p_cmd[2048];
                    // TODO: Quote paths to handle spaces on Windows.
                    sprintf(p_cmd, "%s run -q %s 2>&1", self_path, p_path);

                    FILE *pp = popen(p_cmd, "r");
                    int is_void = 0;
                    if (pp)
                    {
                        char buf[1024];
                        while (fgets(buf, sizeof(buf), pp))
                        {
                            if (strstr(buf, "void") && strstr(buf, "expression"))
                            {
                                is_void = 1;
                            }
                        }
                        pclose(pp);
                    }

                    if (!is_void)
                    {
                        strcat(full_code, "println \"{");
                        strcat(full_code, last_line);
                        strcat(full_code, "}\";");
                    }
                    else
                    {
                        strcat(full_code, last_line);
                    }
                }
                else
                {
                    strcat(full_code, last_line);
                }
                free(probe_code);
            }
            else
            {
                strcat(full_code, last_line);
            }
        }

        if (watches_len > 0)
        {
            strcat(full_code, "; "); // separator.
            for (int i = 0; i < watches_len; i++)
            {
                // Use printf for label, then print "{expr}" for value.
                char wbuf[1024];
                sprintf(wbuf,
                        "printf(\"\\033[90mwatch:%s = \\033[0m\"); print \"{%s}\"; "
                        "printf(\"\\n\"); ",
                        watches[i], watches[i]);
                strcat(full_code, wbuf);
            }
        }

        strcat(full_code, " }");

        const char *temp_dir = zc_get_temp_dir();
        char tmp_path[256];
        snprintf(tmp_path, sizeof(tmp_path), "%szprep_repl_%d.zc", temp_dir, rand());
        FILE *f = fopen(tmp_path, "w");
        if (!f)
        {
            printf("Error: Cannot write temp file\n");
            free(full_code);
            break;
        }
        fprintf(f, "%s", full_code);
        fclose(f);
        free(full_code);

        char cmd[2048];
        // TODO: Quote paths to handle spaces on Windows.
        sprintf(cmd, "%s run -q %s", self_path, tmp_path);

        int ret = system(cmd);
        printf("\n");

        if (0 != ret)
        {
            free(history[--history_len]);
        }
    }

    if (history_path[0])
    {
        FILE *hf = fopen(history_path, "w");
        if (hf)
        {
            for (int i = 0; i < history_len; i++)
            {
                fprintf(hf, "%s\n", history[i]);
            }
            fclose(hf);
        }
    }

    for (int i = 0; i < history_len; i++)
    {
        free(history[i]);
    }
    free(history);
    if (input_buffer)
    {
        free(input_buffer);
    }
}
