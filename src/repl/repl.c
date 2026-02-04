
#include "repl.h"
#include "ast.h"
#include "parser/parser.h"
#include "zprep.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#else
#include <termios.h>
#include <unistd.h>
#include <ctype.h>
#endif

ASTNode *parse_program(ParserContext *ctx, Lexer *l);

static int is_header_line(const char *line)
{
    // Skip whitespace
    while (*line && (*line == ' ' || *line == '\t'))
    {
        line++;
    }
    if (strncmp(line, "struct", 6) == 0)
    {
        return 1;
    }
    if (strncmp(line, "impl", 4) == 0)
    {
        return 1;
    }
    if (strncmp(line, "fn", 2) == 0)
    {
        return 1;
    }
    if (strncmp(line, "use", 3) == 0)
    {
        return 1;
    }
    if (strncmp(line, "include", 7) == 0)
    {
        return 1;
    }
    if (strncmp(line, "typedef", 7) == 0)
    {
        return 1;
    }
    if (strncmp(line, "enum", 4) == 0)
    {
        return 1;
    }
    if (strncmp(line, "const", 5) == 0)
    {
        return 1;
    }
    if (strncmp(line, "def", 3) == 0)
    {
        return 1;
    }
    if (strncmp(line, "#include", 8) == 0)
    {
        return 1;
    }
    if (strncmp(line, "import", 6) == 0)
    {
        return 1;
    }

    return 0;
}

static void repl_error_callback(void *data, Token t, const char *msg)
{
    (void)data;
    (void)t;
    fprintf(stderr, "\033[1;31merror:\033[0m %s\n", msg);
}
#ifndef _WIN32
static struct termios orig_termios;
static int raw_mode_enabled = 0;
#endif
static void disable_raw_mode()
{
#ifndef _WIN32
    if (raw_mode_enabled)
    {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
        raw_mode_enabled = 0;
    }
#endif
}

static void enable_raw_mode()
{
#ifndef _WIN32    
    if (!raw_mode_enabled)
    {
        tcgetattr(STDIN_FILENO, &orig_termios);
        atexit(disable_raw_mode);
        struct termios raw = orig_termios;
        raw.c_lflag &= ~(ECHO | ICANON | IEXTEN | ISIG);
        raw.c_iflag &= ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
        raw.c_oflag &= ~(OPOST);
        raw.c_cflag |= (CS8);
        raw.c_cc[VMIN] = 1;
        raw.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
        raw_mode_enabled = 1;
    }
#endif
}

static const char *KEYWORDS[] = {
    "fn",       "struct",  "var",   "let",   "def",    "const",    "return",  "if",
    "else",     "for",     "while", "do",    "switch", "case",     "default", "break",
    "continue", "typedef", "enum",  "union", "sizeof", "typeof",   "import",  "include",
    "defer",    "guard",   "match", "impl",  "trait",  "comptime", "asm",     "plugin",
    "true",     "false",   "null",  "NULL",  NULL};

static const char *TYPES[] = {"void",  "int",      "char",   "float", "double", "long",
                              "short", "unsigned", "signed", "bool",  NULL};

static int find_matching_brace(const char *buf, int pos)
{
    if (pos < 0 || pos >= (int)strlen(buf))
    {
        return -1;
    }
    char c = buf[pos];
    int dir = 0;
    char match = 0;
    if (c == '{')
    {
        match = '}';
        dir = 1;
    }
    else if (c == '(')
    {
        match = ')';
        dir = 1;
    }
    else if (c == '[')
    {
        match = ']';
        dir = 1;
    }
    else if (c == '}')
    {
        match = '{';
        dir = -1;
    }
    else if (c == ')')
    {
        match = '(';
        dir = -1;
    }
    else if (c == ']')
    {
        match = '[';
        dir = -1;
    }
    else
    {
        return -1;
    }

    int depth = 1;
    int p = pos + dir;
    int len = strlen(buf);
    while (p >= 0 && p < len)
    {
        if (buf[p] == c)
        {
            depth++;
        }
        else if (buf[p] == match)
        {
            depth--;
            if (depth == 0)
            {
                return p;
            }
        }
        p += dir;
    }
    return -1;
}

// Calculate visible length of a string (ignoring ANSI codes)
static int get_visible_length(const char *str)
{
    int len = 0;
    int in_esc = 0;
    while (*str)
    {
        if (*str == '\033')
        {
            in_esc = 1;
        }
        else if (in_esc)
        {
            if (*str == 'm' || *str == 'K') // End of SGR or EL
            {
                in_esc = 0;
            }
            if (isalpha(*str))
            {
                in_esc = 0; // Terminating char
            }
        }
        else
        {
            len++;
        }
        str++;
    }
    return len;
}

// Simple syntax highlighter for the REPL
static void repl_highlight(const char *buf, int cursor_pos);

static int is_definition_of(const char *code, const char *name)
{
    Lexer l;
    lexer_init(&l, code);
    Token t = lexer_next(&l);
    int is_header = 0;

    if (t.type == TOK_UNION)
    {
        is_header = 1;
    }
    else if (t.type == TOK_IDENT)
    {
        if ((t.len == 2 && strncmp(t.start, "fn", 2) == 0) ||
            (t.len == 6 && strncmp(t.start, "struct", 6) == 0) ||
            (t.len == 4 && strncmp(t.start, "enum", 4) == 0) ||
            (t.len == 7 && strncmp(t.start, "typedef", 7) == 0) ||
            (t.len == 5 && strncmp(t.start, "const", 5) == 0))
        {
            is_header = 1;
        }
    }

    if (is_header)
    {
        Token name_tok = lexer_next(&l);
        if (name_tok.type == TOK_IDENT)
        {
            if (strlen(name) == (size_t)name_tok.len &&
                strncmp(name, name_tok.start, name_tok.len) == 0)
            {
                return 1;
            }
        }
    }
    return 0;
}

static void repl_highlight(const char *buf, int cursor_pos)
{
    const char *p = buf;

    int match_pos = -1;
    int brace_pos = -1;

    // Check under cursor
    if (find_matching_brace(buf, cursor_pos) != -1)
    {
        brace_pos = cursor_pos;
        match_pos = find_matching_brace(buf, cursor_pos);
    }
    // Check before cursor (common behavior when typing)
    else if (cursor_pos > 0 && find_matching_brace(buf, cursor_pos - 1) != -1)
    {
        brace_pos = cursor_pos - 1;
        match_pos = find_matching_brace(buf, cursor_pos - 1);
    }

    while (*p)
    {
        long idx = p - buf;

        // Highlight matching braces
        if (idx == brace_pos || idx == match_pos)
        {
            printf("\033[1;44;37m"); // Bright White on Blue background
            putchar(*p);
            printf("\033[0m");
            p++;
            continue;
        }

        if (strncmp(p, "//", 2) == 0)
        {
            printf("\033[1;30m");
            printf("%s", p);
            printf("\033[0m");
            break;
        }
        else if (*p == ':' && isalpha(p[1]))
        {
            printf("\033[1;35m");
            while (*p && !isspace(*p))
            {
                putchar(*p);
                p++;
            }
            printf("\033[0m");
        }
        else if (isdigit(*p))
        {
            printf("\033[1;35m");
            while (isdigit(*p) || *p == '.' || *p == 'x' || *p == 'X')
            {
                putchar(*p);
                p++;
            }
            printf("\033[0m");
        }
        else if (*p == '"' || *p == '\'')
        {
            char quote = *p;
            printf("\033[1;32m");
            putchar(*p);
            p++;
            while (*p && *p != quote)
            {
                if (*p == '\\' && p[1])
                {
                    putchar(*p);
                    p++;
                }
                putchar(*p);
                p++;
            }
            if (*p == quote)
            {
                putchar(*p);
                p++;
            }
            printf("\033[0m");
        }
        else if (strchr(",;.", *p))
        {
            printf("\033[1;30m");
            putchar(*p);
            printf("\033[0m");
            p++;
        }
        else if (strchr("{}[]()", *p))
        {
            printf("\033[0;36m");
            putchar(*p);
            printf("\033[0m");
            p++;
        }
        else if (strchr("+-*/=<>!&|^~%", *p))
        {
            printf("\033[1;37m");
            putchar(*p);
            printf("\033[0m");
            p++;
        }
        else if (isalpha(*p) || *p == '_')
        {
            const char *start = p;
            while (isalnum(*p) || *p == '_')
            {
                p++;
            }
            int len = p - start;
            char word[256];
            if (len < 256)
            {
                strncpy(word, start, len);
                word[len] = 0;

                int is_keyword = 0;
                for (int i = 0; KEYWORDS[i]; i++)
                {
                    if (strcmp(word, KEYWORDS[i]) == 0)
                    {
                        is_keyword = 1;
                        break;
                    }
                }

                int is_type = 0;
                if (!is_keyword)
                {
                    for (int i = 0; TYPES[i]; i++)
                    {
                        if (strcmp(word, TYPES[i]) == 0)
                        {
                            is_type = 1;
                            break;
                        }
                    }
                }

                int is_func = 0;
                if (!is_keyword && !is_type)
                {
                    const char *peek = p;
                    while (*peek && isspace(*peek))
                    {
                        peek++;
                    }
                    if (*peek == '(')
                    {
                        is_func = 1;
                    }
                }

                int is_const = 0;
                if (!is_keyword && !is_type && !is_func && len > 1)
                {
                    int all_upper = 1;
                    int has_upper = 0;
                    for (int i = 0; word[i]; i++)
                    {
                        if (islower(word[i]))
                        {
                            all_upper = 0;
                        }
                        if (isupper(word[i]))
                        {
                            has_upper = 1;
                        }
                    }
                    if (all_upper && has_upper)
                    {
                        is_const = 1;
                    }
                }

                if (is_keyword)
                {
                    printf("\033[1;36m");
                }
                else if (is_type)
                {
                    printf("\033[1;33m");
                }
                else if (is_func)
                {
                    printf("\033[1;34m");
                }
                else if (is_const)
                {
                    printf("\033[1;31m");
                }

                printf("%s", word);
                printf("\033[0m");
            }
            else
            {
                printf("%.*s", len, start);
            }
        }
        else
        {
            putchar(*p);
            p++;
        }
    }
}

static char *repl_complete(const char *buf, int pos)
{
    int start = pos;
    while (start > 0 && (isalnum(buf[start - 1]) || buf[start - 1] == '_' ||
                         buf[start - 1] == ':' || buf[start - 1] == '!'))
    {
        start--;
    }

    int len = pos - start;
    if (len == 0)
    {
        return NULL;
    }

    char prefix[256];
    if (len >= 255)
    {
        return NULL;
    }
    strncpy(prefix, buf + start, len);
    prefix[len] = 0;

    char *match = NULL;
    int match_count = 0;

    for (int i = 0; KEYWORDS[i]; i++)
    {
        if (strncmp(KEYWORDS[i], prefix, len) == 0)
        {
            match = (char *)KEYWORDS[i];
            match_count++;
        }
    }

    static const char *COMMANDS[] = {
        ":help",    ":reset", ":imports", ":vars",  ":funcs", ":structs", ":history", ":type",
        ":time",    ":c",     ":doc",     ":run",   ":edit",  ":save",    ":load",    ":watch",
        ":unwatch", ":undo",  ":delete",  ":clear", ":quit",  NULL};

    if (prefix[0] == ':')
    {
        for (int i = 0; COMMANDS[i]; i++)
        {
            if (strncmp(COMMANDS[i], prefix, len) == 0)
            {
                match = (char *)COMMANDS[i];
                match_count++;
            }
        }
    }

    if (match_count == 1)
    {
        return strdup(match + len);
    }

    return NULL;
}

static char *repl_readline(const char *prompt, char **history, int history_len, int indent_level)
{
    enable_raw_mode();

    int buf_size = 1024;
    char *buf = malloc(buf_size);
    buf[0] = 0;
    int len = 0;
    int pos = 0;

    if (indent_level > 0)
    {
        for (int i = 0; i < indent_level * 4; i++)
        {
            if (len >= buf_size - 1)
            {
                buf_size *= 2;
                buf = realloc(buf, buf_size);
            }
            buf[len++] = ' ';
        }
        buf[len] = 0;
        pos = len;
    }

    int history_idx = history_len;
    char *saved_current_line = NULL;

    int in_search_mode = 0;
    char search_buf[256];
    search_buf[0] = 0;
    int search_match_idx = -1;

    printf("\r\033[K%s", prompt);
    repl_highlight(buf, pos);
    fflush(stdout);

    while (1)
    {
        char c;
        if (read(STDIN_FILENO, &c, 1) != 1)
        {
            break;
        }

        if (c == '\x1b')
        {
            char seq[3];
            if (read(STDIN_FILENO, &seq[0], 1) != 1)
            {
                continue;
            }
            if (read(STDIN_FILENO, &seq[1], 1) != 1)
            {
                continue;
            }

            if (seq[0] == '[')
            {
                if (seq[1] == 'A')
                {
                    if (history_idx > 0)
                    {
                        if (history_idx == history_len)
                        {
                            if (saved_current_line)
                            {
                                free(saved_current_line);
                            }
                            saved_current_line = strdup(buf);
                        }
                        history_idx--;
                        if (history_idx >= 0 && history_idx < history_len)
                        {
                            free(buf);
                            buf = strdup(history[history_idx]);
                            buf_size = strlen(buf) + 1;
                            len = strlen(buf);
                            pos = len;
                        }
                    }
                }
                else if (seq[1] == 'B')
                {
                    if (history_idx < history_len)
                    {
                        history_idx++;
                        free(buf);
                        if (history_idx == history_len)
                        {
                            if (saved_current_line)
                            {
                                buf = strdup(saved_current_line);
                            }
                            else
                            {
                                buf = strdup("");
                            }
                        }
                        else
                        {
                            buf = strdup(history[history_idx]);
                        }
                        buf_size = strlen(buf) + 1;
                        len = strlen(buf);
                        pos = len;
                    }
                }
                else if (seq[1] == 'C')
                {
                    if (pos < len)
                    {
                        pos++;
                    }
                }
                else if (seq[1] == 'D')
                {
                    if (pos > 0)
                    {
                        pos--;
                    }
                }
                else if (seq[1] == 'H')
                {
                    pos = 0;
                }
                else if (seq[1] == 'F')
                {
                    pos = len;
                }
            }
        }
        else if (c == 127 || c == 8)
        {
            if (pos > 0)
            {
                memmove(buf + pos - 1, buf + pos, len - pos + 1);
                len--;
                pos--;
            }
        }
        else if (c == '\r' || c == '\n')
        {
            printf("\r\n");
            break;
        }
        else if (c == 3)
        {
            printf("^C\r\n");
            free(buf);
            if (saved_current_line)
            {
                free(saved_current_line);
            }
            disable_raw_mode();
            return strdup("");
        }
        else if (c == 4)
        {
            if (len == 0)
            {
                free(buf);
                if (saved_current_line)
                {
                    free(saved_current_line);
                }
                disable_raw_mode();
                return NULL;
            }
        }
        else if (c == '\t')
        {
            char *completion = repl_complete(buf, pos);
            if (completion)
            {
                int clen = strlen(completion);
                if (len + clen < buf_size - 1)
                {
                    // Insert completion
                    memmove(buf + pos + clen, buf + pos, len - pos + 1);
                    memcpy(buf + pos, completion, clen);
                    len += clen;
                    pos += clen;
                }
                free(completion);
            }
        }
        else if (c == 18)
        {
            if (!in_search_mode)
            {
                in_search_mode = 1;
                search_buf[0] = 0;
                search_match_idx = history_len;
            }

            int found = -1;
            int start_idx = search_match_idx - 1;
            if (start_idx >= history_len)
            {
                start_idx = history_len - 1;
            }

            for (int i = start_idx; i >= 0; i--)
            {
                if (strstr(history[i], search_buf))
                {
                    found = i;
                    break;
                }
            }

            if (found != -1)
            {
                search_match_idx = found;
                free(buf);
                buf = strdup(history[found]);
                buf_size = strlen(buf) + 1;
                len = strlen(buf);
                pos = len;
                history_idx = found; // Sync history navigation
            }
        }
        else if (in_search_mode)
        {
            if (c == 127 || c == 8) // Backspace
            {
                int sl = strlen(search_buf);
                if (sl > 0)
                {
                    search_buf[sl - 1] = 0;
                    search_match_idx = history_len;
                    int found = -1;
                    for (int i = history_len - 1; i >= 0; i--)
                    {
                        if (strstr(history[i], search_buf))
                        {
                            found = i;
                            break;
                        }
                    }
                    if (found != -1)
                    {
                        search_match_idx = found;
                        free(buf);
                        buf = strdup(history[found]);
                        buf_size = strlen(buf) + 1;
                        len = strlen(buf);
                        pos = len;
                        history_idx = found;
                    }
                }
            }
            else if (c == '\r' || c == '\n' || c == 27 || c == 7 ||
                     c == 3) // Enter/Esc/Ctrl+G/Ctrl+C
            {
                in_search_mode = 0;
                if (c == 3)
                {
                    // Abort
                    free(buf);
                    buf = strdup("");
                    len = 0;
                    pos = 0;
                    printf("^C\r\n");
                    return buf;
                }
                if (c == 7)
                {
                    // Keep current match
                }
                else if (c == '\r' || c == '\n')
                {
                    printf("\r\n");
                    break;
                }
            }
            else if (!iscntrl(c))
            {
                int sl = strlen(search_buf);
                if (sl < 255)
                {
                    search_buf[sl] = c;
                    search_buf[sl + 1] = 0;

                    int found = -1;
                    for (int i = history_len - 1; i >= 0; i--)
                    {
                        if (strstr(history[i], search_buf))
                        {
                            found = i;
                            break;
                        }
                    }
                    if (found != -1)
                    {
                        search_match_idx = found;
                        free(buf);
                        buf = strdup(history[found]);
                        buf_size = strlen(buf) + 1;
                        len = strlen(buf);
                        pos = len;
                        history_idx = found;
                    }
                }
            }
        }
        else if (c == 1)
        {
            pos = 0;
        }
        else if (c == 5)
        {
            pos = len;
        }
        else if (c == 12)
        {
            printf("\033[2J\033[H");
        }
        else if (c == 21)
        {
            if (pos > 0)
            {
                memmove(buf, buf + pos, len - pos + 1);
                len -= pos;
                pos = 0;
            }
        }
        else if (c == 11)
        {
            buf[pos] = 0;
            len = pos;
        }
        else if (c == 14)
        {
            printf("^N\r\n");
            free(buf);
            if (saved_current_line)
            {
                free(saved_current_line);
            }
            disable_raw_mode();
            return strdup(":reset");
        }
        else if (!iscntrl(c))
        {
            if (len >= buf_size - 1)
            {
                buf_size *= 2;
                buf = realloc(buf, buf_size);
            }
            memmove(buf + pos + 1, buf + pos, len - pos + 1);
            buf[pos] = c;
            len++;
            pos++;
        }

        if (in_search_mode)
        {
            printf("\r\033[K(reverse-i-search)`%s': %s", search_buf, buf);
        }
        else
        {
            printf("\r\033[K%s", prompt);
            repl_highlight(buf, pos);
            int prompt_len = get_visible_length(prompt);
            if (pos + prompt_len > 0)
            {
                printf("\r\033[%dC", pos + prompt_len);
            }
            else
            {
                printf("\r");
            }
        }

        fflush(stdout);
    }

    if (saved_current_line)
    {
        free(saved_current_line);
    }
    disable_raw_mode();

    return buf;
}

static void repl_get_code(char **history, int len, char **out_global, char **out_main)
{
    size_t total_len = 0;
    for (int i = 0; i < len; i++)
    {
        total_len += strlen(history[i]) + 2;
    }

    char *global_buf = malloc(total_len + 1);
    char *main_buf = malloc(total_len + 1);
    global_buf[0] = 0;
    main_buf[0] = 0;

    int brace_depth = 0;
    int in_global = 0;

    for (int i = 0; i < len; i++)
    {
        char *line = history[i];

        if (brace_depth == 0)
        {
            if (is_header_line(line))
            {
                in_global = 1;
            }
            else
            {
                in_global = 0;
            }
        }

        if (in_global)
        {
            strcat(global_buf, line);
            strcat(global_buf, "\n");
        }
        else
        {
            strcat(main_buf, line);
            strcat(main_buf, " ");
        }

        for (char *p = line; *p; p++)
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

    *out_global = global_buf;
    *out_main = main_buf;
}

void run_repl(const char *self_path)
{
    printf("\033[1;36mZen C REPL (%s)\033[0m\n", ZEN_VERSION);
    printf("Type 'exit' or 'quit' to leave.\n");
    printf("Type :help for commands.\n");

    int history_cap = 64;
    int history_len = 0;
    char **history = xmalloc(history_cap * sizeof(char *));

    char history_path[512];
    const char *home = getenv("HOME");
    if (z_is_windows() && !home)
    {
        home = getenv("USERPROFILE");
    }
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

    char *watches[16];
    int watches_len = 0;
    for (int i = 0; i < 16; i++)
    {
        watches[i] = NULL;
    }

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
        char cwd[1024];
        char prompt_text[1280];
        if (getcwd(cwd, sizeof(cwd)))
        {
            char *base = strrchr(cwd, '/');
            if (base)
            {
                base++;
            }
            else
            {
                base = cwd;
            }
            snprintf(prompt_text, sizeof(prompt_text), "\033[1;32m%s >>>\033[0m ", base);
        }
        else
        {
            strcpy(prompt_text, "\033[1;32m>>>\033[0m ");
        }

        const char *prompt = (brace_depth > 0 || paren_depth > 0) ? "... " : prompt_text;
        int indent = (brace_depth > 0) ? brace_depth : 0;
        char *rline = repl_readline(prompt, history, history_len, indent);

        if (!rline)
        {
            break;
        }
        strncpy(line_buf, rline, sizeof(line_buf) - 2);
        line_buf[sizeof(line_buf) - 2] = 0;
        strcat(line_buf, "\n");
        free(rline);

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

            if (cmd_buf[0] == '!')
            {
                int ret = system(cmd_buf + 1);
                printf("(exit code: %d)\n", ret);
                continue;
            }
            if (cmd_buf[0] == ':')
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
                    printf("  :undo       Remove last command\n");
                    printf("  :delete <n> Remove command at index n\n");
                    printf("  :watch <x>  Watch expression output\n");
                    printf("  :unwatch <n> Remove watch n\n");
                    printf("  :clear      Clear screen\n");
                    printf("  ! <cmd>     Run shell command\n");
                    printf("  :quit       Exit REPL\n");
                    printf("\nShortcuts:\n");
                    printf("  Up/Down     History navigation\n");
                    printf("  Tab         Completion\n");
                    printf("  Ctrl+A      Go to start\n");
                    printf("  Ctrl+E      Go to end\n");
                    printf("  Ctrl+L      Clear screen\n");
                    printf("  Ctrl+U      Clear line to start\n");
                    printf("  Ctrl+K      Clear line to end\n");
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
                else if (0 == strncmp(cmd_buf, ":show ", 6))
                {
                    char *name = cmd_buf + 6;
                    while (*name && isspace(*name))
                    {
                        name++;
                    }

                    int found = 0;
                    printf("Source definition for '%s':\n", name);

                    for (int i = history_len - 1; i >= 0; i--)
                    {
                        if (is_definition_of(history[i], name))
                        {
                            printf("  \033[90m// Found in history:\033[0m\n");
                            printf("  ");
                            repl_highlight(history[i], -1);
                            printf("\n");
                            found = 1;
                            break;
                        }
                    }

                    if (found)
                    {
                        continue;
                    }

                    printf("Source definition for '%s':\n", name);

                    size_t show_code_size = 4096;
                    for (int i = 0; i < history_len; i++)
                    {
                        show_code_size += strlen(history[i]) + 2;
                    }
                    char *show_code = malloc(show_code_size);
                    strcpy(show_code, "");
                    for (int i = 0; i < history_len; i++)
                    {
                        strcat(show_code, history[i]);
                        strcat(show_code, "\n");
                    }

                    ParserContext ctx = {0};
                    ctx.is_repl = 1;
                    ctx.skip_preamble = 1;
                    ctx.is_fault_tolerant = 1;
                    ctx.on_error = repl_error_callback;
                    Lexer l;
                    lexer_init(&l, show_code);
                    ASTNode *nodes = parse_program(&ctx, &l);

                    ASTNode *search = nodes;
                    if (search && search->type == NODE_ROOT)
                    {
                        search = search->root.children;
                    }

                    for (ASTNode *n = search; n; n = n->next)
                    {
                        if (n->type == NODE_FUNCTION && 0 == strcmp(n->func.name, name))
                        {
                            printf("  fn %s(%s) -> %s\n", n->func.name,
                                   n->func.args ? n->func.args : "",
                                   n->func.ret_type ? n->func.ret_type : "void");
                            found = 1;
                            break;
                        }
                        else if (n->type == NODE_STRUCT && 0 == strcmp(n->strct.name, name))
                        {
                            printf("  struct %s {\n", n->strct.name);
                            for (ASTNode *field = n->strct.fields; field; field = field->next)
                            {
                                if (field->type == NODE_FIELD)
                                {
                                    printf("    %s: %s;\n", field->field.name, field->field.type);
                                }
                                else if (field->type == NODE_VAR_DECL)
                                {
                                    // Fields might be VAR_DECLs in some parses? No, usually
                                    // NODE_FIELD for structs.
                                    printf("    %s: %s;\n", field->var_decl.name,
                                           field->var_decl.type_str);
                                }
                            }
                            printf("  }\n");
                            found = 1;
                            break;
                        }
                    }
                    if (!found)
                    {
                        printf("  (not found)\n");
                    }
                    free(show_code);
                    continue;
                }
                else if (0 == strcmp(cmd_buf, ":clear"))
                {
                    printf("\033[2J\033[H");
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
                    const char *tmpdir = getenv("TEMP");
                    if (!tmpdir)
                    {
                        tmpdir = getenv("TMP");
                    }
                    if (!tmpdir && !z_is_windows())
                    {
                        tmpdir = "/tmp";
                    }
                    if (!tmpdir)
                    {
                        tmpdir = ".";
                    }
                    snprintf(edit_path, sizeof(edit_path), "%s/zprep_edit_%d.zc", tmpdir, rand());
                    FILE *f = fopen(edit_path, "w");
                    if (f)
                    {
                        fprintf(f, "%s", history[idx]);
                        fclose(f);

                        const char *editor = getenv("EDITOR");
                        if (!editor)
                        {
                            editor = "nano";
                        }

                        char cmd[1024];
                        sprintf(cmd, "%s %s", editor, edit_path);
                        int status = system(cmd);

                        if (0 == status)
                        {
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
                        char *global_code = NULL;
                        char *main_code = NULL;
                        repl_get_code(history, history_len, &global_code, &main_code);

                        fprintf(f, "%s\n", global_code);
                        fprintf(f, "\nfn main() {\n%s\n}\n", main_code);

                        free(global_code);
                        free(main_code);
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
                    char *global_code = NULL;
                    char *main_code = NULL;
                    repl_get_code(history, history_len, &global_code, &main_code);

                    size_t code_size = strlen(global_code) + strlen(main_code) + 128;
                    char *code = malloc(code_size);
                    sprintf(code, "%s\nfn main() { %s }", global_code, main_code);
                    free(global_code);
                    free(main_code);

                    ParserContext ctx = {0};
                    ctx.is_repl = 1;
                    ctx.skip_preamble = 1;
                    ctx.is_fault_tolerant = 1;
                    ctx.on_error = repl_error_callback;

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

                        // Generate probe code to print values
                        char *global_code = NULL;
                        char *main_code = NULL;
                        repl_get_code(history, history_len, &global_code, &main_code);

                        // Generate probe code to print values
                        size_t probe_size = strlen(global_code) + strlen(main_code) + 4096;
                        char *probe_code = malloc(probe_size);

                        sprintf(probe_code,
                                "%s\nfn main() { _z_suppress_stdout(); %s _z_restore_stdout(); "
                                "printf(\"Variables:\\n\"); ",
                                global_code, main_code);
                        free(global_code);
                        free(main_code);

                        int found_vars = 0;
                        if (main_func && main_func->func.body &&
                            main_func->func.body->type == NODE_BLOCK)
                        {
                            for (ASTNode *s = main_func->func.body->block.statements; s;
                                 s = s->next)
                            {
                                if (s->type == NODE_VAR_DECL)
                                {
                                    char *t =
                                        s->var_decl.type_str ? s->var_decl.type_str : "Inferred";
                                    // Heuristic for format
                                    char fmt[64];
                                    char val_expr[128];

                                    if (s->var_decl.type_str)
                                    {
                                        if (strcmp(t, "int") == 0 || strcmp(t, "i32") == 0)
                                        {
                                            strcpy(fmt, "%d");
                                            strcpy(val_expr, s->var_decl.name);
                                        }
                                        else if (strcmp(t, "i64") == 0)
                                        {
                                            strcpy(fmt, "%ld");
                                            sprintf(val_expr, "(long)%s", s->var_decl.name);
                                        }
                                        else if (strcmp(t, "float") == 0 ||
                                                 strcmp(t, "double") == 0 ||
                                                 strcmp(t, "f32") == 0 || strcmp(t, "f64") == 0)
                                        {
                                            strcpy(fmt, "%f");
                                            strcpy(val_expr, s->var_decl.name);
                                        }
                                        else if (strcmp(t, "bool") == 0)
                                        {
                                            strcpy(fmt, "%s");
                                            sprintf(val_expr, "%s ? \"true\" : \"false\"",
                                                    s->var_decl.name);
                                        }
                                        else if (strcmp(t, "string") == 0 ||
                                                 strcmp(t, "char*") == 0)
                                        {
                                            strcpy(fmt, "\\\"%s\\\"");
                                            strcpy(val_expr, s->var_decl.name);
                                        } // quote strings
                                        else if (strcmp(t, "char") == 0)
                                        {
                                            strcpy(fmt, "'%c'");
                                            strcpy(val_expr, s->var_decl.name);
                                        }
                                        else
                                        {
                                            // Fallback: address
                                            strcpy(fmt, "@%p");
                                            sprintf(val_expr, "(void*)&%s", s->var_decl.name);
                                        }
                                    }
                                    else
                                    {
                                        // Inferred: Safe fallback? Or try to guess?
                                        // For now, minimal safety: print address
                                        strcpy(fmt, "? @%p");
                                        sprintf(val_expr, "(void*)&%s", s->var_decl.name);
                                    }

                                    char print_stmt[512];
                                    snprintf(print_stmt, sizeof(print_stmt),
                                             "printf(\"  %s (%s): %s\\n\", %s); ", s->var_decl.name,
                                             t, fmt, val_expr);
                                    strcat(probe_code, print_stmt);
                                    found_vars = 1;
                                }
                            }
                        }

                        if (!found_vars)
                        {
                            strcat(probe_code, "printf(\"  (none)\\n\");");
                        }

                        strcat(probe_code, " }");

                        // Execute
                        char tmp_path[256];
                        snprintf(tmp_path, sizeof(tmp_path), "/tmp/zen_repl_vars_%d.zc", getpid());
                        FILE *f = fopen(tmp_path, "w");
                        if (f)
                        {
                            fprintf(f, "%s", probe_code);
                            fclose(f);
                            char cmd[512];
                            snprintf(cmd, sizeof(cmd), "%s run -q %s", self_path, tmp_path);
                            system(cmd);
                            remove(tmp_path);
                        }
                        free(probe_code);
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

                    char *global_code = NULL;
                    char *main_code = NULL;
                    repl_get_code(history, history_len, &global_code, &main_code);

                    size_t probe_size =
                        strlen(global_code) + strlen(main_code) + strlen(expr) + 4096;
                    char *probe_code = malloc(probe_size);

                    sprintf(probe_code, "%s\nfn main() { _z_suppress_stdout(); %s", global_code,
                            main_code);
                    free(global_code);
                    free(main_code);

                    strcat(probe_code, " raw { typedef struct { int _u; } __REVEAL_TYPE__; } ");
                    strcat(probe_code, " let _z_type_probe: __REVEAL_TYPE__; _z_type_probe = (");
                    strcat(probe_code, expr);
                    strcat(probe_code, "); }");

                    char tmp_path[256];
                    const char *tmpdir = getenv("TEMP");
                    if (!tmpdir)
                    {
                        tmpdir = getenv("TMP");
                    }
                    if (!tmpdir && !z_is_windows())
                    {
                        tmpdir = "/tmp";
                    }
                    if (!tmpdir)
                    {
                        tmpdir = ".";
                    }
                    snprintf(tmp_path, sizeof(tmp_path), "%s/zprep_repl_type_%d.zc", tmpdir,
                             rand());
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
                                char *start = strstr(buf, "from type ");
                                char quote = 0;
                                if (!start)
                                {
                                    start = strstr(buf, "incompatible type ");
                                }

                                if (start)
                                {
                                    char *q = strchr(start, '\'');
                                    if (!q)
                                    {
                                        q = strstr(start, "\xe2\x80\x98");
                                    }

                                    if (q)
                                    {
                                        if (*q == '\'')
                                        {
                                            start = q + 1;
                                            quote = '\'';
                                        }
                                        else
                                        {
                                            start = q + 3;
                                            quote = 0;
                                        }

                                        char *end = NULL;
                                        if (quote)
                                        {
                                            end = strchr(start, quote);
                                        }
                                        else
                                        {
                                            end = strstr(start, "\xe2\x80\x99");
                                        }

                                        if (end)
                                        {
                                            *end = 0;
                                            printf("\033[1;36mType: %s\033[0m\n", start);
                                            found = 1;
                                            break;
                                        }
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

                    char *global_code = NULL;
                    char *main_code = NULL;
                    repl_get_code(history, history_len, &global_code, &main_code);

                    size_t code_size =
                        strlen(global_code) + strlen(main_code) + strlen(expr) + 4096;
                    char *code = malloc(code_size);

                    sprintf(code,
                            "%s\ninclude \"time.h\"\nfn main() { _z_suppress_stdout();\n%s "
                            "_z_restore_stdout();\n",
                            global_code, main_code);
                    free(global_code);
                    free(main_code);

                    strcat(code, "raw { clock_t _start = clock(); }\n");
                    strcat(code, "for _i in 0..1000 { ");
                    strcat(code, expr);
                    strcat(code, "; }\n");
                    strcat(code, "raw { clock_t _end = clock(); double _elapsed = (double)(_end - "
                                 "_start) / CLOCKS_PER_SEC; printf(\"1000 iterations: %.4fs "
                                 "(%.6fs/iter)\\n\", _elapsed, _elapsed/1000); }\n");
                    strcat(code, "}");

                    char tmp_path[256];
                    const char *tmpdir = getenv("TEMP");
                    if (!tmpdir)
                    {
                        tmpdir = getenv("TMP");
                    }
                    if (!tmpdir && !z_is_windows())
                    {
                        tmpdir = "/tmp";
                    }
                    if (!tmpdir)
                    {
                        tmpdir = ".";
                    }
                    snprintf(tmp_path, sizeof(tmp_path), "%s/zprep_repl_time_%d.zc", tmpdir,
                             rand());
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
                        char *more = repl_readline("... ", history, history_len, brace_depth);
                        if (!more)
                        {
                            break;
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
                        free(more);
                    }

                    char *global_code = NULL;
                    char *main_code = NULL;
                    repl_get_code(history, history_len, &global_code, &main_code);

                    size_t code_size =
                        strlen(global_code) + strlen(main_code) + strlen(expr_buf) + 128;
                    char *code = malloc(code_size);

                    sprintf(code, "%s\nfn main() { %s %s }", global_code, main_code, expr_buf);
                    free(global_code);
                    free(main_code);
                    free(expr_buf);

                    char tmp_path[256];
                    sprintf(tmp_path, "/tmp/zprep_repl_c_%d.zc", rand());
                    FILE *f = fopen(tmp_path, "w");
                    if (f)
                    {
                        fprintf(f, "%s", code);
                        fclose(f);
                        char cmd[2048];
                        sprintf(cmd,
                                "%s build -q --emit-c -o /tmp/zprep_repl_out %s "
                                "2>/dev/null; sed "
                                "-n '/^int main() {$/,/^}$/p' /tmp/zprep_repl_out.c "
                                "2>/dev/null | "
                                "tail -n +3 | head -n -2 | sed 's/^    //'",
                                self_path, tmp_path);
                        system(cmd);
                    }
                    free(code);
                    continue;
                }
                else if (0 == strcmp(cmd_buf, ":run"))
                {
                    char *global_code = NULL;
                    char *main_code = NULL;
                    repl_get_code(history, history_len, &global_code, &main_code);

                    size_t code_size = strlen(global_code) + strlen(main_code) + 128;
                    char *code = malloc(code_size);

                    sprintf(code, "%s\nfn main() { %s }", global_code, main_code);
                    free(global_code);
                    free(main_code);

                    char tmp_path[256];
                    sprintf(tmp_path, "/tmp/zprep_repl_run_%d.zc", rand());
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

        char *global_code = NULL;
        char *main_code = NULL;
        repl_get_code(history, history_len, &global_code, &main_code);

        size_t total_size = strlen(global_code) + strlen(main_code) + 4096;
        if (watches_len > 0)
        {
            total_size += 16 * 1024;
        }

        char *full_code = malloc(total_size);
        sprintf(full_code, "%s\nfn main() { _z_suppress_stdout(); %s", global_code, main_code);
        free(global_code);
        free(main_code);

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
            ctx.is_fault_tolerant = 1;
            ctx.on_error = repl_error_callback;
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
                char *global_code = NULL;
                char *main_code = NULL;
                repl_get_code(history, history_len - 1, &global_code, &main_code);

                size_t probesz = strlen(global_code) + strlen(main_code) + strlen(last_line) + 4096;
                char *probe_code = malloc(probesz);

                sprintf(probe_code, "%s\nfn main() { _z_suppress_stdout(); %s", global_code,
                        main_code);
                free(global_code);
                free(main_code);

                strcat(probe_code, " raw { typedef struct { int _u; } __REVEAL_TYPE__; } ");
                strcat(probe_code, " var _z_type_probe: __REVEAL_TYPE__; _z_type_probe = (");
                strcat(probe_code, last_line);
                strcat(probe_code, "); }");

                char p_path[256];
                sprintf(p_path, "/tmp/zprep_repl_probe_%d.zc", rand());
                FILE *pf = fopen(p_path, "w");
                if (pf)
                {
                    fprintf(pf, "%s", probe_code);
                    fclose(pf);

                    char p_cmd[2048];
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
            strcat(full_code, "; ");
            for (int i = 0; i < watches_len; i++)
            {
                char wbuf[1024];
                sprintf(wbuf,
                        "printf(\"\\033[90mwatch:%s = \\033[0m\"); print \"{%s}\"; "
                        "printf(\"\\n\"); ",
                        watches[i], watches[i]);
                strcat(full_code, wbuf);
            }
        }

        strcat(full_code, " }");

        char tmp_path[256];
        sprintf(tmp_path, "/tmp/zprep_repl_%d.zc", rand());
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
            int start = history_len > 1000 ? history_len - 1000 : 0;
            for (int i = start; i < history_len; i++)
            {
                fprintf(hf, "%s\n", history[i]);
            }
            fclose(hf);
        }
    }

    if (history)
    {
        for (int i = 0; i < history_len; i++)
        {
            free(history[i]);
        }
        free(history);
    }
    if (input_buffer)
    {
        free(input_buffer);
    }
}
