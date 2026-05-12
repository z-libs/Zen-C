// SPDX-License-Identifier: MIT
/**
 * @file repl_highlight.c
 * @brief Syntax highlighting, keyword tables, and brace matching for the REPL.
 */

#include "repl_state.h"

const char *REPL_KEYWORDS[] = {
    "fn",       "struct",  "var",   "let",   "def",    "const",    "return",  "if",
    "else",     "for",     "while", "do",    "switch", "case",     "default", "break",
    "continue", "typedef", "enum",  "union", "sizeof", "typeof",   "import",  "include",
    "defer",    "guard",   "match", "impl",  "trait",  "comptime", "asm",     "plugin",
    "true",     "false",   "null",  "NULL",  NULL};

const char *REPL_TYPES[] = {"void",  "int",      "char",   "float", "double", "long",
                            "short", "unsigned", "signed", "bool",  NULL};

int find_matching_brace(const char *buf, int pos)
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

int get_visible_length(const char *str)
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

void repl_highlight(const char *buf, int cursor_pos)
{
    static int checked_no_color = 0;
    static int use_color = 1;
    if (!checked_no_color)
    {
        if (getenv("NO_COLOR"))
        {
            use_color = 0;
        }
        checked_no_color = 1;
    }

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
            if (use_color)
            {
                printf("\033[1;44;37m"); // Bright White on Blue background
            }
            putchar(*p);
            if (use_color)
            {
                printf("\033[0m");
            }
            p++;
            continue;
        }

        if (strncmp(p, "//", 2) == 0)
        {
            if (use_color)
            {
                printf("\033[1;30m");
            }
            printf("%s", p);
            if (use_color)
            {
                printf("\033[0m");
            }
            break;
        }
        else if (*p == ':' && isalpha(p[1]))
        {
            if (use_color)
            {
                printf("\033[1;35m");
            }
            while (*p && !isspace(*p))
            {
                putchar(*p);
                p++;
            }
            if (use_color)
            {
                printf("\033[0m");
            }
        }
        else if (isdigit(*p))
        {
            if (use_color)
            {
                printf("\033[1;35m");
            }
            while (isdigit(*p) || *p == '.' || *p == 'x' || *p == 'X')
            {
                putchar(*p);
                p++;
            }
            if (use_color)
            {
                printf("\033[0m");
            }
        }
        else if (*p == '"' || *p == '\'')
        {
            char quote = *p;
            if (use_color)
            {
                printf("\033[1;32m");
            }
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
            if (use_color)
            {
                printf("\033[0m");
            }
        }
        else if (strchr(",;.", *p))
        {
            if (use_color)
            {
                printf("\033[1;30m");
            }
            putchar(*p);
            if (use_color)
            {
                printf("\033[0m");
            }
            p++;
        }
        else if (strchr("{}[]()", *p))
        {
            if (use_color)
            {
                printf("\033[0;36m");
            }
            putchar(*p);
            if (use_color)
            {
                printf("\033[0m");
            }
            p++;
        }
        else if (strchr("+-*/=<>!&|^~%", *p))
        {
            if (use_color)
            {
                printf("\033[1;37m");
            }
            putchar(*p);
            if (use_color)
            {
                printf("\033[0m");
            }
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
            char word[MAX_VAR_NAME_LEN];
            if (len < 256)
            {
                strncpy(word, start, len);
                word[len] = 0;

                int is_keyword = 0;
                for (int i = 0; REPL_KEYWORDS[i]; i++)
                {
                    if (strcmp(word, REPL_KEYWORDS[i]) == 0)
                    {
                        is_keyword = 1;
                        break;
                    }
                }

                int is_type = 0;
                if (!is_keyword)
                {
                    for (int i = 0; REPL_TYPES[i]; i++)
                    {
                        if (strcmp(word, REPL_TYPES[i]) == 0)
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

                if (use_color)
                {
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
                }

                printf("%s", word);
                if (use_color)
                {
                    printf("\033[0m");
                }
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
