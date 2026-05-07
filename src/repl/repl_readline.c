/**
 * @file repl_readline.c
 * @brief Line editing, history navigation, reverse search, and tab completion.
 */

#include "repl_state.h"

extern const char *REPL_KEYWORDS[];

static const char *COMMANDS[] = {":help",    ":reset", ":imports", ":vars",  ":funcs",   ":structs",
                                 ":history", ":type",  ":time",    ":c",     ":doc",     ":run",
                                 ":edit",    ":save",  ":load",    ":watch", ":unwatch", ":undo",
                                 ":delete",  ":clear", ":quit",    ":show",  NULL};

char *repl_complete(ReplState *state, const char *buf, int pos)
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

    char prefix[MAX_VAR_NAME_LEN];
    if (len >= 255)
    {
        return NULL;
    }
    strncpy(prefix, buf + start, len);
    prefix[len] = 0;

    char *match = NULL;
    int match_count = 0;

    /* Keywords */
    for (int i = 0; REPL_KEYWORDS[i]; i++)
    {
        if (strncmp(REPL_KEYWORDS[i], prefix, len) == 0)
        {
            match = (char *)REPL_KEYWORDS[i];
            match_count++;
        }
    }

    /* Commands */
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

    /* Session symbols (user-defined variables, functions, structs, stdlib types) */
    if (state && prefix[0] != ':')
    {
        for (int i = 0; i < state->symbol_count; i++)
        {
            if (strncmp(state->symbols[i], prefix, len) == 0)
            {
                match = state->symbols[i];
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

static int skip_word_forward(const char *buf, int pos, int len)
{
    while (pos < len && isspace(buf[pos]))
    {
        pos++;
    }
    while (pos < len && !isspace(buf[pos]))
    {
        pos++;
    }
    return pos;
}

static int skip_word_backward(const char *buf, int pos)
{
    while (pos > 0 && isspace(buf[pos - 1]))
    {
        pos--;
    }
    while (pos > 0 && !isspace(buf[pos - 1]))
    {
        pos--;
    }
    return pos;
}

char *repl_readline(ReplState *state, const char *prompt, int indent_level)
{
    repl_enable_raw_mode();

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

    int history_idx = state->history_len;
    char *saved_current_line = NULL;

    int in_search_mode = 0;
    char search_buf[MAX_VAR_NAME_LEN];
    search_buf[0] = 0;
    int search_match_idx = -1;

    printf("\r\033[K%s", prompt);
    repl_highlight(buf, pos);
    fflush(stdout);

    while (1)
    {
        char c;
        if (!repl_read_char(&c))
        {
            break;
        }

        if (c == '\x1b')
        {
            char seq[6];
            if (!repl_read_char(&seq[0]))
            {
                continue;
            }
            if (!repl_read_char(&seq[1]))
            {
                continue;
            }

            if (seq[0] == '[')
            {
                if (isdigit(seq[1]))
                {
                    if (repl_read_char(&seq[2]) && seq[2] == ';')
                    {
                        if (repl_read_char(&seq[3]) && repl_read_char(&seq[4]))
                        {
                            if (seq[4] == 'C')
                            { // Ctrl+Right
                                pos = skip_word_forward(buf, pos, len);
                            }
                            else if (seq[4] == 'D')
                            { // Ctrl+Left
                                pos = skip_word_backward(buf, pos);
                            }
                        }
                    }
                    else if (seq[2] == '~')
                    {
                        if (seq[1] == '1' || seq[1] == '7')
                        {
                            pos = 0;
                        }
                        else if (seq[1] == '4' || seq[1] == '8')
                        {
                            pos = len;
                        }
                        else if (seq[1] == '3')
                        { // Delete key
                            if (pos < len)
                            {
                                memmove(buf + pos, buf + pos + 1, len - pos);
                                len--;
                            }
                        }
                    }
                }
                else if (seq[1] == 'A')
                {
                    if (history_idx > 0)
                    {
                        if (history_idx == state->history_len)
                        {
                            if (saved_current_line)
                            {
                                zfree(saved_current_line);
                            }
                            saved_current_line = strdup(buf);
                        }
                        history_idx--;
                        if (history_idx >= 0 && history_idx < state->history_len)
                        {
                            zfree(buf);
                            buf = strdup(state->history[history_idx]);
                            buf_size = strlen(buf) + 1;
                            len = strlen(buf);
                            pos = len;
                        }
                    }
                }
                else if (seq[1] == 'B')
                {
                    if (history_idx < state->history_len)
                    {
                        history_idx++;
                        zfree(buf);
                        if (history_idx == state->history_len)
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
                            buf = strdup(state->history[history_idx]);
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
            else if (seq[0] == 'O')
            {
                if (seq[1] == 'H')
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
            if (state)
            {
                state->aborted = 1;
            }
            zfree(buf);
            if (saved_current_line)
            {
                zfree(saved_current_line);
            }
            repl_disable_raw_mode();
            return strdup("");
        }
        else if (c == 4)
        {
            if (len == 0)
            {
                zfree(buf);
                if (saved_current_line)
                {
                    zfree(saved_current_line);
                }
                repl_disable_raw_mode();
                return NULL;
            }
        }
        else if (c == '\t')
        {
            char *completion = repl_complete(state, buf, pos);
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
                zfree(completion);
            }
        }
        else if (c == 18)
        {
            if (!in_search_mode)
            {
                in_search_mode = 1;
                search_buf[0] = 0;
                search_match_idx = state->history_len;
            }

            int found = -1;
            int start_idx = search_match_idx - 1;
            if (start_idx >= state->history_len)
            {
                start_idx = state->history_len - 1;
            }

            for (int i = start_idx; i >= 0; i--)
            {
                if (strstr(state->history[i], search_buf))
                {
                    found = i;
                    break;
                }
            }

            if (found != -1)
            {
                search_match_idx = found;
                zfree(buf);
                buf = strdup(state->history[found]);
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
                    search_match_idx = state->history_len;
                    int found = -1;
                    for (int i = state->history_len - 1; i >= 0; i--)
                    {
                        if (strstr(state->history[i], search_buf))
                        {
                            found = i;
                            break;
                        }
                    }
                    if (found != -1)
                    {
                        search_match_idx = found;
                        zfree(buf);
                        buf = strdup(state->history[found]);
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
                    zfree(buf);
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
                    for (int i = state->history_len - 1; i >= 0; i--)
                    {
                        if (strstr(state->history[i], search_buf))
                        {
                            found = i;
                            break;
                        }
                    }
                    if (found != -1)
                    {
                        search_match_idx = found;
                        zfree(buf);
                        buf = strdup(state->history[found]);
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
            zfree(buf);
            if (saved_current_line)
            {
                zfree(saved_current_line);
            }
            repl_disable_raw_mode();
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
        zfree(saved_current_line);
    }
    repl_disable_raw_mode();

    return buf;
}
