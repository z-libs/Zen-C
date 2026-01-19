
#include "zprep.h"

void lexer_init(Lexer *l, const char *src)
{
    l->src = src;
    l->pos = 0;
    l->line = 1;
    l->col = 1;
}

static int is_ident_start(char c)
{
    return isalpha(c) || c == '_';
}

static int is_ident_char(char c)
{
    return isalnum(c) || c == '_';
}

Token lexer_next(Lexer *l)
{
    const char *s = l->src + l->pos;
    int start_line = l->line;
    int start_col = l->col;

    while (isspace(*s))
    {
        if (*s == '\n')
        {
            l->line++;
            l->col = 1;
        }
        else
        {
            l->col++;
        }
        l->pos++;
        s++;
        start_line = l->line;
        start_col = l->col;
    }

    // Check for EOF.
    if (!*s)
    {
        return (Token){TOK_EOF, s, 0, start_line, start_col};
    }

    // C preprocessor directives.
    if (*s == '#')
    {
        int len = 0;
        while (s[len] && s[len] != '\n')
        {
            if (s[len] == '\\' && s[len + 1] == '\n')
            {
                len += 2;
                l->line++;
            }
            else
            {
                len++;
            }
        }
        l->pos += len;

        return (Token){TOK_PREPROC, s, len, start_line, start_col};
    }

    // Comments.
    if (s[0] == '/' && s[1] == '/')
    {
        int len = 2;
        while (s[len] && s[len] != '\n')
        {
            len++;
        }
        l->pos += len;
        l->col += len;
        return lexer_next(l);
    }

    // Block Comments.
    if (s[0] == '/' && s[1] == '*')
    {
        // skip two start chars
        l->pos += 2;
        s += 2;

        while (s[0])
        {
            // s[len+1] can be at most the null terminator
            if (s[0] == '*' && s[1] == '/')
            {
                // go over */
                l->pos += 2;
                s += 2;
                break;
            }

            if (s[0] == '\n')
            {
                l->line++;
                l->col = 1;
            }
            else
            {
                l->col++;
            }

            l->pos++;
            s++;
        }

        return lexer_next(l);
    }

    // Identifiers.
    if (is_ident_start(*s))
    {
        int len = 0;
        while (is_ident_char(s[len]))
        {
            len++;
        }

        l->pos += len;
        l->col += len;

        if (len == 4 && strncmp(s, "test", 4) == 0)
        {
            return (Token){TOK_TEST, s, 4, start_line, start_col};
        }
        if (len == 6 && strncmp(s, "assert", 6) == 0)
        {
            return (Token){TOK_ASSERT, s, 6, start_line, start_col};
        }
        if (len == 6 && strncmp(s, "sizeof", 6) == 0)
        {
            return (Token){TOK_SIZEOF, s, 6, start_line, start_col};
        }
        if (len == 5 && strncmp(s, "defer", 5) == 0)
        {
            return (Token){TOK_DEFER, s, 5, start_line, start_col};
        }
        if (len == 8 && strncmp(s, "autofree", 8) == 0)
        {
            return (Token){TOK_AUTOFREE, s, 8, start_line, start_col};
        }
        if (len == 5 && strncmp(s, "alias", 5) == 0)
        {
            return (Token){TOK_ALIAS, s, 5, start_line, start_col};
        }
        if (len == 3 && strncmp(s, "use", 3) == 0)
        {
            return (Token){TOK_USE, s, 3, start_line, start_col};
        }
        if (len == 8 && strncmp(s, "comptime", 8) == 0)
        {
            return (Token){TOK_COMPTIME, s, 8, start_line, start_col};
        }
        if (len == 5 && strncmp(s, "union", 5) == 0)
        {
            return (Token){TOK_UNION, s, 5, start_line, start_col};
        }
        if (len == 3 && strncmp(s, "asm", 3) == 0)
        {
            return (Token){TOK_ASM, s, 3, start_line, start_col};
        }
        if (len == 8 && strncmp(s, "volatile", 8) == 0)
        {
            return (Token){TOK_VOLATILE, s, 8, start_line, start_col};
        }
        if (len == 5 && strncmp(s, "async", 5) == 0)
        {
            return (Token){TOK_ASYNC, s, 5, start_line, start_col};
        }
        if (len == 5 && strncmp(s, "await", 5) == 0)
        {
            return (Token){TOK_AWAIT, s, 5, start_line, start_col};
        }
        if (len == 3 && strncmp(s, "and", 3) == 0)
        {
            return (Token){TOK_AND, s, 3, start_line, start_col};
        }
        if (len == 2 && strncmp(s, "or", 2) == 0)
        {
            return (Token){TOK_OR, s, 2, start_line, start_col};
        }

        // F-Strings
        if (len == 1 && s[0] == 'f' && s[1] == '"')
        {
            // Reset pos/col because we want to parse string
            l->pos -= len;
            l->col -= len;
        }
        else
        {
            return (Token){TOK_IDENT, s, len, start_line, start_col};
        }
    }

    if (s[0] == 'f' && s[1] == '"')
    {
        int len = 2;
        while (s[len] && s[len] != '"')
        {
            if (s[len] == '\\')
            {
                len++;
            }
            len++;
        }
        if (s[len] == '"')
        {
            len++;
        }
        l->pos += len;
        l->col += len;
        return (Token){TOK_FSTRING, s, len, start_line, start_col};
    }

    // Numbers
    if (isdigit(*s))
    {
        int len = 0;
        if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))
        {
            len = 2;
            while (isxdigit(s[len]))
            {
                len++;
            }
        }
        else if (s[0] == '0' && (s[1] == 'b' || s[1] == 'B'))
        {
            len = 2;
            while (s[len] == '0' || s[len] == '1')
            {
                len++;
            }
        }
        else
        {
            while (isdigit(s[len]))
            {
                len++;
            }
            if (s[len] == '.')
            {
                if (s[len + 1] != '.')
                {
                    len++;
                    while (isdigit(s[len]))
                    {
                        len++;
                    }
                    // Consume float suffix (e.g. 1.0f)
                    if (is_ident_start(s[len]))
                    {
                        while (is_ident_char(s[len]))
                        {
                            len++;
                        }
                    }
                    l->pos += len;
                    l->col += len;
                    return (Token){TOK_FLOAT, s, len, start_line, start_col};
                }
            }
        }

        // Consume integer suffix (e.g. 1u, 100u64, 1L)
        if (is_ident_start(s[len]))
        {
            while (is_ident_char(s[len]))
            {
                len++;
            }
        }

        l->pos += len;
        l->col += len;
        return (Token){TOK_INT, s, len, start_line, start_col};
    }

    // Strings
    if (*s == '"')
    {
        int len = 1;
        while (s[len] && s[len] != '"')
        {
            if (s[len] == '\\')
            {
                len++;
            }
            len++;
        }
        if (s[len] == '"')
        {
            len++;
        }
        l->pos += len;
        l->col += len;
        return (Token){TOK_STRING, s, len, start_line, start_col};
    }

    if (*s == '\'')
    {
        int len = 1;
        // Handle escapes like '\n' or regular 'a'
        if (s[len] == '\\')
        {
            len++;
            len++;
        }
        else
        {
            len++;
        }
        if (s[len] == '\'')
        {
            len++;
        }

        l->pos += len;
        l->col += len;
        return (Token){TOK_CHAR, s, len, start_line, start_col};
    }

    // Operators.
    int len = 1;
    ZTokenType type = TOK_OP;

    if (s[0] == '?' && s[1] == '.')
    {
        len = 2;
        type = TOK_Q_DOT;
    }
    else if (s[0] == '?' && s[1] == '?')
    {
        if (s[2] == '=')
        {
            len = 3;
            type = TOK_QQ_EQ;
        }
        else
        {
            len = 2;
            type = TOK_QQ;
        }
    }
    else if (*s == '?')
    {
        type = TOK_QUESTION;
    }
    else if (s[0] == '|' && s[1] == '>')
    {
        len = 2;
        type = TOK_PIPE;
    }
    else if (s[0] == ':' && s[1] == ':')
    {
        len = 2;
        type = TOK_DCOLON;
    }
    else if (s[0] == '.' && s[1] == '.' && s[2] == '.')
    {
        len = 3;
        type = TOK_ELLIPSIS;
    }
    else if (s[0] == '.' && s[1] == '.')
    {
        if (s[2] == '=')
        {
            len = 3;
            type = TOK_DOTDOT_EQ;
        }
        else if (s[2] == '<')
        {
            len = 3;
            type = TOK_DOTDOT_LT;
        }
        else
        {
            len = 2;
            type = TOK_DOTDOT;
        }
    }
    else if ((s[0] == '-' && s[1] == '>') || (s[0] == '=' && s[1] == '>'))
    {
        len = 2;
        type = TOK_ARROW;
    }

    else if ((s[0] == '<' && s[1] == '<') || (s[0] == '>' && s[1] == '>'))
    {
        len = 2;
        if (s[2] == '=')
        {
            len = 3; // Handle <<= and >>=
        }
    }
    else if ((s[0] == '&' && s[1] == '&') || (s[0] == '|' && s[1] == '|') ||
             (s[0] == '+' && s[1] == '+') || (s[0] == '-' && s[1] == '-'))
    {
        len = 2;
    }
    else if (s[1] == '=')
    {
        // This catches: == != <= >= += -= *= /= %= |= &= ^=
        if (strchr("=!<>+-*/%|&^", s[0]))
        {
            len = 2;
        }
    }

    else
    {
        switch (*s)
        {

        case '(':
            type = TOK_LPAREN;
            break;
        case ')':
            type = TOK_RPAREN;
            break;
        case '{':
            type = TOK_LBRACE;
            break;
        case '}':
            type = TOK_RBRACE;
            break;
        case '[':
            type = TOK_LBRACKET;
            break;
        case ']':
            type = TOK_RBRACKET;
            break;
        case '<':
            type = TOK_LANGLE;
            break;
        case '>':
            type = TOK_RANGLE;
            break;
        case ',':
            type = TOK_COMMA;
            break;
        case ':':
            type = TOK_COLON;
            break;
        case ';':
            type = TOK_SEMICOLON;
            break;
        case '@':
            type = TOK_AT;
            break;
        default:
            type = TOK_OP;
            break;
        }
    }

    l->pos += len;
    l->col += len;
    return (Token){type, s, len, start_line, start_col};
}

Token lexer_peek(Lexer *l)
{
    Lexer saved = *l;
    return lexer_next(&saved);
}

Token lexer_peek2(Lexer *l)
{
    Lexer saved = *l;
    lexer_next(&saved);
    return lexer_next(&saved);
}
