
#include "zprep_plugin.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define ZC_COMPAT_IMPLEMENTATION
#include "compat/compat.h"

static long stack[256];
static int sp = 0;
static double fstack[256];
static int fsp = 0;
static long zmemory[4096];
static int zhere = 0;
static long loop_stack[32];
static int lsp = 0;

typedef struct
{
    char name[32];
    char body[512];
} ForthWord;
static ForthWord forth_dict[128];
static int forth_dict_count = 0;

// Stack operations
static void push(long v)
{
    if (sp < 256)
    {
        stack[sp++] = v;
    }
}
static long pop()
{
    return sp > 0 ? stack[--sp] : 0;
}
static long peek()
{
    return sp > 0 ? stack[sp - 1] : 0;
}
static void fpush(double v)
{
    if (fsp < 256)
    {
        fstack[fsp++] = v;
    }
}
static double fpop()
{
    return fsp > 0 ? fstack[--fsp] : 0.0;
}
static double fpeek()
{
    return fsp > 0 ? fstack[fsp - 1] : 0.0;
}

// String utilities
static char *xstrdup(const char *s)
{
    return s ? zc_strdup(s) : NULL;
}
static void *xmalloc(size_t n)
{
    void *p = malloc(n);
    if (!p)
    {
        exit(1);
    }
    return p;
}

// Tokenizer
static char *next_token(char **ptr)
{
    if (!ptr || !*ptr)
    {
        return NULL;
    }
    while (**ptr && (**ptr == ' ' || **ptr == '\t' || **ptr == '\n'))
    {
        (*ptr)++;
    }
    if (**ptr == '\0')
    {
        return NULL;
    }
    char *start = *ptr;
    while (**ptr && **ptr != ' ' && **ptr != '\t' && **ptr != '\n')
    {
        (*ptr)++;
    }
    if (**ptr)
    {
        **ptr = '\0';
        (*ptr)++;
    }
    return start;
}

static void skip_tokens_until(char **ptr, const char *target, const char *alt)
{
    char *t;
    while ((t = next_token(ptr)))
    {
        if (strcmp(t, target) == 0)
        {
            return;
        }
        if (alt && strcmp(t, alt) == 0)
        {
            return;
        }
    }
}

void eval_forth_token(char *token, char **scan_ptr, char **out_ptr)
{
    if (0 == strcmp(token, "+"))
    {
        push(pop() + pop());
    }
    else if (0 == strcmp(token, "-"))
    {
        long b = pop();
        push(pop() - b);
    }
    else if (0 == strcmp(token, "*"))
    {
        push(pop() * pop());
    }
    else if (0 == strcmp(token, "/"))
    {
        long b = pop();
        push(b ? pop() / b : 0);
    }
    else if (0 == strcmp(token, "%"))
    {
        long b = pop();
        push(b ? pop() % b : 0);
    }
    else if (0 == strcmp(token, "<<"))
    {
        long b = pop();
        push(pop() << b);
    }
    else if (0 == strcmp(token, ">>"))
    {
        long b = pop();
        push(pop() >> b);
    }
    else if (0 == strcmp(token, "&"))
    {
        push(pop() & pop());
    }
    else if (0 == strcmp(token, "|"))
    {
        push(pop() | pop());
    }
    else if (0 == strcmp(token, "^"))
    {
        push(pop() ^ pop());
    }
    else if (0 == strcmp(token, "~"))
    {
        push(~pop());
    }
    else if (0 == strcmp(token, "="))
    {
        push(pop() == pop());
    }
    else if (0 == strcmp(token, ">"))
    {
        long b = pop();
        push(pop() > b);
    }
    else if (0 == strcmp(token, "<"))
    {
        long b = pop();
        push(pop() < b);
    }
    else if (0 == strcmp(token, "choose"))
    {
        long f = pop();
        long t = pop();
        long c = pop();
        push(c ? t : f);
    }
    else if (0 == strcmp(token, "if"))
    {
        if (pop() == 0)
        {
            skip_tokens_until(scan_ptr, "then", "else");
        }
    }
    else if (0 == strcmp(token, "else"))
    {
        skip_tokens_until(scan_ptr, "then", NULL);
    }
    else if (0 == strcmp(token, "then"))
    {
        /* Marker. */
    }
    else if (0 == strcmp(token, "("))
    {
        char *t;
        while (NULL != (t = next_token(scan_ptr)))
        {
            if (strchr(t, ')'))
            {
                break;
            }
        }
    }
    else if (0 == strcmp(token, "do"))
    {
        long start = pop();
        long limit = pop();
        char *rest = *scan_ptr;
        char *temp = xstrdup(rest);
        char *curs = temp;
        char *t;
        int depth = 0;
        char *end = NULL;
        while (NULL != (t = next_token(&curs)))
        {
            if (0 == strcmp(t, "do"))
            {
                depth++;
            }
            if (0 == strcmp(t, "loop"))
            {
                if (depth == 0)
                {
                    end = curs;
                    break;
                }
                depth--;
            }
        }
        if (!end)
        {
            fprintf(stderr, "[Forth Error] 'do' without matching 'loop'\n");
            exit(1);
        }
        long len = end - temp;
        *scan_ptr += len;
        char *body = xmalloc(len + 1);
        strncpy(body, rest, len);
        body[len] = '\0';
        for (long i = start; i < limit; i++)
        {
            if (lsp < 32)
            {
                loop_stack[lsp++] = i;
            }
            else
            {
                fprintf(stderr, "[Error] Loop Stack Overflow\n");
                exit(1);
            }
            char *runnable = xstrdup(body);
            char *rc = runnable;
            char *rt;
            while (NULL != (rt = next_token(&rc)))
            {
                eval_forth_token(rt, &rc, out_ptr);
            }
            free(runnable);
            lsp--;
        }
        free(body);
        free(temp);
    }
    else if (0 == strcmp(token, "loop"))
    {
        /* Marker. */
    }
    else if (0 == strcmp(token, "i"))
    {
        if (lsp > 0)
        {
            push(loop_stack[lsp - 1]);
        }
        else
        {
            fprintf(stderr, "[Error] 'i' used outside of loop\n");
            exit(1);
        }
    }
    else if (0 == strcmp(token, "f+"))
    {
        fpush(fpop() + fpop());
    }
    else if (0 == strcmp(token, "f-"))
    {
        double b = fpop();
        fpush(fpop() - b);
    }
    else if (0 == strcmp(token, "f*"))
    {
        fpush(fpop() * fpop());
    }
    else if (0 == strcmp(token, "f/"))
    {
        double b = fpop();
        fpush(b != 0.0 ? fpop() / b : 0.0);
    }
    else if (0 == strcmp(token, "fsqrt"))
    {
        fpush(sqrt(fpop()));
    }
    else if (0 == strcmp(token, "dup"))
    {
        push(peek());
    }
    else if (0 == strcmp(token, "drop"))
    {
        pop();
    }
    else if (0 == strcmp(token, "swap"))
    {
        long a = pop();
        long b = pop();
        push(a);
        push(b);
    }
    else if (0 == strcmp(token, "rot"))
    {
        long a = pop();
        long b = pop();
        long c = pop();
        push(b);
        push(a);
        push(c);
    }
    else if (0 == strcmp(token, "-rot"))
    {
        long a = pop();
        long b = pop();
        long c = pop();
        push(a);
        push(c);
        push(b);
    }
    else if (0 == strcmp(token, "fdup"))
    {
        fpush(fpeek());
    }
    else if (0 == strcmp(token, "fdrop"))
    {
        fpop();
    }
    else if (0 == strcmp(token, "fswap"))
    {
        double a = fpop();
        double b = fpop();
        fpush(a);
        fpush(b);
    }
    else if (0 == strcmp(token, "i>f"))
    {
        fpush((double)pop());
    }
    else if (0 == strcmp(token, "f>i"))
    {
        push((long)fpop());
    }
    else if (0 == strcmp(token, "emit"))
    {
        *out_ptr += sprintf(*out_ptr, "%ld", pop());
    }
    else if (0 == strcmp(token, "femit"))
    {
        *out_ptr += sprintf(*out_ptr, "%f", fpop());
    }
    else if (0 == strcmp(token, ","))
    {
        *out_ptr += sprintf(*out_ptr, ", ");
    }
    else if (0 == strcmp(token, "."))
    {
        fprintf(stderr, "[Debug] Int: %ld\n", peek());
    }
    else if (0 == strcmp(token, "f."))
    {
        fprintf(stderr, "[Debug] Float: %f\n", fpeek());
    }
    else if (0 == strcmp(token, ".\""))
    {
        char *p = *scan_ptr;
        if (' ' == *p)
        {
            p++;
        }
        char *end = strchr(p, '"');
        if (end)
        {
            int len = end - p;
            strncpy(*out_ptr, p, len);
            *out_ptr += len;
            **out_ptr = '\0';
            *scan_ptr = end + 1;
        }
        else
        {
            fprintf(stderr, "[Forth Error] Unterminated string literal\n");
            exit(1);
        }
    }
    else if (0 == strcmp(token, "s\""))
    {
        char *p = *scan_ptr;
        if (' ' == *p)
        {
            p++;
        }
        char *end = strchr(p, '"');
        if (end)
        {
            int len = end - p;
            long start_addr = zhere;
            for (int i = 0; i < len; i++)
            {
                zmemory[zhere++] = (long)p[i];
            }
            push(start_addr);
            push(len);
            *scan_ptr = end + 1;
        }
        else
        {
            fprintf(stderr, "[Forth Error] Unterminated string literal (s\")\n");
            exit(1);
        }
    }
    else if (0 == strcmp(token, "c@"))
    {
        long addr = pop();
        push(zmemory[addr]);
    }
    else if (0 == strcmp(token, "!"))
    {
        long addr = pop();
        long val = pop();
        zmemory[addr] = val;
    }
    else if (0 == strcmp(token, "@"))
    {
        long addr = pop();
        push(zmemory[addr]);
    }
    else if (0 == strcmp(token, "?"))
    {
        long addr = pop();
        fprintf(stderr, "[Debug] @%ld = %ld\n", addr, zmemory[addr]);
    }
    else if (0 == strcmp(token, "allot"))
    {
        zhere += pop();
    }
    else if (0 == strcmp(token, "variable"))
    {
        char *name = next_token(scan_ptr);
        if (!name)
        {
            fprintf(stderr, "[Error] 'variable' needs a name\n");
            exit(1);
        }
        if (forth_dict_count < 128)
        {
            strcpy(forth_dict[forth_dict_count].name, name);
            sprintf(forth_dict[forth_dict_count].body, "%d", zhere);
            forth_dict_count++;
            zhere++;
        }
    }
    else if (0 == strcmp(token, "constant"))
    {
        long val = pop();
        char *name = next_token(scan_ptr);
        if (!name)
        {
            fprintf(stderr, "[Error] 'constant' needs a name\n");
            exit(1);
        }
        if (forth_dict_count < 128)
        {
            strcpy(forth_dict[forth_dict_count].name, name);
            sprintf(forth_dict[forth_dict_count].body, "%ld", val);
            forth_dict_count++;
        }
    }
    else
    {
        if (strchr(token, '.'))
        {
            char *end;
            double fval = strtod(token, &end);
            if ('\0' == *end)
            {
                fpush(fval);
                return;
            }
        }
        char *end;
        long val = strtol(token, &end, 0);
        if ('\0' == *end)
        {
            push(val);
        }
        else
        {
            int found = 0;
            for (int i = forth_dict_count - 1; i >= 0; i--)
            {
                if (0 == strcmp(forth_dict[i].name, token))
                {
                    char body_copy[512];
                    strcpy(body_copy, forth_dict[i].body);
                    char *sub_scan = body_copy;
                    char *sub_token;
                    while (NULL != (sub_token = next_token(&sub_scan)))
                    {
                        eval_forth_token(sub_token, &sub_scan, out_ptr);
                    }
                    found = 1;
                    break;
                }
            }
            if (!found)
            {
                *out_ptr += sprintf(*out_ptr, "%s ", token);
            }
        }
    }
}

void process_forth_source(char *src, char **out_ptr)
{
    char *cursor = src;
    char *token;
    int state = 0;
    char name[32];
    char body[512];
    body[0] = '\0';

    while (NULL != (token = next_token(&cursor)))
    {
        if (0 == strcmp(token, "#end"))
        {
            break;
        }

        if (0 == state)
        {
            if (0 == strcmp(token, ":"))
            {
                state = 1;
            }
            else
            {
                eval_forth_token(token, &cursor, out_ptr);
            }
        }
        else if (1 == state)
        {
            strncpy(name, token, 31);
            name[31] = '\0';
            body[0] = '\0';
            state = 2;
        }
        else if (2 == state)
        {
            if (0 == strcmp(token, ";"))
            {
                if (forth_dict_count < 128)
                {
                    strcpy(forth_dict[forth_dict_count].name, name);
                    strcpy(forth_dict[forth_dict_count].body, body);
                    forth_dict_count++;
                }
                state = 0;
            }
            else
            {
                if (strlen(body) > 0)
                {
                    strcat(body, " ");
                }
                if (strlen(body) + strlen(token) < 512)
                {
                    strcat(body, token);
                }
            }
        }
    }
}

void load_prelude()
{
    char prelude[] = ": squared fdup f* ; : hypot squared fswap squared f+ fsqrt "
                     "; : over swap dup "
                     "rot rot ; : panic 0 1 - exit ; ";
    char dummy_buf[1024];
    dummy_buf[0] = '\0';
    char *ptr = dummy_buf;
    process_forth_source(prelude, &ptr);
}

void zprep_forth_plugin_fn(const char *input_body, const ZApi *api)
{
    size_t cap = 8192;
    char *output_buffer = malloc(cap);
    if (!output_buffer)
    {
        return;
    }
    output_buffer[0] = '\0';

    char *write_ptr = output_buffer;
    char *input_copy = xstrdup(input_body);

    process_forth_source(input_copy, &write_ptr);

    free(input_copy);

    fprintf(api->out, "%s", output_buffer);

    free(output_buffer);
}

ZPlugin forth_plugin = {.name = "forth", .fn = zprep_forth_plugin_fn};

ZPlugin *z_plugin_init()
{
    return &forth_plugin;
}
