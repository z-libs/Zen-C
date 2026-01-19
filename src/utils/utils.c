
#include "parser.h"
#include "zprep.h"
#ifdef _WIN32
#include <io.h>
#include <malloc.h>
#define popen _popen
#define pclose _pclose
#else
#include <alloca.h>
#endif

char *g_current_filename = "unknown";
ParserContext *g_parser_ctx = NULL;

// ** Arena Implementation **
#define ARENA_BLOCK_SIZE (1024 * 1024)

typedef struct ArenaBlock
{
    struct ArenaBlock *next;
    size_t used;
    size_t cap;
    char data[];
} ArenaBlock;

static ArenaBlock *current_block = NULL;

static void *arena_alloc_raw(size_t size)
{
    size_t actual_size = size + sizeof(size_t);
    actual_size = (actual_size + 7) & ~7;

    if (!current_block || (current_block->used + actual_size > current_block->cap))
    {
        size_t block_size = actual_size > ARENA_BLOCK_SIZE ? actual_size : ARENA_BLOCK_SIZE;
#undef malloc
        ArenaBlock *new_block = malloc(sizeof(ArenaBlock) + block_size);
        if (!new_block)
        {
            fprintf(stderr, "Fatal: Out of memory\n");
            exit(1);
        }

        new_block->cap = block_size;
        new_block->used = 0;
        new_block->next = current_block;
        current_block = new_block;
    }

    void *ptr = current_block->data + current_block->used;
    current_block->used += actual_size;
    *(size_t *)ptr = size;
    return (char *)ptr + sizeof(size_t);
}

void *xmalloc(size_t size)
{
    return arena_alloc_raw(size);
}

void *xcalloc(size_t n, size_t size)
{
    size_t total = n * size;
    void *p = arena_alloc_raw(total);
    memset(p, 0, total);
    return p;
}

void *xrealloc(void *ptr, size_t new_size)
{
    if (!ptr)
    {
        return xmalloc(new_size);
    }
    size_t *header = (size_t *)((char *)ptr - sizeof(size_t));
    size_t old_size = *header;
    if (new_size <= old_size)
    {
        return ptr;
    }
    void *new_ptr = xmalloc(new_size);
    memcpy(new_ptr, ptr, old_size);
    return new_ptr;
}

char *xstrdup(const char *s)
{
    if (!s)
    {
        return NULL;
    }
    size_t len = strlen(s);
    char *d = xmalloc(len + 1);
    memcpy(d, s, len);
    d[len] = 0;
    return d;
}

void zpanic(const char *fmt, ...)
{
    va_list a;
    va_start(a, fmt);
    fprintf(stderr, COLOR_RED "error: " COLOR_RESET COLOR_BOLD);
    vfprintf(stderr, fmt, a);
    fprintf(stderr, COLOR_RESET "\n");
    va_end(a);
    exit(1);
}

// Warning system (non-fatal).
void zwarn(const char *fmt, ...)
{
    if (g_config.quiet)
    {
        return;
    }
    g_warning_count++;
    va_list a;
    va_start(a, fmt);
    fprintf(stderr, COLOR_YELLOW "warning: " COLOR_RESET COLOR_BOLD);
    vfprintf(stderr, fmt, a);
    fprintf(stderr, COLOR_RESET "\n");
    va_end(a);
}

void zwarn_at(Token t, const char *fmt, ...)
{
    if (g_config.quiet)
    {
        return;
    }
    // Header: 'warning: message'.
    g_warning_count++;
    va_list a;
    va_start(a, fmt);
    fprintf(stderr, COLOR_YELLOW "warning: " COLOR_RESET COLOR_BOLD);
    vfprintf(stderr, fmt, a);
    fprintf(stderr, COLOR_RESET "\n");
    va_end(a);

    // Location.
    fprintf(stderr, COLOR_BLUE "  --> " COLOR_RESET "%s:%d:%d\n", g_current_filename, t.line,
            t.col);

    // Context. Only if token has valid data.
    if (t.start)
    {
        const char *line_start = t.start - (t.col - 1);
        const char *line_end = t.start;
        while (*line_end && *line_end != '\n')
        {
            line_end++;
        }
        int line_len = line_end - line_start;

        fprintf(stderr, COLOR_BLUE "   |\n" COLOR_RESET);
        fprintf(stderr, COLOR_BLUE "%-3d| " COLOR_RESET "%.*s\n", t.line, line_len, line_start);
        fprintf(stderr, COLOR_BLUE "   | " COLOR_RESET);

        // Caret.
        for (int i = 0; i < t.col - 1; i++)
        {
            fprintf(stderr, " ");
        }
        fprintf(stderr, COLOR_YELLOW "^ here" COLOR_RESET "\n");
        fprintf(stderr, COLOR_BLUE "   |\n" COLOR_RESET);
    }
}

void zpanic_at(Token t, const char *fmt, ...)
{
    // Header: 'error: message'.
    va_list a;
    va_start(a, fmt);
    fprintf(stderr, COLOR_RED "error: " COLOR_RESET COLOR_BOLD);
    vfprintf(stderr, fmt, a);
    fprintf(stderr, COLOR_RESET "\n");
    va_end(a);

    // Location: '--> file:line:col'.
    fprintf(stderr, COLOR_BLUE "  --> " COLOR_RESET "%s:%d:%d\n", g_current_filename, t.line,
            t.col);

    // Context line.
    const char *line_start = t.start - (t.col - 1);
    const char *line_end = t.start;
    while (*line_end && *line_end != '\n')
    {
        line_end++;
    }
    int line_len = line_end - line_start;

    // Visual bar.
    fprintf(stderr, COLOR_BLUE "   |\n" COLOR_RESET);
    fprintf(stderr, COLOR_BLUE "%-3d| " COLOR_RESET "%.*s\n", t.line, line_len, line_start);
    fprintf(stderr, COLOR_BLUE "   | " COLOR_RESET);

    // caret
    for (int i = 0; i < t.col - 1; i++)
    {
        fprintf(stderr, " ");
    }
    fprintf(stderr, COLOR_RED "^ here" COLOR_RESET "\n");
    fprintf(stderr, COLOR_BLUE "   |\n" COLOR_RESET);

    if (g_parser_ctx && g_parser_ctx->is_fault_tolerant && g_parser_ctx->on_error)
    {
        // Construct error message buffer
        char msg[1024];
        va_list args2;
        va_start(args2, fmt);
        vsnprintf(msg, sizeof(msg), fmt, args2);
        va_end(args2);

        g_parser_ctx->on_error(g_parser_ctx->error_callback_data, t, msg);
        return; // Recover!
    }

    exit(1);
}

// Enhanced error with suggestion.
void zpanic_with_suggestion(Token t, const char *msg, const char *suggestion)
{
    // Header.
    fprintf(stderr, COLOR_RED "error: " COLOR_RESET COLOR_BOLD "%s" COLOR_RESET "\n", msg);

    // Location.
    fprintf(stderr, COLOR_BLUE "  --> " COLOR_RESET "%s:%d:%d\n", g_current_filename, t.line,
            t.col);

    // Context.
    const char *line_start = t.start - (t.col - 1);
    const char *line_end = t.start;
    while (*line_end && *line_end != '\n')
    {
        line_end++;
    }
    int line_len = line_end - line_start;

    fprintf(stderr, COLOR_BLUE "   |\n" COLOR_RESET);
    fprintf(stderr, COLOR_BLUE "%-3d| " COLOR_RESET "%.*s\n", t.line, line_len, line_start);
    fprintf(stderr, COLOR_BLUE "   | " COLOR_RESET);
    for (int i = 0; i < t.col - 1; i++)
    {
        fprintf(stderr, " ");
    }
    fprintf(stderr, COLOR_RED "^ here" COLOR_RESET "\n");

    // Suggestion.
    if (suggestion)
    {
        fprintf(stderr, COLOR_BLUE "   |\n" COLOR_RESET);
        fprintf(stderr, COLOR_CYAN "   = help: " COLOR_RESET "%s\n", suggestion);
    }

    exit(1);
}

// Specific error types with helpful messages.
void error_undefined_function(Token t, const char *func_name, const char *suggestion)
{
    char msg[256];
    sprintf(msg, "Undefined function '%s'", func_name);

    if (suggestion)
    {
        char help[512];
        sprintf(help, "Did you mean '%s'?", suggestion);
        zpanic_with_suggestion(t, msg, help);
    }
    else
    {
        zpanic_with_suggestion(t, msg, "Check if the function is defined or imported");
    }
}

void error_wrong_arg_count(Token t, const char *func_name, int expected, int got)
{
    char msg[256];
    sprintf(msg, "Wrong number of arguments to function '%s'", func_name);

    char help[256];
    sprintf(help, "Expected %d argument%s, but got %d", expected, expected == 1 ? "" : "s", got);

    zpanic_with_suggestion(t, msg, help);
}

void error_undefined_field(Token t, const char *struct_name, const char *field_name,
                           const char *suggestion)
{
    char msg[256];
    sprintf(msg, "Struct '%s' has no field '%s'", struct_name, field_name);

    if (suggestion)
    {
        char help[256];
        sprintf(help, "Did you mean '%s'?", suggestion);
        zpanic_with_suggestion(t, msg, help);
    }
    else
    {
        zpanic_with_suggestion(t, msg, "Check the struct definition");
    }
}

void error_type_expected(Token t, const char *expected, const char *got)
{
    char msg[256];
    sprintf(msg, "Type mismatch");

    char help[512];
    sprintf(help, "Expected type '%s', but found '%s'", expected, got);

    zpanic_with_suggestion(t, msg, help);
}

void error_cannot_index(Token t, const char *type_name)
{
    char msg[256];
    sprintf(msg, "Cannot index into type '%s'", type_name);

    zpanic_with_suggestion(t, msg, "Only arrays and slices can be indexed");
}

void warn_unused_variable(Token t, const char *var_name)
{
    if (g_config.quiet)
    {
        return;
    }
    char msg[256];
    sprintf(msg, "Unused variable '%s'", var_name);
    zwarn_at(t, "%s", msg);
    fprintf(stderr,
            COLOR_CYAN "   = note: " COLOR_RESET "Consider removing it or prefixing with '_'\n");
}

void warn_shadowing(Token t, const char *var_name)
{
    if (g_config.quiet)
    {
        return;
    }
    char msg[256];
    sprintf(msg, "Variable '%s' shadows a previous declaration", var_name);
    zwarn_at(t, "%s", msg);
    fprintf(stderr, COLOR_CYAN "   = note: " COLOR_RESET "This can lead to confusion\n");
}

void warn_unreachable_code(Token t)
{
    if (g_config.quiet)
    {
        return;
    }
    zwarn_at(t, "Unreachable code detected");
    fprintf(stderr, COLOR_CYAN "   = note: " COLOR_RESET "This code will never execute\n");
}

void warn_implicit_conversion(Token t, const char *from_type, const char *to_type)
{
    if (g_config.quiet)
    {
        return;
    }
    char msg[256];
    sprintf(msg, "Implicit conversion from '%s' to '%s'", from_type, to_type);
    zwarn_at(t, "%s", msg);
    fprintf(stderr, COLOR_CYAN "   = note: " COLOR_RESET "Consider using an explicit cast\n");
}

void warn_missing_return(Token t, const char *func_name)
{
    if (g_config.quiet)
    {
        return;
    }
    char msg[256];
    sprintf(msg, "Function '%s' may not return a value in all paths", func_name);
    zwarn_at(t, "%s", msg);
    fprintf(stderr, COLOR_CYAN "   = note: " COLOR_RESET
                               "Add a return statement or make the function return 'void'\n");
}

void warn_comparison_always_true(Token t, const char *reason)
{
    if (g_config.quiet)
    {
        return;
    }
    zwarn_at(t, "Comparison is always true");
    if (reason)
    {
        fprintf(stderr, COLOR_CYAN "   = note: " COLOR_RESET "%s\n", reason);
    }
}

void warn_comparison_always_false(Token t, const char *reason)
{
    if (g_config.quiet)
    {
        return;
    }
    zwarn_at(t, "Comparison is always false");
    if (reason)
    {
        fprintf(stderr, COLOR_CYAN "   = note: " COLOR_RESET "%s\n", reason);
    }
}

void warn_unused_parameter(Token t, const char *param_name, const char *func_name)
{
    if (g_config.quiet)
    {
        return;
    }
    char msg[256];
    sprintf(msg, "Unused parameter '%s' in function '%s'", param_name, func_name);
    zwarn_at(t, "%s", msg);
    fprintf(stderr, COLOR_CYAN "   = note: " COLOR_RESET
                               "Consider prefixing with '_' if intentionally unused\n");
}

void warn_narrowing_conversion(Token t, const char *from_type, const char *to_type)
{
    if (g_config.quiet)
    {
        return;
    }
    char msg[256];
    sprintf(msg, "Narrowing conversion from '%s' to '%s'", from_type, to_type);
    zwarn_at(t, "%s", msg);
    fprintf(stderr, COLOR_CYAN "   = note: " COLOR_RESET "This may cause data loss\n");
}

void warn_division_by_zero(Token t)
{
    if (g_config.quiet)
    {
        return;
    }
    zwarn_at(t, "Division by zero");
    fprintf(stderr,
            COLOR_CYAN "   = note: " COLOR_RESET "This will cause undefined behavior at runtime\n");
}

void warn_integer_overflow(Token t, const char *type_name, long long value)
{
    if (g_config.quiet)
    {
        return;
    }
    char msg[256];
    sprintf(msg, "Integer literal %lld overflows type '%s'", value, type_name);
    zwarn_at(t, "%s", msg);
    fprintf(stderr, COLOR_CYAN "   = note: " COLOR_RESET "Value will be truncated\n");
}

void warn_array_bounds(Token t, int index, int size)
{
    if (g_config.quiet)
    {
        return;
    }
    char msg[256];
    sprintf(msg, "Array index %d is out of bounds for array of size %d", index, size);
    zwarn_at(t, "%s", msg);
    fprintf(stderr, COLOR_CYAN "   = note: " COLOR_RESET "Valid indices are 0 to %d\n", size - 1);
}

void warn_format_string(Token t, int arg_num, const char *expected, const char *got)
{
    if (g_config.quiet)
    {
        return;
    }
    char msg[256];
    sprintf(msg, "Format argument %d: expected '%s', got '%s'", arg_num, expected, got);
    zwarn_at(t, "%s", msg);
    fprintf(stderr, COLOR_CYAN "   = note: " COLOR_RESET
                               "Mismatched format specifier may cause undefined behavior\n");
}

void warn_null_pointer(Token t, const char *expr)
{
    if (g_config.quiet)
    {
        return;
    }
    char msg[256];
    sprintf(msg, "Potential null pointer access in '%s'", expr);
    zwarn_at(t, "%s", msg);
    fprintf(stderr, COLOR_CYAN "   = note: " COLOR_RESET "Add a null check before accessing\n");
}

char *load_file(const char *fn)
{
    FILE *f = fopen(fn, "rb");
    if (!f)
    {
        char *root = getenv("ZC_ROOT");
        if (root)
        {
            char path[1024];
            snprintf(path, sizeof(path), "%s/%s", root, fn);
            f = fopen(path, "rb");
        }
    }
    if (!f)
    {
        char path[1024];
        snprintf(path, sizeof(path), "/usr/local/share/zenc/%s", fn);
        f = fopen(path, "rb");
    }
    if (!f)
    {
        char path[1024];
        snprintf(path, sizeof(path), "/usr/share/zenc/%s", fn);
        f = fopen(path, "rb");
    }

    if (!f)
    {
        return 0;
    }
    fseek(f, 0, SEEK_END);
    long l = ftell(f);
    rewind(f);
    char *b = xmalloc(l + 1);
    fread(b, 1, l, f);
    b[l] = 0;
    fclose(f);
    return b;
}

// ** Build Directives **
char g_link_flags[MAX_FLAGS_SIZE] = "";
char g_cflags[MAX_FLAGS_SIZE] = "";
int g_warning_count = 0;
CompilerConfig g_config = {0};

// Helper for environment expansion
static void expand_env_vars(char *dest, size_t dest_size, const char *src)
{
    char *d = dest;
    const char *s = src;
    size_t remaining = dest_size - 1;

    while (*s && remaining > 0)
    {
        if (*s == '$' && *(s + 1) == '{')
        {
            const char *end = strchr(s + 2, '}');
            if (end)
            {
                char var_name[256];
                int len = end - (s + 2);
                if (len < 255)
                {
                    strncpy(var_name, s + 2, len);
                    var_name[len] = 0;
                    char *val = getenv(var_name);
                    if (val)
                    {
                        size_t val_len = strlen(val);
                        if (val_len < remaining)
                        {
                            strcpy(d, val);
                            d += val_len;
                            remaining -= val_len;
                            s = end + 1;
                            continue;
                        }
                    }
                }
            }
        }
        *d++ = *s++;
        remaining--;
    }
    *d = 0;
}

// Helper to determine active OS
static int is_os_active(const char *os_name)
{
    if (0 == strcmp(os_name, "linux"))
    {
#ifdef __linux__
        return 1;
#else
        return 0;
#endif
    }
    else if (0 == strcmp(os_name, "windows"))
    {
#ifdef _WIN32
        return 1;
#else
        return 0;
#endif
    }
    else if (0 == strcmp(os_name, "macos") || 0 == strcmp(os_name, "darwin"))
    {
#ifdef __APPLE__
        return 1;
#else
        return 0;
#endif
    }
    return 0;
}

void scan_build_directives(ParserContext *ctx, const char *src)
{
    (void)ctx;
    const char *p = src;
    while (*p)
    {
        if (p[0] == '/' && p[1] == '/' && p[2] == '>')
        {
            p += 3;
            while (*p == ' ')
            {
                p++;
            }

            const char *start = p;
            int len = 0;
            while (p[len] && p[len] != '\n')
            {
                len++;
            }

            char raw_line[2048];
            if (len >= 2047)
            {
                len = 2047;
            }
            strncpy(raw_line, start, len);
            raw_line[len] = 0;

            char line[2048];
            expand_env_vars(line, sizeof(line), raw_line);

            char *directive = line;
            char *colon = strchr(line, ':');
            if (colon)
            {
                *colon = 0;
                if (is_os_active(line))
                {
                    directive = colon + 1;
                    while (*directive == ' ')
                    {
                        directive++;
                    }
                }
                else if (0 == strcmp(line, "linux") || 0 == strcmp(line, "windows") ||
                         0 == strcmp(line, "macos"))
                {
                    goto next_line;
                }
                else
                {
                    *colon = ':';
                    directive = line;
                }
            }

            // Process Directive
            if (0 == strncmp(directive, "link:", 5))
            {
                if (strlen(g_link_flags) > 0)
                {
                    strcat(g_link_flags, " ");
                }
                strcat(g_link_flags, directive + 5);
            }
            else if (0 == strncmp(directive, "cflags:", 7))
            {
                if (strlen(g_cflags) > 0)
                {
                    strcat(g_cflags, " ");
                }
                strcat(g_cflags, directive + 7);
            }
            else if (0 == strncmp(directive, "include:", 8))
            {
                char flags[2048];
                sprintf(flags, "-I%s", directive + 8);
                if (strlen(g_cflags) > 0)
                {
                    strcat(g_cflags, " ");
                }
                strcat(g_cflags, flags);
            }
            else if (strncmp(directive, "lib:", 4) == 0)
            {
                char flags[2048];
                sprintf(flags, "-L%s", directive + 4);
                if (strlen(g_link_flags) > 0)
                {
                    strcat(g_link_flags, " ");
                }
                strcat(g_link_flags, flags);
            }
            else if (strncmp(directive, "framework:", 10) == 0)
            {
                char flags[2048];
                sprintf(flags, "-framework %s", directive + 10);
                if (strlen(g_link_flags) > 0)
                {
                    strcat(g_link_flags, " ");
                }
                strcat(g_link_flags, flags);
            }
            else if (strncmp(directive, "define:", 7) == 0)
            {
                char flags[2048];
                sprintf(flags, "-D%s", directive + 7);
                if (strlen(g_cflags) > 0)
                {
                    strcat(g_cflags, " ");
                }
                strcat(g_cflags, flags);
            }
            else if (0 == strncmp(directive, "shell:", 6))
            {
                if (system(directive + 6) != 0)
                {
                    zwarn("Shell directive failed: %s", directive + 6);
                }
            }
            else if (strncmp(directive, "get:", 4) == 0)
            {
                char *url = directive + 4;
                while (*url == ' ')
                {
                    url++;
                }
                char *filename = strrchr(url, '/');
                if (!filename)
                {
                    filename = "downloaded_file";
                }
                else
                {
                    filename++;
                }
                FILE *f = fopen(filename, "r");
                if (f)
                {
                    fclose(f);
                }
                else
                {
                    char cmd[8192];
                    if (z_is_windows())
                    {
                        sprintf(cmd, "curl -s -L \"%s\" -o \"%s\"", url, filename);
                    }
                    else
                    {
                        sprintf(cmd, "wget -q \"%s\" -O \"%s\" || curl -s -L \"%s\" -o \"%s\"", url,
                                filename, url, filename);
                    }
                    if (system(cmd) != 0)
                    {
                        zwarn("Failed to download %s", url);
                    }
                }
            }
            else if (strncmp(directive, "pkg-config:", 11) == 0)
            {
                char *libs = directive + 11;
                char cmd[4096];
                sprintf(cmd, "pkg-config --cflags %s", libs);
                FILE *fp = popen(cmd, "r");
                if (fp)
                {
                    char buf[1024];
                    if (fgets(buf, sizeof(buf), fp))
                    {
                        size_t l = strlen(buf);
                        if (l > 0 && buf[l - 1] == '\n')
                        {
                            buf[l - 1] = 0;
                        }
                        if (strlen(g_cflags) > 0)
                        {
                            strcat(g_cflags, " ");
                        }
                        strcat(g_cflags, buf);
                    }
                    pclose(fp);
                }

                sprintf(cmd, "pkg-config --libs %s", libs);
                fp = popen(cmd, "r");
                if (fp)
                {
                    char buf[1024];
                    if (fgets(buf, sizeof(buf), fp))
                    {
                        size_t l = strlen(buf);
                        if (l > 0 && buf[l - 1] == '\n')
                        {
                            buf[l - 1] = 0;
                        }
                        if (strlen(g_link_flags) > 0)
                        {
                            strcat(g_link_flags, " ");
                        }
                        strcat(g_link_flags, buf);
                    }
                    pclose(fp);
                }
            }
            else
            {
                zwarn("Unknown build directive: '%s'", directive);
            }

            p += len;
        }
    next_line:
        while (*p && *p != '\n')
        {
            p++;
        }
        if (*p == '\n')
        {
            p++;
        }
    }
}

// Levenshtein distance for "did you mean?" suggestions.
int levenshtein(const char *s1, const char *s2)
{
    size_t len1 = strlen(s1);
    size_t len2 = strlen(s2);

    // Quick optimization.
    if (len1 > len2 ? (len1 - len2 > 3) : (len2 - len1 > 3))
    {
        return 999;
    }

    size_t rows = len1 + 1;
    size_t cols = len2 + 1;
    int *matrix = alloca(rows * cols * sizeof(int));

    for (size_t i = 0; i <= len1; i++)
    {
        matrix[i * cols + 0] = (int)i;
    }
    for (size_t j = 0; j <= len2; j++)
    {
        matrix[0 * cols + j] = (int)j;
    }

    for (size_t i = 1; i <= len1; i++)
    {
        for (size_t j = 1; j <= len2; j++)
        {
            int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
            int del = matrix[(i - 1) * cols + j] + 1;
            int ins = matrix[i * cols + (j - 1)] + 1;
            int sub = matrix[(i - 1) * cols + (j - 1)] + cost;

            int val = (del < ins) ? del : ins;
            if (sub < val)
            {
                val = sub;
            }
            matrix[i * cols + j] = val;
        }
    }

    int result = matrix[len1 * cols + len2];
    return result;
}
