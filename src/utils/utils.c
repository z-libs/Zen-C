
#include "parser.h"
#include "zprep.h"

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
            zfatal("Out of memory");
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

#include <time.h>
#include "../platform/os.h"

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

static void append_flag(char *dest, size_t max_size, const char *flag)
{
    size_t current_len = strlen(dest);
    size_t flag_len = strlen(flag);
    if (current_len > 0)
    {
        if (current_len + flag_len + 2 >= max_size)
        {
            zwarn("Build flags buffer overflow prevented.");
            return;
        }
        strcat(dest, " ");
        strcat(dest, flag);
    }
    else
    {
        if (flag_len + 1 >= max_size)
        {
            zwarn("Build flags buffer overflow prevented.");
            return;
        }
        strcat(dest, flag);
    }
}

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

            // Strip trailing \r (Windows CRLF)
            int rlen = strlen(raw_line);
            if (rlen > 0 && raw_line[rlen - 1] == '\r')
            {
                raw_line[rlen - 1] = 0;
            }

            char line[2048];
            expand_env_vars(line, sizeof(line), raw_line);

            char *directive = line;
            char *colon = strchr(line, ':');
            if (colon)
            {
                *colon = 0; // split the string temporarily
                if (0 == strcmp(line, "linux") || 0 == strcmp(line, "windows") ||
                    0 == strcmp(line, "macos") || 0 == strcmp(line, "darwin"))
                {
                    if (is_os_active(line))
                    {
                        directive = colon + 1;
                        while (*directive == ' ')
                        {
                            directive++;
                        }
                    }
                    else
                    {
                        // OS specified but not active, skip this directive completely
                        goto next_line;
                    }
                }
                else
                {
                    // Not an OS prefix, restore the colon
                    *colon = ':';
                    directive = line;
                }
            }

            char *directive_val = NULL;
            // Process Directive
            if (0 == strncmp(directive, "link:", 5))
            {
                directive_val = directive + 5;
                while (*directive_val == ' ')
                {
                    directive_val++;
                }
                append_flag(g_link_flags, sizeof(g_link_flags), directive_val);
            }
            else if (0 == strncmp(directive, "cflags:", 7))
            {
                directive_val = directive + 7;
                while (*directive_val == ' ')
                {
                    directive_val++;
                }
                append_flag(g_cflags, sizeof(g_cflags), directive_val);
            }
            else if (0 == strncmp(directive, "include:", 8))
            {
                directive_val = directive + 8;
                while (*directive_val == ' ')
                {
                    directive_val++;
                }
                char flags[2048];
                snprintf(flags, sizeof(flags), "-I%s", directive_val);
                append_flag(g_cflags, sizeof(g_cflags), flags);
            }
            else if (strncmp(directive, "lib:", 4) == 0)
            {
                directive_val = directive + 4;
                while (*directive_val == ' ')
                {
                    directive_val++;
                }
                char flags[2048];
                snprintf(flags, sizeof(flags), "-L%s", directive_val);
                append_flag(g_link_flags, sizeof(g_link_flags), flags);
            }
            else if (strncmp(directive, "framework:", 10) == 0)
            {
                directive_val = directive + 10;
                while (*directive_val == ' ')
                {
                    directive_val++;
                }
                char flags[2048];
                snprintf(flags, sizeof(flags), "-framework %s", directive_val);
                append_flag(g_link_flags, sizeof(g_link_flags), flags);
            }
            else if (strncmp(directive, "define:", 7) == 0)
            {
                directive_val = directive + 7;
                while (*directive_val == ' ')
                {
                    directive_val++;
                }
                char flags[2048];
                snprintf(flags, sizeof(flags), "-D%s", directive_val);
                append_flag(g_cflags, sizeof(g_cflags), flags);

                if (g_config.cfg_define_count < 64)
                {
                    char *name = xstrdup(directive_val);
                    char *eq = strchr(name, '=');
                    if (eq)
                    {
                        *eq = '\0';
                    }
                    g_config.cfg_defines[g_config.cfg_define_count++] = name;
                }
            }
            else if (0 == strncmp(directive, "shell:", 6))
            {
                directive_val = directive + 6;
                while (*directive_val == ' ')
                {
                    directive_val++;
                }
                zwarn("Security Alert: Execution of 'shell:' directive (%s) was BLOCKED by default "
                      "to prevent Remote Code Execution.",
                      directive_val);
                // Intentionally ignored system() call for security reasons
            }
            else if (strncmp(directive, "get:", 4) == 0)
            {
                char *url = directive + 4;
                while (*url == ' ')
                {
                    url++;
                }
                zwarn("Security Alert: Execution of 'get:' directive (%s) was BLOCKED. Please "
                      "download external dependencies manually.",
                      url);
                // Intentionally ignored external network hit for security reasons
            }
            else if (strncmp(directive, "pkg-config:", 11) == 0)
            {
                char *libs = directive + 11;

                // Security check for malicious pkg-config commands containing bash injections
                int is_safe = 1;
                for (int i = 0; libs[i]; i++)
                {
                    if (!isalnum(libs[i]) && libs[i] != '-' && libs[i] != '_' && libs[i] != ' ' &&
                        libs[i] != '.')
                    {
                        is_safe = 0;
                        break;
                    }
                }

                if (!is_safe)
                {
                    zwarn("Security Alert: Execution of 'pkg-config:' directive with invalid chars "
                          "(%s) was BLOCKED.",
                          libs);
                }
                else
                {
                    char cmd[4096];
                    snprintf(cmd, sizeof(cmd), "pkg-config --cflags %s", libs);
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
                            append_flag(g_cflags, sizeof(g_cflags), buf);
                        }
                        pclose(fp);
                    }

                    snprintf(cmd, sizeof(cmd), "pkg-config --libs %s", libs);
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
                            append_flag(g_link_flags, sizeof(g_link_flags), buf);
                        }
                        pclose(fp);
                    }
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
    int len1 = strlen(s1);
    int len2 = strlen(s2);

    // Quick optimization.
    if (abs(len1 - len2) > 3)
    {
        return 999;
    }

    int matrix[len1 + 1][len2 + 1];

    for (int i = 0; i <= len1; i++)
    {
        matrix[i][0] = i;
    }
    for (int j = 0; j <= len2; j++)
    {
        matrix[0][j] = j;
    }

    for (int i = 1; i <= len1; i++)
    {
        for (int j = 1; j <= len2; j++)
        {
            int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
            int del = matrix[i - 1][j] + 1;
            int ins = matrix[i][j - 1] + 1;
            int sub = matrix[i - 1][j - 1] + cost;

            matrix[i][j] = (del < ins) ? del : ins;
            if (sub < matrix[i][j])
            {
                matrix[i][j] = sub;
            }
        }
    }

    return matrix[len1][len2];
}
