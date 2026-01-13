#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "compat.h"
#include <stdlib.h>
#include <string.h>

// TODO: Not thread-safe; concurrent dlopen/dlsym will race.
static char s_dlerror[512] = "";

zc_dl_handle zc_dlopen(const char *path, int flags)
{
    (void)flags;
    HMODULE handle = LoadLibraryA(path);
    if (!handle)
    {
        DWORD err = GetLastError();
        FormatMessageA(
            FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            s_dlerror, sizeof(s_dlerror), NULL);
    }
    return (zc_dl_handle)handle;
}

void *zc_dlsym(zc_dl_handle handle, const char *name)
{
    void *sym = (void *)GetProcAddress((HMODULE)handle, name);
    if (!sym)
    {
        DWORD err = GetLastError();
        FormatMessageA(
            FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            s_dlerror, sizeof(s_dlerror), NULL);
    }
    return sym;
}

int zc_dlclose(zc_dl_handle handle)
{
    return FreeLibrary((HMODULE)handle) ? 0 : -1;
}

const char *zc_dlerror(void)
{
    return s_dlerror;
}

ssize_t zc_getline(char **lineptr, size_t *n, FILE *stream)
{
    if (!lineptr || !n || !stream)
        return -1;

    if (*lineptr == NULL || *n == 0)
    {
        *n = 128;
        *lineptr = malloc(*n);
        if (!*lineptr)
            return -1;
    }

    size_t pos = 0;
    int c;
    while ((c = fgetc(stream)) != EOF)
    {
        if (pos + 1 >= *n)
        {
            size_t new_size = *n * 2;
            char *new_ptr = realloc(*lineptr, new_size);
            if (!new_ptr)
                return -1;
            *lineptr = new_ptr;
            *n = new_size;
        }
        (*lineptr)[pos++] = (char)c;
        if (c == '\n')
            break;
    }

    if (pos == 0 && c == EOF)
        return -1;

    (*lineptr)[pos] = '\0';
    return (ssize_t)pos;
}

void zc_seed_random(void)
{
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    srand((unsigned int)(counter.QuadPart ^ GetCurrentProcessId()));
}

static char s_temp_dir[MAX_PATH] = "";

const char *zc_get_temp_dir(void)
{
    if (s_temp_dir[0] == '\0')
    {
        DWORD len = GetTempPathA(MAX_PATH, s_temp_dir);
        if (len == 0 || len >= MAX_PATH)
        {
            s_temp_dir[0] = '.';
            s_temp_dir[1] = '\\';
            s_temp_dir[2] = '\0';
        }
    }
    return s_temp_dir;
}

char *extract_main_body(const char *c_file_path)
{
    FILE *f = fopen(c_file_path, "r");
    if (!f)
        return NULL;

    char *result = NULL;
    size_t result_size = 0;
    size_t result_cap = 4096;
    result = malloc(result_cap);
    if (!result)
    {
        fclose(f);
        return NULL;
    }
    result[0] = '\0';

    char *line = NULL;
    size_t line_cap = 0;
    int in_main = 0;
    int brace_depth = 0;
    int skip_lines = 2;

    while (zc_getline(&line, &line_cap, f) != -1)
    {
        if (!in_main)
        {
            if (strstr(line, "int main(") != NULL && strstr(line, "{") != NULL)
            {
                in_main = 1;
                brace_depth = 1;
                skip_lines = 2;
            }
            continue;
        }

        for (char *p = line; *p; p++)
        {
            if (*p == '{')
                brace_depth++;
            else if (*p == '}')
                brace_depth--;
        }

        if (brace_depth == 0)
            break;

        if (skip_lines > 0)
        {
            skip_lines--;
            continue;
        }

        const char *content = line;
        if (strncmp(content, "    ", 4) == 0)
            content += 4;

        size_t len = strlen(content);
        if (result_size + len + 1 > result_cap)
        {
            result_cap *= 2;
            char *new_result = realloc(result, result_cap);
            if (!new_result)
            {
                free(result);
                free(line);
                fclose(f);
                return NULL;
            }
            result = new_result;
        }
        memcpy(result + result_size, content, len);
        result_size += len;
        result[result_size] = '\0';
    }

    free(line);
    fclose(f);
    return result;
}

#endif
