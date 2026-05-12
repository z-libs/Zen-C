// SPDX-License-Identifier: MIT
#define ZEN_DISABLE_COLORS_WRAPPER
#include "colors.h"
#include "../zprep.h"
#include <stdlib.h>
#include <string.h>
#include "platform/arch.h"

#if ZC_OS_WINDOWS
#include <io.h>
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif

#undef printf
#undef fprintf
#undef vprintf
#undef vfprintf

static void strip_ansi_codes(char *str)
{
    char *read_ptr = str;
    char *write_ptr = str;
    while (*read_ptr)
    {
        if (*read_ptr == '\033' && *(read_ptr + 1) == '[')
        {
            read_ptr += 2;
            while (*read_ptr && *read_ptr != 'm')
            {
                read_ptr++;
            }
            if (*read_ptr == 'm')
            {
                read_ptr++;
            }
        }
        else
        {
            *write_ptr++ = *read_ptr++;
        }
    }
    *write_ptr = '\0';
}

int zvfprintf(FILE *stream, const char *format, va_list args)
{
    int should_strip = !isatty(fileno(stream));

    if (!should_strip)
    {
        return vfprintf(stream, format, args);
    }

    char stack_buf[4096];
    va_list args_copy;
    va_copy(args_copy, args);
    int len = vsnprintf(stack_buf, sizeof(stack_buf), format, args_copy);
    va_end(args_copy);

    if (len < 0)
    {
        return len;
    }

    char *work_buf = stack_buf;
    if (len >= (int)sizeof(stack_buf))
    {
        work_buf = malloc(len + 1);
        if (!work_buf)
        {
            return -1;
        }
        vsnprintf(work_buf, len + 1, format, args);
    }

    strip_ansi_codes(work_buf);
    int stripped_len = strlen(work_buf);
    int ret = fwrite(work_buf, 1, stripped_len, stream);

    if (work_buf != stack_buf)
    {
        zfree(work_buf);
    }

    return ret >= 0 ? stripped_len : -1;
}

int zfprintf(FILE *stream, const char *format, ...)
{
    va_list args;
    va_start(args, format);
    int ret = zvfprintf(stream, format, args);
    va_end(args);
    return ret;
}

int zprintf(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    int ret = zvfprintf(stdout, format, args);
    va_end(args);
    return ret;
}

int zvprintf(const char *format, va_list args)
{
    return zvfprintf(stdout, format, args);
}
