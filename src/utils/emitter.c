
#include "emitter.h"
#include <stdlib.h>
#include <string.h>

void emitter_init_file(Emitter *e, FILE *file)
{
    if (e)
    {
        e->mode = EMITTER_FILE;
        e->file = file;
    }
}

void emitter_init_buffer(Emitter *e)
{
    if (e)
    {
        e->mode = EMITTER_BUFFER;
        e->buffer.buf = NULL;
        e->buffer.len = 0;
        e->buffer.cap = 0;
    }
}

void emitter_printf(Emitter *e, const char *fmt, ...)
{
    if (!e)
    {
        return;
    }
    va_list args;
    va_start(args, fmt);
    if (e->mode == EMITTER_FILE)
    {
        if (e->file)
        {
            vfprintf(e->file, fmt, args);
        }
    }
    else
    {
        va_list args_copy;
        va_copy(args_copy, args);
        int len = vsnprintf(NULL, 0, fmt, args_copy);
        va_end(args_copy);
        if (len < 0)
        {
            va_end(args);
            return;
        }
        size_t needed = e->buffer.len + (size_t)len + 1;
        if (needed > e->buffer.cap)
        {
            e->buffer.cap = needed + (needed >> 1);
            e->buffer.buf = (realloc)(e->buffer.buf, e->buffer.cap);
        }
        vsnprintf(e->buffer.buf + e->buffer.len, (size_t)(len + 1), fmt, args);
        e->buffer.len += (size_t)len;
    }
    va_end(args);
}

void emitter_puts(Emitter *e, const char *s)
{
    if (!e || !s)
    {
        return;
    }
    if (e->mode == EMITTER_FILE)
    {
        if (e->file)
        {
            fputs(s, e->file);
        }
    }
    else
    {
        size_t slen = strlen(s);
        size_t needed = e->buffer.len + slen + 1;
        if (needed > e->buffer.cap)
        {
            e->buffer.cap = needed + (needed >> 1);
            e->buffer.buf = (realloc)(e->buffer.buf, e->buffer.cap);
        }
        memcpy(e->buffer.buf + e->buffer.len, s, slen);
        e->buffer.len += slen;
        e->buffer.buf[e->buffer.len] = '\0';
    }
}

void emitter_write(Emitter *e, const void *ptr, size_t size)
{
    if (!e || !ptr)
    {
        return;
    }
    if (e->mode == EMITTER_FILE)
    {
        if (e->file)
        {
            fwrite(ptr, 1, size, e->file);
        }
    }
    else
    {
        size_t needed = e->buffer.len + size + 1;
        if (needed > e->buffer.cap)
        {
            e->buffer.cap = needed + (needed >> 1);
            e->buffer.buf = (realloc)(e->buffer.buf, e->buffer.cap);
        }
        memcpy(e->buffer.buf + e->buffer.len, ptr, size);
        e->buffer.len += size;
        e->buffer.buf[e->buffer.len] = '\0';
    }
}

char *emitter_take_string(Emitter *e)
{
    if (!e || e->mode != EMITTER_BUFFER)
    {
        return NULL;
    }
    char *result = e->buffer.buf;
    e->buffer.buf = NULL;
    e->buffer.len = 0;
    e->buffer.cap = 0;
    return result;
}

void emitter_release(Emitter *e)
{
    if (!e)
    {
        return;
    }
    if (e->mode == EMITTER_BUFFER)
    {
        e->buffer.buf = NULL;
        e->buffer.len = 0;
        e->buffer.cap = 0;
    }
}
