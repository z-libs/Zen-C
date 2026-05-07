
#include "emitter.h"

void emitter_init(Emitter *e, FILE *out)
{
    if (e)
    {
        e->out = out;
    }
}

void emitter_printf(Emitter *e, const char *fmt, ...)
{
    if (!e || !e->out)
    {
        return;
    }
    va_list args;
    va_start(args, fmt);
    vfprintf(e->out, fmt, args);
    va_end(args);
}

void emitter_puts(Emitter *e, const char *s)
{
    if (!e || !e->out)
    {
        return;
    }
    fputs(s, e->out);
}

void emitter_write(Emitter *e, const void *ptr, size_t size)
{
    if (!e || !e->out)
    {
        return;
    }
    fwrite(ptr, 1, size, e->out);
}
