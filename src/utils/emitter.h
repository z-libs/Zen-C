
#ifndef EMITTER_H
#define EMITTER_H

#include <stdio.h>
#include <stdarg.h>

typedef enum
{
    EMITTER_FILE,
    EMITTER_BUFFER
} EmitterMode;

typedef struct Emitter
{
    EmitterMode mode;
    union
    {
        FILE *file;
        struct
        {
            char *buf;
            size_t len;
            size_t cap;
        } buffer;
    };
} Emitter;

void emitter_init_file(Emitter *e, FILE *file);
void emitter_init_buffer(Emitter *e);
void emitter_printf(Emitter *e, const char *fmt, ...);
void emitter_puts(Emitter *e, const char *s);
void emitter_write(Emitter *e, const void *ptr, size_t size);
char *emitter_take_string(Emitter *e);
void emitter_release(Emitter *e);

#endif
