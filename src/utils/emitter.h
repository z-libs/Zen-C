// SPDX-License-Identifier: MIT

#ifndef ZC_ALLOW_INTERNAL
#error "utils/emitter.h is internal to Zen C. Include the appropriate public header instead."
#endif

#ifndef EMITTER_H
#define EMITTER_H

#include <stdio.h>
#include <stdarg.h>

#define EMITTER_INDENT_SIZE 4
#define EMITTER_SAVED_STACK_MAX 8
#define EMITTER_INITIAL_BUF_CAP 256

typedef enum
{
    EMITTER_FILE,
    EMITTER_BUFFER
} EmitterMode;

typedef struct EmitterSavedState
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
    int indent_level;
    size_t output_line;
} EmitterSavedState;

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
    int indent_level;
    int pending_indent;
    size_t output_line;
    EmitterSavedState saved_stack[EMITTER_SAVED_STACK_MAX];
    int saved_count;
} Emitter;

void emitter_init_file(Emitter *e, FILE *file);
void emitter_init_buffer(Emitter *e);
void emitter_printf(Emitter *e, const char *fmt, ...);
void emitter_vprintf(Emitter *e, const char *fmt, va_list args);
void emitter_puts(Emitter *e, const char *s);
void emitter_putc(Emitter *e, char c);
void emitter_write(Emitter *e, const void *ptr, size_t size);
char *emitter_take_string(Emitter *e);
void emitter_indent(Emitter *e);
void emitter_dedent(Emitter *e);
int emitter_push(Emitter *e);
int emitter_pop(Emitter *e);
void emitter_release(Emitter *e);

#endif
