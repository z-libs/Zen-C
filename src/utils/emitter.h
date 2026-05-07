
#ifndef EMITTER_H
#define EMITTER_H

#include <stdio.h>
#include <stdarg.h>

/**
 * @brief Emitter structure for structured code generation.
 *
 * This allows us to redirect output easily (e.g. for f-strings or comptime).
 */
typedef struct Emitter
{
    FILE *out;
} Emitter;

/**
 * @brief Initialize an emitter with a file stream.
 */
void emitter_init(Emitter *e, FILE *out);

/**
 * @brief Emit formatted text to the emitter.
 */
void emitter_printf(Emitter *e, const char *fmt, ...);

void emitter_puts(Emitter *e, const char *s);

/**
 * @brief Write raw bytes to the emitter.
 */
void emitter_write(Emitter *e, const void *ptr, size_t size);

#endif
