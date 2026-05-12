// SPDX-License-Identifier: MIT
#ifndef ZC_UTILS_H
#define ZC_UTILS_H

/**
 * @file zc_utils.h
 * @brief Public API for libzc-utils: memory, vectors, file I/O.
 */

#include <stddef.h>

// ============================================================================
// Memory allocation (from zalloc.h arena system)
// ============================================================================

void *xmalloc(size_t size);
void *xrealloc(void *ptr, size_t new_size);
void *xcalloc(size_t n, size_t size);
char *xstrdup(const char *s);

/**
 * @brief No-op free for arena-allocated memory.
 * Actually frees nothing — the arena is reclaimed as a whole.
 */
#define zfree(ptr) ((void)0)

/**
 * @brief Redirect standard allocation to the arena.
 */
#define malloc(sz) xmalloc(sz)
#define realloc(p, s) xrealloc(p, s)
#define calloc(n, s) xcalloc(n, s)

// ============================================================================
// Dynamic string vector
// ============================================================================

#include "../utils/zvec.h"

ZVEC_GENERATE_IMPL(char *, Str)

// ============================================================================
// Emitter (output buffer)
// ============================================================================

typedef struct Emitter Emitter;
void emitter_init_file(Emitter *e, FILE *f);
void emitter_init_buffer(Emitter *e);
void emitter_printf(Emitter *e, const char *fmt, ...);
char *emitter_take_string(Emitter *e);

// ============================================================================
// File I/O
// ============================================================================

char *load_file(const char *fn);
char *z_resolve_path(const char *fn, const char *relative_to, struct CompilerConfig *cfg);

#endif // ZC_UTILS_H
