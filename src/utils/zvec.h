// SPDX-License-Identifier: MIT
/*
#ifndef ZC_ALLOW_INTERNAL
#error "utils/zvec.h is internal to Zen C. Include the appropriate public header instead."
#endif

 * zvec.h — Type-safe, zero-overhead dynamic arrays
 * Part of Zen Development Kit (ZDK)
 *
 * This is a macro-generated, single-header dynamic array library that produces
 * fully type-safe vector implementations at compile time. It supports both C and C++
 * with zero-cost abstraction.
 *
 * License: MIT
 * Author: Zuhaitz
 * Repository: https://github.com/z-libs/zvec.h
 * Version: 1.0.3
 */

#ifndef ZVEC_H
#define ZVEC_H

#include <string.h>
#include <stdlib.h>
#include <assert.h>

#ifndef Z_OK
#define Z_OK 0
#endif

#ifndef Z_ENOMEM
#define Z_ENOMEM -1
#endif

#ifndef Z_GROWTH_FACTOR
#define Z_GROWTH_FACTOR(cap) ((cap) == 0 ? 8 : (cap) * 2)
#endif

/* Allocator overrides */
#ifndef ZVEC_MALLOC
#define ZVEC_MALLOC(sz) malloc(sz)
#endif

#ifndef ZVEC_REALLOC
#define ZVEC_REALLOC(p, sz) realloc(p, sz)
#endif

#ifndef ZVEC_FREE
#define ZVEC_FREE(p) free(p)
#endif

#define ZVEC_GENERATE_IMPL(T, Name)                                                                \
    typedef struct                                                                                 \
    {                                                                                              \
        T *data;                                                                                   \
        size_t length;                                                                             \
        size_t capacity;                                                                           \
    } zvec_##Name;                                                                                 \
                                                                                                   \
    static inline int zvec_reserve_##Name(zvec_##Name *v, size_t new_cap)                          \
    {                                                                                              \
        if (new_cap <= v->capacity)                                                                \
            return Z_OK;                                                                           \
        T *new_data = (T *)ZVEC_REALLOC(v->data, new_cap * sizeof(T));                             \
        if (!new_data)                                                                             \
            return Z_ENOMEM;                                                                       \
        v->data = new_data;                                                                        \
        v->capacity = new_cap;                                                                     \
        return Z_OK;                                                                               \
    }                                                                                              \
                                                                                                   \
    static inline void zvec_free_##Name(zvec_##Name *v)                                            \
    {                                                                                              \
        ZVEC_FREE(v->data);                                                                        \
        v->data = NULL;                                                                            \
        v->length = 0;                                                                             \
        v->capacity = 0;                                                                           \
    }                                                                                              \
                                                                                                   \
    static inline int zvec_push_##Name(zvec_##Name *v, T value)                                    \
    {                                                                                              \
        if (v->length >= v->capacity)                                                              \
        {                                                                                          \
            size_t new_cap = Z_GROWTH_FACTOR(v->capacity);                                         \
            if (Z_OK != zvec_reserve_##Name(v, new_cap))                                           \
                return Z_ENOMEM;                                                                   \
        }                                                                                          \
        v->data[v->length++] = value;                                                              \
        return Z_OK;                                                                               \
    }                                                                                              \
                                                                                                   \
    static inline T zvec_pop_##Name(zvec_##Name *v)                                                \
    {                                                                                              \
        assert(v->length > 0);                                                                     \
        return v->data[--v->length];                                                               \
    }                                                                                              \
                                                                                                   \
    static inline void zvec_clear_##Name(zvec_##Name *v)                                           \
    {                                                                                              \
        v->length = 0;                                                                             \
    }

#endif /* ZVEC_H */
