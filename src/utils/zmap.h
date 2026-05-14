/*
 * GENERATED FILE - DO NOT EDIT DIRECTLY
 * Source: zmap.c
 *
 * This file is part of the z-libs collection: https://github.com/z-libs
 * Licensed under the MIT License.
 */

/* ============================================================================
   z-libs Common Definitions (Bundled)
   This block is auto-generated. It is guarded so that if you include multiple
   z-libs it is only defined once.
   ============================================================================ */
#ifndef Z_COMMON_BUNDLED
#define Z_COMMON_BUNDLED

/*
 * zcommon.h — Common definitions for the Zen Development Kit (ZDK)
 * Part of ZDK
 *
 * This header defines shared macros, error codes, and compiler extensions
 * used across all ZDK libraries.
 *
 * License: MIT
 * Author: Zuhaitz
 * Repository: https://github.com/z-libs/z-core
 * Version: 1.0.0
 */

#ifndef ZCOMMON_H
#define ZCOMMON_H

#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

// Return codes and error handling.
#ifndef Z_OK
#define Z_OK 0
#endif
#ifndef Z_FOUND
#define Z_FOUND 1 // Element found (positive).
#endif
#ifndef Z_ERR
#define Z_ERR -1 // Generic error.
#endif
#ifndef Z_ENOMEM
#define Z_ENOMEM -2 // Out of memory (malloc/realloc failed).
#endif
#ifndef Z_EOOB
#define Z_EOOB -3 // Out of bounds / range error.
#endif
#ifndef Z_EEMPTY
#define Z_EEMPTY -4 // Container is empty.
#endif
#ifndef Z_ENOTFOUND
#define Z_ENOTFOUND -5 // Element not found.
#endif
#ifndef Z_EINVAL
#define Z_EINVAL -6 // Invalid argument / parameter.
#endif
#ifndef Z_EEXIST
#define Z_EEXIST -7 // Element already exists.
#endif

// Memory management.

/* * If the user hasn't defined their own allocator, use the standard C library.
 * To override globally, define these macros before including any ZDK header.
 */
#ifndef Z_MALLOC
#include <stdlib.h>
#define Z_MALLOC(sz) malloc(sz)
#define Z_CALLOC(n, sz) calloc(n, sz)
#define Z_REALLOC(p, sz) realloc(p, sz)
#define Z_FREE(p) free(p)
#endif

// Compiler extensions and optimization.

// Type inference (typeof)
#ifdef __cplusplus
#include <type_traits>
#define Z_TYPEOF(x) typename std::remove_reference<decltype(x)>::type
#define Z_HAS_TYPEOF 1
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L
#define Z_TYPEOF(x) typeof(x)
#define Z_HAS_TYPEOF 1
#elif defined(__GNUC__) || defined(__clang__) || defined(__TINYC__)
#define Z_TYPEOF(x) __typeof__(x)
#define Z_HAS_TYPEOF 1
#else
#define Z_HAS_TYPEOF 0
#endif

// Extensions (cleanup, attributes, branch prediction)
#if !defined(Z_NO_EXTENSIONS) && (defined(__GNUC__) || defined(__clang__) || defined(__TINYC__))

#define Z_HAS_CLEANUP 1
#define Z_CLEANUP(func) __attribute__((cleanup(func)))
#define Z_NODISCARD __attribute__((warn_unused_result))

// TCC supports attributes but NOT __builtin_expect
#if defined(__TINYC__)
#define Z_LIKELY(x) (x)
#define Z_UNLIKELY(x) (x)
#else
#define Z_LIKELY(x) __builtin_expect(!!(x), 1)
#define Z_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

#else
// Fallback for MSVC or strict standard C.
#define Z_HAS_CLEANUP 0
#define Z_CLEANUP(func)
#define Z_NODISCARD
#define Z_LIKELY(x) (x)
#define Z_UNLIKELY(x) (x)

#endif

// Metaprogramming and internal utils.

/* * Markers for the Z-Scanner tool to find type definitions.
 * For the C compiler, they are no-ops (they compile to nothing).
 */
#define DEFINE_VEC_TYPE(T, Name)
#define DEFINE_LIST_TYPE(T, Name)
#define DEFINE_MAP_TYPE(Key, Val, Name)
#define DEFINE_STABLE_MAP_TYPE(Key, Val, Name)
#define DEFINE_TREE_TYPE(Key, Val, Name)

// Token concatenation macros (useful for unique variable names in macros).
#define Z_CONCAT_(a, b) a##b
#define Z_CONCAT(a, b) Z_CONCAT_(a, b)
#define Z_UNIQUE(prefix) Z_CONCAT(prefix, __LINE__)

// Growth strategy.

/* * Determines how containers expand when full.
 * Default is 2.0x (Geometric Growth).
 *
 * Optimization note:
 * 2.0x minimizes realloc calls but can waste memory.
 * 1.5x is often better for memory fragmentation and reuse.
 */
#ifndef Z_GROWTH_FACTOR
// Default: Double capacity (2.0x).
#define Z_GROWTH_FACTOR(cap) ((cap) == 0 ? 32 : (cap) * 2)

// Alternative: 1.5x Growth (Uncomment to use in your project).
// #define Z_GROWTH_FACTOR(cap) ((cap) == 0 ? 32 : (cap) + (cap) / 2)
#endif

#endif // ZCOMMON_H

#endif // Z_COMMON_BUNDLED
/* ============================================================================ */

/*
 * zmap.h — Type-safe, zero-overhead hash map
 * Part of Zen Development Kit (ZDK)
 *
 * This is a macro-generated, single-header hash map library.
 * It produces fully type-safe map implementations at compile time using
 * Robin Hood hashing with backward shift deletion.
 *
 * Features:
 * • Open addressing with Robin Hood hashing (high load factors)
 * • Two storage modes:
 * 1. Standard: Keys/Values stored inline (fastest, cache-friendly)
 * 2. Stable: Values stored via pointer (stable addresses, like std::map)
 * • C++ z_map::map<K,V> with RAII and STL-compatible iterators
 * • C++ complex type support (constructors/destructors called)
 * • Allocation failure returns Z_ENOMEM (fast path)
 *
 * License: MIT
 * Author: Zuhaitz
 * Repository: https://github.com/z-libs/zmap.h
 * Version: 1.1.0
 */

#ifndef ZMAP_H
#define ZMAP_H
// [Bundled] "zcommon.h" is included inline in this same file
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#if defined(__has_include) && __has_include("zerror.h")
#include "zerror.h"
#define Z_HAS_ZERROR 1
#elif defined(ZERROR_H)
#define Z_HAS_ZERROR 1
#else
#define Z_HAS_ZERROR 0
#endif

// Shared enum.
typedef enum
{
    ZMAP_EMPTY = 0,
    ZMAP_OCCUPIED
} zmap_state;

// C++ interop preamble.
#ifdef __cplusplus
#include <stdexcept>
#include <iterator>
#include <utility>
#include <type_traits>
#include <new>
#include <algorithm>

namespace z_map
{
// Forward declarations.
template <typename K, typename V> struct map;
template <typename K, typename V> class map_iterator;

template <typename K, typename V> struct traits
{
    static_assert(0 == sizeof(K), "No zmap implementation registered for this key/value pair.");
};

template <typename K, typename V> class map_iterator
{
public:
    static constexpr bool is_const = std::is_const<K>::value || std::is_const<V>::value;

    using KeyT = typename std::remove_const<K>::type;
    using ValT = typename std::remove_const<V>::type;

    using MapTraits = traits<KeyT, ValT>;
    using CMap = typename MapTraits::map_type;
    using CBucket = typename MapTraits::bucket_type;

    using iterator_category = std::forward_iterator_tag;
    using value_type = CBucket;
    using difference_type = ptrdiff_t;

    using reference = typename std::conditional<is_const, const CBucket &, CBucket &>::type;
    using pointer = typename std::conditional<is_const, const CBucket *, CBucket *>::type;

    map_iterator(CMap *m, size_t idx) : map_ptr(m), index(idx)
    {
        if (map_ptr && index < map_ptr->capacity)
        {
            if (ZMAP_EMPTY == map_ptr->buckets[index].state)
            {
                advance();
            }
        }
    }

    reference operator*() const
    {
        return map_ptr->buckets[index];
    }

    pointer operator->() const
    {
        return &map_ptr->buckets[index];
    }

    bool operator==(const map_iterator &other) const
    {
        return map_ptr == other.map_ptr && index == other.index;
    }

    bool operator!=(const map_iterator &other) const
    {
        return !(*this == other);
    }

    map_iterator &operator++()
    {
        if (index < map_ptr->capacity)
        {
            index++;
            advance();
        }
        return *this;
    }

    map_iterator operator++(int)
    {
        map_iterator temp = *this;
        ++(*this);
        return temp;
    }

private:
    void advance()
    {
        while (index < map_ptr->capacity && ZMAP_EMPTY == map_ptr->buckets[index].state)
        {
            index++;
        }
    }

    CMap *map_ptr;
    size_t index;
};

template <typename K, typename V> struct map
{
    using Traits = traits<K, V>;
    using c_map = typename Traits::map_type;
    using iterator = map_iterator<K, V>;
    using const_iterator = map_iterator<const K, const V>;

    c_map inner;

    using HashFunc = uint32_t (*)(K, uint32_t);
    using CmpFunc = int (*)(K, K);

    map(HashFunc h, CmpFunc c, uint32_t seed = 0xCAFEBABE, float load_factor = 0.85f)
        : inner(Traits::init(h, c, load_factor))
    {
        Traits::set_seed(&inner, seed);
    }

    map(map &&other) noexcept : inner(other.inner)
    {
        other.inner = Traits::init(inner.hash_func, inner.cmp_func, inner.load_factor);
    }

    ~map()
    {
        Traits::free(&inner);
    }

    map &operator=(map &&other) noexcept
    {
        if (this != &other)
        {
            Traits::free(&inner);
            inner = other.inner;
            other.inner = Traits::init(inner.hash_func, inner.cmp_func, inner.load_factor);
        }
        return *this;
    }

    map(const map &) = delete;
    map &operator=(const map &) = delete;

    void put(const K &key, const V &val)
    {
        if (Z_OK != Traits::put(&inner, key, val))
        {
            throw std::bad_alloc();
        }
    }

    void insert_or_assign(const K &key, const V &val)
    {
        put(key, val);
    }

    V *get(const K &key)
    {
        return Traits::get(&inner, key);
    }

    const V *get(const K &key) const
    {
        return Traits::get((c_map *)&inner, key);
    }

    bool contains(const K &key) const
    {
        return NULL != Traits::get((c_map *)&inner, key);
    }

    V &operator[](const K &key)
    {
        V *ptr = get(key);
        if (!ptr)
        {
            put(key, V{});
            ptr = get(key);
        }
        return *ptr;
    }

    V &at(const K &key)
    {
        V *ptr = get(key);
        if (!ptr)
        {
            throw std::out_of_range("z_map::map::at");
        }
        return *ptr;
    }

    void erase(const K &key)
    {
        Traits::remove(&inner, key);
    }

    void clear()
    {
        Traits::clear(&inner);
    }

    size_t size() const
    {
        return inner.count;
    }

    bool empty() const
    {
        return 0 == inner.count;
    }

    iterator begin()
    {
        return iterator(&inner, 0);
    }

    iterator end()
    {
        return iterator(&inner, inner.capacity);
    }

    const_iterator begin() const
    {
        return const_iterator((c_map *)&inner, 0);
    }

    const_iterator end() const
    {
        return const_iterator((c_map *)&inner, inner.capacity);
    }
};
} // namespace z_map
extern "C"
{
#endif // __cplusplus

    // Configuration and allocators.

#ifndef ZMAP_MALLOC
#define ZMAP_MALLOC(sz) Z_MALLOC(sz)
#endif

#ifndef ZMAP_CALLOC
#define ZMAP_CALLOC(n, sz) Z_CALLOC(n, sz)
#endif

#ifndef ZMAP_REALLOC
#define ZMAP_REALLOC(p, sz) Z_REALLOC(p, sz)
#endif

#ifndef ZMAP_FREE
#define ZMAP_FREE(p) Z_FREE(p)
#endif

// Load factor configuration.
#define ZMAP_DEFAULT_LOAD 0.85f

// Hashing helpers.
#if defined(__has_include) && __has_include("zhash.h")
#include "zhash.h"
#define ZMAP_HAS_ZHASH 1
#elif defined(ZHASH_H)
#define ZMAP_HAS_ZHASH 1
#else
#define ZMAP_HAS_ZHASH 0
#endif

#ifndef ZMAP_HASH_FUNC
#if ZMAP_HAS_ZHASH
#define ZMAP_HASH_FUNC(key, len, seed) zhash_fast(key, len, seed)
#else
    // FNV-1a inline fallback.
    static inline uint32_t zmap_default_hash(const void *key, size_t len, uint32_t seed)
    {
        uint32_t hash = 2166136261u ^ seed;
        const uint8_t *data = (const uint8_t *)key;
        for (size_t i = 0; i < len; i++)
        {
            hash ^= data[i];
            hash *= 16777619;
        }
        return hash;
    }
#define ZMAP_HASH_FUNC(key, len, seed) zmap_default_hash(key, len, seed)
#endif
#endif

#define ZMAP_HASH_SCALAR(k, s) ZMAP_HASH_FUNC(&(k), sizeof(k), s)
#define ZMAP_HASH_STR(k, s) ZMAP_HASH_FUNC((k), strlen(k), s)

    static inline size_t zmap_next_pow2(size_t n)
    {
        if (0 == n)
        {
            return 16;
        }
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
#if UINTPTR_MAX > 0xFFFFFFFF
        n |= n >> 32;
#endif
        return n + 1;
    }

#define ZMAP_FIB_CONST 0x9E3779B9U

    static inline size_t zmap_fib_index(uint32_t hash, uint32_t bits)
    {
        return (size_t)((hash * ZMAP_FIB_CONST) >> (32 - bits));
    }

    static inline size_t zmap_dist(size_t index, size_t capacity, uint32_t hash, uint32_t bits)
    {
        size_t home = zmap_fib_index(hash, bits);
        if (index >= home)
        {
            return index - home;
        }
        return (index + capacity) - home;
    }

// Safe API logic.
#if Z_HAS_ZERROR
    static inline zerr zmap_err_impl(int code, const char *msg, const char *file, int line,
                                     const char *func)
    {
        return zerr_create_impl(code, file, line, func, "%s", msg);
    }

#define ZMAP_GEN_SAFE_IMPL(KeyT, ValT, Name)                                                       \
    DEFINE_RESULT(ValT *, ResPtr_##Name)                                                           \
                                                                                                   \
    static inline zres zmap_put_safe_##Name(zmap_##Name *m, KeyT k, ValT v, const char *f, int l,  \
                                            const char *fn)                                        \
    {                                                                                              \
        if (zmap_put_##Name(m, k, v) != Z_OK)                                                      \
        {                                                                                          \
            return zres_err(zmap_err_impl(Z_ENOMEM, "Map Put OOM", f, l, fn));                     \
        }                                                                                          \
        return zres_ok();                                                                          \
    }                                                                                              \
                                                                                                   \
    static inline ResPtr_##Name zmap_get_safe_##Name(zmap_##Name *m, KeyT k, const char *f, int l, \
                                                     const char *fn)                               \
    {                                                                                              \
        ValT *ptr = zmap_get_##Name(m, k);                                                         \
        if (!ptr)                                                                                  \
        {                                                                                          \
            return ResPtr_##Name##_err(zmap_err_impl(Z_ENOTFOUND, "Key not found", f, l, fn));     \
        }                                                                                          \
        return ResPtr_##Name##_ok(ptr);                                                            \
    }
#else
#define ZMAP_GEN_SAFE_IMPL(KeyT, ValT, Name)
#endif

/* * Implementation injection (C vs C++).
 * C++ uses new/delete/move for proper RAII.
 * C uses calloc/free/struct-copy.
 */
#ifdef __cplusplus
#define ZMAP_IMPL_OPS(KeyT, ValT, Name)                                                            \
    static inline void zmap_free_##Name(zmap_##Name *m)                                            \
    {                                                                                              \
        if (m->buckets)                                                                            \
        {                                                                                          \
            delete[] m->buckets;                                                                   \
        }                                                                                          \
        m->buckets = nullptr;                                                                      \
        m->count = 0;                                                                              \
        m->capacity = 0;                                                                           \
    }                                                                                              \
                                                                                                   \
    static inline void zmap_clear_##Name(zmap_##Name *m)                                           \
    {                                                                                              \
        if (m->buckets)                                                                            \
        {                                                                                          \
            delete[] m->buckets;                                                                   \
        }                                                                                          \
        m->buckets = nullptr;                                                                      \
        m->count = 0;                                                                              \
        m->capacity = 0;                                                                           \
    }                                                                                              \
                                                                                                   \
    static inline int zmap_resize_##Name(zmap_##Name *m, size_t new_cap)                           \
    {                                                                                              \
        try                                                                                        \
        {                                                                                          \
            zmap_bucket_##Name *new_buckets = new zmap_bucket_##Name[new_cap]();                   \
            uint32_t new_bits = 0;                                                                 \
            size_t temp = new_cap;                                                                 \
            while (temp >>= 1)                                                                     \
            {                                                                                      \
                new_bits++;                                                                        \
            }                                                                                      \
            for (size_t i = 0; i < m->capacity; i++)                                               \
            {                                                                                      \
                if (ZMAP_OCCUPIED == m->buckets[i].state)                                          \
                {                                                                                  \
                    zmap_bucket_##Name entry = std::move(m->buckets[i]);                           \
                    size_t idx = zmap_fib_index(entry.stored_hash, new_bits);                      \
                    size_t dist = 0;                                                               \
                    for (;;)                                                                       \
                    {                                                                              \
                        if (ZMAP_EMPTY == new_buckets[idx].state)                                  \
                        {                                                                          \
                            new_buckets[idx] = std::move(entry);                                   \
                            new_buckets[idx].state = ZMAP_OCCUPIED;                                \
                            break;                                                                 \
                        }                                                                          \
                        size_t existing_dist =                                                     \
                            zmap_dist(idx, new_cap, new_buckets[idx].stored_hash, new_bits);       \
                        if (dist > existing_dist)                                                  \
                        {                                                                          \
                            std::swap(new_buckets[idx], entry);                                    \
                            dist = existing_dist;                                                  \
                        }                                                                          \
                        idx = (idx + 1) & (new_cap - 1);                                           \
                        dist++;                                                                    \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
            if (m->buckets)                                                                        \
            {                                                                                      \
                delete[] m->buckets;                                                               \
            }                                                                                      \
            m->buckets = new_buckets;                                                              \
            m->capacity = new_cap;                                                                 \
            m->bits = new_bits;                                                                    \
            m->threshold = (size_t)(new_cap * m->load_factor);                                     \
            return Z_OK;                                                                           \
        }                                                                                          \
        catch (...)                                                                                \
        {                                                                                          \
            return Z_ENOMEM;                                                                       \
        }                                                                                          \
    }                                                                                              \
                                                                                                   \
    static inline int zmap_put_##Name(zmap_##Name *m, KeyT key, ValT val)                          \
    {                                                                                              \
        try                                                                                        \
        {                                                                                          \
            if (m->count >= m->threshold)                                                          \
            {                                                                                      \
                size_t new_cap = zmap_next_pow2(Z_GROWTH_FACTOR(m->capacity));                     \
                if (Z_OK != zmap_resize_##Name(m, new_cap))                                        \
                {                                                                                  \
                    return Z_ENOMEM;                                                               \
                }                                                                                  \
            }                                                                                      \
            uint32_t hash = m->hash_func(key, m->seed);                                            \
            size_t idx = zmap_fib_index(hash, m->bits);                                            \
            size_t dist = 0;                                                                       \
            zmap_bucket_##Name entry;                                                              \
            entry.key = key;                                                                       \
            entry.value = val;                                                                     \
            entry.stored_hash = hash;                                                              \
            entry.state = ZMAP_OCCUPIED;                                                           \
            for (;;)                                                                               \
            {                                                                                      \
                if (ZMAP_EMPTY == m->buckets[idx].state)                                           \
                {                                                                                  \
                    m->buckets[idx] = std::move(entry);                                            \
                    m->count++;                                                                    \
                    return Z_OK;                                                                   \
                }                                                                                  \
                if (m->buckets[idx].stored_hash == hash &&                                         \
                    0 == m->cmp_func(m->buckets[idx].key, key))                                    \
                {                                                                                  \
                    m->buckets[idx].value = val;                                                   \
                    return Z_OK;                                                                   \
                }                                                                                  \
                size_t existing_dist =                                                             \
                    zmap_dist(idx, m->capacity, m->buckets[idx].stored_hash, m->bits);             \
                if (dist > existing_dist)                                                          \
                {                                                                                  \
                    std::swap(m->buckets[idx], entry);                                             \
                    dist = existing_dist;                                                          \
                }                                                                                  \
                idx = (idx + 1) & (m->capacity - 1);                                               \
                dist++;                                                                            \
            }                                                                                      \
        }                                                                                          \
        catch (...)                                                                                \
        {                                                                                          \
            return Z_ENOMEM;                                                                       \
        }                                                                                          \
    }

#define ZMAP_IMPL_STABLE_OPS(KeyT, ValT, Name)                                                     \
    static inline void zmap_free_stable_##Name(zmap_stable_##Name *m)                              \
    {                                                                                              \
        if (m->buckets)                                                                            \
        {                                                                                          \
            for (size_t i = 0; i < m->capacity; i++)                                               \
            {                                                                                      \
                if (ZMAP_OCCUPIED == m->buckets[i].state)                                          \
                {                                                                                  \
                    delete m->buckets[i].value;                                                    \
                }                                                                                  \
            }                                                                                      \
            delete[] m->buckets;                                                                   \
        }                                                                                          \
        *m = (zmap_stable_##Name){0};                                                              \
    }                                                                                              \
                                                                                                   \
    static inline void zmap_clear_stable_##Name(zmap_stable_##Name *m)                             \
    {                                                                                              \
        zmap_free_stable_##Name(m);                                                                \
    }                                                                                              \
                                                                                                   \
    static inline int zmap_resize_stable_##Name(zmap_stable_##Name *m, size_t new_cap)             \
    {                                                                                              \
        try                                                                                        \
        {                                                                                          \
            zmap_bucket_stable_##Name *new_buckets = new zmap_bucket_stable_##Name[new_cap]();     \
            uint32_t new_bits = 0;                                                                 \
            size_t temp = new_cap;                                                                 \
            while (temp >>= 1)                                                                     \
            {                                                                                      \
                new_bits++;                                                                        \
            }                                                                                      \
            for (size_t i = 0; i < m->capacity; i++)                                               \
            {                                                                                      \
                if (ZMAP_OCCUPIED == m->buckets[i].state)                                          \
                {                                                                                  \
                    zmap_bucket_stable_##Name entry = m->buckets[i];                               \
                    size_t idx = zmap_fib_index(entry.stored_hash, new_bits);                      \
                    size_t dist = 0;                                                               \
                    for (;;)                                                                       \
                    {                                                                              \
                        if (ZMAP_EMPTY == new_buckets[idx].state)                                  \
                        {                                                                          \
                            new_buckets[idx] = entry;                                              \
                            new_buckets[idx].state = ZMAP_OCCUPIED;                                \
                            break;                                                                 \
                        }                                                                          \
                        size_t existing_dist =                                                     \
                            zmap_dist(idx, new_cap, new_buckets[idx].stored_hash, new_bits);       \
                        if (dist > existing_dist)                                                  \
                        {                                                                          \
                            std::swap(new_buckets[idx], entry);                                    \
                            dist = existing_dist;                                                  \
                        }                                                                          \
                        idx = (idx + 1) & (new_cap - 1);                                           \
                        dist++;                                                                    \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
            if (m->buckets)                                                                        \
            {                                                                                      \
                delete[] m->buckets;                                                               \
            }                                                                                      \
            m->buckets = new_buckets;                                                              \
            m->capacity = new_cap;                                                                 \
            m->bits = new_bits;                                                                    \
            m->threshold = (size_t)(new_cap * m->load_factor);                                     \
            return Z_OK;                                                                           \
        }                                                                                          \
        catch (...)                                                                                \
        {                                                                                          \
            return Z_ENOMEM;                                                                       \
        }                                                                                          \
    }                                                                                              \
                                                                                                   \
    static inline int zmap_put_stable_##Name(zmap_stable_##Name *m, KeyT key, ValT val)            \
    {                                                                                              \
        try                                                                                        \
        {                                                                                          \
            if (m->count >= m->threshold)                                                          \
            {                                                                                      \
                size_t new_cap = zmap_next_pow2(Z_GROWTH_FACTOR(m->capacity));                     \
                if (Z_OK != zmap_resize_stable_##Name(m, new_cap))                                 \
                {                                                                                  \
                    return Z_ENOMEM;                                                               \
                }                                                                                  \
            }                                                                                      \
            uint32_t hash = m->hash_func(key, m->seed);                                            \
            size_t idx = zmap_fib_index(hash, m->bits);                                            \
            size_t dist = 0;                                                                       \
            zmap_bucket_stable_##Name entry;                                                       \
            entry.key = key;                                                                       \
            entry.value = nullptr;                                                                 \
            entry.stored_hash = hash;                                                              \
            entry.state = ZMAP_OCCUPIED;                                                           \
            for (;;)                                                                               \
            {                                                                                      \
                if (ZMAP_EMPTY == m->buckets[idx].state)                                           \
                {                                                                                  \
                    if (!entry.value)                                                              \
                    {                                                                              \
                        entry.value = new ValT(val);                                               \
                    }                                                                              \
                    m->buckets[idx] = entry;                                                       \
                    m->count++;                                                                    \
                    return Z_OK;                                                                   \
                }                                                                                  \
                if (m->buckets[idx].stored_hash == hash &&                                         \
                    0 == m->cmp_func(m->buckets[idx].key, key))                                    \
                {                                                                                  \
                    *m->buckets[idx].value = val;                                                  \
                    return Z_OK;                                                                   \
                }                                                                                  \
                size_t existing_dist =                                                             \
                    zmap_dist(idx, m->capacity, m->buckets[idx].stored_hash, m->bits);             \
                if (dist > existing_dist)                                                          \
                {                                                                                  \
                    if (!entry.value)                                                              \
                    {                                                                              \
                        entry.value = new ValT(val);                                               \
                    }                                                                              \
                    std::swap(m->buckets[idx], entry);                                             \
                    dist = existing_dist;                                                          \
                }                                                                                  \
                idx = (idx + 1) & (m->capacity - 1);                                               \
                dist++;                                                                            \
            }                                                                                      \
        }                                                                                          \
        catch (...)                                                                                \
        {                                                                                          \
            return Z_ENOMEM;                                                                       \
        }                                                                                          \
    }                                                                                              \
    static inline void zmap_remove_val_stable_##Name(ValT *ptr)                                    \
    {                                                                                              \
        delete ptr;                                                                                \
    }
#else
#define ZMAP_IMPL_OPS(KeyT, ValT, Name)                                                            \
    static inline void zmap_free_##Name(zmap_##Name *m)                                            \
    {                                                                                              \
        ZMAP_FREE(m->buckets);                                                                     \
        *m = (zmap_##Name){0};                                                                     \
    }                                                                                              \
                                                                                                   \
    static inline void zmap_clear_##Name(zmap_##Name *m)                                           \
    {                                                                                              \
        if (m->capacity > 0)                                                                       \
        {                                                                                          \
            memset(m->buckets, 0, m->capacity * sizeof(zmap_bucket_##Name));                       \
        }                                                                                          \
        m->count = 0;                                                                              \
    }                                                                                              \
                                                                                                   \
    static inline int zmap_resize_##Name(zmap_##Name *m, size_t new_cap)                           \
    {                                                                                              \
        zmap_bucket_##Name *new_buckets =                                                          \
            (zmap_bucket_##Name *)ZMAP_CALLOC(new_cap, sizeof(zmap_bucket_##Name));                \
        if (!new_buckets)                                                                          \
        {                                                                                          \
            return Z_ENOMEM;                                                                       \
        }                                                                                          \
        uint32_t new_bits = 0;                                                                     \
        size_t temp = new_cap;                                                                     \
        while (temp >>= 1)                                                                         \
        {                                                                                          \
            new_bits++;                                                                            \
        }                                                                                          \
        for (size_t i = 0; i < m->capacity; i++)                                                   \
        {                                                                                          \
            if (ZMAP_OCCUPIED == m->buckets[i].state)                                              \
            {                                                                                      \
                zmap_bucket_##Name entry = m->buckets[i];                                          \
                size_t idx = zmap_fib_index(entry.stored_hash, new_bits);                          \
                size_t dist = 0;                                                                   \
                for (;;)                                                                           \
                {                                                                                  \
                    if (ZMAP_EMPTY == new_buckets[idx].state)                                      \
                    {                                                                              \
                        new_buckets[idx] = entry;                                                  \
                        new_buckets[idx].state = ZMAP_OCCUPIED;                                    \
                        break;                                                                     \
                    }                                                                              \
                    size_t existing_dist =                                                         \
                        zmap_dist(idx, new_cap, new_buckets[idx].stored_hash, new_bits);           \
                    if (dist > existing_dist)                                                      \
                    {                                                                              \
                        zmap_bucket_##Name swap_tmp = new_buckets[idx];                            \
                        new_buckets[idx] = entry;                                                  \
                        entry = swap_tmp;                                                          \
                        dist = existing_dist;                                                      \
                    }                                                                              \
                    idx = (idx + 1) & (new_cap - 1);                                               \
                    dist++;                                                                        \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
        ZMAP_FREE(m->buckets);                                                                     \
        m->buckets = new_buckets;                                                                  \
        m->capacity = new_cap;                                                                     \
        m->bits = new_bits;                                                                        \
        m->threshold = (size_t)(new_cap * m->load_factor);                                         \
        return Z_OK;                                                                               \
    }                                                                                              \
                                                                                                   \
    static inline int zmap_put_##Name(zmap_##Name *m, KeyT key, ValT val)                          \
    {                                                                                              \
        if (m->count >= m->threshold)                                                              \
        {                                                                                          \
            size_t new_cap = zmap_next_pow2(Z_GROWTH_FACTOR(m->capacity));                         \
            if (Z_OK != zmap_resize_##Name(m, new_cap))                                            \
            {                                                                                      \
                return Z_ENOMEM;                                                                   \
            }                                                                                      \
        }                                                                                          \
        uint32_t hash = m->hash_func(key, m->seed);                                                \
        size_t idx = zmap_fib_index(hash, m->bits);                                                \
        size_t dist = 0;                                                                           \
        zmap_bucket_##Name entry = (zmap_bucket_##Name){                                           \
            .key = key, .value = val, .stored_hash = hash, .state = ZMAP_OCCUPIED};                \
        for (;;)                                                                                   \
        {                                                                                          \
            if (ZMAP_EMPTY == m->buckets[idx].state)                                               \
            {                                                                                      \
                m->buckets[idx] = entry;                                                           \
                m->count++;                                                                        \
                return Z_OK;                                                                       \
            }                                                                                      \
            if (m->buckets[idx].stored_hash == hash && 0 == m->cmp_func(m->buckets[idx].key, key)) \
            {                                                                                      \
                m->buckets[idx].value = val;                                                       \
                return Z_OK;                                                                       \
            }                                                                                      \
            size_t existing_dist =                                                                 \
                zmap_dist(idx, m->capacity, m->buckets[idx].stored_hash, m->bits);                 \
            if (dist > existing_dist)                                                              \
            {                                                                                      \
                zmap_bucket_##Name swap_tmp = m->buckets[idx];                                     \
                m->buckets[idx] = entry;                                                           \
                entry = swap_tmp;                                                                  \
                dist = existing_dist;                                                              \
            }                                                                                      \
            idx = (idx + 1) & (m->capacity - 1);                                                   \
            dist++;                                                                                \
        }                                                                                          \
    }

#define ZMAP_IMPL_STABLE_OPS(KeyT, ValT, Name)                                                     \
    static inline void zmap_free_stable_##Name(zmap_stable_##Name *m)                              \
    {                                                                                              \
        if (m->buckets)                                                                            \
        {                                                                                          \
            for (size_t i = 0; i < m->capacity; i++)                                               \
            {                                                                                      \
                if (ZMAP_OCCUPIED == m->buckets[i].state)                                          \
                {                                                                                  \
                    ZMAP_FREE(m->buckets[i].value);                                                \
                }                                                                                  \
            }                                                                                      \
            ZMAP_FREE(m->buckets);                                                                 \
        }                                                                                          \
        *m = (zmap_stable_##Name){0};                                                              \
    }                                                                                              \
                                                                                                   \
    static inline void zmap_clear_stable_##Name(zmap_stable_##Name *m)                             \
    {                                                                                              \
        zmap_free_stable_##Name(m);                                                                \
    }                                                                                              \
                                                                                                   \
    static inline int zmap_resize_stable_##Name(zmap_stable_##Name *m, size_t new_cap)             \
    {                                                                                              \
        zmap_bucket_stable_##Name *new_buckets =                                                   \
            (zmap_bucket_stable_##Name *)ZMAP_CALLOC(new_cap, sizeof(zmap_bucket_stable_##Name));  \
        if (!new_buckets)                                                                          \
        {                                                                                          \
            return Z_ENOMEM;                                                                       \
        }                                                                                          \
        uint32_t new_bits = 0;                                                                     \
        size_t temp = new_cap;                                                                     \
        while (temp >>= 1)                                                                         \
        {                                                                                          \
            new_bits++;                                                                            \
        }                                                                                          \
        for (size_t i = 0; i < m->capacity; i++)                                                   \
        {                                                                                          \
            if (ZMAP_OCCUPIED == m->buckets[i].state)                                              \
            {                                                                                      \
                zmap_bucket_stable_##Name entry = m->buckets[i];                                   \
                size_t idx = zmap_fib_index(entry.stored_hash, new_bits);                          \
                size_t dist = 0;                                                                   \
                for (;;)                                                                           \
                {                                                                                  \
                    if (ZMAP_EMPTY == new_buckets[idx].state)                                      \
                    {                                                                              \
                        new_buckets[idx] = entry;                                                  \
                        new_buckets[idx].state = ZMAP_OCCUPIED;                                    \
                        break;                                                                     \
                    }                                                                              \
                    size_t existing_dist =                                                         \
                        zmap_dist(idx, new_cap, new_buckets[idx].stored_hash, new_bits);           \
                    if (dist > existing_dist)                                                      \
                    {                                                                              \
                        zmap_bucket_stable_##Name tmp = new_buckets[idx];                          \
                        new_buckets[idx] = entry;                                                  \
                        entry = tmp;                                                               \
                        dist = existing_dist;                                                      \
                    }                                                                              \
                    idx = (idx + 1) & (new_cap - 1);                                               \
                    dist++;                                                                        \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
        ZMAP_FREE(m->buckets);                                                                     \
        m->buckets = new_buckets;                                                                  \
        m->capacity = new_cap;                                                                     \
        m->bits = new_bits;                                                                        \
        m->threshold = (size_t)(new_cap * m->load_factor);                                         \
        return Z_OK;                                                                               \
    }                                                                                              \
                                                                                                   \
    static inline int zmap_put_stable_##Name(zmap_stable_##Name *m, KeyT key, ValT val)            \
    {                                                                                              \
        if (m->count >= m->threshold)                                                              \
        {                                                                                          \
            size_t new_cap = zmap_next_pow2(Z_GROWTH_FACTOR(m->capacity));                         \
            if (Z_OK != zmap_resize_stable_##Name(m, new_cap))                                     \
            {                                                                                      \
                return Z_ENOMEM;                                                                   \
            }                                                                                      \
        }                                                                                          \
        uint32_t hash = m->hash_func(key, m->seed);                                                \
        size_t idx = zmap_fib_index(hash, m->bits);                                                \
        size_t dist = 0;                                                                           \
        zmap_bucket_stable_##Name entry = (zmap_bucket_stable_##Name){                             \
            .key = key, .value = NULL, .stored_hash = hash, .state = ZMAP_OCCUPIED};               \
        for (;;)                                                                                   \
        {                                                                                          \
            if (ZMAP_EMPTY == m->buckets[idx].state)                                               \
            {                                                                                      \
                if (!entry.value)                                                                  \
                {                                                                                  \
                    entry.value = (ValT *)ZMAP_MALLOC(sizeof(ValT));                               \
                    if (!entry.value)                                                              \
                    {                                                                              \
                        return Z_ENOMEM;                                                           \
                    }                                                                              \
                    *entry.value = val;                                                            \
                }                                                                                  \
                m->buckets[idx] = entry;                                                           \
                m->count++;                                                                        \
                return Z_OK;                                                                       \
            }                                                                                      \
            if (m->buckets[idx].stored_hash == hash && 0 == m->cmp_func(m->buckets[idx].key, key)) \
            {                                                                                      \
                *m->buckets[idx].value = val;                                                      \
                return Z_OK;                                                                       \
            }                                                                                      \
            size_t existing_dist =                                                                 \
                zmap_dist(idx, m->capacity, m->buckets[idx].stored_hash, m->bits);                 \
            if (dist > existing_dist)                                                              \
            {                                                                                      \
                if (!entry.value)                                                                  \
                {                                                                                  \
                    entry.value = (ValT *)ZMAP_MALLOC(sizeof(ValT));                               \
                    if (!entry.value)                                                              \
                    {                                                                              \
                        return Z_ENOMEM;                                                           \
                    }                                                                              \
                    *entry.value = val;                                                            \
                }                                                                                  \
                zmap_bucket_stable_##Name temp = m->buckets[idx];                                  \
                m->buckets[idx] = entry;                                                           \
                entry = temp;                                                                      \
                dist = existing_dist;                                                              \
            }                                                                                      \
            idx = (idx + 1) & (m->capacity - 1);                                                   \
            dist++;                                                                                \
        }                                                                                          \
    }                                                                                              \
    static inline void zmap_remove_val_stable_##Name(ValT *ptr)                                    \
    {                                                                                              \
        ZMAP_FREE(ptr);                                                                            \
    }
#endif

/*
 * ZMAP_GENERATE_IMPL
 * Standard In-Place Map Generator.
 */
#define ZMAP_GENERATE_IMPL(KeyT, ValT, Name)                                                       \
    typedef struct                                                                                 \
    {                                                                                              \
        KeyT key;                                                                                  \
        ValT value;                                                                                \
        uint32_t stored_hash;                                                                      \
        zmap_state state;                                                                          \
    } zmap_bucket_##Name;                                                                          \
                                                                                                   \
    typedef struct                                                                                 \
    {                                                                                              \
        zmap_bucket_##Name *buckets;                                                               \
        size_t capacity;                                                                           \
        size_t count;                                                                              \
        size_t threshold;                                                                          \
        uint32_t bits;                                                                             \
        float load_factor;                                                                         \
        uint32_t seed;                                                                             \
        uint32_t (*hash_func)(KeyT, uint32_t);                                                     \
        int (*cmp_func)(KeyT, KeyT);                                                               \
    } zmap_##Name;                                                                                 \
                                                                                                   \
    typedef struct                                                                                 \
    {                                                                                              \
        zmap_##Name *map;                                                                          \
        size_t index;                                                                              \
    } zmap_iter_##Name;                                                                            \
                                                                                                   \
    static inline zmap_##Name zmap_init_ext_##Name(uint32_t (*h)(KeyT, uint32_t),                  \
                                                   int (*c)(KeyT, KeyT), float load)               \
    {                                                                                              \
        return (zmap_##Name){.buckets = NULL,                                                      \
                             .capacity = 0,                                                        \
                             .count = 0,                                                           \
                             .threshold = 0,                                                       \
                             .bits = 0,                                                            \
                             .load_factor =                                                        \
                                 (load <= 0.1f || load > 0.95f) ? ZMAP_DEFAULT_LOAD : load,        \
                             .seed = 0xCAFEBABE,                                                   \
                             .hash_func = h,                                                       \
                             .cmp_func = c};                                                       \
    }                                                                                              \
                                                                                                   \
    static inline zmap_##Name zmap_init_##Name(uint32_t (*h)(KeyT, uint32_t),                      \
                                               int (*c)(KeyT, KeyT))                               \
    {                                                                                              \
        return zmap_init_ext_##Name(h, c, ZMAP_DEFAULT_LOAD);                                      \
    }                                                                                              \
                                                                                                   \
    static inline void zmap_set_seed_##Name(zmap_##Name *m, uint32_t s)                            \
    {                                                                                              \
        m->seed = s;                                                                               \
    }                                                                                              \
                                                                                                   \
    ZMAP_IMPL_OPS(KeyT, ValT, Name)                                                                \
                                                                                                   \
    static inline ValT *zmap_get_##Name(zmap_##Name *m, KeyT key)                                  \
    {                                                                                              \
        if (0 == m->count)                                                                         \
        {                                                                                          \
            return NULL;                                                                           \
        }                                                                                          \
        uint32_t hash = m->hash_func(key, m->seed);                                                \
        size_t idx = zmap_fib_index(hash, m->bits);                                                \
        size_t dist = 0;                                                                           \
        for (;;)                                                                                   \
        {                                                                                          \
            if (ZMAP_EMPTY == m->buckets[idx].state)                                               \
            {                                                                                      \
                return NULL;                                                                       \
            }                                                                                      \
            size_t existing_dist =                                                                 \
                zmap_dist(idx, m->capacity, m->buckets[idx].stored_hash, m->bits);                 \
            if (dist > existing_dist)                                                              \
            {                                                                                      \
                return NULL;                                                                       \
            }                                                                                      \
            if (m->buckets[idx].stored_hash == hash && 0 == m->cmp_func(m->buckets[idx].key, key)) \
            {                                                                                      \
                return &m->buckets[idx].value;                                                     \
            }                                                                                      \
            idx = (idx + 1) & (m->capacity - 1);                                                   \
            dist++;                                                                                \
        }                                                                                          \
    }                                                                                              \
                                                                                                   \
    static inline void zmap_remove_##Name(zmap_##Name *m, KeyT key)                                \
    {                                                                                              \
        if (0 == m->count)                                                                         \
        {                                                                                          \
            return;                                                                                \
        }                                                                                          \
        uint32_t hash = m->hash_func(key, m->seed);                                                \
        size_t idx = zmap_fib_index(hash, m->bits);                                                \
        size_t dist = 0;                                                                           \
        for (;;)                                                                                   \
        {                                                                                          \
            if (ZMAP_EMPTY == m->buckets[idx].state)                                               \
            {                                                                                      \
                return;                                                                            \
            }                                                                                      \
            size_t existing_dist =                                                                 \
                zmap_dist(idx, m->capacity, m->buckets[idx].stored_hash, m->bits);                 \
            if (dist > existing_dist)                                                              \
            {                                                                                      \
                return;                                                                            \
            }                                                                                      \
            if (m->buckets[idx].stored_hash == hash && 0 == m->cmp_func(m->buckets[idx].key, key)) \
            {                                                                                      \
                m->count--;                                                                        \
                for (;;)                                                                           \
                {                                                                                  \
                    size_t next = (idx + 1) & (m->capacity - 1);                                   \
                    if (ZMAP_EMPTY == m->buckets[next].state)                                      \
                    {                                                                              \
                        m->buckets[idx].state = ZMAP_EMPTY;                                        \
                        return;                                                                    \
                    }                                                                              \
                    size_t next_dist =                                                             \
                        zmap_dist(next, m->capacity, m->buckets[next].stored_hash, m->bits);       \
                    if (0 == next_dist)                                                            \
                    {                                                                              \
                        m->buckets[idx].state = ZMAP_EMPTY;                                        \
                        return;                                                                    \
                    }                                                                              \
                    m->buckets[idx] = m->buckets[next];                                            \
                    idx = next;                                                                    \
                }                                                                                  \
            }                                                                                      \
            idx = (idx + 1) & (m->capacity - 1);                                                   \
            dist++;                                                                                \
        }                                                                                          \
    }                                                                                              \
                                                                                                   \
    static inline zmap_iter_##Name zmap_iter_init_##Name(zmap_##Name *m)                           \
    {                                                                                              \
        return (zmap_iter_##Name){.map = m, .index = 0};                                           \
    }                                                                                              \
                                                                                                   \
    static inline bool zmap_iter_next_##Name(zmap_iter_##Name *it, KeyT *out_k, ValT *out_v)       \
    {                                                                                              \
        if (!it->map || !it->map->buckets)                                                         \
        {                                                                                          \
            return false;                                                                          \
        }                                                                                          \
        while (it->index < it->map->capacity)                                                      \
        {                                                                                          \
            size_t i = it->index++;                                                                \
            if (ZMAP_OCCUPIED == it->map->buckets[i].state)                                        \
            {                                                                                      \
                if (out_k)                                                                         \
                    *out_k = it->map->buckets[i].key;                                              \
                if (out_v)                                                                         \
                    *out_v = it->map->buckets[i].value;                                            \
                return true;                                                                       \
            }                                                                                      \
        }                                                                                          \
        return false;                                                                              \
    }                                                                                              \
                                                                                                   \
    static inline size_t zmap_size_##Name(zmap_##Name *m)                                          \
    {                                                                                              \
        return m->count;                                                                           \
    }                                                                                              \
                                                                                                   \
    ZMAP_GEN_SAFE_IMPL(KeyT, ValT, Name)

/*
 * ZMAP_GENERATE_STABLE_IMPL
 * Stable Map Generator. Values are heap-allocated pointers.
 */
#define ZMAP_GENERATE_STABLE_IMPL(KeyT, ValT, Name)                                                \
    typedef struct                                                                                 \
    {                                                                                              \
        KeyT key;                                                                                  \
        ValT *value;                                                                               \
        uint32_t stored_hash;                                                                      \
        zmap_state state;                                                                          \
    } zmap_bucket_stable_##Name;                                                                   \
                                                                                                   \
    typedef struct                                                                                 \
    {                                                                                              \
        zmap_bucket_stable_##Name *buckets;                                                        \
        size_t capacity;                                                                           \
        size_t count;                                                                              \
        size_t threshold;                                                                          \
        uint32_t bits;                                                                             \
        float load_factor;                                                                         \
        uint32_t seed;                                                                             \
        uint32_t (*hash_func)(KeyT, uint32_t);                                                     \
        int (*cmp_func)(KeyT, KeyT);                                                               \
    } zmap_stable_##Name;                                                                          \
                                                                                                   \
    typedef struct                                                                                 \
    {                                                                                              \
        zmap_stable_##Name *map;                                                                   \
        size_t index;                                                                              \
    } zmap_iter_stable_##Name;                                                                     \
                                                                                                   \
    static inline zmap_stable_##Name zmap_init_ext_stable_##Name(uint32_t (*h)(KeyT, uint32_t),    \
                                                                 int (*c)(KeyT, KeyT), float load) \
    {                                                                                              \
        return (zmap_stable_##Name){.buckets = NULL,                                               \
                                    .capacity = 0,                                                 \
                                    .count = 0,                                                    \
                                    .threshold = 0,                                                \
                                    .bits = 0,                                                     \
                                    .load_factor =                                                 \
                                        (load <= 0.1f || load > 0.95f) ? ZMAP_DEFAULT_LOAD : load, \
                                    .seed = 0xCAFEBABE,                                            \
                                    .hash_func = h,                                                \
                                    .cmp_func = c};                                                \
    }                                                                                              \
                                                                                                   \
    static inline zmap_stable_##Name zmap_init_stable_##Name(uint32_t (*h)(KeyT, uint32_t),        \
                                                             int (*c)(KeyT, KeyT))                 \
    {                                                                                              \
        return zmap_init_ext_stable_##Name(h, c, ZMAP_DEFAULT_LOAD);                               \
    }                                                                                              \
                                                                                                   \
    static inline void zmap_set_seed_stable_##Name(zmap_stable_##Name *m, uint32_t s)              \
    {                                                                                              \
        m->seed = s;                                                                               \
    }                                                                                              \
                                                                                                   \
    ZMAP_IMPL_STABLE_OPS(KeyT, ValT, Name)                                                         \
                                                                                                   \
    static inline ValT *zmap_get_stable_##Name(zmap_stable_##Name *m, KeyT key)                    \
    {                                                                                              \
        if (0 == m->count)                                                                         \
        {                                                                                          \
            return NULL;                                                                           \
        }                                                                                          \
        uint32_t hash = m->hash_func(key, m->seed);                                                \
        size_t idx = zmap_fib_index(hash, m->bits);                                                \
        size_t dist = 0;                                                                           \
        for (;;)                                                                                   \
        {                                                                                          \
            if (ZMAP_EMPTY == m->buckets[idx].state)                                               \
            {                                                                                      \
                return NULL;                                                                       \
            }                                                                                      \
            size_t existing_dist =                                                                 \
                zmap_dist(idx, m->capacity, m->buckets[idx].stored_hash, m->bits);                 \
            if (dist > existing_dist)                                                              \
            {                                                                                      \
                return NULL;                                                                       \
            }                                                                                      \
            if (m->buckets[idx].stored_hash == hash && 0 == m->cmp_func(m->buckets[idx].key, key)) \
            {                                                                                      \
                return m->buckets[idx].value;                                                      \
            }                                                                                      \
            idx = (idx + 1) & (m->capacity - 1);                                                   \
            dist++;                                                                                \
        }                                                                                          \
    }                                                                                              \
                                                                                                   \
    static inline void zmap_remove_stable_##Name(zmap_stable_##Name *m, KeyT key)                  \
    {                                                                                              \
        if (0 == m->count)                                                                         \
        {                                                                                          \
            return;                                                                                \
        }                                                                                          \
        uint32_t hash = m->hash_func(key, m->seed);                                                \
        size_t idx = zmap_fib_index(hash, m->bits);                                                \
        size_t dist = 0;                                                                           \
        for (;;)                                                                                   \
        {                                                                                          \
            if (ZMAP_EMPTY == m->buckets[idx].state)                                               \
            {                                                                                      \
                return;                                                                            \
            }                                                                                      \
            size_t existing_dist =                                                                 \
                zmap_dist(idx, m->capacity, m->buckets[idx].stored_hash, m->bits);                 \
            if (dist > existing_dist)                                                              \
            {                                                                                      \
                return;                                                                            \
            }                                                                                      \
            if (m->buckets[idx].stored_hash == hash && 0 == m->cmp_func(m->buckets[idx].key, key)) \
            {                                                                                      \
                zmap_remove_val_stable_##Name(m->buckets[idx].value);                              \
                m->count--;                                                                        \
                for (;;)                                                                           \
                {                                                                                  \
                    size_t next = (idx + 1) & (m->capacity - 1);                                   \
                    if (ZMAP_EMPTY == m->buckets[next].state)                                      \
                    {                                                                              \
                        m->buckets[idx].state = ZMAP_EMPTY;                                        \
                        return;                                                                    \
                    }                                                                              \
                    size_t next_dist =                                                             \
                        zmap_dist(next, m->capacity, m->buckets[next].stored_hash, m->bits);       \
                    if (0 == next_dist)                                                            \
                    {                                                                              \
                        m->buckets[idx].state = ZMAP_EMPTY;                                        \
                        return;                                                                    \
                    }                                                                              \
                    m->buckets[idx] = m->buckets[next];                                            \
                    idx = next;                                                                    \
                }                                                                                  \
            }                                                                                      \
            idx = (idx + 1) & (m->capacity - 1);                                                   \
            dist++;                                                                                \
        }                                                                                          \
    }                                                                                              \
                                                                                                   \
    static inline size_t zmap_size_stable_##Name(zmap_stable_##Name *m)                            \
    {                                                                                              \
        return m->count;                                                                           \
    }                                                                                              \
                                                                                                   \
    static inline zmap_iter_stable_##Name zmap_iter_init_stable_##Name(zmap_stable_##Name *m)      \
    {                                                                                              \
        return (zmap_iter_stable_##Name){m, 0};                                                    \
    }                                                                                              \
                                                                                                   \
    static inline bool zmap_iter_next_stable_##Name(zmap_iter_stable_##Name *it, KeyT *out_k,      \
                                                    ValT *out_v)                                   \
    {                                                                                              \
        if (!it->map || !it->map->buckets)                                                         \
        {                                                                                          \
            return false;                                                                          \
        }                                                                                          \
        while (it->index < it->map->capacity)                                                      \
        {                                                                                          \
            size_t i = it->index++;                                                                \
            if (ZMAP_OCCUPIED == it->map->buckets[i].state)                                        \
            {                                                                                      \
                if (out_k)                                                                         \
                {                                                                                  \
                    *out_k = it->map->buckets[i].key;                                              \
                }                                                                                  \
                if (out_v)                                                                         \
                {                                                                                  \
                    *out_v = *it->map->buckets[i].value;                                           \
                }                                                                                  \
                return true;                                                                       \
            }                                                                                      \
        }                                                                                          \
        return false;                                                                              \
    }

// Dispatch entries.
#define M_PUT_ENTRY(K, V, N) zmap_##N * : zmap_put_##N,
#define M_GET_ENTRY(K, V, N) zmap_##N * : zmap_get_##N,
#define M_REM_ENTRY(K, V, N) zmap_##N * : zmap_remove_##N,
#define M_FREE_ENTRY(K, V, N) zmap_##N * : zmap_free_##N,
#define M_SIZE_ENTRY(K, V, N) zmap_##N * : zmap_size_##N,
#define M_CLEAR_ENTRY(K, V, N) zmap_##N * : zmap_clear_##N,
#define M_SEED_ENTRY(K, V, N) zmap_##N * : zmap_set_seed_##N,
#define M_ITER_INIT(K, V, N) zmap_##N * : zmap_iter_init_##N,
#define M_ITER_NEXT(K, V, N) zmap_iter_##N * : zmap_iter_next_##N,

#define S_PUT_ENTRY(K, V, N) zmap_stable_##N * : zmap_put_stable_##N,
#define S_GET_ENTRY(K, V, N) zmap_stable_##N * : zmap_get_stable_##N,
#define S_REM_ENTRY(K, V, N) zmap_stable_##N * : zmap_remove_stable_##Name,
#define S_FREE_ENTRY(K, V, N) zmap_stable_##N * : zmap_free_stable_##Name,
#define S_SIZE_ENTRY(K, V, N) zmap_stable_##N * : zmap_size_stable_##Name,
#define S_CLEAR_ENTRY(K, V, N) zmap_stable_##N * : zmap_clear_stable_##Name,
#define S_SEED_ENTRY(K, V, N) zmap_stable_##N * : zmap_set_seed_stable_##Name,
#define S_ITER_INIT(K, V, N) zmap_stable_##N * : zmap_iter_init_stable_##Name,
#define S_ITER_NEXT(K, V, N) zmap_iter_stable_##N * : zmap_iter_next_stable_##Name,

#if Z_HAS_ZERROR
    static inline zres zmap_err_dummy(void *v, ...)
    {
        (void)v;
        return zres_err(zerr_create(-1, "Unknown Map Type"));
    }

#define M_PUT_SAFE_ENTRY(K, V, N) zmap_##N * : zmap_put_safe_##N,
#define M_GET_SAFE_ENTRY(K, V, N) zmap_##N * : zmap_get_safe_##N,
#endif

// Registry Hooks.
#ifndef REGISTER_ZMAP_TYPES
#if defined(__has_include) && __has_include("z_registry.h")
#include "z_registry.h"
#endif
#endif

#ifndef REGISTER_ZMAP_TYPES
#define REGISTER_ZMAP_TYPES(X)
#endif
#ifndef Z_AUTOGEN_MAPS
#define Z_AUTOGEN_MAPS(X)
#endif
#ifndef REGISTER_STABLE_MAPS
#define REGISTER_STABLE_MAPS(X)
#endif
#ifndef Z_AUTOGEN_STABLE_MAPS
#define Z_AUTOGEN_STABLE_MAPS(X)
#endif

#define Z_ALL_MAPS(X) Z_AUTOGEN_MAPS(X) REGISTER_ZMAP_TYPES(X)
#define Z_ALL_STABLE_MAPS(X) Z_AUTOGEN_STABLE_MAPS(X) REGISTER_STABLE_MAPS(X)

    Z_ALL_MAPS(ZMAP_GENERATE_IMPL)
    Z_ALL_STABLE_MAPS(ZMAP_GENERATE_STABLE_IMPL)

// API Macros.
#define zmap_init(Name, h, c) zmap_init_##Name(h, c)
#define zmap_init_stable(Name, h, c) zmap_init_stable_##Name(h, c)

#if defined(Z_HAS_CLEANUP) && Z_HAS_CLEANUP
#define zmap_autofree(Name) Z_CLEANUP(zmap_free_##Name) zmap_##Name
#define zmap_autofree_stable(Name) Z_CLEANUP(zmap_free_stable_##Name) zmap_stable_##Name
#endif

#define zmap_put(m, k, v)                                                                          \
    _Generic((m), Z_ALL_MAPS(M_PUT_ENTRY) Z_ALL_STABLE_MAPS(S_PUT_ENTRY) default: 0)(m, k, v)
#define zmap_get(m, k)                                                                             \
    _Generic((m), Z_ALL_MAPS(M_GET_ENTRY) Z_ALL_STABLE_MAPS(S_GET_ENTRY) default: (void *)0)(m, k)
#define zmap_remove(m, k)                                                                          \
    _Generic((m), Z_ALL_MAPS(M_REM_ENTRY) Z_ALL_STABLE_MAPS(S_REM_ENTRY) default: (void)0)(m, k)
#define zmap_free(m)                                                                               \
    _Generic((m), Z_ALL_MAPS(M_FREE_ENTRY) Z_ALL_STABLE_MAPS(S_FREE_ENTRY) default: (void)0)(m)
#define zmap_size(m)                                                                               \
    _Generic((m), Z_ALL_MAPS(M_SIZE_ENTRY) Z_ALL_STABLE_MAPS(S_SIZE_ENTRY) default: 0)(m)
#define zmap_clear(m)                                                                              \
    _Generic((m), Z_ALL_MAPS(M_CLEAR_ENTRY) Z_ALL_STABLE_MAPS(S_CLEAR_ENTRY) default: (void)0)(m)
#define zmap_set_seed(m, s)                                                                        \
    _Generic((m), Z_ALL_MAPS(M_SEED_ENTRY) Z_ALL_STABLE_MAPS(S_SEED_ENTRY) default: (void)0)(m, s)

#if Z_HAS_ZERROR
#define zmap_put_safe(m, k, v)                                                                     \
    _Generic((m), Z_ALL_MAPS(M_PUT_SAFE_ENTRY) default: zmap_err_dummy)(m, k, v, __FILE__,         \
                                                                        __LINE__, __func__)
#define zmap_get_safe(m, k)                                                                        \
    _Generic((m), Z_ALL_MAPS(M_GET_SAFE_ENTRY) default: zmap_err_dummy)(m, k, __FILE__, __LINE__,  \
                                                                        __func__)
#endif

// Iterators.
#define zmap_iter_init(Name, m)                                                                    \
    _Generic((m), Z_ALL_MAPS(M_ITER_INIT) Z_ALL_STABLE_MAPS(S_ITER_INIT) default: 0)(m)
#define zmap_iter_next(it, k, v)                                                                   \
    _Generic((it), Z_ALL_MAPS(M_ITER_NEXT) Z_ALL_STABLE_MAPS(S_ITER_NEXT) default: false)(it, k, v)

/* * zmap_foreach(Name, m, k_ptr, v_ptr)
 * Iterates over the map. k_ptr and v_ptr are assigned pointers to key and value.
 * v_ptr will be `ValT*` for standard maps and `ValT*` (dereferenced from internal ptr) for stable
 * maps.
 */
#define zmap_foreach(Name, m, k_ptr, v_ptr)                                                        \
    for (size_t _i_##Name = 0; _i_##Name < (m)->capacity; ++_i_##Name)                             \
        if ((m)->buckets[_i_##Name].state == ZMAP_OCCUPIED &&                                      \
            ((k_ptr) = &(m)->buckets[_i_##Name].key) &&                                            \
            ((v_ptr) = (void *)&(m)->buckets[_i_##Name].value))

// Optional short names.
#ifdef ZMAP_SHORT_NAMES
#define map(Name) zmap_##Name
#define map_stable(Name) zmap_stable_##Name
#define map_init zmap_init
#define map_init_stable zmap_init_stable
#define map_autofree zmap_autofree
#define map_autofree_stable zmap_autofree_stable
#define map_put zmap_put
#define map_get zmap_get
#define map_remove zmap_remove
#define map_free zmap_free
#define map_size zmap_size
#define map_clear zmap_clear
#define map_set_seed zmap_set_seed

#define map_iter_init zmap_iter_init
#define map_iter_next zmap_iter_next

#if Z_HAS_ZERROR
#define map_put_safe zmap_put_safe
#define map_get_safe zmap_get_safe
#endif
#endif

#ifdef __cplusplus
} // extern "C"

namespace z_map
{
#define ZMAP_CPP_TRAITS(Key, Val, Name)                                                            \
    template <> struct traits<Key, Val>                                                            \
    {                                                                                              \
        using map_type = ::zmap_##Name;                                                            \
        using bucket_type = ::zmap_bucket_##Name;                                                  \
        static constexpr auto init = ::zmap_init_ext_##Name;                                       \
        static constexpr auto put = ::zmap_put_##Name;                                             \
        static constexpr auto get = ::zmap_get_##Name;                                             \
        static constexpr auto remove = ::zmap_remove_##Name;                                       \
        static constexpr auto clear = ::zmap_clear_##Name;                                         \
        static constexpr auto free = ::zmap_free_##Name;                                           \
        static constexpr auto set_seed = ::zmap_set_seed_##Name;                                   \
    };

Z_ALL_MAPS(ZMAP_CPP_TRAITS)
} // namespace z_map
#endif // __cplusplus

#endif // ZMAP_H
