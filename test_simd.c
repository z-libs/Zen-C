#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <fcntl.h>
#define ZC_SIMD(T, N) T __attribute__((vector_size(N * sizeof(T))))
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202300L
#define ZC_AUTO auto
#define ZC_AUTO_INIT(var, init) auto var = (init)
#else
#define ZC_AUTO __auto_type
#define ZC_AUTO_INIT(var, init) __auto_type var = (init)
#endif
#define ZC_CAST(T, x) ((T)(x))
#ifdef __TINYC__
#undef ZC_AUTO
#define ZC_AUTO __auto_type
#undef ZC_AUTO_INIT
#define ZC_AUTO_INIT(var, init) __typeof__((init)) var = (init)

#ifndef __builtin_expect
#define __builtin_expect(x, v) (x)
#endif

#ifndef __builtin_unreachable
#define __builtin_unreachable()
#endif
#else
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202300L
#define ZC_AUTO_INIT(var, init) auto var = (init)
#else
#define ZC_AUTO_INIT(var, init) __auto_type var = (init)
#endif
#endif
static inline const char *_z_bool_str(_Bool b)
{
    return b ? "true" : "false";
}
#ifdef __OBJC__
#define _z_objc_map , id : "%s", Class : "%s", SEL : "%s"
#define _z_objc_arg_map(x)                                                                         \
    , id : [(id)(x) description].UTF8String, Class : class_getName((Class)(x)),                    \
                                                     SEL : sel_getName((SEL)(x))
#else
#define _z_objc_map
#define _z_objc_arg_map(x)
#endif

#define _z_str(x)                                                                                  \
    _Generic((x),                                                                                  \
        _Bool: "%s",                                                                               \
        char: "%c",                                                                                \
        signed char: "%c",                                                                         \
        unsigned char: "%u",                                                                       \
        short: "%d",                                                                               \
        unsigned short: "%u",                                                                      \
        int: "%d",                                                                                 \
        unsigned int: "%u",                                                                        \
        long: "%ld",                                                                               \
        unsigned long: "%lu",                                                                      \
        long long: "%lld",                                                                         \
        unsigned long long: "%llu",                                                                \
        float: "%f",                                                                               \
        double: "%f",                                                                              \
        char *: "%s",                                                                              \
        const char *: "%s",                                                                        \
        void *: "%p" _z_objc_map)
#define _z_arg(x) _Generic((x), _Bool: _z_bool_str(x) _z_objc_arg_map(x), default: (x))
typedef size_t usize;
typedef char *string;
typedef struct
{
    void *func;
    void *ctx;
} z_closure_T;
typedef void U0;
typedef int8_t I8;
typedef uint8_t U8;
typedef int16_t I16;
typedef uint16_t U16;
typedef int32_t I32;
typedef uint32_t U32;
typedef int64_t I64;
typedef uint64_t U64;
#define F32 float
#define F64 double
#define z_malloc malloc
#define z_realloc realloc
#define z_free free
#define z_print printf
void z_panic(const char *msg)
{
    fprintf(stderr, "Panic: %s\n", msg);
    exit(1);
}
#if defined(__APPLE__)
#define _ZC_SEC __attribute__((used, section("__DATA,__zarch")))
#elif defined(_WIN32)
#define _ZC_SEC __attribute__((used))
#else
#define _ZC_SEC __attribute__((used, section(".note.zarch")))
#endif
static const unsigned char _zc_abi_v1[] _ZC_SEC = {0x07, 0xd5, 0x59, 0x30, 0x7c, 0x7f, 0x66,
                                                   0x75, 0x30, 0x69, 0x7f, 0x65, 0x3c, 0x30,
                                                   0x59, 0x7c, 0x79, 0x7e, 0x73, 0x71};
void _z_autofree_impl(void *p)
{
    void **pp = (void **)p;
    if (*pp)
    {
        z_free(*pp);
        *pp = NULL;
    }
}
#define assert(cond, ...)                                                                          \
    if (!(cond))                                                                                   \
    {                                                                                              \
        fprintf(stderr, "Assertion failed: " __VA_ARGS__);                                         \
        exit(1);                                                                                   \
    }
string _z_readln_raw()
{
    size_t cap = 64;
    size_t len = 0;
    char *line = z_malloc(cap);
    if (!line)
    {
        return NULL;
    }
    int c;
    while ((c = fgetc(stdin)) != EOF)
    {
        if (c == '\n')
        {
            break;
        }
        if (len + 1 >= cap)
        {
            cap *= 2;
            char *n = z_realloc(line, cap);
            if (!n)
            {
                z_free(line);
                return NULL;
            }
            line = n;
        }
        line[len++] = c;
    }
    if (len == 0 && c == EOF)
    {
        z_free(line);
        return NULL;
    }
    line[len] = 0;
    return line;
}
int _z_scan_helper(const char *fmt, ...)
{
    char *l = _z_readln_raw();
    if (!l)
    {
        return 0;
    }
    va_list ap;
    va_start(ap, fmt);
    int r = vsscanf(l, fmt, ap);
    va_end(ap);
    z_free(l);
    return r;
}
int _z_orig_stdout = -1;
void _z_suppress_stdout()
{
    fflush(stdout);
    if (_z_orig_stdout == -1)
    {
        _z_orig_stdout = dup(STDOUT_FILENO);
    }
    int nullfd = open("/dev/null", O_WRONLY);
    dup2(nullfd, STDOUT_FILENO);
    close(nullfd);
}
void _z_restore_stdout()
{
    fflush(stdout);
    if (_z_orig_stdout != -1)
    {
        dup2(_z_orig_stdout, STDOUT_FILENO);
        close(_z_orig_stdout);
        _z_orig_stdout = -1;
    }
}
typedef char *string;
typedef struct
{
    void **data;
    int len;
    int cap;
} Vec;
#define Vec_new() (Vec){.data = 0, .len = 0, .cap = 0}
void _z_vec_push(Vec *v, void *item)
{
    if (v->len >= v->cap)
    {
        v->cap = v->cap ? v->cap * 2 : 8;
        v->data = z_realloc(v->data, v->cap * sizeof(void *));
    }
    v->data[v->len++] = item;
}
static inline Vec _z_make_vec(int count, ...)
{
    Vec v = {0};
    v.cap = count > 8 ? count : 8;
    v.data = z_malloc(v.cap * sizeof(void *));
    v.len = 0;
    va_list args;
    va_start(args, count);
    for (int i = 0; i < count; i++)
    {
        v.data[v.len++] = va_arg(args, void *);
    }
    va_end(args);
    return v;
}
#define Vec_push(v, i) _z_vec_push(&(v), (void *)(long)(i))
static inline long _z_check_bounds(long index, long limit)
{
    if (index < 0 || index >= limit)
    {
        fprintf(stderr, "Index out of bounds: %ld (limit %ld)\n", index, limit);
        exit(1);
    }
    return index;
}

typedef ZC_SIMD(uint8_t, 16) u8x16;
typedef ZC_SIMD(int8_t, 16) i8x16;
typedef ZC_SIMD(int32_t, 8) i32x8;
typedef ZC_SIMD(float, 8) f32x8;
typedef ZC_SIMD(int32_t, 4) i32x4;
typedef ZC_SIMD(float, 4) f32x4;
int main(void);

int main(void)
{
    {
        f32x4 a = (f32x4){1.000000, 1.000000, 1.000000, 1.000000};
        f32x4 b = (f32x4){2.000000, 2.000000, 2.000000, 2.000000};
        f32x4 c = (a + b);
        float c0 = c[0];
        ({
            fprintf(stdout, "%s", "f32x4 addition result[0]: ");
            fprintf(stdout, "%f", c0);
            fprintf(stdout, "%s", " (expected 3.0)");
            fprintf(stdout, "\n");
            0;
        });
        assert((c0 == 3.000000), "SIMD addition failed");
        i32x4 x = (i32x4){255, 255, 255, 255};
        i32x4 y = (i32x4){240, 240, 240, 240};
        i32x4 z = (x & y);
        int32_t z0 = z[0];
        ({
            fprintf(stdout, "%s", "i32x4 bitwise AND result[0]: ");
            fprintf(stdout, "%d", z0);
            fprintf(stdout, "%s", " (expected 240)");
            fprintf(stdout, "\n");
            0;
        });
        assert((z0 == 240), "SIMD bitwise AND failed");
        f32x4 d = (a * b);
        float d0 = d[0];
        ({
            fprintf(stdout, "%s", "f32x4 multiplication result[0]: ");
            fprintf(stdout, "%f", d0);
            fprintf(stdout, "%s", " (expected 2.0)");
            fprintf(stdout, "\n");
            0;
        });
        assert((d0 == 2.000000), "SIMD multiplication failed");
        ({
            fprintf(stdout, "%s", "PASS: SIMD verification successful");
            fprintf(stdout, "\n");
            0;
        });
    }
}
