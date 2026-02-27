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
size_t _z_check_bounds(size_t index, size_t size)
{
    if (index >= size)
    {
        fprintf(stderr, "Index out of bounds: %zu >= %zu\n", index, size);
        exit(1);
    }
    return index;
}
void yield(const char *s)
{
    printf("%s", s);
}
void code(const char *s)
{
    printf("%s", s);
}
void compile_error(const char *s)
{
    fprintf(stderr, "Compile-time error: %s\n", s);
    exit(1);
}
void compile_warn(const char *s)
{
    fprintf(stderr, "Compile-time warning: %s\n", s);
}
#define __COMPTIME_TARGET__ "linux"
#define __COMPTIME_FILE__ "tests/language/functions/test_attributes.zc"
int main()
{
    ZC_AUTO val = comp_helper();
    assert((val == 123), "Comptime check failed!");
    return 0;
}
