#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdbool.h>
#ifdef __has_builtin
#if __has_builtin(__builtin_pow)
#define _zc_pow __builtin_pow
#endif
#endif
#ifndef _zc_pow
extern double pow(double, double);
#define _zc_pow pow
#endif
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
static inline const char* _z_bool_str(_Bool b) { return b ? "true" : "false"; }
#ifdef __SIZEOF_INT128__
static inline const char *_z_u128_str(unsigned __int128 val) {
    static _Thread_local char buf[40];
    if (val == 0) return "0";
    int i = 38;
    buf[39] = 0;
    while (val > 0) { buf[i--] = (char)((val % 10) + '0'); val /= 10; }
    return &buf[i + 1];
}
static inline const char *_z_i128_str(__int128 val) {
    static _Thread_local char buf[41];
    if (val == 0) return "0";
    int neg = val < 0;
    unsigned __int128 uval = neg ? -val : val;
    int i = 39;
    buf[40] = 0;
    while (uval > 0) { buf[i--] = (char)((uval % 10) + '0'); uval /= 10; }
    if (neg) buf[i--] = '-';
    return &buf[i + 1];
}
#define _z_128_map ,__int128: "%s", unsigned __int128: "%s"
#else
#define _z_128_map
#endif
#ifdef __OBJC__
#define _z_objc_map ,id: "%s", Class: "%s", SEL: "%s"
#define _z_objc_arg_map(x) ,id: [(id)(x) description].UTF8String, Class: class_getName((Class)(x)), SEL: sel_getName((SEL)(x))
#else
#define _z_objc_map
#define _z_objc_arg_map(x)
#endif

#define _z_str(x) _Generic((x), _Bool: "%s", char: "%c", signed char: "%c", unsigned char: "%u", short: "%d", unsigned short: "%u", int: "%d", unsigned int: "%u", long: "%ld", unsigned long: "%lu", long long: "%lld", unsigned long long: "%llu", float: "%f", double: "%f", char*: "%s", const char*: "%s", void*: "%p" _z_128_map _z_objc_map)
#ifdef __SIZEOF_INT128__
#define _z_safe_i128(x) _Generic((x), __int128: (x), default: (__int128)0)
#define _z_safe_u128(x) _Generic((x), unsigned __int128: (x), default: (unsigned __int128)0)
#define _z_128_arg_map(x) ,__int128: _z_i128_str(_z_safe_i128(x)), unsigned __int128: _z_u128_str(_z_safe_u128(x))
#else
#define _z_128_arg_map(x)
#endif
#define _z_safe_bool(x) _Generic((x), _Bool: (x), default: (_Bool)0)
#define _z_arg(x) _Generic((x), _Bool: _z_bool_str(_z_safe_bool(x)) _z_128_arg_map(x) _z_objc_arg_map(x), default: (x))
typedef size_t usize;
typedef char* string;
typedef struct { void *func; void *ctx; void (*drop)(void*); } z_closure_T;
static void *_z_closure_ctx_stash[256];
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
void __zenc_panic(const char* msg) { fprintf(stderr, "Panic: %s\n", msg); exit(1); }
#if defined(__APPLE__)
#define _ZC_SEC __attribute__((used,section("__DATA,__zarch")))
#elif defined(_WIN32)
#define _ZC_SEC __attribute__((used))
#else
#define _ZC_SEC __attribute__((used,section(".note.zarch")))
#endif
static const unsigned char _zc_abi_v1[] _ZC_SEC = {0x07,0xd5,0x59,0x30,0x7c,0x7f,0x66,0x75,0x30,0x69,0x7f,0x65,0x3c,0x30,0x59,0x7c,0x79,0x7e,0x73,0x71};
void _z_autofree_impl(void *p) { void **pp = (void**)p; if(*pp) { z_free(*pp); *pp = NULL; } }
#define __zenc_assert(cond, ...) if (!(cond)) { fprintf(stderr, "\"Assertion failed: \" " __VA_ARGS__); exit(1); }
string _z_readln_raw() { size_t cap = 64; size_t len = 0; char *line = z_malloc(cap); if(!line) return NULL; int c; while((c = fgetc(stdin)) != EOF) { if(c == '\n') break; if(len + 1 >= cap) { cap *= 2; char *n = z_realloc(line, cap); if(!n) { z_free(line); return NULL; } line = n; } line[len++] = c; } if(len == 0 && c == EOF) { z_free(line); return NULL; } line[len] = 0; return line; }
int _z_scan_helper(const char *fmt, ...) { char *l = _z_readln_raw(); if(!l) return 0; va_list ap; va_start(ap, fmt); int r = vsscanf(l, fmt, ap); va_end(ap); z_free(l); return r; }
int _z_orig_stdout = -1;
void _z_suppress_stdout() {
    fflush(stdout);
    if (_z_orig_stdout == -1) _z_orig_stdout = dup(STDOUT_FILENO);
    int nullfd = open("/dev/null", O_WRONLY);
    dup2(nullfd, STDOUT_FILENO);
    close(nullfd);
}
void _z_restore_stdout() {
    fflush(stdout);
    if (_z_orig_stdout != -1) {
        dup2(_z_orig_stdout, STDOUT_FILENO);
        close(_z_orig_stdout);
        _z_orig_stdout = -1;
    }
}
#ifndef ZC_CFG_linux
#define ZC_CFG_linux 1
#endif
#include "std/third-party/tre/include/tre.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdarg.h>
typedef char* string;
typedef struct { void **data; int len; int cap; } Vec;
#define Vec_new() (Vec){.data=0, .len=0, .cap=0}
void _z_vec_push(Vec *v, void *item) { if(v->len >= v->cap) { v->cap = v->cap?v->cap*2:8; v->data = z_realloc(v->data, v->cap * sizeof(void*)); } v->data[v->len++] = item; }
static inline Vec _z_make_vec(int count, ...) { Vec v = {0}; v.cap = count > 8 ? count : 8; v.data = z_malloc(v.cap * sizeof(void*)); v.len = 0; va_list args; va_start(args, count); for(int i=0; i<count; i++) { v.data[v.len++] = va_arg(args, void*); } va_end(args); return v; }
#define Vec_push(v, i) _z_vec_push(&(v), (void*)(uintptr_t)(i))
static inline long _z_check_bounds(long index, long limit) { if(index < 0 || index >= limit) { fprintf(stderr, "Index out of bounds: %ld (limit %ld)\n", index, limit); exit(1); } return index; }
typedef struct VecIterResult__String VecIterResult__String;
typedef struct VecIterRef__String VecIterRef__String;
typedef struct VecIter__String VecIter__String;
typedef struct Vec__String Vec__String;
typedef struct VecIterResult__int32_t VecIterResult__int32_t;
typedef struct VecIterRef__int32_t VecIterRef__int32_t;
typedef struct VecIter__int32_t VecIter__int32_t;
typedef struct Vec__int32_t Vec__int32_t;
typedef struct VecIterResult__size_t VecIterResult__size_t;
typedef struct VecIterRef__size_t VecIterRef__size_t;
typedef struct VecIter__size_t VecIter__size_t;
typedef struct Vec__size_t Vec__size_t;
typedef struct Option__size_t Option__size_t;
typedef struct Option__int32_t Option__int32_t;
typedef struct VecIterResult__char VecIterResult__char;
typedef struct VecIterRef__char VecIterRef__char;
typedef struct Option__char Option__char;
typedef struct VecIter__char VecIter__char;
typedef struct Vec__char Vec__char;
typedef struct Regex Regex;
typedef struct ZRegMatch ZRegMatch;
typedef struct Match Match;
typedef struct StringCharsIter StringCharsIter;
typedef struct Sort Sort;
typedef struct Option__Match Option__Match;
typedef struct String String;
typedef struct Option__String Option__String;


typedef tre_str_source tre_str_source;
typedef regamatch_t regamatch_t;
typedef regaparams_t regaparams_t;
typedef reg_errcode_t reg_errcode_t;
typedef regmatch_t regmatch_t;
typedef regex_t regex_t;
typedef struct Clone_VTable {
    void* (*clone)(void* self);
} Clone_VTable;
typedef struct Clone { void *self; Clone_VTable *vtable; } Clone;

typedef struct Copy_VTable {
} Copy_VTable;
typedef struct Copy { void *self; Copy_VTable *vtable; } Copy;

typedef struct Drop_VTable {
    void (*drop)(void* self);
} Drop_VTable;
typedef struct Drop { void *self; Drop_VTable *vtable; } Drop;


    #define ZC_IMPL_SORT(T) \
    static int _z_partition_##T(T* arr, int low, int high) { \
        T pivot = arr[high]; \
        int i = low - 1; \
        for (int j = low; j < high; j++) { \
            if (arr[j] < pivot) { \
                i++; \
                T temp = arr[i]; \
                arr[i] = arr[j]; \
                arr[j] = temp; \
            } \
        } \
        T temp2 = arr[i + 1]; \
        arr[i + 1] = arr[high]; \
        arr[high] = temp2; \
        return i + 1; \
    } \
    static void _z_quick_sort_##T(T* arr, int low, int high) { \
        if (low < high) { \
            int pivot_idx = _z_partition_##T(arr, low, high); \
            _z_quick_sort_##T(arr, low, pivot_idx - 1); \
            _z_quick_sort_##T(arr, pivot_idx + 1, high); \
        } \
    } \
    void sort_##T(T* arr, size_t len) { \
        if (len > 1) { \
            _z_quick_sort_##T(arr, 0, (int)len - 1); \
        } \
    }

    // Pre-declare standard library types
    ZC_IMPL_SORT(int)
    ZC_IMPL_SORT(long)
    ZC_IMPL_SORT(float)
    ZC_IMPL_SORT(double)

struct VecIterResult__String {
    String* ptr;
};

struct VecIterRef__String {
    String* data;
    size_t count;
    size_t idx;
};

struct VecIter__String {
    String* data;
    size_t count;
    size_t idx;
};

struct Vec__String {
    String* data;
    size_t len;
    size_t cap;
};

struct VecIterResult__int32_t {
    int32_t* ptr;
};

struct VecIterRef__int32_t {
    int32_t* data;
    size_t count;
    size_t idx;
};

struct VecIter__int32_t {
    int32_t* data;
    size_t count;
    size_t idx;
};

struct Vec__int32_t {
    int32_t* data;
    size_t len;
    size_t cap;
};

struct VecIterResult__size_t {
    size_t* ptr;
};

struct VecIterRef__size_t {
    size_t* data;
    size_t count;
    size_t idx;
};

struct VecIter__size_t {
    size_t* data;
    size_t count;
    size_t idx;
};

struct Vec__size_t {
    size_t* data;
    size_t len;
    size_t cap;
};

struct Option__size_t {
    bool is_some;
    size_t val;
};

struct Option__int32_t {
    bool is_some;
    int32_t val;
};

struct VecIterResult__char {
    char* ptr;
};

struct VecIterRef__char {
    char* data;
    size_t count;
    size_t idx;
};

struct Option__char {
    bool is_some;
    char val;
};

struct VecIter__char {
    char* data;
    size_t count;
    size_t idx;
};

struct Vec__char {
    char* data;
    size_t len;
    size_t cap;
};

struct Regex {
    void* preg;
    char* pattern;
    int32_t flags;
};

struct ZRegMatch {
    int32_t rm_so;
    int32_t rm_eo;
};

struct Match {
    char* text;
    int32_t start;
    int32_t len;
};

struct StringCharsIter {
    char* data;
    size_t len;
    size_t pos;
};

struct Sort {
    char _placeholder;
};

struct Option__Match {
    bool is_some;
    Match val;
};

struct String {
    Vec__char vec;
};

struct Option__String {
    bool is_some;
    String val;
};


    // Include TRE implementation for static linking.
    // This ensures linking works without precompiled libraries across all platforms (Windows/Linux/macOS).
    #include "std/third-party/tre/tre_full.c"

Clone Clone__clone(Clone* self) {
    void* res = self->vtable->clone(self->self);
    return (Clone){.self = res, .vtable = self->vtable};
}



void Drop__drop(Drop* self) {
    return self->vtable->drop(self->self);
}


Option__Match Option__Match__Some(Match v);
Option__Match Option__Match__None(void);
bool Option__Match__is_some(Option__Match* self);
bool Option__Match__is_none(Option__Match* self);
void Option__Match__forget(Option__Match* self);
Match Option__Match__unwrap(Option__Match* self);
Match* Option__Match__unwrap_ref(Option__Match* self);
Match Option__Match__unwrap_or(Option__Match* self, Match def_val);
Match Option__Match__expect(Option__Match* self, char* msg);
Option__Match Option__Match__or_else(Option__Match* self, Option__Match other);
Vec__String Vec__String__new(void);
Vec__String Vec__String__with_capacity(size_t cap);
void Vec__String__grow(Vec__String* self);
void Vec__String__grow_to_fit(Vec__String* self, size_t new_len);
VecIter__String Vec__String__iterator(Vec__String* self);
VecIterRef__String Vec__String__iter_ref(Vec__String* self);
void Vec__String__push(Vec__String* self, String item);
void Vec__String__insert(Vec__String* self, size_t idx, String item);
String Vec__String__pop(Vec__String* self);
Option__String Vec__String__pop_opt(Vec__String* self);
String Vec__String__remove(Vec__String* self, size_t idx);
void Vec__String__append(Vec__String* self, Vec__String other);
String Vec__String__get(Vec__String* self, size_t idx);
String Vec__String__index(Vec__String* self, size_t idx);
String* Vec__String__get_ref(Vec__String* self, size_t idx);
String Vec__String__last(Vec__String* self);
size_t Vec__String__length(Vec__String* self);
bool Vec__String__contains(Vec__String* self, String item);
bool Vec__String__is_empty(Vec__String* self);
void Vec__String__clear(Vec__String* self);
void Vec__String__free(Vec__String* self);
String Vec__String__first(Vec__String* self);
void Vec__String__set(Vec__String* self, size_t idx, String item);
void Vec__String__reverse(Vec__String* self);
bool Vec__String__eq(Vec__String* self, Vec__String* other);
void Vec__String__forget(Vec__String* self);
Vec__String Vec__String__add(Vec__String* self, Vec__String* other);
void Vec__String__add_assign(Vec__String* self, Vec__String* other);
bool Vec__String__neq(Vec__String* self, Vec__String* other);
void Vec__String__shl(Vec__String* self, String item);
void Vec__String__shr(Vec__String* self, String* out_item);
Vec__String Vec__String__mul(Vec__String* self, size_t count);
void Vec__String__mul_assign(Vec__String* self, size_t count);
Vec__String Vec__String__clone(Vec__String* self);
VecIterResult__String VecIterRef__String__next(VecIterRef__String* self);
VecIterRef__String VecIterRef__String__iterator(VecIterRef__String* self);
bool VecIterResult__String__is_none(VecIterResult__String* self);
String* VecIterResult__String__unwrap(VecIterResult__String* self);
Option__String VecIter__String__next(VecIter__String* self);
VecIter__String VecIter__String__iterator(VecIter__String* self);
Option__String Option__String__Some(String v);
Option__String Option__String__None(void);
bool Option__String__is_some(Option__String* self);
bool Option__String__is_none(Option__String* self);
void Option__String__forget(Option__String* self);
String Option__String__unwrap(Option__String* self);
String* Option__String__unwrap_ref(Option__String* self);
String Option__String__unwrap_or(Option__String* self, String def_val);
String Option__String__expect(Option__String* self, char* msg);
Option__String Option__String__or_else(Option__String* self, Option__String other);
void Vec__String__Drop__drop(Vec__String* self);
Vec__int32_t Vec__int32_t__new(void);
Vec__int32_t Vec__int32_t__with_capacity(size_t cap);
void Vec__int32_t__grow(Vec__int32_t* self);
void Vec__int32_t__grow_to_fit(Vec__int32_t* self, size_t new_len);
VecIter__int32_t Vec__int32_t__iterator(Vec__int32_t* self);
VecIterRef__int32_t Vec__int32_t__iter_ref(Vec__int32_t* self);
void Vec__int32_t__push(Vec__int32_t* self, int32_t item);
void Vec__int32_t__insert(Vec__int32_t* self, size_t idx, int32_t item);
int32_t Vec__int32_t__pop(Vec__int32_t* self);
Option__int32_t Vec__int32_t__pop_opt(Vec__int32_t* self);
int32_t Vec__int32_t__remove(Vec__int32_t* self, size_t idx);
void Vec__int32_t__append(Vec__int32_t* self, Vec__int32_t other);
int32_t Vec__int32_t__get(Vec__int32_t* self, size_t idx);
int32_t Vec__int32_t__index(Vec__int32_t* self, size_t idx);
int32_t* Vec__int32_t__get_ref(Vec__int32_t* self, size_t idx);
int32_t Vec__int32_t__last(Vec__int32_t* self);
size_t Vec__int32_t__length(Vec__int32_t* self);
bool Vec__int32_t__contains(Vec__int32_t* self, int32_t item);
bool Vec__int32_t__is_empty(Vec__int32_t* self);
void Vec__int32_t__clear(Vec__int32_t* self);
void Vec__int32_t__free(Vec__int32_t* self);
int32_t Vec__int32_t__first(Vec__int32_t* self);
void Vec__int32_t__set(Vec__int32_t* self, size_t idx, int32_t item);
void Vec__int32_t__reverse(Vec__int32_t* self);
bool Vec__int32_t__eq(Vec__int32_t* self, Vec__int32_t* other);
void Vec__int32_t__forget(Vec__int32_t* self);
Vec__int32_t Vec__int32_t__add(Vec__int32_t* self, Vec__int32_t* other);
void Vec__int32_t__add_assign(Vec__int32_t* self, Vec__int32_t* other);
bool Vec__int32_t__neq(Vec__int32_t* self, Vec__int32_t* other);
void Vec__int32_t__shl(Vec__int32_t* self, int32_t item);
void Vec__int32_t__shr(Vec__int32_t* self, int32_t* out_item);
Vec__int32_t Vec__int32_t__mul(Vec__int32_t* self, size_t count);
void Vec__int32_t__mul_assign(Vec__int32_t* self, size_t count);
Vec__int32_t Vec__int32_t__clone(Vec__int32_t* self);
VecIterResult__int32_t VecIterRef__int32_t__next(VecIterRef__int32_t* self);
VecIterRef__int32_t VecIterRef__int32_t__iterator(VecIterRef__int32_t* self);
bool VecIterResult__int32_t__is_none(VecIterResult__int32_t* self);
int32_t* VecIterResult__int32_t__unwrap(VecIterResult__int32_t* self);
Option__int32_t VecIter__int32_t__next(VecIter__int32_t* self);
VecIter__int32_t VecIter__int32_t__iterator(VecIter__int32_t* self);
void Vec__int32_t__Drop__drop(Vec__int32_t* self);
Vec__size_t Vec__size_t__new(void);
Vec__size_t Vec__size_t__with_capacity(size_t cap);
void Vec__size_t__grow(Vec__size_t* self);
void Vec__size_t__grow_to_fit(Vec__size_t* self, size_t new_len);
VecIter__size_t Vec__size_t__iterator(Vec__size_t* self);
VecIterRef__size_t Vec__size_t__iter_ref(Vec__size_t* self);
void Vec__size_t__push(Vec__size_t* self, size_t item);
void Vec__size_t__insert(Vec__size_t* self, size_t idx, size_t item);
size_t Vec__size_t__pop(Vec__size_t* self);
Option__size_t Vec__size_t__pop_opt(Vec__size_t* self);
size_t Vec__size_t__remove(Vec__size_t* self, size_t idx);
void Vec__size_t__append(Vec__size_t* self, Vec__size_t other);
size_t Vec__size_t__get(Vec__size_t* self, size_t idx);
size_t Vec__size_t__index(Vec__size_t* self, size_t idx);
size_t* Vec__size_t__get_ref(Vec__size_t* self, size_t idx);
size_t Vec__size_t__last(Vec__size_t* self);
size_t Vec__size_t__length(Vec__size_t* self);
bool Vec__size_t__contains(Vec__size_t* self, size_t item);
bool Vec__size_t__is_empty(Vec__size_t* self);
void Vec__size_t__clear(Vec__size_t* self);
void Vec__size_t__free(Vec__size_t* self);
size_t Vec__size_t__first(Vec__size_t* self);
void Vec__size_t__set(Vec__size_t* self, size_t idx, size_t item);
void Vec__size_t__reverse(Vec__size_t* self);
bool Vec__size_t__eq(Vec__size_t* self, Vec__size_t* other);
void Vec__size_t__forget(Vec__size_t* self);
Vec__size_t Vec__size_t__add(Vec__size_t* self, Vec__size_t* other);
void Vec__size_t__add_assign(Vec__size_t* self, Vec__size_t* other);
bool Vec__size_t__neq(Vec__size_t* self, Vec__size_t* other);
void Vec__size_t__shl(Vec__size_t* self, size_t item);
void Vec__size_t__shr(Vec__size_t* self, size_t* out_item);
Vec__size_t Vec__size_t__mul(Vec__size_t* self, size_t count);
void Vec__size_t__mul_assign(Vec__size_t* self, size_t count);
Vec__size_t Vec__size_t__clone(Vec__size_t* self);
VecIterResult__size_t VecIterRef__size_t__next(VecIterRef__size_t* self);
VecIterRef__size_t VecIterRef__size_t__iterator(VecIterRef__size_t* self);
bool VecIterResult__size_t__is_none(VecIterResult__size_t* self);
size_t* VecIterResult__size_t__unwrap(VecIterResult__size_t* self);
Option__size_t VecIter__size_t__next(VecIter__size_t* self);
VecIter__size_t VecIter__size_t__iterator(VecIter__size_t* self);
void Vec__size_t__Drop__drop(Vec__size_t* self);
Option__size_t Option__size_t__Some(size_t v);
Option__size_t Option__size_t__None(void);
bool Option__size_t__is_some(Option__size_t* self);
bool Option__size_t__is_none(Option__size_t* self);
void Option__size_t__forget(Option__size_t* self);
size_t Option__size_t__unwrap(Option__size_t* self);
size_t* Option__size_t__unwrap_ref(Option__size_t* self);
size_t Option__size_t__unwrap_or(Option__size_t* self, size_t def_val);
size_t Option__size_t__expect(Option__size_t* self, char* msg);
Option__size_t Option__size_t__or_else(Option__size_t* self, Option__size_t other);
Option__int32_t Option__int32_t__Some(int32_t v);
Option__int32_t Option__int32_t__None(void);
bool Option__int32_t__is_some(Option__int32_t* self);
bool Option__int32_t__is_none(Option__int32_t* self);
void Option__int32_t__forget(Option__int32_t* self);
int32_t Option__int32_t__unwrap(Option__int32_t* self);
int32_t* Option__int32_t__unwrap_ref(Option__int32_t* self);
int32_t Option__int32_t__unwrap_or(Option__int32_t* self, int32_t def_val);
int32_t Option__int32_t__expect(Option__int32_t* self, char* msg);
Option__int32_t Option__int32_t__or_else(Option__int32_t* self, Option__int32_t other);
Vec__char Vec__char__new(void);
Vec__char Vec__char__with_capacity(size_t cap);
void Vec__char__grow(Vec__char* self);
void Vec__char__grow_to_fit(Vec__char* self, size_t new_len);
VecIter__char Vec__char__iterator(Vec__char* self);
VecIterRef__char Vec__char__iter_ref(Vec__char* self);
void Vec__char__push(Vec__char* self, char item);
void Vec__char__insert(Vec__char* self, size_t idx, char item);
char Vec__char__pop(Vec__char* self);
Option__char Vec__char__pop_opt(Vec__char* self);
char Vec__char__remove(Vec__char* self, size_t idx);
void Vec__char__append(Vec__char* self, Vec__char other);
char Vec__char__get(Vec__char* self, size_t idx);
char Vec__char__index(Vec__char* self, size_t idx);
char* Vec__char__get_ref(Vec__char* self, size_t idx);
char Vec__char__last(Vec__char* self);
size_t Vec__char__length(Vec__char* self);
bool Vec__char__contains(Vec__char* self, char item);
bool Vec__char__is_empty(Vec__char* self);
void Vec__char__clear(Vec__char* self);
void Vec__char__free(Vec__char* self);
char Vec__char__first(Vec__char* self);
void Vec__char__set(Vec__char* self, size_t idx, char item);
void Vec__char__reverse(Vec__char* self);
bool Vec__char__eq(Vec__char* self, Vec__char* other);
void Vec__char__forget(Vec__char* self);
Vec__char Vec__char__add(Vec__char* self, Vec__char* other);
void Vec__char__add_assign(Vec__char* self, Vec__char* other);
bool Vec__char__neq(Vec__char* self, Vec__char* other);
void Vec__char__shl(Vec__char* self, char item);
void Vec__char__shr(Vec__char* self, char* out_item);
Vec__char Vec__char__mul(Vec__char* self, size_t count);
void Vec__char__mul_assign(Vec__char* self, size_t count);
Vec__char Vec__char__clone(Vec__char* self);
VecIterResult__char VecIterRef__char__next(VecIterRef__char* self);
VecIterRef__char VecIterRef__char__iterator(VecIterRef__char* self);
bool VecIterResult__char__is_none(VecIterResult__char* self);
char* VecIterResult__char__unwrap(VecIterResult__char* self);
Option__char VecIter__char__next(VecIter__char* self);
VecIter__char VecIter__char__iterator(VecIter__char* self);
Option__char Option__char__Some(char v);
Option__char Option__char__None(void);
bool Option__char__is_some(Option__char* self);
bool Option__char__is_none(Option__char* self);
void Option__char__forget(Option__char* self);
char Option__char__unwrap(Option__char* self);
char* Option__char__unwrap_ref(Option__char* self);
char Option__char__unwrap_or(Option__char* self, char def_val);
char Option__char__expect(Option__char* self, char* msg);
Option__char Option__char__or_else(Option__char* self, Option__char other);
void Vec__char__Drop__drop(Vec__char* self);
Vec__String regex_split(char* pattern, char* text);
int32_t regex_count(char* pattern, char* text);
Option__Match regex_find(char* pattern, char* text);
bool regex_match(char* pattern, char* text);
int32_t _z_internal_str_case_cmp(char* s1, char* s2);
void sort_double(double* arr, size_t len);
void sort_float(float* arr, size_t len);
void sort_long(long* arr, size_t len);
void sort_int(int32_t* arr, size_t len);
void __zenc_todo_impl(const char* file, int32_t line, const char* func, const char* msg);
void __zenc_panic_impl(const char* file, int32_t line, const char* func, const char* msg);
Regex Regex__compile(char* pattern);
Regex Regex__compile_with_flags(char* pattern, int32_t flags);
bool Regex__is_valid(Regex* self);
bool Regex__match(Regex* self, char* text);
bool Regex__match_full(Regex* self, char* text);
bool Regex__match_at(Regex* self, char* text, int32_t offset);
bool Regex__is_match(Regex* self, char* text);
Option__Match Regex__find(Regex* self, char* text);
Option__Match Regex__find_at(Regex* self, char* text, int32_t start);
int32_t Regex__count(Regex* self, char* text);
Vec__String Regex__split(Regex* self, char* text);
char* Regex__pattern(Regex* self);
int32_t Regex__flags(Regex* self);
bool Regex__is_valid_pattern(char* pattern);
void Regex__destroy(Regex* self);
Match Match__new(char* text, int32_t start, int32_t len);
char* Match__as_string(Match* self);
int32_t Match__end(Match* self);
String String__new(char* s);
String String__from(char* s);
String String__from_rune(int32_t r);
String String__from_runes(int32_t* runes, size_t count);
char* String__c_str(String* self);
char* String__to_string(String* self);
void String__destroy(String* self);
void String__forget(String* self);
void String__append(String* self, String* other);
void String__append_c(String* self, char* s);
void String__push_rune(String* self, int32_t r);
void String__append_c_ptr(String* ptr, char* s);
String String__add(String* self, String* other);
void String__add_assign(String* self, String* other);
bool String__eq(String* self, String* other);
bool String__neq(String* self, String* other);
int32_t String__compare(String* self, String* other);
bool String__lt(String* self, String* other);
bool String__gt(String* self, String* other);
bool String__le(String* self, String* other);
bool String__ge(String* self, String* other);
int32_t String__compare_ignore_case(String* self, String* other);
bool String__eq_ignore_case(String* self, String* other);
bool String__eq_str(String* self, char* s);
size_t String__length(String* self);
String String__substring(String* self, size_t start, size_t len);
bool String__contains_str(String* self, char* target);
Option__size_t String__find_str(String* self, char* target);
Vec__size_t String__find_all_str(String* self, char* target);
String String__to_lowercase(String* self);
String String__pad_right(String* self, size_t target_len, char pad_char);
String String__pad_left(String* self, size_t target_len, char pad_char);
String String__to_uppercase(String* self);
Option__size_t String__find(String* self, char target);
void String__print(String* self);
void String__println(String* self);
bool String__is_empty(String* self);
bool String__contains(String* self, char target);
bool String__starts_with(String* self, char* prefix);
bool String__ends_with(String* self, char* suffix);
void String__reserve(String* self, size_t cap);
void String__free(String* self);
size_t String__utf8_seq_len(char first_byte);
size_t String__utf8_len(String* self);
String String__utf8_at(String* self, size_t idx);
int32_t String__utf8_get(String* self, size_t idx);
Vec__int32_t String__runes(String* self);
StringCharsIter String__iterator(String* self);
StringCharsIter String__chars(String* self);
String String__from_runes_vec(Vec__int32_t runes);
void String__insert_rune(String* self, size_t idx, int32_t r);
int32_t String__remove_rune_at(String* self, size_t idx);
String String__utf8_substr(String* self, size_t start_idx, size_t num_chars);
Vec__String String__split(String* self, char delim);
String String__trim(String* self);
String String__replace(String* self, char* target, char* replacement);
Option__int32_t StringCharsIter__next(StringCharsIter* self);
StringCharsIter StringCharsIter__iterator(StringCharsIter* self);
// Auto-Generated RAII Glue for Vec__String
void Vec__String__Drop__glue(Vec__String *self) {
    Vec__String__Drop__drop(self);
}


// Auto-Generated RAII Glue for Vec__int32_t
void Vec__int32_t__Drop__glue(Vec__int32_t *self) {
    Vec__int32_t__Drop__drop(self);
}


// Auto-Generated RAII Glue for Vec__size_t
void Vec__size_t__Drop__glue(Vec__size_t *self) {
    Vec__size_t__Drop__drop(self);
}


// Auto-Generated RAII Glue for Vec__char
void Vec__char__Drop__glue(Vec__char *self) {
    Vec__char__Drop__drop(self);
}


// Auto-Generated RAII Glue for String
void String__Drop__glue(String *self) {
    Vec__char__Drop__glue(&self->vec);
}


// Auto-Generated RAII Glue for Option__String
void Option__String__Drop__glue(Option__String *self) {
    String__Drop__glue(&self->val);
}


// Global Generic Drop Dispatch
#define _z_drop(x) _Generic((x), \
    Vec__String: Vec__String__Drop__glue((void*)&(x)), \
    Vec__int32_t: Vec__int32_t__Drop__glue((void*)&(x)), \
    Vec__size_t: Vec__size_t__Drop__glue((void*)&(x)), \
    Vec__char: Vec__char__Drop__glue((void*)&(x)), \
    String: String__Drop__glue((void*)&(x)), \
    Option__String: Option__String__Drop__glue((void*)&(x)), \
    default: (void)0)

static void _z_test_0() {
    {

#line 5 "tests/std/test_tre_regex.zc"
    Regex re = 
#line 5 "tests/std/test_tre_regex.zc"
Regex__compile("a+b");

#line 7 "tests/std/test_tre_regex.zc"
if ((!
#line 7 "tests/std/test_tre_regex.zc"
Regex__is_valid((&re))))     {
__zenc_panic("Failed to compile regex");
    }

#line 9 "tests/std/test_tre_regex.zc"
if ((!
#line 9 "tests/std/test_tre_regex.zc"
Regex__match((&re), "aaab")))     {
__zenc_panic("Match 1: FAILED");
    }

#line 10 "tests/std/test_tre_regex.zc"
if (
#line 10 "tests/std/test_tre_regex.zc"
Regex__match((&re), "acb"))     {
__zenc_panic("Match 2: FAILED");
    }

#line 12 "tests/std/test_tre_regex.zc"
Regex__destroy((&re));

#line 15 "tests/std/test_tre_regex.zc"
    Regex re2 = 
#line 15 "tests/std/test_tre_regex.zc"
Regex__compile("([0-9]+)");

#line 16 "tests/std/test_tre_regex.zc"
    Option__Match found = 
#line 16 "tests/std/test_tre_regex.zc"
Regex__find((&re2), "Year: 2023");

#line 17 "tests/std/test_tre_regex.zc"
if ((!
#line 17 "tests/std/test_tre_regex.zc"
Option__Match__is_some((&found))))     {
__zenc_panic("Find: FAILED (Not found)");
    }

#line 18 "tests/std/test_tre_regex.zc"
    Match m = 
#line 18 "tests/std/test_tre_regex.zc"
Option__Match__unwrap((&found));

#line 26 "tests/std/test_tre_regex.zc"
    char* s = 
#line 26 "tests/std/test_tre_regex.zc"
Match__as_string((&m));

#line 27 "tests/std/test_tre_regex.zc"
if ((
#line 27 "tests/std/test_tre_regex.zc"
strcmp(s, "2023") != 0))     {
__zenc_panic("Find: FAILED (Wrong content)");
    }

#line 28 "tests/std/test_tre_regex.zc"
free(s);

#line 29 "tests/std/test_tre_regex.zc"
Regex__destroy((&re2));
    }
}

void _z_run_tests() {
    _z_test_0();
}


#line 10 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

Option__Match Option__Match__Some(Match v)
{
    {

#line 11 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return (Option__Match){.is_some = true, .val = v};
    }
}

#line 14 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

Option__Match Option__Match__None(void)
{
    {

#line 15 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    Option__Match opt = {0};

#line 16 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
(opt.is_some = false);

#line 17 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
memset((&opt.val), 0, sizeof(Match));

#line 18 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return opt;
    }
}

#line 21 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

bool Option__Match__is_some(Option__Match* self)
{
    {

#line 22 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return self->is_some;
    }
}

#line 25 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

bool Option__Match__is_none(Option__Match* self)
{
    {

#line 26 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return (!self->is_some);
    }
}

#line 29 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

void Option__Match__forget(Option__Match* self)
{
    {

#line 30 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
memset((&self->val), 0, sizeof(Match));
    }
}

#line 33 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

Match Option__Match__unwrap(Option__Match* self)
{
    {

#line 34 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
if ((!self->is_some))     {

#line 35 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    ({ fprintf(stderr, "%s", "Panic: unwrap called on None"); fprintf(stderr, "\n"); 0; });

#line 36 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
exit(1);
    }

#line 38 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    Match v = self->val;

#line 40 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return v;
    }
}

#line 43 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

Match* Option__Match__unwrap_ref(Option__Match* self)
{
    {

#line 44 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
if ((!self->is_some))     {

#line 45 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    ({ fprintf(stderr, "%s", "Panic: unwrap_ref called on None"); fprintf(stderr, "\n"); 0; });

#line 46 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
exit(1);
    }

#line 48 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return (&self->val);
    }
}

#line 51 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

Match Option__Match__unwrap_or(Option__Match* self, Match def_val)
{
    {

#line 52 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
if (self->is_some)     {

#line 53 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return self->val;
    }

#line 55 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return def_val;
    }
}

#line 58 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

Match Option__Match__expect(Option__Match* self, char* msg)
{
    {

#line 59 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
if ((!self->is_some))     {

#line 60 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    ({ fprintf(stderr, "%s", "Panic: "); fprintf(stderr, "%s", msg); fprintf(stderr, "\n"); 0; });

#line 61 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
exit(1);
    }

#line 63 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return self->val;
    }
}

#line 66 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

Option__Match Option__Match__or_else(Option__Match* self, Option__Match other)
{
    {

#line 67 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
if (self->is_some)     {

#line 67 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return *self;
    }

#line 68 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return other;
    }
}

#line 73 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

Vec__String Vec__String__new(void)
{
    {

#line 74 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (Vec__String){.data = NULL};
    }
}

#line 77 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

Vec__String Vec__String__with_capacity(size_t cap)
{
    {

#line 78 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((cap == 0))     {

#line 79 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (Vec__String){.data = NULL};
    }

#line 81 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (Vec__String){.data = ((String*)(
#line 82 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
malloc((cap * sizeof(String))))), .cap = cap};
    }
}

#line 88 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__String__grow(Vec__String* self)
{
    {

#line 89 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->cap == 0))     {

#line 89 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->cap = 8);
    }
 else     {

#line 90 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->cap = (self->cap * 2));
    }

#line 91 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data = ((String*)(
#line 91 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
realloc(self->data, (self->cap * sizeof(String))))));
    }
}

#line 94 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__String__grow_to_fit(Vec__String* self, size_t new_len)
{
    {

#line 95 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->cap >= new_len))     {

#line 96 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return ;
    }

#line 99 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->cap == 0))     {

#line 99 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->cap = 8);
    }

#line 100 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
while ((self->cap < new_len))     {

#line 101 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->cap = (self->cap * 2));
    }

#line 104 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data = ((String*)(
#line 104 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
realloc(self->data, (self->cap * sizeof(String))))));
    }
}

#line 107 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

VecIter__String Vec__String__iterator(Vec__String* self)
{
    {

#line 108 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (VecIter__String){.data = self->data, .count = self->len};
    }
}

#line 115 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

VecIterRef__String Vec__String__iter_ref(Vec__String* self)
{
    {

#line 116 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (VecIterRef__String){.data = self->data, .count = self->len};
    }
}

#line 123 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__String__push(Vec__String* self, String item)
{

#line 123 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    int __z_drop_flag_item = 1;
    {

#line 124 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len >= self->cap))     {

#line 125 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__String__grow(self);
    }

#line 127 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data[self->len] = ({ __z_drop_flag_item = 0; item; }));

#line 128 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = (self->len + 1));
    }

#line 123 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_item) String__Drop__glue(&item);
}

#line 131 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__String__insert(Vec__String* self, size_t idx, String item)
{

#line 131 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    int __z_drop_flag_item = 1;
    {

#line 132 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((idx > self->len))     {

#line 133 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: Insert index out of bounds"); fprintf(stderr, "\n"); 0; });

#line 134 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 136 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len >= self->cap))     {

#line 137 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__String__grow(self);
    }

#line 140 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((idx < self->len))     {

#line 141 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
memmove(((self->data + idx) + 1), (self->data + idx), ((self->len - idx) * sizeof(String)));
    }

#line 143 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data[idx] = ({ __z_drop_flag_item = 0; item; }));

#line 144 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = (self->len + 1));
    }

#line 131 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_item) String__Drop__glue(&item);
}

#line 147 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

String Vec__String__pop(Vec__String* self)
{
    {

#line 148 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len == 0))     {

#line 149 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: pop called on empty Vec"); fprintf(stderr, "\n"); 0; });

#line 150 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 152 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = (self->len - 1));

#line 153 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return self->data[self->len];
    }
}

#line 156 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

Option__String Vec__String__pop_opt(Vec__String* self)
{
    {

#line 157 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len == 0))     {

#line 158 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return 
#line 158 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Option__String__None();
    }

#line 160 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = (self->len - 1));

#line 161 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return 
#line 161 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Option__String__Some(self->data[self->len]);
    }
}

#line 164 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

String Vec__String__remove(Vec__String* self, size_t idx)
{
    {

#line 165 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((idx >= self->len))     {

#line 166 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: Remove index out of bounds"); fprintf(stderr, "\n"); 0; });

#line 167 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 169 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    int __z_drop_flag_item = 1; String item = self->data[idx];

#line 171 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((idx < (self->len - 1)))     {

#line 172 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
memmove((self->data + idx), ((self->data + idx) + 1), (((self->len - idx) - 1) * sizeof(String)));
    }

#line 174 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = (self->len - 1));

#line 175 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return ({ ZC_AUTO _z_ret_mv = item; memset(&item, 0, sizeof(_z_ret_mv)); __z_drop_flag_item = 0; 
#line 169 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_item) String__Drop__glue(&item);
_z_ret_mv; });

#line 169 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_item) String__Drop__glue(&item);
    }
}

#line 180 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__String__append(Vec__String* self, Vec__String other)
{

#line 180 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    int __z_drop_flag_other = 1;
    {

#line 181 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t new_len = (self->len + other.len);

#line 182 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__String__grow_to_fit(self, new_len);

#line 184 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
memcpy((self->data + self->len), other.data, (other.len * sizeof(String)));

#line 185 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = new_len);

#line 186 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__String__forget(&other);
    }

#line 180 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_other) Vec__String__Drop__glue(&other);
}

#line 189 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

String Vec__String__get(Vec__String* self, size_t idx)
{
    {

#line 190 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((idx >= self->len))     {

#line 191 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: Index out of bounds"); fprintf(stderr, "\n"); 0; });

#line 192 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 194 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return self->data[idx];
    }
}

#line 197 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

String Vec__String__index(Vec__String* self, size_t idx)
{
    {

#line 198 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return 
#line 198 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__String__get(self, idx);
    }
}

#line 201 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

String* Vec__String__get_ref(Vec__String* self, size_t idx)
{
    {

#line 202 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((idx >= self->len))     {

#line 203 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: Index out of bounds"); fprintf(stderr, "\n"); 0; });

#line 204 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 206 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (&self->data[idx]);
    }
}

#line 209 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

String Vec__String__last(Vec__String* self)
{
    {

#line 210 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len == 0))     {

#line 211 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: last called on empty Vec"); fprintf(stderr, "\n"); 0; });

#line 212 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 214 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return self->data[(self->len - 1)];
    }
}

#line 217 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

size_t Vec__String__length(Vec__String* self)
{
    {

#line 218 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return self->len;
    }
}

#line 221 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

bool Vec__String__contains(Vec__String* self, String item)
{

#line 221 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    int __z_drop_flag_item = 1;
    {

#line 222 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t i = 0;

#line 223 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
while ((i < self->len))     {

#line 224 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((
#line 224 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
memcmp((&self->data[i]), (&item), sizeof(String)) == 0))     {

#line 224 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    { bool _z_ret = true; 
#line 221 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_item) String__Drop__glue(&item);
return _z_ret; }
    }
(i++);
    }

#line 227 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    { bool _z_ret = false; 
#line 221 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_item) String__Drop__glue(&item);
return _z_ret; }
    }

#line 221 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_item) String__Drop__glue(&item);
}

#line 230 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

bool Vec__String__is_empty(Vec__String* self)
{
    {

#line 231 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (self->len == 0);
    }
}

#line 234 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__String__clear(Vec__String* self)
{
    {

#line 235 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = 0);
    }
}

#line 238 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__String__free(Vec__String* self)
{
    {

#line 239 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if (self->data)     {

#line 239 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
free(self->data);
    }

#line 240 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data = NULL);

#line 241 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = 0);

#line 242 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->cap = 0);
    }
}

#line 245 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

String Vec__String__first(Vec__String* self)
{
    {

#line 246 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len == 0))     {

#line 247 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: first called on empty Vec"); fprintf(stderr, "\n"); 0; });

#line 248 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 250 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return self->data[0];
    }
}

#line 253 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__String__set(Vec__String* self, size_t idx, String item)
{

#line 253 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    int __z_drop_flag_item = 1;
    {

#line 254 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((idx >= self->len))     {

#line 255 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: set index out of bounds"); fprintf(stderr, "\n"); 0; });

#line 256 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 258 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data[idx] = ({ __z_drop_flag_item = 0; item; }));
    }

#line 253 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_item) String__Drop__glue(&item);
}

#line 261 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__String__reverse(Vec__String* self)
{
    {

#line 262 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t i = 0;

#line 263 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t j = (self->len - 1);

#line 264 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
while ((i < j))     {

#line 265 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    int __z_drop_flag_tmp = 1; String tmp = self->data[i];

#line 266 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data[i] = self->data[j]);

#line 267 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data[j] = ({ __z_drop_flag_tmp = 0; tmp; }));
(i++);
(j--);

#line 265 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_tmp) String__Drop__glue(&tmp);
    }
    }
}

#line 275 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

bool Vec__String__eq(Vec__String* self, Vec__String* other)
{
    {

#line 276 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len != (*other).len))     {

#line 276 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return false;
    }

#line 277 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t i = 0;

#line 278 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
while ((i < self->len))     {

#line 279 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((
#line 279 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
memcmp((&self->data[i]), (&(*other).data[i]), sizeof(String)) != 0))     {

#line 279 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return false;
    }

#line 280 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(i = (i + 1));
    }

#line 282 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return true;
    }
}

#line 286 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__String__forget(Vec__String* self)
{
    {

#line 287 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data = NULL);

#line 288 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = 0);

#line 289 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->cap = 0);
    }
}

#line 295 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

Vec__String Vec__String__add(Vec__String* self, Vec__String* other)
{
    {

#line 296 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    int __z_drop_flag_result = 1; Vec__String result = 
#line 296 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__String__clone(self);

#line 297 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__String__append(&result, Vec__String__clone(other));

#line 298 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return ({ ZC_AUTO _z_ret_mv = result; memset(&result, 0, sizeof(_z_ret_mv)); __z_drop_flag_result = 0; 
#line 296 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_result) Vec__String__Drop__glue(&result);
_z_ret_mv; });

#line 296 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_result) Vec__String__Drop__glue(&result);
    }
}

#line 302 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__String__add_assign(Vec__String* self, Vec__String* other)
{
    {

#line 303 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__String__append(self, Vec__String__clone(other));
    }
}

#line 306 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

bool Vec__String__neq(Vec__String* self, Vec__String* other)
{
    {

#line 307 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (!
#line 307 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__String__eq(self, other));
    }
}

#line 311 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__String__shl(Vec__String* self, String item)
{

#line 311 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    int __z_drop_flag_item = 1;
    {

#line 312 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__String__push(self, ({ __z_drop_flag_item = 0; item; }));
    }

#line 311 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_item) String__Drop__glue(&item);
}

#line 316 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__String__shr(Vec__String* self, String* out_item)
{
    {

#line 317 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((out_item != NULL))     {

#line 318 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
((*out_item) = 
#line 318 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__String__pop(self));
    }
 else     {

#line 320 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__String__pop(self);
    }
    }
}

#line 327 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

Vec__String Vec__String__mul(Vec__String* self, size_t count)
{
    {

#line 328 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    int __z_drop_flag_result = 1; Vec__String result = 
#line 328 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__String__with_capacity((self->len * count));

#line 329 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t c = 0;

#line 330 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
while ((c < count))     {

#line 331 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__String__append(&result, Vec__String__clone(self));

#line 332 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(c = (c + 1));
    }

#line 334 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return ({ ZC_AUTO _z_ret_mv = result; memset(&result, 0, sizeof(_z_ret_mv)); __z_drop_flag_result = 0; 
#line 328 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_result) Vec__String__Drop__glue(&result);
_z_ret_mv; });

#line 328 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_result) Vec__String__Drop__glue(&result);
    }
}

#line 340 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__String__mul_assign(Vec__String* self, size_t count)
{
    {

#line 341 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((count == 0))     {

#line 342 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__String__clear(self);

#line 343 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return ;
    }

#line 345 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((count == 1))     {

#line 346 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return ;
    }

#line 348 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t original_len = self->len;

#line 349 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__String__grow_to_fit(self, (self->len * count));

#line 350 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t c = 1;

#line 351 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
while ((c < count))     {

#line 352 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
memcpy((self->data + (original_len * c)), self->data, (original_len * sizeof(String)));

#line 353 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = (self->len + original_len));

#line 354 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(c = (c + 1));
    }
    }
}

#line 359 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

Vec__String Vec__String__clone(Vec__String* self)
{
    {

#line 360 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len == 0))     {

#line 361 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (Vec__String){.data = NULL};
    }

#line 363 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    String* new_data = ((String*)(
#line 363 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
malloc((self->len * sizeof(String)))));

#line 364 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t i = 0;

#line 365 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
while ((i < self->len))     {

#line 366 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(new_data[i] = self->data[i]);

#line 367 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(i = (i + 1));
    }

#line 369 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (Vec__String){.data = new_data, .len = self->len, .cap = self->len};
    }
}

#line 58 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

VecIterResult__String VecIterRef__String__next(VecIterRef__String* self)
{
    {

#line 59 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->idx < self->count))     {

#line 60 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    String* item = (&self->data[self->idx]);

#line 61 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->idx = (self->idx + 1));

#line 62 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (VecIterResult__String){.ptr = item};
    }

#line 64 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (VecIterResult__String){.ptr = NULL};
    }
}

#line 67 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

VecIterRef__String VecIterRef__String__iterator(VecIterRef__String* self)
{
    {

#line 68 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return *self;
    }
}

#line 23 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

bool VecIterResult__String__is_none(VecIterResult__String* self)
{
    {

#line 24 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (self->ptr == NULL);
    }
}

#line 27 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

String* VecIterResult__String__unwrap(VecIterResult__String* self)
{
    {

#line 28 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->ptr == NULL))     {

#line 29 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: unwrap called on null VecIterResult"); fprintf(stderr, "\n"); 0; });

#line 30 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 32 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return self->ptr;
    }
}

#line 43 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

Option__String VecIter__String__next(VecIter__String* self)
{
    {

#line 44 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->idx < self->count))     {

#line 45 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    int __z_drop_flag_item = 1; String item = self->data[self->idx];

#line 46 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->idx = (self->idx + 1));

#line 47 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    { Option__String _z_ret = 
#line 47 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Option__String__Some(({ __z_drop_flag_item = 0; item; })); 
#line 45 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_item) String__Drop__glue(&item);
return _z_ret; }

#line 45 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_item) String__Drop__glue(&item);
    }

#line 49 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return 
#line 49 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Option__String__None();
    }
}

#line 52 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

VecIter__String VecIter__String__iterator(VecIter__String* self)
{
    {

#line 53 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return *self;
    }
}

#line 10 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

Option__String Option__String__Some(String v)
{

#line 10 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    int __z_drop_flag_v = 1;
    {

#line 11 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    { Option__String _z_ret = (Option__String){.is_some = true, .val = ({ __z_drop_flag_v = 0; v; })}; 
#line 10 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    if (__z_drop_flag_v) String__Drop__glue(&v);
return _z_ret; }
    }

#line 10 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    if (__z_drop_flag_v) String__Drop__glue(&v);
}

#line 14 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

Option__String Option__String__None(void)
{
    {

#line 15 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    int __z_drop_flag_opt = 1; Option__String opt = {0};

#line 16 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
(opt.is_some = false);

#line 17 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
memset((&opt.val), 0, sizeof(String));

#line 18 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return ({ ZC_AUTO _z_ret_mv = opt; memset(&opt, 0, sizeof(_z_ret_mv)); __z_drop_flag_opt = 0; 
#line 15 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    if (__z_drop_flag_opt) Option__String__Drop__glue(&opt);
_z_ret_mv; });

#line 15 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    if (__z_drop_flag_opt) Option__String__Drop__glue(&opt);
    }
}

#line 21 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

bool Option__String__is_some(Option__String* self)
{
    {

#line 22 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return self->is_some;
    }
}

#line 25 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

bool Option__String__is_none(Option__String* self)
{
    {

#line 26 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return (!self->is_some);
    }
}

#line 29 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

void Option__String__forget(Option__String* self)
{
    {

#line 30 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
memset((&self->val), 0, sizeof(String));
    }
}

#line 33 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

String Option__String__unwrap(Option__String* self)
{
    {

#line 34 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
if ((!self->is_some))     {

#line 35 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    ({ fprintf(stderr, "%s", "Panic: unwrap called on None"); fprintf(stderr, "\n"); 0; });

#line 36 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
exit(1);
    }

#line 38 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    int __z_drop_flag_v = 1; String v = self->val;
memset(&self->val, 0, sizeof(self->val));

#line 40 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return ({ ZC_AUTO _z_ret_mv = v; memset(&v, 0, sizeof(_z_ret_mv)); __z_drop_flag_v = 0; 
#line 38 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    if (__z_drop_flag_v) String__Drop__glue(&v);
_z_ret_mv; });

#line 38 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    if (__z_drop_flag_v) String__Drop__glue(&v);
    }
}

#line 43 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

String* Option__String__unwrap_ref(Option__String* self)
{
    {

#line 44 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
if ((!self->is_some))     {

#line 45 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    ({ fprintf(stderr, "%s", "Panic: unwrap_ref called on None"); fprintf(stderr, "\n"); 0; });

#line 46 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
exit(1);
    }

#line 48 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return (&self->val);
    }
}

#line 51 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

String Option__String__unwrap_or(Option__String* self, String def_val)
{

#line 51 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    int __z_drop_flag_def_val = 1;
    {

#line 52 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
if (self->is_some)     {

#line 53 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    { String _z_ret = self->val; 
#line 51 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    if (__z_drop_flag_def_val) String__Drop__glue(&def_val);
return _z_ret; }
    }

#line 55 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return ({ ZC_AUTO _z_ret_mv = def_val; memset(&def_val, 0, sizeof(_z_ret_mv)); __z_drop_flag_def_val = 0; 
#line 51 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    if (__z_drop_flag_def_val) String__Drop__glue(&def_val);
_z_ret_mv; });
    }

#line 51 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    if (__z_drop_flag_def_val) String__Drop__glue(&def_val);
}

#line 58 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

String Option__String__expect(Option__String* self, char* msg)
{
    {

#line 59 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
if ((!self->is_some))     {

#line 60 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    ({ fprintf(stderr, "%s", "Panic: "); fprintf(stderr, "%s", msg); fprintf(stderr, "\n"); 0; });

#line 61 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
exit(1);
    }

#line 63 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return self->val;
    }
}

#line 66 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

Option__String Option__String__or_else(Option__String* self, Option__String other)
{

#line 66 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    int __z_drop_flag_other = 1;
    {

#line 67 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
if (self->is_some)     {

#line 67 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    { Option__String _z_ret = *self; 
#line 66 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    if (__z_drop_flag_other) Option__String__Drop__glue(&other);
return _z_ret; }
    }

#line 68 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return ({ ZC_AUTO _z_ret_mv = other; memset(&other, 0, sizeof(_z_ret_mv)); __z_drop_flag_other = 0; 
#line 66 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    if (__z_drop_flag_other) Option__String__Drop__glue(&other);
_z_ret_mv; });
    }

#line 66 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    if (__z_drop_flag_other) Option__String__Drop__glue(&other);
}

#line 378 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__String__Drop__drop(Vec__String* self)
{
    {

#line 379 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__String__free(self);
    }
}

#line 73 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

Vec__int32_t Vec__int32_t__new(void)
{
    {

#line 74 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (Vec__int32_t){.data = NULL};
    }
}

#line 77 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

Vec__int32_t Vec__int32_t__with_capacity(size_t cap)
{
    {

#line 78 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((cap == 0))     {

#line 79 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (Vec__int32_t){.data = NULL};
    }

#line 81 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (Vec__int32_t){.data = ((int32_t*)(
#line 82 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
malloc((cap * sizeof(int32_t))))), .cap = cap};
    }
}

#line 88 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__int32_t__grow(Vec__int32_t* self)
{
    {

#line 89 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->cap == 0))     {

#line 89 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->cap = 8);
    }
 else     {

#line 90 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->cap = (self->cap * 2));
    }

#line 91 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data = ((int32_t*)(
#line 91 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
realloc(self->data, (self->cap * sizeof(int32_t))))));
    }
}

#line 94 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__int32_t__grow_to_fit(Vec__int32_t* self, size_t new_len)
{
    {

#line 95 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->cap >= new_len))     {

#line 96 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return ;
    }

#line 99 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->cap == 0))     {

#line 99 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->cap = 8);
    }

#line 100 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
while ((self->cap < new_len))     {

#line 101 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->cap = (self->cap * 2));
    }

#line 104 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data = ((int32_t*)(
#line 104 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
realloc(self->data, (self->cap * sizeof(int32_t))))));
    }
}

#line 107 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

VecIter__int32_t Vec__int32_t__iterator(Vec__int32_t* self)
{
    {

#line 108 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (VecIter__int32_t){.data = self->data, .count = self->len};
    }
}

#line 115 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

VecIterRef__int32_t Vec__int32_t__iter_ref(Vec__int32_t* self)
{
    {

#line 116 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (VecIterRef__int32_t){.data = self->data, .count = self->len};
    }
}

#line 123 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__int32_t__push(Vec__int32_t* self, int32_t item)
{
    {

#line 124 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len >= self->cap))     {

#line 125 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__int32_t__grow(self);
    }

#line 127 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data[self->len] = item);

#line 128 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = (self->len + 1));
    }
}

#line 131 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__int32_t__insert(Vec__int32_t* self, size_t idx, int32_t item)
{
    {

#line 132 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((idx > self->len))     {

#line 133 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: Insert index out of bounds"); fprintf(stderr, "\n"); 0; });

#line 134 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 136 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len >= self->cap))     {

#line 137 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__int32_t__grow(self);
    }

#line 140 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((idx < self->len))     {

#line 141 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
memmove(((self->data + idx) + 1), (self->data + idx), ((self->len - idx) * sizeof(int32_t)));
    }

#line 143 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data[idx] = item);

#line 144 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = (self->len + 1));
    }
}

#line 147 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

int32_t Vec__int32_t__pop(Vec__int32_t* self)
{
    {

#line 148 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len == 0))     {

#line 149 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: pop called on empty Vec"); fprintf(stderr, "\n"); 0; });

#line 150 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 152 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = (self->len - 1));

#line 153 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return self->data[self->len];
    }
}

#line 156 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

Option__int32_t Vec__int32_t__pop_opt(Vec__int32_t* self)
{
    {

#line 157 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len == 0))     {

#line 158 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return 
#line 158 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Option__int32_t__None();
    }

#line 160 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = (self->len - 1));

#line 161 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return 
#line 161 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Option__int32_t__Some(self->data[self->len]);
    }
}

#line 164 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

int32_t Vec__int32_t__remove(Vec__int32_t* self, size_t idx)
{
    {

#line 165 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((idx >= self->len))     {

#line 166 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: Remove index out of bounds"); fprintf(stderr, "\n"); 0; });

#line 167 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 169 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    int32_t item = self->data[idx];

#line 171 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((idx < (self->len - 1)))     {

#line 172 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
memmove((self->data + idx), ((self->data + idx) + 1), (((self->len - idx) - 1) * sizeof(int32_t)));
    }

#line 174 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = (self->len - 1));

#line 175 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return item;
    }
}

#line 180 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__int32_t__append(Vec__int32_t* self, Vec__int32_t other)
{

#line 180 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    int __z_drop_flag_other = 1;
    {

#line 181 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t new_len = (self->len + other.len);

#line 182 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__int32_t__grow_to_fit(self, new_len);

#line 184 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
memcpy((self->data + self->len), other.data, (other.len * sizeof(int32_t)));

#line 185 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = new_len);

#line 186 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__int32_t__forget(&other);
    }

#line 180 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_other) Vec__int32_t__Drop__glue(&other);
}

#line 189 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

int32_t Vec__int32_t__get(Vec__int32_t* self, size_t idx)
{
    {

#line 190 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((idx >= self->len))     {

#line 191 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: Index out of bounds"); fprintf(stderr, "\n"); 0; });

#line 192 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 194 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return self->data[idx];
    }
}

#line 197 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

int32_t Vec__int32_t__index(Vec__int32_t* self, size_t idx)
{
    {

#line 198 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return 
#line 198 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__int32_t__get(self, idx);
    }
}

#line 201 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

int32_t* Vec__int32_t__get_ref(Vec__int32_t* self, size_t idx)
{
    {

#line 202 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((idx >= self->len))     {

#line 203 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: Index out of bounds"); fprintf(stderr, "\n"); 0; });

#line 204 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 206 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (&self->data[idx]);
    }
}

#line 209 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

int32_t Vec__int32_t__last(Vec__int32_t* self)
{
    {

#line 210 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len == 0))     {

#line 211 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: last called on empty Vec"); fprintf(stderr, "\n"); 0; });

#line 212 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 214 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return self->data[(self->len - 1)];
    }
}

#line 217 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

size_t Vec__int32_t__length(Vec__int32_t* self)
{
    {

#line 218 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return self->len;
    }
}

#line 221 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

bool Vec__int32_t__contains(Vec__int32_t* self, int32_t item)
{
    {

#line 222 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t i = 0;

#line 223 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
while ((i < self->len))     {

#line 224 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((
#line 224 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
memcmp((&self->data[i]), (&item), sizeof(int32_t)) == 0))     {

#line 224 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return true;
    }
(i++);
    }

#line 227 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return false;
    }
}

#line 230 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

bool Vec__int32_t__is_empty(Vec__int32_t* self)
{
    {

#line 231 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (self->len == 0);
    }
}

#line 234 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__int32_t__clear(Vec__int32_t* self)
{
    {

#line 235 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = 0);
    }
}

#line 238 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__int32_t__free(Vec__int32_t* self)
{
    {

#line 239 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if (self->data)     {

#line 239 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
free(self->data);
    }

#line 240 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data = NULL);

#line 241 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = 0);

#line 242 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->cap = 0);
    }
}

#line 245 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

int32_t Vec__int32_t__first(Vec__int32_t* self)
{
    {

#line 246 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len == 0))     {

#line 247 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: first called on empty Vec"); fprintf(stderr, "\n"); 0; });

#line 248 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 250 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return self->data[0];
    }
}

#line 253 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__int32_t__set(Vec__int32_t* self, size_t idx, int32_t item)
{
    {

#line 254 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((idx >= self->len))     {

#line 255 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: set index out of bounds"); fprintf(stderr, "\n"); 0; });

#line 256 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 258 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data[idx] = item);
    }
}

#line 261 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__int32_t__reverse(Vec__int32_t* self)
{
    {

#line 262 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t i = 0;

#line 263 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t j = (self->len - 1);

#line 264 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
while ((i < j))     {

#line 265 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    int32_t tmp = self->data[i];

#line 266 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data[i] = self->data[j]);

#line 267 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data[j] = tmp);
(i++);
(j--);
    }
    }
}

#line 275 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

bool Vec__int32_t__eq(Vec__int32_t* self, Vec__int32_t* other)
{
    {

#line 276 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len != (*other).len))     {

#line 276 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return false;
    }

#line 277 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t i = 0;

#line 278 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
while ((i < self->len))     {

#line 279 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((
#line 279 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
memcmp((&self->data[i]), (&(*other).data[i]), sizeof(int32_t)) != 0))     {

#line 279 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return false;
    }

#line 280 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(i = (i + 1));
    }

#line 282 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return true;
    }
}

#line 286 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__int32_t__forget(Vec__int32_t* self)
{
    {

#line 287 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data = NULL);

#line 288 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = 0);

#line 289 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->cap = 0);
    }
}

#line 295 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

Vec__int32_t Vec__int32_t__add(Vec__int32_t* self, Vec__int32_t* other)
{
    {

#line 296 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    int __z_drop_flag_result = 1; Vec__int32_t result = 
#line 296 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__int32_t__clone(self);

#line 297 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__int32_t__append(&result, Vec__int32_t__clone(other));

#line 298 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return ({ ZC_AUTO _z_ret_mv = result; memset(&result, 0, sizeof(_z_ret_mv)); __z_drop_flag_result = 0; 
#line 296 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_result) Vec__int32_t__Drop__glue(&result);
_z_ret_mv; });

#line 296 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_result) Vec__int32_t__Drop__glue(&result);
    }
}

#line 302 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__int32_t__add_assign(Vec__int32_t* self, Vec__int32_t* other)
{
    {

#line 303 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__int32_t__append(self, Vec__int32_t__clone(other));
    }
}

#line 306 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

bool Vec__int32_t__neq(Vec__int32_t* self, Vec__int32_t* other)
{
    {

#line 307 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (!
#line 307 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__int32_t__eq(self, other));
    }
}

#line 311 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__int32_t__shl(Vec__int32_t* self, int32_t item)
{
    {

#line 312 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__int32_t__push(self, item);
    }
}

#line 316 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__int32_t__shr(Vec__int32_t* self, int32_t* out_item)
{
    {

#line 317 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((out_item != NULL))     {

#line 318 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
((*out_item) = 
#line 318 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__int32_t__pop(self));
    }
 else     {

#line 320 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__int32_t__pop(self);
    }
    }
}

#line 327 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

Vec__int32_t Vec__int32_t__mul(Vec__int32_t* self, size_t count)
{
    {

#line 328 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    int __z_drop_flag_result = 1; Vec__int32_t result = 
#line 328 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__int32_t__with_capacity((self->len * count));

#line 329 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t c = 0;

#line 330 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
while ((c < count))     {

#line 331 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__int32_t__append(&result, Vec__int32_t__clone(self));

#line 332 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(c = (c + 1));
    }

#line 334 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return ({ ZC_AUTO _z_ret_mv = result; memset(&result, 0, sizeof(_z_ret_mv)); __z_drop_flag_result = 0; 
#line 328 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_result) Vec__int32_t__Drop__glue(&result);
_z_ret_mv; });

#line 328 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_result) Vec__int32_t__Drop__glue(&result);
    }
}

#line 340 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__int32_t__mul_assign(Vec__int32_t* self, size_t count)
{
    {

#line 341 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((count == 0))     {

#line 342 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__int32_t__clear(self);

#line 343 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return ;
    }

#line 345 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((count == 1))     {

#line 346 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return ;
    }

#line 348 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t original_len = self->len;

#line 349 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__int32_t__grow_to_fit(self, (self->len * count));

#line 350 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t c = 1;

#line 351 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
while ((c < count))     {

#line 352 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
memcpy((self->data + (original_len * c)), self->data, (original_len * sizeof(int32_t)));

#line 353 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = (self->len + original_len));

#line 354 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(c = (c + 1));
    }
    }
}

#line 359 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

Vec__int32_t Vec__int32_t__clone(Vec__int32_t* self)
{
    {

#line 360 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len == 0))     {

#line 361 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (Vec__int32_t){.data = NULL};
    }

#line 363 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    int32_t* new_data = ((int32_t*)(
#line 363 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
malloc((self->len * sizeof(int32_t)))));

#line 364 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t i = 0;

#line 365 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
while ((i < self->len))     {

#line 366 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(new_data[i] = self->data[i]);

#line 367 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(i = (i + 1));
    }

#line 369 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (Vec__int32_t){.data = new_data, .len = self->len, .cap = self->len};
    }
}

#line 58 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

VecIterResult__int32_t VecIterRef__int32_t__next(VecIterRef__int32_t* self)
{
    {

#line 59 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->idx < self->count))     {

#line 60 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    int32_t* item = (&self->data[self->idx]);

#line 61 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->idx = (self->idx + 1));

#line 62 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (VecIterResult__int32_t){.ptr = item};
    }

#line 64 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (VecIterResult__int32_t){.ptr = NULL};
    }
}

#line 67 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

VecIterRef__int32_t VecIterRef__int32_t__iterator(VecIterRef__int32_t* self)
{
    {

#line 68 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return *self;
    }
}

#line 23 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

bool VecIterResult__int32_t__is_none(VecIterResult__int32_t* self)
{
    {

#line 24 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (self->ptr == NULL);
    }
}

#line 27 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

int32_t* VecIterResult__int32_t__unwrap(VecIterResult__int32_t* self)
{
    {

#line 28 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->ptr == NULL))     {

#line 29 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: unwrap called on null VecIterResult"); fprintf(stderr, "\n"); 0; });

#line 30 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 32 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return self->ptr;
    }
}

#line 43 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

Option__int32_t VecIter__int32_t__next(VecIter__int32_t* self)
{
    {

#line 44 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->idx < self->count))     {

#line 45 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    int32_t item = self->data[self->idx];

#line 46 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->idx = (self->idx + 1));

#line 47 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return 
#line 47 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Option__int32_t__Some(item);
    }

#line 49 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return 
#line 49 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Option__int32_t__None();
    }
}

#line 52 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

VecIter__int32_t VecIter__int32_t__iterator(VecIter__int32_t* self)
{
    {

#line 53 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return *self;
    }
}

#line 378 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__int32_t__Drop__drop(Vec__int32_t* self)
{
    {

#line 379 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__int32_t__free(self);
    }
}

#line 73 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

Vec__size_t Vec__size_t__new(void)
{
    {

#line 74 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (Vec__size_t){.data = NULL};
    }
}

#line 77 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

Vec__size_t Vec__size_t__with_capacity(size_t cap)
{
    {

#line 78 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((cap == 0))     {

#line 79 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (Vec__size_t){.data = NULL};
    }

#line 81 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (Vec__size_t){.data = ((size_t*)(
#line 82 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
malloc((cap * sizeof(size_t))))), .cap = cap};
    }
}

#line 88 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__size_t__grow(Vec__size_t* self)
{
    {

#line 89 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->cap == 0))     {

#line 89 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->cap = 8);
    }
 else     {

#line 90 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->cap = (self->cap * 2));
    }

#line 91 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data = ((size_t*)(
#line 91 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
realloc(self->data, (self->cap * sizeof(size_t))))));
    }
}

#line 94 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__size_t__grow_to_fit(Vec__size_t* self, size_t new_len)
{
    {

#line 95 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->cap >= new_len))     {

#line 96 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return ;
    }

#line 99 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->cap == 0))     {

#line 99 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->cap = 8);
    }

#line 100 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
while ((self->cap < new_len))     {

#line 101 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->cap = (self->cap * 2));
    }

#line 104 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data = ((size_t*)(
#line 104 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
realloc(self->data, (self->cap * sizeof(size_t))))));
    }
}

#line 107 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

VecIter__size_t Vec__size_t__iterator(Vec__size_t* self)
{
    {

#line 108 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (VecIter__size_t){.data = self->data, .count = self->len};
    }
}

#line 115 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

VecIterRef__size_t Vec__size_t__iter_ref(Vec__size_t* self)
{
    {

#line 116 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (VecIterRef__size_t){.data = self->data, .count = self->len};
    }
}

#line 123 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__size_t__push(Vec__size_t* self, size_t item)
{
    {

#line 124 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len >= self->cap))     {

#line 125 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__size_t__grow(self);
    }

#line 127 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data[self->len] = item);

#line 128 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = (self->len + 1));
    }
}

#line 131 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__size_t__insert(Vec__size_t* self, size_t idx, size_t item)
{
    {

#line 132 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((idx > self->len))     {

#line 133 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: Insert index out of bounds"); fprintf(stderr, "\n"); 0; });

#line 134 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 136 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len >= self->cap))     {

#line 137 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__size_t__grow(self);
    }

#line 140 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((idx < self->len))     {

#line 141 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
memmove(((self->data + idx) + 1), (self->data + idx), ((self->len - idx) * sizeof(size_t)));
    }

#line 143 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data[idx] = item);

#line 144 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = (self->len + 1));
    }
}

#line 147 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

size_t Vec__size_t__pop(Vec__size_t* self)
{
    {

#line 148 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len == 0))     {

#line 149 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: pop called on empty Vec"); fprintf(stderr, "\n"); 0; });

#line 150 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 152 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = (self->len - 1));

#line 153 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return self->data[self->len];
    }
}

#line 156 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

Option__size_t Vec__size_t__pop_opt(Vec__size_t* self)
{
    {

#line 157 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len == 0))     {

#line 158 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return 
#line 158 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Option__size_t__None();
    }

#line 160 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = (self->len - 1));

#line 161 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return 
#line 161 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Option__size_t__Some(self->data[self->len]);
    }
}

#line 164 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

size_t Vec__size_t__remove(Vec__size_t* self, size_t idx)
{
    {

#line 165 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((idx >= self->len))     {

#line 166 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: Remove index out of bounds"); fprintf(stderr, "\n"); 0; });

#line 167 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 169 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t item = self->data[idx];

#line 171 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((idx < (self->len - 1)))     {

#line 172 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
memmove((self->data + idx), ((self->data + idx) + 1), (((self->len - idx) - 1) * sizeof(size_t)));
    }

#line 174 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = (self->len - 1));

#line 175 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return item;
    }
}

#line 180 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__size_t__append(Vec__size_t* self, Vec__size_t other)
{

#line 180 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    int __z_drop_flag_other = 1;
    {

#line 181 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t new_len = (self->len + other.len);

#line 182 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__size_t__grow_to_fit(self, new_len);

#line 184 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
memcpy((self->data + self->len), other.data, (other.len * sizeof(size_t)));

#line 185 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = new_len);

#line 186 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__size_t__forget(&other);
    }

#line 180 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_other) Vec__size_t__Drop__glue(&other);
}

#line 189 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

size_t Vec__size_t__get(Vec__size_t* self, size_t idx)
{
    {

#line 190 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((idx >= self->len))     {

#line 191 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: Index out of bounds"); fprintf(stderr, "\n"); 0; });

#line 192 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 194 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return self->data[idx];
    }
}

#line 197 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

size_t Vec__size_t__index(Vec__size_t* self, size_t idx)
{
    {

#line 198 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return 
#line 198 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__size_t__get(self, idx);
    }
}

#line 201 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

size_t* Vec__size_t__get_ref(Vec__size_t* self, size_t idx)
{
    {

#line 202 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((idx >= self->len))     {

#line 203 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: Index out of bounds"); fprintf(stderr, "\n"); 0; });

#line 204 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 206 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (&self->data[idx]);
    }
}

#line 209 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

size_t Vec__size_t__last(Vec__size_t* self)
{
    {

#line 210 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len == 0))     {

#line 211 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: last called on empty Vec"); fprintf(stderr, "\n"); 0; });

#line 212 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 214 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return self->data[(self->len - 1)];
    }
}

#line 217 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

size_t Vec__size_t__length(Vec__size_t* self)
{
    {

#line 218 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return self->len;
    }
}

#line 221 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

bool Vec__size_t__contains(Vec__size_t* self, size_t item)
{
    {

#line 222 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t i = 0;

#line 223 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
while ((i < self->len))     {

#line 224 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((
#line 224 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
memcmp((&self->data[i]), (&item), sizeof(size_t)) == 0))     {

#line 224 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return true;
    }
(i++);
    }

#line 227 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return false;
    }
}

#line 230 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

bool Vec__size_t__is_empty(Vec__size_t* self)
{
    {

#line 231 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (self->len == 0);
    }
}

#line 234 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__size_t__clear(Vec__size_t* self)
{
    {

#line 235 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = 0);
    }
}

#line 238 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__size_t__free(Vec__size_t* self)
{
    {

#line 239 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if (self->data)     {

#line 239 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
free(self->data);
    }

#line 240 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data = NULL);

#line 241 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = 0);

#line 242 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->cap = 0);
    }
}

#line 245 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

size_t Vec__size_t__first(Vec__size_t* self)
{
    {

#line 246 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len == 0))     {

#line 247 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: first called on empty Vec"); fprintf(stderr, "\n"); 0; });

#line 248 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 250 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return self->data[0];
    }
}

#line 253 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__size_t__set(Vec__size_t* self, size_t idx, size_t item)
{
    {

#line 254 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((idx >= self->len))     {

#line 255 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: set index out of bounds"); fprintf(stderr, "\n"); 0; });

#line 256 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 258 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data[idx] = item);
    }
}

#line 261 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__size_t__reverse(Vec__size_t* self)
{
    {

#line 262 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t i = 0;

#line 263 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t j = (self->len - 1);

#line 264 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
while ((i < j))     {

#line 265 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t tmp = self->data[i];

#line 266 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data[i] = self->data[j]);

#line 267 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data[j] = tmp);
(i++);
(j--);
    }
    }
}

#line 275 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

bool Vec__size_t__eq(Vec__size_t* self, Vec__size_t* other)
{
    {

#line 276 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len != (*other).len))     {

#line 276 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return false;
    }

#line 277 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t i = 0;

#line 278 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
while ((i < self->len))     {

#line 279 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((
#line 279 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
memcmp((&self->data[i]), (&(*other).data[i]), sizeof(size_t)) != 0))     {

#line 279 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return false;
    }

#line 280 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(i = (i + 1));
    }

#line 282 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return true;
    }
}

#line 286 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__size_t__forget(Vec__size_t* self)
{
    {

#line 287 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data = NULL);

#line 288 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = 0);

#line 289 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->cap = 0);
    }
}

#line 295 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

Vec__size_t Vec__size_t__add(Vec__size_t* self, Vec__size_t* other)
{
    {

#line 296 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    int __z_drop_flag_result = 1; Vec__size_t result = 
#line 296 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__size_t__clone(self);

#line 297 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__size_t__append(&result, Vec__size_t__clone(other));

#line 298 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return ({ ZC_AUTO _z_ret_mv = result; memset(&result, 0, sizeof(_z_ret_mv)); __z_drop_flag_result = 0; 
#line 296 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_result) Vec__size_t__Drop__glue(&result);
_z_ret_mv; });

#line 296 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_result) Vec__size_t__Drop__glue(&result);
    }
}

#line 302 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__size_t__add_assign(Vec__size_t* self, Vec__size_t* other)
{
    {

#line 303 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__size_t__append(self, Vec__size_t__clone(other));
    }
}

#line 306 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

bool Vec__size_t__neq(Vec__size_t* self, Vec__size_t* other)
{
    {

#line 307 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (!
#line 307 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__size_t__eq(self, other));
    }
}

#line 311 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__size_t__shl(Vec__size_t* self, size_t item)
{
    {

#line 312 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__size_t__push(self, item);
    }
}

#line 316 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__size_t__shr(Vec__size_t* self, size_t* out_item)
{
    {

#line 317 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((out_item != NULL))     {

#line 318 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
((*out_item) = 
#line 318 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__size_t__pop(self));
    }
 else     {

#line 320 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__size_t__pop(self);
    }
    }
}

#line 327 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

Vec__size_t Vec__size_t__mul(Vec__size_t* self, size_t count)
{
    {

#line 328 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    int __z_drop_flag_result = 1; Vec__size_t result = 
#line 328 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__size_t__with_capacity((self->len * count));

#line 329 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t c = 0;

#line 330 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
while ((c < count))     {

#line 331 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__size_t__append(&result, Vec__size_t__clone(self));

#line 332 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(c = (c + 1));
    }

#line 334 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return ({ ZC_AUTO _z_ret_mv = result; memset(&result, 0, sizeof(_z_ret_mv)); __z_drop_flag_result = 0; 
#line 328 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_result) Vec__size_t__Drop__glue(&result);
_z_ret_mv; });

#line 328 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_result) Vec__size_t__Drop__glue(&result);
    }
}

#line 340 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__size_t__mul_assign(Vec__size_t* self, size_t count)
{
    {

#line 341 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((count == 0))     {

#line 342 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__size_t__clear(self);

#line 343 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return ;
    }

#line 345 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((count == 1))     {

#line 346 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return ;
    }

#line 348 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t original_len = self->len;

#line 349 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__size_t__grow_to_fit(self, (self->len * count));

#line 350 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t c = 1;

#line 351 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
while ((c < count))     {

#line 352 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
memcpy((self->data + (original_len * c)), self->data, (original_len * sizeof(size_t)));

#line 353 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = (self->len + original_len));

#line 354 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(c = (c + 1));
    }
    }
}

#line 359 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

Vec__size_t Vec__size_t__clone(Vec__size_t* self)
{
    {

#line 360 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len == 0))     {

#line 361 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (Vec__size_t){.data = NULL};
    }

#line 363 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t* new_data = ((size_t*)(
#line 363 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
malloc((self->len * sizeof(size_t)))));

#line 364 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t i = 0;

#line 365 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
while ((i < self->len))     {

#line 366 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(new_data[i] = self->data[i]);

#line 367 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(i = (i + 1));
    }

#line 369 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (Vec__size_t){.data = new_data, .len = self->len, .cap = self->len};
    }
}

#line 58 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

VecIterResult__size_t VecIterRef__size_t__next(VecIterRef__size_t* self)
{
    {

#line 59 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->idx < self->count))     {

#line 60 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t* item = (&self->data[self->idx]);

#line 61 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->idx = (self->idx + 1));

#line 62 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (VecIterResult__size_t){.ptr = item};
    }

#line 64 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (VecIterResult__size_t){.ptr = NULL};
    }
}

#line 67 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

VecIterRef__size_t VecIterRef__size_t__iterator(VecIterRef__size_t* self)
{
    {

#line 68 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return *self;
    }
}

#line 23 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

bool VecIterResult__size_t__is_none(VecIterResult__size_t* self)
{
    {

#line 24 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (self->ptr == NULL);
    }
}

#line 27 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

size_t* VecIterResult__size_t__unwrap(VecIterResult__size_t* self)
{
    {

#line 28 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->ptr == NULL))     {

#line 29 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: unwrap called on null VecIterResult"); fprintf(stderr, "\n"); 0; });

#line 30 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 32 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return self->ptr;
    }
}

#line 43 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

Option__size_t VecIter__size_t__next(VecIter__size_t* self)
{
    {

#line 44 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->idx < self->count))     {

#line 45 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t item = self->data[self->idx];

#line 46 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->idx = (self->idx + 1));

#line 47 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return 
#line 47 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Option__size_t__Some(item);
    }

#line 49 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return 
#line 49 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Option__size_t__None();
    }
}

#line 52 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

VecIter__size_t VecIter__size_t__iterator(VecIter__size_t* self)
{
    {

#line 53 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return *self;
    }
}

#line 378 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__size_t__Drop__drop(Vec__size_t* self)
{
    {

#line 379 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__size_t__free(self);
    }
}

#line 10 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

Option__size_t Option__size_t__Some(size_t v)
{
    {

#line 11 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return (Option__size_t){.is_some = true, .val = v};
    }
}

#line 14 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

Option__size_t Option__size_t__None(void)
{
    {

#line 15 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    Option__size_t opt = {0};

#line 16 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
(opt.is_some = false);

#line 17 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
memset((&opt.val), 0, sizeof(size_t));

#line 18 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return opt;
    }
}

#line 21 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

bool Option__size_t__is_some(Option__size_t* self)
{
    {

#line 22 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return self->is_some;
    }
}

#line 25 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

bool Option__size_t__is_none(Option__size_t* self)
{
    {

#line 26 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return (!self->is_some);
    }
}

#line 29 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

void Option__size_t__forget(Option__size_t* self)
{
    {

#line 30 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
memset((&self->val), 0, sizeof(size_t));
    }
}

#line 33 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

size_t Option__size_t__unwrap(Option__size_t* self)
{
    {

#line 34 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
if ((!self->is_some))     {

#line 35 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    ({ fprintf(stderr, "%s", "Panic: unwrap called on None"); fprintf(stderr, "\n"); 0; });

#line 36 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
exit(1);
    }

#line 38 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    size_t v = self->val;

#line 40 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return v;
    }
}

#line 43 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

size_t* Option__size_t__unwrap_ref(Option__size_t* self)
{
    {

#line 44 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
if ((!self->is_some))     {

#line 45 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    ({ fprintf(stderr, "%s", "Panic: unwrap_ref called on None"); fprintf(stderr, "\n"); 0; });

#line 46 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
exit(1);
    }

#line 48 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return (&self->val);
    }
}

#line 51 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

size_t Option__size_t__unwrap_or(Option__size_t* self, size_t def_val)
{
    {

#line 52 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
if (self->is_some)     {

#line 53 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return self->val;
    }

#line 55 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return def_val;
    }
}

#line 58 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

size_t Option__size_t__expect(Option__size_t* self, char* msg)
{
    {

#line 59 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
if ((!self->is_some))     {

#line 60 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    ({ fprintf(stderr, "%s", "Panic: "); fprintf(stderr, "%s", msg); fprintf(stderr, "\n"); 0; });

#line 61 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
exit(1);
    }

#line 63 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return self->val;
    }
}

#line 66 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

Option__size_t Option__size_t__or_else(Option__size_t* self, Option__size_t other)
{
    {

#line 67 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
if (self->is_some)     {

#line 67 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return *self;
    }

#line 68 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return other;
    }
}

#line 10 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

Option__int32_t Option__int32_t__Some(int32_t v)
{
    {

#line 11 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return (Option__int32_t){.is_some = true, .val = v};
    }
}

#line 14 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

Option__int32_t Option__int32_t__None(void)
{
    {

#line 15 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    Option__int32_t opt = {0};

#line 16 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
(opt.is_some = false);

#line 17 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
memset((&opt.val), 0, sizeof(int32_t));

#line 18 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return opt;
    }
}

#line 21 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

bool Option__int32_t__is_some(Option__int32_t* self)
{
    {

#line 22 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return self->is_some;
    }
}

#line 25 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

bool Option__int32_t__is_none(Option__int32_t* self)
{
    {

#line 26 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return (!self->is_some);
    }
}

#line 29 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

void Option__int32_t__forget(Option__int32_t* self)
{
    {

#line 30 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
memset((&self->val), 0, sizeof(int32_t));
    }
}

#line 33 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

int32_t Option__int32_t__unwrap(Option__int32_t* self)
{
    {

#line 34 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
if ((!self->is_some))     {

#line 35 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    ({ fprintf(stderr, "%s", "Panic: unwrap called on None"); fprintf(stderr, "\n"); 0; });

#line 36 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
exit(1);
    }

#line 38 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    int32_t v = self->val;

#line 40 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return v;
    }
}

#line 43 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

int32_t* Option__int32_t__unwrap_ref(Option__int32_t* self)
{
    {

#line 44 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
if ((!self->is_some))     {

#line 45 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    ({ fprintf(stderr, "%s", "Panic: unwrap_ref called on None"); fprintf(stderr, "\n"); 0; });

#line 46 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
exit(1);
    }

#line 48 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return (&self->val);
    }
}

#line 51 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

int32_t Option__int32_t__unwrap_or(Option__int32_t* self, int32_t def_val)
{
    {

#line 52 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
if (self->is_some)     {

#line 53 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return self->val;
    }

#line 55 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return def_val;
    }
}

#line 58 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

int32_t Option__int32_t__expect(Option__int32_t* self, char* msg)
{
    {

#line 59 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
if ((!self->is_some))     {

#line 60 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    ({ fprintf(stderr, "%s", "Panic: "); fprintf(stderr, "%s", msg); fprintf(stderr, "\n"); 0; });

#line 61 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
exit(1);
    }

#line 63 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return self->val;
    }
}

#line 66 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

Option__int32_t Option__int32_t__or_else(Option__int32_t* self, Option__int32_t other)
{
    {

#line 67 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
if (self->is_some)     {

#line 67 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return *self;
    }

#line 68 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return other;
    }
}

#line 73 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

Vec__char Vec__char__new(void)
{
    {

#line 74 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (Vec__char){.data = NULL};
    }
}

#line 77 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

Vec__char Vec__char__with_capacity(size_t cap)
{
    {

#line 78 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((cap == 0))     {

#line 79 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (Vec__char){.data = NULL};
    }

#line 81 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (Vec__char){.data = ((char*)(
#line 82 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
malloc((cap * sizeof(char))))), .cap = cap};
    }
}

#line 88 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__char__grow(Vec__char* self)
{
    {

#line 89 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->cap == 0))     {

#line 89 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->cap = 8);
    }
 else     {

#line 90 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->cap = (self->cap * 2));
    }

#line 91 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data = ((char*)(
#line 91 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
realloc(self->data, (self->cap * sizeof(char))))));
    }
}

#line 94 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__char__grow_to_fit(Vec__char* self, size_t new_len)
{
    {

#line 95 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->cap >= new_len))     {

#line 96 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return ;
    }

#line 99 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->cap == 0))     {

#line 99 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->cap = 8);
    }

#line 100 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
while ((self->cap < new_len))     {

#line 101 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->cap = (self->cap * 2));
    }

#line 104 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data = ((char*)(
#line 104 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
realloc(self->data, (self->cap * sizeof(char))))));
    }
}

#line 107 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

VecIter__char Vec__char__iterator(Vec__char* self)
{
    {

#line 108 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (VecIter__char){.data = self->data, .count = self->len};
    }
}

#line 115 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

VecIterRef__char Vec__char__iter_ref(Vec__char* self)
{
    {

#line 116 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (VecIterRef__char){.data = self->data, .count = self->len};
    }
}

#line 123 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__char__push(Vec__char* self, char item)
{
    {

#line 124 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len >= self->cap))     {

#line 125 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__char__grow(self);
    }

#line 127 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data[self->len] = item);

#line 128 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = (self->len + 1));
    }
}

#line 131 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__char__insert(Vec__char* self, size_t idx, char item)
{
    {

#line 132 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((idx > self->len))     {

#line 133 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: Insert index out of bounds"); fprintf(stderr, "\n"); 0; });

#line 134 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 136 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len >= self->cap))     {

#line 137 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__char__grow(self);
    }

#line 140 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((idx < self->len))     {

#line 141 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
memmove(((self->data + idx) + 1), (self->data + idx), ((self->len - idx) * sizeof(char)));
    }

#line 143 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data[idx] = item);

#line 144 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = (self->len + 1));
    }
}

#line 147 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

char Vec__char__pop(Vec__char* self)
{
    {

#line 148 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len == 0))     {

#line 149 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: pop called on empty Vec"); fprintf(stderr, "\n"); 0; });

#line 150 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 152 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = (self->len - 1));

#line 153 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return self->data[self->len];
    }
}

#line 156 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

Option__char Vec__char__pop_opt(Vec__char* self)
{
    {

#line 157 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len == 0))     {

#line 158 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return 
#line 158 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Option__char__None();
    }

#line 160 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = (self->len - 1));

#line 161 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return 
#line 161 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Option__char__Some(self->data[self->len]);
    }
}

#line 164 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

char Vec__char__remove(Vec__char* self, size_t idx)
{
    {

#line 165 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((idx >= self->len))     {

#line 166 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: Remove index out of bounds"); fprintf(stderr, "\n"); 0; });

#line 167 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 169 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    char item = self->data[idx];

#line 171 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((idx < (self->len - 1)))     {

#line 172 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
memmove((self->data + idx), ((self->data + idx) + 1), (((self->len - idx) - 1) * sizeof(char)));
    }

#line 174 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = (self->len - 1));

#line 175 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return item;
    }
}

#line 180 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__char__append(Vec__char* self, Vec__char other)
{

#line 180 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    int __z_drop_flag_other = 1;
    {

#line 181 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t new_len = (self->len + other.len);

#line 182 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__char__grow_to_fit(self, new_len);

#line 184 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
memcpy((self->data + self->len), other.data, (other.len * sizeof(char)));

#line 185 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = new_len);

#line 186 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__char__forget(&other);
    }

#line 180 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_other) Vec__char__Drop__glue(&other);
}

#line 189 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

char Vec__char__get(Vec__char* self, size_t idx)
{
    {

#line 190 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((idx >= self->len))     {

#line 191 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: Index out of bounds"); fprintf(stderr, "\n"); 0; });

#line 192 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 194 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return self->data[idx];
    }
}

#line 197 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

char Vec__char__index(Vec__char* self, size_t idx)
{
    {

#line 198 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return 
#line 198 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__char__get(self, idx);
    }
}

#line 201 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

char* Vec__char__get_ref(Vec__char* self, size_t idx)
{
    {

#line 202 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((idx >= self->len))     {

#line 203 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: Index out of bounds"); fprintf(stderr, "\n"); 0; });

#line 204 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 206 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (&self->data[idx]);
    }
}

#line 209 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

char Vec__char__last(Vec__char* self)
{
    {

#line 210 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len == 0))     {

#line 211 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: last called on empty Vec"); fprintf(stderr, "\n"); 0; });

#line 212 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 214 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return self->data[(self->len - 1)];
    }
}

#line 217 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

size_t Vec__char__length(Vec__char* self)
{
    {

#line 218 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return self->len;
    }
}

#line 221 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

bool Vec__char__contains(Vec__char* self, char item)
{
    {

#line 222 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t i = 0;

#line 223 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
while ((i < self->len))     {

#line 224 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((
#line 224 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
memcmp((&self->data[i]), (&item), sizeof(char)) == 0))     {

#line 224 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return true;
    }
(i++);
    }

#line 227 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return false;
    }
}

#line 230 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

bool Vec__char__is_empty(Vec__char* self)
{
    {

#line 231 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (self->len == 0);
    }
}

#line 234 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__char__clear(Vec__char* self)
{
    {

#line 235 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = 0);
    }
}

#line 238 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__char__free(Vec__char* self)
{
    {

#line 239 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if (self->data)     {

#line 239 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
free(self->data);
    }

#line 240 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data = NULL);

#line 241 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = 0);

#line 242 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->cap = 0);
    }
}

#line 245 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

char Vec__char__first(Vec__char* self)
{
    {

#line 246 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len == 0))     {

#line 247 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: first called on empty Vec"); fprintf(stderr, "\n"); 0; });

#line 248 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 250 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return self->data[0];
    }
}

#line 253 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__char__set(Vec__char* self, size_t idx, char item)
{
    {

#line 254 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((idx >= self->len))     {

#line 255 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: set index out of bounds"); fprintf(stderr, "\n"); 0; });

#line 256 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 258 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data[idx] = item);
    }
}

#line 261 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__char__reverse(Vec__char* self)
{
    {

#line 262 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t i = 0;

#line 263 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t j = (self->len - 1);

#line 264 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
while ((i < j))     {

#line 265 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    char tmp = self->data[i];

#line 266 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data[i] = self->data[j]);

#line 267 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data[j] = tmp);
(i++);
(j--);
    }
    }
}

#line 275 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

bool Vec__char__eq(Vec__char* self, Vec__char* other)
{
    {

#line 276 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len != (*other).len))     {

#line 276 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return false;
    }

#line 277 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t i = 0;

#line 278 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
while ((i < self->len))     {

#line 279 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((
#line 279 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
memcmp((&self->data[i]), (&(*other).data[i]), sizeof(char)) != 0))     {

#line 279 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return false;
    }

#line 280 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(i = (i + 1));
    }

#line 282 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return true;
    }
}

#line 286 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__char__forget(Vec__char* self)
{
    {

#line 287 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->data = NULL);

#line 288 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = 0);

#line 289 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->cap = 0);
    }
}

#line 295 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

Vec__char Vec__char__add(Vec__char* self, Vec__char* other)
{
    {

#line 296 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    int __z_drop_flag_result = 1; Vec__char result = 
#line 296 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__char__clone(self);

#line 297 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__char__append(&result, Vec__char__clone(other));

#line 298 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return ({ ZC_AUTO _z_ret_mv = result; memset(&result, 0, sizeof(_z_ret_mv)); __z_drop_flag_result = 0; 
#line 296 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_result) Vec__char__Drop__glue(&result);
_z_ret_mv; });

#line 296 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_result) Vec__char__Drop__glue(&result);
    }
}

#line 302 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__char__add_assign(Vec__char* self, Vec__char* other)
{
    {

#line 303 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__char__append(self, Vec__char__clone(other));
    }
}

#line 306 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

bool Vec__char__neq(Vec__char* self, Vec__char* other)
{
    {

#line 307 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (!
#line 307 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__char__eq(self, other));
    }
}

#line 311 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__char__shl(Vec__char* self, char item)
{
    {

#line 312 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__char__push(self, item);
    }
}

#line 316 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__char__shr(Vec__char* self, char* out_item)
{
    {

#line 317 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((out_item != NULL))     {

#line 318 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
((*out_item) = 
#line 318 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__char__pop(self));
    }
 else     {

#line 320 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__char__pop(self);
    }
    }
}

#line 327 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

Vec__char Vec__char__mul(Vec__char* self, size_t count)
{
    {

#line 328 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    int __z_drop_flag_result = 1; Vec__char result = 
#line 328 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__char__with_capacity((self->len * count));

#line 329 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t c = 0;

#line 330 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
while ((c < count))     {

#line 331 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__char__append(&result, Vec__char__clone(self));

#line 332 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(c = (c + 1));
    }

#line 334 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return ({ ZC_AUTO _z_ret_mv = result; memset(&result, 0, sizeof(_z_ret_mv)); __z_drop_flag_result = 0; 
#line 328 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_result) Vec__char__Drop__glue(&result);
_z_ret_mv; });

#line 328 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    if (__z_drop_flag_result) Vec__char__Drop__glue(&result);
    }
}

#line 340 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__char__mul_assign(Vec__char* self, size_t count)
{
    {

#line 341 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((count == 0))     {

#line 342 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__char__clear(self);

#line 343 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return ;
    }

#line 345 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((count == 1))     {

#line 346 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return ;
    }

#line 348 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t original_len = self->len;

#line 349 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__char__grow_to_fit(self, (self->len * count));

#line 350 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t c = 1;

#line 351 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
while ((c < count))     {

#line 352 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
memcpy((self->data + (original_len * c)), self->data, (original_len * sizeof(char)));

#line 353 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->len = (self->len + original_len));

#line 354 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(c = (c + 1));
    }
    }
}

#line 359 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

Vec__char Vec__char__clone(Vec__char* self)
{
    {

#line 360 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->len == 0))     {

#line 361 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (Vec__char){.data = NULL};
    }

#line 363 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    char* new_data = ((char*)(
#line 363 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
malloc((self->len * sizeof(char)))));

#line 364 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    size_t i = 0;

#line 365 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
while ((i < self->len))     {

#line 366 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(new_data[i] = self->data[i]);

#line 367 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(i = (i + 1));
    }

#line 369 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (Vec__char){.data = new_data, .len = self->len, .cap = self->len};
    }
}

#line 58 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

VecIterResult__char VecIterRef__char__next(VecIterRef__char* self)
{
    {

#line 59 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->idx < self->count))     {

#line 60 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    char* item = (&self->data[self->idx]);

#line 61 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->idx = (self->idx + 1));

#line 62 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (VecIterResult__char){.ptr = item};
    }

#line 64 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (VecIterResult__char){.ptr = NULL};
    }
}

#line 67 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

VecIterRef__char VecIterRef__char__iterator(VecIterRef__char* self)
{
    {

#line 68 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return *self;
    }
}

#line 23 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

bool VecIterResult__char__is_none(VecIterResult__char* self)
{
    {

#line 24 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return (self->ptr == NULL);
    }
}

#line 27 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

char* VecIterResult__char__unwrap(VecIterResult__char* self)
{
    {

#line 28 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->ptr == NULL))     {

#line 29 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    ({ fprintf(stderr, "%s", "Panic: unwrap called on null VecIterResult"); fprintf(stderr, "\n"); 0; });

#line 30 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
exit(1);
    }

#line 32 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return self->ptr;
    }
}

#line 43 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

Option__char VecIter__char__next(VecIter__char* self)
{
    {

#line 44 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
if ((self->idx < self->count))     {

#line 45 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    char item = self->data[self->idx];

#line 46 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
(self->idx = (self->idx + 1));

#line 47 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return 
#line 47 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Option__char__Some(item);
    }

#line 49 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return 
#line 49 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Option__char__None();
    }
}

#line 52 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

VecIter__char VecIter__char__iterator(VecIter__char* self)
{
    {

#line 53 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
    return *self;
    }
}

#line 10 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

Option__char Option__char__Some(char v)
{
    {

#line 11 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return (Option__char){.is_some = true, .val = v};
    }
}

#line 14 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

Option__char Option__char__None(void)
{
    {

#line 15 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    Option__char opt = {0};

#line 16 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
(opt.is_some = false);

#line 17 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
memset((&opt.val), 0, sizeof(char));

#line 18 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return opt;
    }
}

#line 21 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

bool Option__char__is_some(Option__char* self)
{
    {

#line 22 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return self->is_some;
    }
}

#line 25 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

bool Option__char__is_none(Option__char* self)
{
    {

#line 26 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return (!self->is_some);
    }
}

#line 29 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

void Option__char__forget(Option__char* self)
{
    {

#line 30 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
memset((&self->val), 0, sizeof(char));
    }
}

#line 33 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

char Option__char__unwrap(Option__char* self)
{
    {

#line 34 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
if ((!self->is_some))     {

#line 35 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    ({ fprintf(stderr, "%s", "Panic: unwrap called on None"); fprintf(stderr, "\n"); 0; });

#line 36 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
exit(1);
    }

#line 38 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    char v = self->val;

#line 40 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return v;
    }
}

#line 43 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

char* Option__char__unwrap_ref(Option__char* self)
{
    {

#line 44 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
if ((!self->is_some))     {

#line 45 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    ({ fprintf(stderr, "%s", "Panic: unwrap_ref called on None"); fprintf(stderr, "\n"); 0; });

#line 46 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
exit(1);
    }

#line 48 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return (&self->val);
    }
}

#line 51 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

char Option__char__unwrap_or(Option__char* self, char def_val)
{
    {

#line 52 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
if (self->is_some)     {

#line 53 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return self->val;
    }

#line 55 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return def_val;
    }
}

#line 58 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

char Option__char__expect(Option__char* self, char* msg)
{
    {

#line 59 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
if ((!self->is_some))     {

#line 60 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    ({ fprintf(stderr, "%s", "Panic: "); fprintf(stderr, "%s", msg); fprintf(stderr, "\n"); 0; });

#line 61 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
exit(1);
    }

#line 63 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return self->val;
    }
}

#line 66 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"

Option__char Option__char__or_else(Option__char* self, Option__char other)
{
    {

#line 67 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
if (self->is_some)     {

#line 67 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return *self;
    }

#line 68 "/home/zuhaitz/zenc-lang/zenc/std/option.zc"
    return other;
    }
}

#line 378 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"

void Vec__char__Drop__drop(Vec__char* self)
{
    {

#line 379 "/home/zuhaitz/zenc-lang/zenc/std/vec.zc"
Vec__char__free(self);
    }
}

#line 203 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"

Vec__String regex_split(char* pattern, char* text)
{
    {

#line 204 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    Regex re = 
#line 204 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
Regex__compile(pattern);

#line 205 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    int __z_drop_flag_parts = 1; Vec__String parts = 
#line 205 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
Regex__split((&re), text);

#line 206 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
Regex__destroy((&re));

#line 207 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    return ({ ZC_AUTO _z_ret_mv = parts; memset(&parts, 0, sizeof(_z_ret_mv)); __z_drop_flag_parts = 0; 
#line 205 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    if (__z_drop_flag_parts) Vec__String__Drop__glue(&parts);
_z_ret_mv; });

#line 205 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    if (__z_drop_flag_parts) Vec__String__Drop__glue(&parts);
    }
}

#line 196 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"

int32_t regex_count(char* pattern, char* text)
{
    {

#line 197 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    Regex re = 
#line 197 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
Regex__compile(pattern);

#line 198 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    int32_t count = 
#line 198 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
Regex__count((&re), text);

#line 199 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
Regex__destroy((&re));

#line 200 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    return count;
    }
}

#line 189 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"

Option__Match regex_find(char* pattern, char* text)
{
    {

#line 190 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    Regex re = 
#line 190 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
Regex__compile(pattern);

#line 191 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    Option__Match result = 
#line 191 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
Regex__find((&re), text);

#line 192 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
Regex__destroy((&re));

#line 193 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    return result;
    }
}

#line 182 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"

bool regex_match(char* pattern, char* text)
{
    {

#line 183 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    Regex re = 
#line 183 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
Regex__compile(pattern);

#line 184 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    bool result = 
#line 184 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
Regex__match((&re), text);

#line 185 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
Regex__destroy((&re));

#line 186 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    return result;
    }
}

#line 5 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

int32_t _z_internal_str_case_cmp(char* s1, char* s2)
{
    {

#line 6 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    int32_t i = 0;

#line 7 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
while (true)     {

#line 8 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    int32_t c1 = ((int32_t)(((uint8_t)(s1[i]))));

#line 9 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    int32_t c2 = ((int32_t)(((uint8_t)(s2[i]))));

#line 12 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    ZC_AUTO lc1 = ({ __typeof__((c1 + 32)) _ifval; if (((c1 >= ((int32_t)('A'))) && (c1 <= ((int32_t)('Z'))))) { _ifval = (c1 + 32); } else { _ifval = c1; } _ifval; });

#line 13 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    ZC_AUTO lc2 = ({ __typeof__((c2 + 32)) _ifval; if (((c2 >= ((int32_t)('A'))) && (c2 <= ((int32_t)('Z'))))) { _ifval = (c2 + 32); } else { _ifval = c2; } _ifval; });

#line 15 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((lc1 != lc2))     {

#line 16 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return (lc1 - lc2);
    }

#line 19 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((c1 == 0))     {

#line 20 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
break;
    }

#line 22 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(i = (i + 1));
    }

#line 24 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 0;
    }
}

#line 13 "/home/zuhaitz/zenc-lang/zenc/std/core.zc"

void __zenc_todo_impl(const char* file, int32_t line, const char* func, const char* msg)
{
    {

#line 14 "/home/zuhaitz/zenc-lang/zenc/std/core.zc"
fprintf(stderr, "todo: %s\n  at %s:%d in %s()\n", msg, file, line, func);

#line 15 "/home/zuhaitz/zenc-lang/zenc/std/core.zc"
exit(1);
    }
}

#line 8 "/home/zuhaitz/zenc-lang/zenc/std/core.zc"

void __zenc_panic_impl(const char* file, int32_t line, const char* func, const char* msg)
{
    {

#line 9 "/home/zuhaitz/zenc-lang/zenc/std/core.zc"
fprintf(stderr, "panic: %s\n  at %s:%d in %s()\n", msg, file, line, func);

#line 10 "/home/zuhaitz/zenc-lang/zenc/std/core.zc"
exit(1);
    }
}

#line 48 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"

Regex Regex__compile(char* pattern)
{
    {

#line 49 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    return 
#line 49 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
Regex__compile_with_flags(pattern, (1 | 2));
    }
}

#line 52 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"

Regex Regex__compile_with_flags(char* pattern, int32_t flags)
{
    {

#line 53 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    void* preg = 
#line 53 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
malloc(1024);

#line 54 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    ZC_AUTO status = 
#line 54 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
tre_regcomp(preg, pattern, flags);

#line 55 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
if ((status != 0))     {

#line 56 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
free(preg);

#line 57 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    return (struct Regex){.preg = NULL, .pattern = NULL, .flags = flags};
    }

#line 59 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    return (struct Regex){.preg = preg, .pattern = pattern, .flags = flags};
    }
}

#line 62 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"

bool Regex__is_valid(Regex* self)
{
    {

#line 63 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    return (self->preg != NULL);
    }
}

#line 66 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"

bool Regex__match(Regex* self, char* text)
{
    {

#line 67 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
if ((self->preg == NULL))     {

#line 67 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    return false;
    }

#line 68 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    return (
#line 68 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
tre_regexec(self->preg, text, 0, 0, 0) == 0);
    }
}

#line 71 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"

bool Regex__match_full(Regex* self, char* text)
{
    {

#line 72 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    return 
#line 72 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
Regex__match(self, text);
    }
}

#line 75 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"

bool Regex__match_at(Regex* self, char* text, int32_t offset)
{
    {

#line 76 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
if ((self->preg == NULL))     {

#line 76 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    return false;
    }

#line 77 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    ZC_AUTO len = 
#line 77 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
strlen(text);

#line 78 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
if (((offset < 0) || (offset > len)))     {

#line 78 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    return false;
    }

#line 79 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    return (
#line 79 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
tre_regexec(self->preg, (text + offset), 0, 0, 0) == 0);
    }
}

#line 82 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"

bool Regex__is_match(Regex* self, char* text)
{
    {

#line 83 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    return 
#line 83 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
Regex__match(self, text);
    }
}

#line 86 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"

Option__Match Regex__find(Regex* self, char* text)
{
    {

#line 87 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
if ((self->preg == NULL))     {

#line 87 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    return 
#line 87 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
Option__Match__None();
    }

#line 89 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    ZRegMatch pmatch[1] = {0};

#line 90 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
if ((
#line 90 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
tre_regexec(self->preg, text, 1, ((void*)(pmatch)), 0) == 0))     {

#line 91 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    int32_t start = pmatch[_z_check_bounds(0, 1)].rm_so;

#line 92 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    int32_t end = pmatch[_z_check_bounds(0, 1)].rm_eo;

#line 93 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
if ((start != (-1)))     {

#line 94 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    return 
#line 94 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
Option__Match__Some(Match__new(text, start, (end - start)));
    }
    }

#line 97 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    return 
#line 97 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
Option__Match__None();
    }
}

#line 100 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"

Option__Match Regex__find_at(Regex* self, char* text, int32_t start)
{
    {

#line 101 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    ZC_AUTO len = 
#line 101 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
strlen(text);

#line 102 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
if (((start < 0) || (start >= len)))     {

#line 103 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    return 
#line 103 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
Option__Match__None();
    }

#line 105 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    return 
#line 105 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
Regex__find(self, (text + start));
    }
}

#line 108 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"

int32_t Regex__count(Regex* self, char* text)
{
    {

#line 109 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
if ((self->preg == NULL))     {

#line 109 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    return 0;
    }

#line 110 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    int32_t count = 0;

#line 111 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    int32_t pos = 0;

#line 112 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    ZC_AUTO t_len = 
#line 112 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
strlen(text);

#line 113 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
while ((pos < t_len))     {

#line 114 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    char* sub = (text + pos);

#line 115 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
if ((
#line 115 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
tre_regexec(self->preg, sub, 0, 0, 0) == 0))     {

#line 116 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
(count = (count + 1));

#line 117 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
(pos = (pos + 1));
    }
 else     {

#line 119 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
break;
    }
    }

#line 122 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    return count;
    }
}

#line 125 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"

Vec__String Regex__split(Regex* self, char* text)
{
    {

#line 126 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    int __z_drop_flag_parts = 1; Vec__String parts = 
#line 126 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
Vec__String__new();

#line 127 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
if ((self->preg == NULL))     {

#line 128 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
Vec__String__push((&parts), String__from(text));

#line 129 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    return ({ ZC_AUTO _z_ret_mv = parts; memset(&parts, 0, sizeof(_z_ret_mv)); __z_drop_flag_parts = 0; 
#line 126 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    if (__z_drop_flag_parts) Vec__String__Drop__glue(&parts);
_z_ret_mv; });
    }

#line 131 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    ZC_AUTO t_len = 
#line 131 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
strlen(text);

#line 132 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    int32_t last_pos = 0;

#line 133 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    int32_t pos = 0;

#line 134 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
while ((pos < t_len))     {

#line 135 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    char* sub = (text + pos);

#line 136 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
if ((
#line 136 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
tre_regexec(self->preg, sub, 0, 0, 0) == 0))     {

#line 137 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
if ((pos > last_pos))     {

#line 138 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    char* before = (text + last_pos);

#line 139 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    int32_t part_len = (pos - last_pos);

#line 140 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    int __z_drop_flag_v = 1; Vec__char v = 
#line 140 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
Vec__char__new();

#line 141 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
for (int i = (int)(0); i < part_len; i = (i + 1))     {

#line 142 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
Vec__char__push((&v), before[i]);
    }

#line 144 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
Vec__char__push((&v), 0);

#line 145 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
Vec__String__push((&parts), (struct String){.vec = ({ __z_drop_flag_v = 0; v; })});

#line 140 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    if (__z_drop_flag_v) Vec__char__Drop__glue(&v);
    }

#line 147 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
(last_pos = (pos + 1));

#line 148 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
(pos = (pos + 1));
    }
 else     {

#line 150 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
(pos = (pos + 1));
    }
    }

#line 153 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
if ((last_pos < t_len))     {

#line 154 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
Vec__String__push((&parts), String__from((text + last_pos)));
    }

#line 156 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    return ({ ZC_AUTO _z_ret_mv = parts; memset(&parts, 0, sizeof(_z_ret_mv)); __z_drop_flag_parts = 0; 
#line 126 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    if (__z_drop_flag_parts) Vec__String__Drop__glue(&parts);
_z_ret_mv; });

#line 126 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    if (__z_drop_flag_parts) Vec__String__Drop__glue(&parts);
    }
}

#line 159 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"

char* Regex__pattern(Regex* self)
{
    {

#line 160 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    return self->pattern;
    }
}

#line 163 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"

int32_t Regex__flags(Regex* self)
{
    {

#line 164 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    return self->flags;
    }
}

#line 167 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"

bool Regex__is_valid_pattern(char* pattern)
{
    {

#line 168 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    Regex test_regex = 
#line 168 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
Regex__compile(pattern);

#line 169 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    bool valid = 
#line 169 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
Regex__is_valid((&test_regex));

#line 170 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
Regex__destroy(&test_regex);

#line 171 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    return valid;
    }
}

#line 174 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"

void Regex__destroy(Regex* self)
{
    {

#line 175 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
if ((self->preg != NULL))     {

#line 176 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
tre_regfree(self->preg);

#line 177 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
free(self->preg);
    }
    }
}

#line 25 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"

Match Match__new(char* text, int32_t start, int32_t len)
{
    {

#line 26 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    return (struct Match){.text = text, .start = start, .len = len};
    }
}

#line 29 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"

char* Match__as_string(Match* self)
{
    {

#line 30 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    char* s = 
#line 30 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
malloc((self->len + 1));

#line 31 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
memcpy(s, (self->text + self->start), self->len);

#line 32 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
(s[self->len] = 0);

#line 33 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    return s;
    }
}

#line 36 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"

int32_t Match__end(Match* self)
{
    {

#line 37 "/home/zuhaitz/zenc-lang/zenc/std/regex.zc"
    return (self->start + self->len);
    }
}

#line 71 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

String String__new(char* s)
{
    {

#line 72 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    ZC_AUTO len = 
#line 72 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
strlen(s);

#line 73 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    int __z_drop_flag_v = 1; Vec__char v = 
#line 73 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__new();

#line 75 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
for (int i = (int)(0); i < len; i = (i + 1))     {

#line 76 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&v), s[i]);
    }

#line 78 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&v), 0);

#line 81 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    char* d = v.data;

#line 82 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t l = v.len;

#line 83 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t c = v.cap;

#line 86 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__forget((&v));

#line 88 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    { String _z_ret = (struct String){.vec = (Vec__char){.data = d, .len = l, .cap = c}}; 
#line 73 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_v) Vec__char__Drop__glue(&v);
return _z_ret; }

#line 73 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_v) Vec__char__Drop__glue(&v);
    }
}

#line 91 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

String String__from(char* s)
{
    {

#line 92 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 
#line 92 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__new(s);
    }
}

#line 95 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

String String__from_rune(int32_t r)
{
    {

#line 96 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    int __z_drop_flag_s = 1; String s = 
#line 96 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__new("");

#line 97 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__push_rune(&s, r);

#line 98 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return ({ ZC_AUTO _z_ret_mv = s; memset(&s, 0, sizeof(_z_ret_mv)); __z_drop_flag_s = 0; 
#line 96 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_s) String__Drop__glue(&s);
_z_ret_mv; });

#line 96 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_s) String__Drop__glue(&s);
    }
}

#line 101 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

String String__from_runes(int32_t* runes, size_t count)
{
    {

#line 102 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    int __z_drop_flag_s = 1; String s = 
#line 102 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__new("");

#line 103 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
for (size_t i = (size_t)(0); i < count; i = (i + 1))     {

#line 104 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__push_rune(&s, runes[i]);
    }

#line 106 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return ({ ZC_AUTO _z_ret_mv = s; memset(&s, 0, sizeof(_z_ret_mv)); __z_drop_flag_s = 0; 
#line 102 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_s) String__Drop__glue(&s);
_z_ret_mv; });

#line 102 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_s) String__Drop__glue(&s);
    }
}

#line 109 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

char* String__c_str(String* self)
{
    {

#line 110 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return self->vec.data;
    }
}

#line 113 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

char* String__to_string(String* self)
{
    {

#line 114 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 
#line 114 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__c_str(self);
    }
}

#line 117 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

void String__destroy(String* self)
{
    {

#line 118 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__free((&self->vec));
    }
}

#line 121 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

void String__forget(String* self)
{
    {

#line 122 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__forget((&self->vec));
    }
}

#line 125 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

void String__append(String* self, String* other)
{
    {

#line 127 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((self->vec.len > 0))     {

#line 128 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(self->vec.len = (self->vec.len - 1));
    }

#line 131 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t other_len = (*other).vec.len;

#line 132 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
for (int i = (int)(0); i < other_len; i = (i + 1))     {

#line 133 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&self->vec), Vec__char__get((&(*other).vec), i));
    }
    }
}

#line 137 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

void String__append_c(String* self, char* s)
{
    {

#line 138 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((self->vec.len > 0))     {

#line 139 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(self->vec.len = (self->vec.len - 1));
    }

#line 141 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    ZC_AUTO len = 
#line 141 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
strlen(s);

#line 142 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
for (int i = (int)(0); i < len; i = (i + 1))     {

#line 143 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&self->vec), s[i]);
    }

#line 145 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&self->vec), 0);
    }
}

#line 148 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

void String__push_rune(String* self, int32_t r)
{
    {

#line 149 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((self->vec.len > 0))     {

#line 150 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(self->vec.len = (self->vec.len - 1));
    }

#line 153 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    uint32_t val = ((uint32_t)(r));

#line 154 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((val < 128))     {

#line 155 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&self->vec), ((char)(val)));
    }

#line 156 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
 else if ((val < 2048))     {

#line 157 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&self->vec), ((char)((192 | (val >> 6)))));

#line 158 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&self->vec), ((char)((128 | (val & 63)))));
    }

#line 159 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
 else if ((val < 65536))     {

#line 160 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&self->vec), ((char)((224 | (val >> 12)))));

#line 161 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&self->vec), ((char)((128 | ((val >> 6) & 63)))));

#line 162 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&self->vec), ((char)((128 | (val & 63)))));
    }
 else     {

#line 164 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&self->vec), ((char)((240 | (val >> 18)))));

#line 165 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&self->vec), ((char)((128 | ((val >> 12) & 63)))));

#line 166 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&self->vec), ((char)((128 | ((val >> 6) & 63)))));

#line 167 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&self->vec), ((char)((128 | (val & 63)))));
    }

#line 169 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&self->vec), 0);
    }
}

#line 172 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

void String__append_c_ptr(String* ptr, char* s)
{
    {

#line 173 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((ptr->vec.len > 0))     {

#line 174 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(ptr->vec.len = (ptr->vec.len - 1));
    }

#line 176 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    ZC_AUTO len = 
#line 176 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
strlen(s);

#line 177 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
for (int i = (int)(0); i < len; i = (i + 1))     {

#line 178 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&ptr->vec), s[i]);
    }

#line 180 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&ptr->vec), 0);
    }
}

#line 183 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

String String__add(String* self, String* other)
{
    {

#line 184 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    int __z_drop_flag_new_s = 1; String new_s = 
#line 184 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__from(String__c_str(self));

#line 185 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__append((&new_s), other);

#line 187 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    char* d = new_s.vec.data;

#line 188 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t l = new_s.vec.len;

#line 189 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t c = new_s.vec.cap;

#line 190 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__forget((&new_s));

#line 192 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    { String _z_ret = (struct String){.vec = (Vec__char){.data = d, .len = l, .cap = c}}; 
#line 184 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_new_s) String__Drop__glue(&new_s);
return _z_ret; }

#line 184 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_new_s) String__Drop__glue(&new_s);
    }
}

#line 195 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

void String__add_assign(String* self, String* other)
{
    {

#line 196 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__append(self, other);
    }
}

#line 199 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

bool String__eq(String* self, String* other)
{
    {

#line 200 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    int zero = 0;

#line 201 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return (
#line 201 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
strcmp(String__c_str(self), String__c_str((&(*other)))) == zero);
    }
}

#line 204 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

bool String__neq(String* self, String* other)
{
    {

#line 205 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return (!
#line 205 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__eq(self, other));
    }
}

#line 208 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

int32_t String__compare(String* self, String* other)
{
    {

#line 209 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 
#line 209 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
strcmp(String__c_str(self), String__c_str((&(*other))));
    }
}

#line 212 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

bool String__lt(String* self, String* other)
{
    {

#line 213 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return (
#line 213 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__compare(self, other) < 0);
    }
}

#line 216 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

bool String__gt(String* self, String* other)
{
    {

#line 217 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return (
#line 217 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__compare(self, other) > 0);
    }
}

#line 220 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

bool String__le(String* self, String* other)
{
    {

#line 221 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return (
#line 221 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__compare(self, other) <= 0);
    }
}

#line 224 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

bool String__ge(String* self, String* other)
{
    {

#line 225 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return (
#line 225 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__compare(self, other) >= 0);
    }
}

#line 228 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

int32_t String__compare_ignore_case(String* self, String* other)
{
    {

#line 229 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 
#line 229 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
_z_internal_str_case_cmp(self->vec.data, (*other).vec.data);
    }
}

#line 232 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

bool String__eq_ignore_case(String* self, String* other)
{
    {

#line 233 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    int32_t zero = 0;

#line 234 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return (
#line 234 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__compare_ignore_case(self, other) == zero);
    }
}

#line 237 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

bool String__eq_str(String* self, char* s)
{
    {

#line 238 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    int zero = 0;

#line 239 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return (
#line 239 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
strcmp(String__c_str(self), s) == zero);
    }
}

#line 242 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

size_t String__length(String* self)
{
    {

#line 243 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((self->vec.len == 0))     {

#line 243 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 0;
    }

#line 244 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return (self->vec.len - 1);
    }
}

#line 247 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

String String__substring(String* self, size_t start, size_t len)
{
    {

#line 248 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if (((start + len) > 
#line 248 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__length(self)))     {

#line 249 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
__zenc_panic("substring out of bounds");
    }

#line 251 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    int __z_drop_flag_v = 1; Vec__char v = 
#line 251 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__new();

#line 252 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
for (size_t i = (size_t)(0); i < len; i = (i + 1))     {

#line 253 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&v), Vec__char__get((&self->vec), (start + i)));
    }

#line 255 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&v), 0);

#line 257 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    char* d = v.data;

#line 258 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t l = v.len;

#line 259 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t c = v.cap;

#line 260 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__forget((&v));

#line 262 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    { String _z_ret = (struct String){.vec = (Vec__char){.data = d, .len = l, .cap = c}}; 
#line 251 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_v) Vec__char__Drop__glue(&v);
return _z_ret; }

#line 251 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_v) Vec__char__Drop__glue(&v);
    }
}

#line 264 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

bool String__contains_str(String* self, char* target)
{
    {

#line 265 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 
#line 265 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Option__size_t__is_some((Option__size_t[]){String__find_str(self, target)});
    }
}

#line 268 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

Option__size_t String__find_str(String* self, char* target)
{
    {

#line 269 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    ZC_AUTO t_len = 
#line 269 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
strlen(target);

#line 270 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((t_len == 0))     {

#line 270 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 
#line 270 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Option__size_t__Some(0);
    }

#line 271 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t s_len = 
#line 271 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__length(self);

#line 272 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((t_len > s_len))     {

#line 272 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 
#line 272 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Option__size_t__None();
    }

#line 274 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
for (size_t i = (size_t)(0); i <= (s_len - t_len); i = (i + 1))     {

#line 275 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    bool is_match = true;

#line 276 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
for (size_t k = (size_t)(0); k < t_len; k = (k + 1))     {

#line 277 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((
#line 277 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__get((&self->vec), (i + k)) != target[k]))     {

#line 278 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(is_match = false);

#line 279 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
break;
    }
    }

#line 282 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if (is_match)     {

#line 282 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 
#line 282 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Option__size_t__Some(i);
    }
    }

#line 284 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 
#line 284 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Option__size_t__None();
    }
}

#line 287 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

Vec__size_t String__find_all_str(String* self, char* target)
{
    {

#line 288 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    int __z_drop_flag_indices = 1; Vec__size_t indices = 
#line 288 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__size_t__new();

#line 289 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    ZC_AUTO t_len = 
#line 289 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
strlen(target);

#line 290 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((t_len == 0))     {

#line 290 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return ({ ZC_AUTO _z_ret_mv = indices; memset(&indices, 0, sizeof(_z_ret_mv)); __z_drop_flag_indices = 0; 
#line 288 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_indices) Vec__size_t__Drop__glue(&indices);
_z_ret_mv; });
    }

#line 291 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t s_len = 
#line 291 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__length(self);

#line 292 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((t_len > s_len))     {

#line 292 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return ({ ZC_AUTO _z_ret_mv = indices; memset(&indices, 0, sizeof(_z_ret_mv)); __z_drop_flag_indices = 0; 
#line 288 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_indices) Vec__size_t__Drop__glue(&indices);
_z_ret_mv; });
    }

#line 294 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
for (size_t i = (size_t)(0); i <= (s_len - t_len); i = (i + 1))     {

#line 295 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    bool is_match = true;

#line 296 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
for (size_t k = (size_t)(0); k < t_len; k = (k + 1))     {

#line 297 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((
#line 297 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__get((&self->vec), (i + k)) != target[k]))     {

#line 298 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(is_match = false);

#line 299 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
break;
    }
    }

#line 302 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if (is_match)     {

#line 303 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__size_t__push((&indices), i);
    }
    }

#line 307 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return ({ ZC_AUTO _z_ret_mv = indices; memset(&indices, 0, sizeof(_z_ret_mv)); __z_drop_flag_indices = 0; 
#line 288 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_indices) Vec__size_t__Drop__glue(&indices);
_z_ret_mv; });

#line 288 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_indices) Vec__size_t__Drop__glue(&indices);
    }
}

#line 310 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

String String__to_lowercase(String* self)
{
    {

#line 311 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t len = 
#line 311 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__length(self);

#line 312 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    int __z_drop_flag_v = 1; Vec__char v = 
#line 312 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__new();

#line 313 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
for (size_t i = (size_t)(0); i < len; i = (i + 1))     {

#line 314 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    char c = 
#line 314 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__get((&self->vec), i);

#line 315 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if (((c >= 'A') && (c <= 'Z')))     {

#line 316 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&v), ((char)((((int32_t)(c)) + 32))));
    }
 else     {

#line 318 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&v), c);
    }
    }

#line 321 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&v), 0);

#line 322 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    char* d = v.data;

#line 323 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t l = v.len;

#line 324 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t c_cap = v.cap;

#line 325 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__forget((&v));

#line 326 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    { String _z_ret = (struct String){.vec = (Vec__char){.data = d, .len = l, .cap = c_cap}}; 
#line 312 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_v) Vec__char__Drop__glue(&v);
return _z_ret; }

#line 312 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_v) Vec__char__Drop__glue(&v);
    }
}

#line 329 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

String String__pad_right(String* self, size_t target_len, char pad_char)
{
    {

#line 330 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t current_len = 
#line 330 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__length(self);

#line 331 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((current_len >= target_len))     {

#line 332 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 
#line 332 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__from(String__c_str(self));
    }

#line 335 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    int __z_drop_flag_v = 1; Vec__char v = 
#line 335 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__new();

#line 336 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
for (size_t i = (size_t)(0); i < current_len; i = (i + 1))     {

#line 337 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&v), Vec__char__get((&self->vec), i));
    }

#line 339 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
for (size_t i = (size_t)(current_len); i < target_len; i = (i + 1))     {

#line 340 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&v), pad_char);
    }

#line 342 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&v), 0);

#line 344 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    char* d = v.data;

#line 345 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t l = v.len;

#line 346 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t c_cap = v.cap;

#line 347 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__forget((&v));

#line 348 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    { String _z_ret = (struct String){.vec = (Vec__char){.data = d, .len = l, .cap = c_cap}}; 
#line 335 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_v) Vec__char__Drop__glue(&v);
return _z_ret; }

#line 335 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_v) Vec__char__Drop__glue(&v);
    }
}

#line 351 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

String String__pad_left(String* self, size_t target_len, char pad_char)
{
    {

#line 352 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t current_len = 
#line 352 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__length(self);

#line 353 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((current_len >= target_len))     {

#line 354 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 
#line 354 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__from(String__c_str(self));
    }

#line 357 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    int __z_drop_flag_v = 1; Vec__char v = 
#line 357 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__new();

#line 358 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t diff = (target_len - current_len);

#line 359 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
for (size_t i = (size_t)(0); i < diff; i = (i + 1))     {

#line 360 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&v), pad_char);
    }

#line 362 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
for (size_t i = (size_t)(0); i < current_len; i = (i + 1))     {

#line 363 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&v), Vec__char__get((&self->vec), i));
    }

#line 365 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&v), 0);

#line 367 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    char* d = v.data;

#line 368 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t l = v.len;

#line 369 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t c_cap = v.cap;

#line 370 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__forget((&v));

#line 371 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    { String _z_ret = (struct String){.vec = (Vec__char){.data = d, .len = l, .cap = c_cap}}; 
#line 357 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_v) Vec__char__Drop__glue(&v);
return _z_ret; }

#line 357 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_v) Vec__char__Drop__glue(&v);
    }
}

#line 374 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

String String__to_uppercase(String* self)
{
    {

#line 375 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t len = 
#line 375 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__length(self);

#line 376 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    int __z_drop_flag_v = 1; Vec__char v = 
#line 376 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__new();

#line 377 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
for (size_t i = (size_t)(0); i < len; i = (i + 1))     {

#line 378 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    char c = 
#line 378 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__get((&self->vec), i);

#line 379 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if (((c >= 'a') && (c <= 'z')))     {

#line 380 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&v), ((char)((((int32_t)(c)) - 32))));
    }
 else     {

#line 382 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&v), c);
    }
    }

#line 385 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&v), 0);

#line 386 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    char* d = v.data;

#line 387 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t l = v.len;

#line 388 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t c_cap = v.cap;

#line 389 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__forget((&v));

#line 390 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    { String _z_ret = (struct String){.vec = (Vec__char){.data = d, .len = l, .cap = c_cap}}; 
#line 376 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_v) Vec__char__Drop__glue(&v);
return _z_ret; }

#line 376 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_v) Vec__char__Drop__glue(&v);
    }
}

#line 393 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

Option__size_t String__find(String* self, char target)
{
    {

#line 394 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t len = 
#line 394 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__length(self);

#line 395 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
for (size_t i = (size_t)(0); i < len; i = (i + 1))     {

#line 396 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((
#line 396 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__get((&self->vec), i) == target))     {

#line 397 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 
#line 397 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Option__size_t__Some(i);
    }
    }

#line 400 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 
#line 400 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Option__size_t__None();
    }
}

#line 403 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

void String__print(String* self)
{
    {

#line 404 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
printf("%s", String__c_str(self));

#line 405 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
fflush(stdout);
    }
}

#line 408 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

void String__println(String* self)
{
    {

#line 409 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
printf("%s\n", String__c_str(self));
    }
}

#line 412 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

bool String__is_empty(String* self)
{
    {

#line 413 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return (
#line 413 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__length(self) == 0);
    }
}

#line 416 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

bool String__contains(String* self, char target)
{
    {

#line 417 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 
#line 417 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Option__size_t__is_some((__typeof__((String__find(self, target)))[]){String__find(self, target)});
    }
}

#line 420 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

bool String__starts_with(String* self, char* prefix)
{
    {

#line 421 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    ZC_AUTO plen = 
#line 421 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
strlen(prefix);

#line 422 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((plen > 
#line 422 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__length(self)))     {

#line 422 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return false;
    }

#line 423 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    int zero = 0;

#line 424 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return (
#line 424 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
strncmp(String__c_str(self), prefix, plen) == zero);
    }
}

#line 427 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

bool String__ends_with(String* self, char* suffix)
{
    {

#line 428 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    ZC_AUTO slen = 
#line 428 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
strlen(suffix);

#line 429 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t len = 
#line 429 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__length(self);

#line 430 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((slen > len))     {

#line 430 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return false;
    }

#line 431 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    int32_t offset = ((int32_t)((len - slen)));

#line 432 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    int zero = 0;

#line 433 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return (
#line 433 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
strcmp((String__c_str(self) + offset), suffix) == zero);
    }
}

#line 436 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

void String__reserve(String* self, size_t cap)
{
    {

#line 437 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__grow_to_fit((&self->vec), (cap + 1));
    }
}

#line 440 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

void String__free(String* self)
{
    {

#line 441 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__free((&self->vec));
    }
}

#line 444 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

size_t String__utf8_seq_len(char first_byte)
{
    {

#line 445 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    int32_t b = ((int32_t)(first_byte));

#line 446 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if (((b & 128) == 0))     {

#line 446 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 1;
    }

#line 447 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if (((b & 224) == 192))     {

#line 447 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 2;
    }

#line 448 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if (((b & 240) == 224))     {

#line 448 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 3;
    }

#line 449 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if (((b & 248) == 240))     {

#line 449 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 4;
    }

#line 450 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 1;
    }
}

#line 453 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

size_t String__utf8_len(String* self)
{
    {

#line 454 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t count = 0;

#line 455 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t i = 0;

#line 456 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t len = 
#line 456 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__length(self);

#line 457 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
while ((i < len))     {

#line 458 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    char c = 
#line 458 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__get((&self->vec), i);

#line 459 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(i = (i + 
#line 459 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__utf8_seq_len(c)));

#line 460 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(count = (count + 1));
    }

#line 462 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return count;
    }
}

#line 465 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

String String__utf8_at(String* self, size_t idx)
{
    {

#line 466 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t count = 0;

#line 467 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t i = 0;

#line 468 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t len = 
#line 468 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__length(self);

#line 469 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
while ((i < len))     {

#line 470 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    char c = 
#line 470 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__get((&self->vec), i);

#line 471 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t seq = 
#line 471 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__utf8_seq_len(c);

#line 473 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((count == idx))     {

#line 474 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 
#line 474 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__substring(self, i, seq);
    }

#line 477 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(i = (i + seq));

#line 478 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(count = (count + 1));
    }

#line 480 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 
#line 480 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__new("");
    }
}

#line 483 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

int32_t String__utf8_get(String* self, size_t idx)
{
    {

#line 484 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t count = 0;

#line 485 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t i = 0;

#line 486 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t len = 
#line 486 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__length(self);

#line 487 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
while ((i < len))     {

#line 488 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    uint8_t c = ((uint8_t)(
#line 488 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__get((&self->vec), i)));

#line 489 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t seq = 
#line 489 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__utf8_seq_len(((char)(c)));

#line 491 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((count == idx))     {

#line 492 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((seq == 1))     {

#line 492 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return ((int32_t)(c));
    }

#line 493 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((seq == 2))     {

#line 493 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return ((int32_t)(((((int32_t)((c & 31))) << 6) | (((int32_t)(((uint8_t)(
#line 493 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__get((&self->vec), (i + 1)))))) & 63))));
    }

#line 494 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((seq == 3))     {

#line 494 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return ((int32_t)((((((int32_t)((c & 15))) << 12) | ((((int32_t)(((uint8_t)(
#line 494 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__get((&self->vec), (i + 1)))))) & 63) << 6)) | (((int32_t)(((uint8_t)(Vec__char__get((&self->vec), (i + 2)))))) & 63))));
    }

#line 495 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((seq == 4))     {

#line 495 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return ((int32_t)(((((((int32_t)((c & 7))) << 18) | ((((int32_t)(((uint8_t)(
#line 495 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__get((&self->vec), (i + 1)))))) & 63) << 12)) | ((((int32_t)(((uint8_t)(Vec__char__get((&self->vec), (i + 2)))))) & 63) << 6)) | (((int32_t)(((uint8_t)(Vec__char__get((&self->vec), (i + 3)))))) & 63))));
    }
    }

#line 498 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(i = (i + seq));

#line 499 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(count = (count + 1));
    }

#line 501 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return ((int32_t)(0));
    }
}

#line 504 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

Vec__int32_t String__runes(String* self)
{
    {

#line 505 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    int __z_drop_flag_v = 1; Vec__int32_t v = 
#line 505 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__int32_t__new();

#line 506 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t i = 0;

#line 507 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t len = 
#line 507 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__length(self);

#line 508 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
while ((i < len))     {

#line 509 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    uint8_t c = ((uint8_t)(
#line 509 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__get((&self->vec), i)));

#line 510 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t seq = 
#line 510 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__utf8_seq_len(((char)(c)));

#line 511 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    int32_t val = 0;

#line 512 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((seq == 1))     {

#line 513 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(val = ((int32_t)(c)));
    }

#line 514 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
 else if ((seq == 2))     {

#line 515 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(val = ((int32_t)(((((int32_t)((c & 31))) << 6) | (((int32_t)(((uint8_t)(
#line 515 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__get((&self->vec), (i + 1)))))) & 63)))));
    }

#line 516 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
 else if ((seq == 3))     {

#line 517 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(val = ((int32_t)((((((int32_t)((c & 15))) << 12) | ((((int32_t)(((uint8_t)(
#line 517 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__get((&self->vec), (i + 1)))))) & 63) << 6)) | (((int32_t)(((uint8_t)(Vec__char__get((&self->vec), (i + 2)))))) & 63)))));
    }

#line 518 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
 else if ((seq == 4))     {

#line 519 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(val = ((int32_t)(((((((int32_t)((c & 7))) << 18) | ((((int32_t)(((uint8_t)(
#line 519 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__get((&self->vec), (i + 1)))))) & 63) << 12)) | ((((int32_t)(((uint8_t)(Vec__char__get((&self->vec), (i + 2)))))) & 63) << 6)) | (((int32_t)(((uint8_t)(Vec__char__get((&self->vec), (i + 3)))))) & 63)))));
    }

#line 521 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__int32_t__push((&v), val);

#line 522 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(i = (i + seq));
    }

#line 524 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return ({ ZC_AUTO _z_ret_mv = v; memset(&v, 0, sizeof(_z_ret_mv)); __z_drop_flag_v = 0; 
#line 505 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_v) Vec__int32_t__Drop__glue(&v);
_z_ret_mv; });

#line 505 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_v) Vec__int32_t__Drop__glue(&v);
    }
}

#line 527 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

StringCharsIter String__iterator(String* self)
{
    {

#line 528 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 
#line 528 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__chars(self);
    }
}

#line 531 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

StringCharsIter String__chars(String* self)
{
    {

#line 532 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return (struct StringCharsIter){.data = 
#line 533 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__c_str(self), .len = 
#line 534 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__length(self)};
    }
}

#line 539 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

String String__from_runes_vec(Vec__int32_t runes)
{

#line 539 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    int __z_drop_flag_runes = 1;
    {

#line 540 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    int __z_drop_flag_s = 1; String s = 
#line 540 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__new("");

#line 541 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
for (size_t i = (size_t)(0); i < runes.len; i = (i + 1))     {

#line 542 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__push_rune((&s), Vec__int32_t__get((&runes), i));
    }

#line 544 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return ({ ZC_AUTO _z_ret_mv = s; memset(&s, 0, sizeof(_z_ret_mv)); __z_drop_flag_s = 0; 
#line 540 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_s) String__Drop__glue(&s);

#line 539 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_runes) Vec__int32_t__Drop__glue(&runes);
_z_ret_mv; });

#line 540 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_s) String__Drop__glue(&s);
    }

#line 539 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_runes) Vec__int32_t__Drop__glue(&runes);
}

#line 547 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

void String__insert_rune(String* self, size_t idx, int32_t r)
{
    {

#line 548 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t i = 0;

#line 549 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t count = 0;

#line 550 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t len = 
#line 550 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__length(self);

#line 551 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
while (((i < len) && (count < idx)))     {

#line 552 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(i = (i + 
#line 552 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__utf8_seq_len(Vec__char__get((&self->vec), i))));

#line 553 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(count = (count + 1));
    }

#line 556 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    uint32_t val = ((uint32_t)(r));

#line 557 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((val < 128))     {

#line 558 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__insert((&self->vec), i, ((char)(val)));
    }

#line 559 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
 else if ((val < 2048))     {

#line 560 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__insert((&self->vec), i, ((char)((192 | (val >> 6)))));

#line 561 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__insert((&self->vec), (i + 1), ((char)((128 | (val & 63)))));
    }

#line 562 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
 else if ((val < 65536))     {

#line 563 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__insert((&self->vec), i, ((char)((224 | (val >> 12)))));

#line 564 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__insert((&self->vec), (i + 1), ((char)((128 | ((val >> 6) & 63)))));

#line 565 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__insert((&self->vec), (i + 2), ((char)((128 | (val & 63)))));
    }
 else     {

#line 567 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__insert((&self->vec), i, ((char)((240 | (val >> 18)))));

#line 568 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__insert((&self->vec), (i + 1), ((char)((128 | ((val >> 12) & 63)))));

#line 569 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__insert((&self->vec), (i + 2), ((char)((128 | ((val >> 6) & 63)))));

#line 570 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__insert((&self->vec), (i + 3), ((char)((128 | (val & 63)))));
    }
    }
}

#line 574 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

int32_t String__remove_rune_at(String* self, size_t idx)
{
    {

#line 575 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t i = 0;

#line 576 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t count = 0;

#line 577 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t len = 
#line 577 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__length(self);

#line 578 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
while (((i < len) && (count < idx)))     {

#line 579 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(i = (i + 
#line 579 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__utf8_seq_len(Vec__char__get((&self->vec), i))));

#line 580 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(count = (count + 1));
    }

#line 583 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((i >= len))     {

#line 583 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return ((int32_t)(0));
    }

#line 585 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    uint8_t c = ((uint8_t)(
#line 585 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__get((&self->vec), i)));

#line 586 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t seq = 
#line 586 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__utf8_seq_len(((char)(c)));

#line 587 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    int32_t val = 0;

#line 588 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((seq == 1))     {

#line 588 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(val = ((int32_t)(c)));
    }

#line 589 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
 else if ((seq == 2))     {

#line 589 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(val = ((int32_t)(((((int32_t)((c & 31))) << 6) | (((int32_t)(((uint8_t)(
#line 589 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__get((&self->vec), (i + 1)))))) & 63)))));
    }

#line 590 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
 else if ((seq == 3))     {

#line 590 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(val = ((int32_t)((((((int32_t)((c & 15))) << 12) | ((((int32_t)(((uint8_t)(
#line 590 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__get((&self->vec), (i + 1)))))) & 63) << 6)) | (((int32_t)(((uint8_t)(Vec__char__get((&self->vec), (i + 2)))))) & 63)))));
    }

#line 591 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
 else if ((seq == 4))     {

#line 591 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(val = ((int32_t)(((((((int32_t)((c & 7))) << 18) | ((((int32_t)(((uint8_t)(
#line 591 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__get((&self->vec), (i + 1)))))) & 63) << 12)) | ((((int32_t)(((uint8_t)(Vec__char__get((&self->vec), (i + 2)))))) & 63) << 6)) | (((int32_t)(((uint8_t)(Vec__char__get((&self->vec), (i + 3)))))) & 63)))));
    }

#line 593 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
for (size_t j = (size_t)(0); j < seq; j = (j + 1))     {

#line 594 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__remove((&self->vec), i);
    }

#line 596 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return val;
    }
}

#line 599 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

String String__utf8_substr(String* self, size_t start_idx, size_t num_chars)
{
    {

#line 600 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((num_chars == 0))     {

#line 600 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 
#line 600 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__new("");
    }

#line 602 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t byte_start = 0;

#line 603 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t byte_len = 0;

#line 605 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t count = 0;

#line 606 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t i = 0;

#line 607 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t len = 
#line 607 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__length(self);

#line 608 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    bool found_start = false;

#line 610 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
while ((i < len))     {

#line 612 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if (((!found_start) && (count == start_idx)))     {

#line 613 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(byte_start = i);

#line 614 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(found_start = true);

#line 616 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(count = 0);
    }

#line 617 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
 else if ((!found_start))     {

#line 619 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    char c = 
#line 619 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__get((&self->vec), i);

#line 620 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(i = (i + 
#line 620 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__utf8_seq_len(c)));

#line 621 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(count = (count + 1));

#line 622 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
continue;
    }

#line 626 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((count < num_chars))     {

#line 627 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    char c = 
#line 627 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__get((&self->vec), i);

#line 628 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t seq = 
#line 628 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__utf8_seq_len(c);

#line 629 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(byte_len = (byte_len + seq));

#line 630 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(i = (i + seq));

#line 631 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(count = (count + 1));
    }
 else     {

#line 633 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
break;
    }
    }

#line 637 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((!found_start))     {

#line 637 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 
#line 637 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__new("");
    }

#line 639 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 
#line 639 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__substring(self, byte_start, byte_len);
    }
}

#line 641 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

Vec__String String__split(String* self, char delim)
{
    {

#line 642 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    int __z_drop_flag_parts = 1; Vec__String parts = 
#line 642 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__String__new();

#line 643 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t len = 
#line 643 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__length(self);

#line 644 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((len == 0))     {

#line 644 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return ({ ZC_AUTO _z_ret_mv = parts; memset(&parts, 0, sizeof(_z_ret_mv)); __z_drop_flag_parts = 0; 
#line 642 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_parts) Vec__String__Drop__glue(&parts);
_z_ret_mv; });
    }

#line 646 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t start = 0;

#line 647 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t i = 0;

#line 649 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
while ((i < len))     {

#line 650 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((
#line 650 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__get((&self->vec), i) == delim))     {

#line 652 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__String__push((&parts), String__substring(self, start, (i - start)));

#line 653 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(start = (i + 1));
    }

#line 655 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(i = (i + 1));
    }

#line 659 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((start <= len))     {

#line 660 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__String__push((&parts), String__substring(self, start, (len - start)));
    }

#line 663 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return ({ ZC_AUTO _z_ret_mv = parts; memset(&parts, 0, sizeof(_z_ret_mv)); __z_drop_flag_parts = 0; 
#line 642 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_parts) Vec__String__Drop__glue(&parts);
_z_ret_mv; });

#line 642 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_parts) Vec__String__Drop__glue(&parts);
    }
}

#line 666 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

String String__trim(String* self)
{
    {

#line 667 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t start = 0;

#line 668 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t len = 
#line 668 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__length(self);

#line 669 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t end = len;

#line 672 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
while ((start < len))     {

#line 673 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    char c = 
#line 673 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__get((&self->vec), start);

#line 674 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if (((((c != ' ') && (c != '\t')) && (c != '\n')) && (c != '\r')))     {

#line 675 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
break;
    }

#line 677 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(start = (start + 1));
    }

#line 680 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((start == len))     {

#line 681 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 
#line 681 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__new("");
    }

#line 685 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
while ((end > start))     {

#line 686 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    char c = 
#line 686 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__get((&self->vec), (end - 1));

#line 687 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if (((((c != ' ') && (c != '\t')) && (c != '\n')) && (c != '\r')))     {

#line 688 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
break;
    }

#line 690 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(end = (end - 1));
    }

#line 693 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 
#line 693 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__substring(self, start, (end - start));
    }
}

#line 696 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

String String__replace(String* self, char* target, char* replacement)
{
    {

#line 697 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    ZC_AUTO t_len = 
#line 697 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
strlen(target);

#line 698 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((t_len == 0))     {

#line 698 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 
#line 698 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__substring(self, 0, String__length(self));
    }

#line 701 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t s_len = 
#line 701 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__length(self);

#line 702 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    int __z_drop_flag_result = 1; String result = 
#line 702 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__new("");

#line 704 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t i = 0;

#line 705 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
while ((i < s_len))     {

#line 707 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if (((i + t_len) <= s_len))     {

#line 708 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    bool is_match = true;

#line 710 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
for (size_t k = (size_t)(0); k < t_len; k = (k + 1))     {

#line 711 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((
#line 711 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__get((&self->vec), (i + k)) != target[k]))     {

#line 712 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(is_match = false);

#line 713 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
break;
    }
    }

#line 717 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if (is_match)     {

#line 718 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    int __z_drop_flag_r_str = 1; String r_str = 
#line 718 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__new(replacement);

#line 719 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__append((&result), (&r_str));

#line 720 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(i = (i + t_len));

#line 721 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

#line 718 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_r_str) String__Drop__glue(&r_str);
continue;

#line 718 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_r_str) String__Drop__glue(&r_str);
    }
    }

#line 726 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    int __z_drop_flag_v = 1; Vec__char v = 
#line 726 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__new();

#line 727 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&v), Vec__char__get((&self->vec), i));

#line 728 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Vec__char__push((&v), 0);

#line 729 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    int __z_drop_flag_ch_s = 1; String ch_s = 
#line 729 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__new(v.data);

#line 730 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__append((&result), (&ch_s));

#line 731 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__destroy((&ch_s));

#line 732 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(i = (i + 1));

#line 729 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_ch_s) String__Drop__glue(&ch_s);

#line 726 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_v) Vec__char__Drop__glue(&v);
    }

#line 734 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return ({ ZC_AUTO _z_ret_mv = result; memset(&result, 0, sizeof(_z_ret_mv)); __z_drop_flag_result = 0; 
#line 702 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_result) String__Drop__glue(&result);
_z_ret_mv; });

#line 702 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    if (__z_drop_flag_result) String__Drop__glue(&result);
    }
}

#line 39 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

Option__int32_t StringCharsIter__next(StringCharsIter* self)
{
    {

#line 40 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((self->pos >= self->len))     {

#line 40 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 
#line 40 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Option__int32_t__None();
    }

#line 42 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    uint8_t c = ((uint8_t)(self->data[self->pos]));

#line 43 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    size_t seq = 
#line 43 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
String__utf8_seq_len(((char)(c)));

#line 45 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if (((self->pos + seq) > self->len))     {

#line 46 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(self->pos = self->len);

#line 47 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 
#line 47 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Option__int32_t__None();
    }

#line 50 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    int32_t val = 0;

#line 51 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
if ((seq == 1))     {

#line 52 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(val = ((int32_t)(c)));
    }

#line 53 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
 else if ((seq == 2))     {

#line 54 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(val = ((int32_t)(((((int32_t)((c & 31))) << 6) | (((int32_t)(((uint8_t)(self->data[(self->pos + 1)])))) & 63)))));
    }

#line 55 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
 else if ((seq == 3))     {

#line 56 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(val = ((int32_t)((((((int32_t)((c & 15))) << 12) | ((((int32_t)(((uint8_t)(self->data[(self->pos + 1)])))) & 63) << 6)) | (((int32_t)(((uint8_t)(self->data[(self->pos + 2)])))) & 63)))));
    }

#line 57 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
 else if ((seq == 4))     {

#line 58 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(val = ((int32_t)(((((((int32_t)((c & 7))) << 18) | ((((int32_t)(((uint8_t)(self->data[(self->pos + 1)])))) & 63) << 12)) | ((((int32_t)(((uint8_t)(self->data[(self->pos + 2)])))) & 63) << 6)) | (((int32_t)(((uint8_t)(self->data[(self->pos + 3)])))) & 63)))));
    }

#line 61 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
(self->pos = (self->pos + seq));

#line 62 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return 
#line 62 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
Option__int32_t__Some(val);
    }
}

#line 65 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"

StringCharsIter StringCharsIter__iterator(StringCharsIter* self)
{
    {

#line 66 "/home/zuhaitz/zenc-lang/zenc/std/string.zc"
    return (*self);
    }
}

int main() { _z_run_tests(); return 0; }
