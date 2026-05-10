
#ifndef ZC_COMPAT_H
#define ZC_COMPAT_H

#include "platform/compiler.h"

/* File extensions for mixed source compilation */
#define ZC_EXT_C ".c"
#define ZC_EXT_CPP ".cpp"
#define ZC_EXT_M ".m"
#define ZC_EXT_MM ".mm"
#define ZC_EXT_O ".o"
#define ZC_EXT_A ".a"
#define ZC_EXT_DYLIB ".dylib"
#define ZC_EXT_SO ".so"

#define ZC_IS_BACKEND_EXT(ext)                                                                     \
    (strcmp(ext, ZC_EXT_C) == 0 || strcmp(ext, ZC_EXT_CPP) == 0 || strcmp(ext, ZC_EXT_M) == 0 ||   \
     strcmp(ext, ZC_EXT_MM) == 0 || strcmp(ext, ZC_EXT_O) == 0 || strcmp(ext, ZC_EXT_A) == 0 ||    \
     strcmp(ext, ZC_EXT_DYLIB) == 0 || strcmp(ext, ZC_EXT_SO) == 0)

/* Centralized string definition for codegen emission */
#define ZC_TCC_COMPAT_STR                                                                          \
    "#ifdef __TINYC__\n"                                                                           \
    "#undef ZC_AUTO\n"                                                                             \
    "#define ZC_AUTO __auto_type\n"                                                                \
    "#undef ZC_AUTO_INIT\n"                                                                        \
    "#define ZC_AUTO_INIT(var, init) __typeof__((init)) var = (init)\n"                            \
    "\n"                                                                                           \
    "#ifndef __builtin_expect\n"                                                                   \
    "#define __builtin_expect(x, v) (x)\n"                                                         \
    "#endif\n"                                                                                     \
    "\n"                                                                                           \
    "#ifndef __builtin_unreachable\n"                                                              \
    "#define __builtin_unreachable()\n"                                                            \
    "#endif\n"                                                                                     \
    "#else\n"                                                                                      \
    "#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202300L\n"                               \
    "#define ZC_AUTO_INIT(var, init) auto var = (init)\n"                                          \
    "#else\n"                                                                                      \
    "#define ZC_AUTO_INIT(var, init) __auto_type var = (init)\n"                                   \
    "#endif\n"                                                                                     \
    "#endif\n"

#ifdef __SIZEOF_INT128__
#define _z_safe_i128(x) _Generic((x), __int128: (x), default: (__int128)0)
#define _z_safe_u128(x) _Generic((x), unsigned __int128: (x), default: (unsigned __int128)0)
#define _z_128_map , __int128 : "%s", unsigned __int128 : "%s"
#define _z_128_arg_map(x)                                                                          \
    , __int128 : _z_i128_str(_z_safe_i128(x)), unsigned __int128 : _z_u128_str(_z_safe_u128(x))
#else
#define _z_128_map
#define _z_128_arg_map(x)
#endif

#define ZC_C_GENERIC_STR                                                                           \
    "#ifdef __SIZEOF_INT128__\n"                                                                   \
    "static inline const char *_z_u128_str(unsigned __int128 val) {\n"                             \
    "    static _Thread_local char buf[40];\n"                                                     \
    "    if (val == 0) return \"0\";\n"                                                            \
    "    int i = 38;\n"                                                                            \
    "    buf[39] = 0;\n"                                                                           \
    "    while (val > 0) { buf[i--] = (char)((val % 10) + '0'); val /= 10; }\n"                    \
    "    return &buf[i + 1];\n"                                                                    \
    "}\n"                                                                                          \
    "static inline const char *_z_i128_str(__int128 val) {\n"                                      \
    "    static _Thread_local char buf[41];\n"                                                     \
    "    if (val == 0) return \"0\";\n"                                                            \
    "    int neg = val < 0;\n"                                                                     \
    "    unsigned __int128 uval = neg ? -val : val;\n"                                             \
    "    int i = 39;\n"                                                                            \
    "    buf[40] = 0;\n"                                                                           \
    "    while (uval > 0) { buf[i--] = (char)((uval % 10) + '0'); uval /= 10; }\n"                 \
    "    if (neg) buf[i--] = '-';\n"                                                               \
    "    return &buf[i + 1];\n"                                                                    \
    "}\n"                                                                                          \
    "#define _z_128_map ,__int128: \"%s\", unsigned __int128: \"%s\"\n"                            \
    "#else\n"                                                                                      \
    "#define _z_128_map\n"                                                                         \
    "#endif\n"                                                                                     \
    "#ifdef __OBJC__\n"                                                                            \
    "#define _z_objc_map ,id: \"%s\"\n"                                                            \
    "#define _z_objc_arg_map(x) ,id: [(id)(x) description].UTF8String\n"                           \
    "#else\n"                                                                                      \
    "#define _z_objc_map\n"                                                                        \
    "#define _z_objc_arg_map(x)\n"                                                                 \
    "#endif\n"                                                                                     \
    "\n"                                                                                           \
    "#define _z_str(x) _Generic((x), _Bool: \"%s\", char: \"%c\", "                                \
    "signed char: \"%c\", unsigned char: \"%u\", short: \"%d\", "                                  \
    "unsigned short: \"%u\", int: \"%d\", unsigned int: \"%u\", "                                  \
    "long: \"%ld\", unsigned long: \"%lu\", long long: \"%lld\", "                                 \
    "unsigned long long: \"%llu\", float: \"%f\", double: \"%f\", "                                \
    "char*: \"%s\", const char*: \"%s\", void*: \"%p\" _z_128_map _z_objc_map)\n"

#define ZC_C_ARG_GENERIC_STR                                                                       \
    "#ifdef __SIZEOF_INT128__\n"                                                                   \
    "#define _z_safe_i128(x) _Generic((x), __int128: (x), default: (__int128)0)\n"                 \
    "#define _z_safe_u128(x) _Generic((x), unsigned __int128: (x), default: (unsigned "            \
    "__int128)0)\n"                                                                                \
    "#define _z_128_arg_map(x) ,__int128: _z_i128_str(_z_safe_i128(x)), unsigned __int128: "       \
    "_z_u128_str(_z_safe_u128(x))\n"                                                               \
    "#else\n"                                                                                      \
    "#define _z_128_arg_map(x)\n"                                                                  \
    "#endif\n"                                                                                     \
    "#define _z_safe_bool(x) _Generic((x), _Bool: (x), default: (_Bool)0)\n"                       \
    "#define _z_arg(x) _Generic((x), _Bool: _z_bool_str(_z_safe_bool(x)) _z_128_arg_map(x), "      \
    "default: (x))\n"

#ifdef __cplusplus
#include <type_traits>

inline const char *_zc_fmt(bool)
{
    return "%d";
}
inline const char *_zc_fmt(char)
{
    return "%c";
}
inline const char *_zc_fmt(signed char)
{
    return "%c";
}
inline const char *_zc_fmt(unsigned char)
{
    return "%u";
}
inline const char *_zc_fmt(short)
{
    return "%d";
}
inline const char *_zc_fmt(unsigned short)
{
    return "%u";
}
inline const char *_zc_fmt(int)
{
    return "%d";
}
inline const char *_zc_fmt(unsigned int)
{
    return "%u";
}
inline const char *_zc_fmt(long)
{
    return "%ld";
}
inline const char *_zc_fmt(unsigned long)
{
    return "%lu";
}
inline const char *_zc_fmt(long long)
{
    return "%lld";
}
inline const char *_zc_fmt(unsigned long long)
{
    return "%llu";
}
inline const char *_zc_fmt(float)
{
    return "%f";
}
inline const char *_zc_fmt(double)
{
    return "%f";
}
inline const char *_zc_fmt(char *)
{
    return "%s";
}
inline const char *_zc_fmt(const char *)
{
    return "%s";
}
inline const char *_zc_fmt(void *)
{
    return "%p";
}

#define _z_str(x) _zc_fmt(x)

#ifdef __OBJC__
#include <objc/objc.h>
#include <objc/runtime.h>
#include <objc/message.h> // for direct calls if needed, but [x description] is fine

inline const char *_zc_fmt(id x)
{
    return [[x description] UTF8String];
}
inline const char *_zc_fmt(Class x)
{
    return class_getName(x);
}
inline const char *_zc_fmt(SEL x)
{
    return sel_getName(x);
}
// BOOL is signed char usually, already handled?
// "typedef signed char BOOL;" on standard apple headers.
// If it maps to signed char, `_zc_fmt(signed char)` handles it ("%c").
// We might want "YES"/"NO" for BOOL.
// But we can't distinguish typedefs in C++ function overloads easily if underlying type is same.
// We'll leave BOOL as %c or %d for now to avoid ambiguity errors.
#endif

#endif

#endif
