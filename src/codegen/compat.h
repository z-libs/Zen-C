
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

/* Generic selection string for C mode */
#define ZC_C_GENERIC_STR                                                                           \
    "#ifdef __OBJC__\n"                                                                            \
    "#define _z_objc_map ,id: \"%s\", Class: \"%s\", SEL: \"%s\"\n"                                \
    "#define _z_objc_arg_map(x) ,id: [(id)(x) description].UTF8String, Class: "                    \
    "class_getName((Class)(x)), SEL: sel_getName((SEL)(x))\n"                                      \
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
    "char*: \"%s\", const char*: \"%s\", void*: \"%p\" _z_objc_map)\n"

#define ZC_C_ARG_GENERIC_STR                                                                       \
    "#define _z_arg(x) _Generic((x), _Bool: _z_bool_str(x) _z_objc_arg_map(x), default: (x))\n"

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
