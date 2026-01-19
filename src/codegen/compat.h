
#ifndef ZC_COMPAT_H
#define ZC_COMPAT_H

#ifdef __cplusplus
/* C++ mode */
#define ZC_AUTO auto
#define ZC_CAST(T, x) static_cast<T>(x)
#define ZC_REINTERPRET(T, x) reinterpret_cast<T>(x)
#define ZC_EXTERN_C extern "C"
#define ZC_EXTERN_C_BEGIN                                                                          \
    extern "C"                                                                                     \
    {
#define ZC_EXTERN_C_END }
#else
/* C mode */
#define ZC_AUTO __auto_type
#define ZC_CAST(T, x) ((T)(x))
#define ZC_REINTERPRET(T, x) ((T)(x))
#define ZC_EXTERN_C
#define ZC_EXTERN_C_BEGIN
#define ZC_EXTERN_C_END
#endif

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
#endif

#endif
