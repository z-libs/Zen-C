// compat.h - Compatibility layer for different platforms and compilers
#ifndef ZC_COMPAT_H
#define ZC_COMPAT_H

#include <stddef.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>


#ifdef __cplusplus
extern "C" {
#endif

#if defined(__clang__)
    #define ZC_COMPILER_CLANG
#elif defined(__GNUC__) || defined(__GNUG__)
    #define ZC_COMPILER_GCC 
#elif defined(_MSC_VER)
    #define ZC_COMPILER_MSVC
#elif defined(__TINYC__)
    #define ZC_COMPILER_TINYC
#elif defined(__ZIG__)
    #define ZC_COMPILER_ZIG
#else
    #define ZC_COMPILER_UNKNOWN
#endif

#if defined(ZC_COMPILER_TINYC) || defined(ZC_COMPILER_MSVC)
#define __auto_type __typeof__
#endif

/* --- Platform Detection & Base Headers --- */
#if defined(_WIN32) || defined (_WIN64)
#   define ZC_WINDOWS 1
#   define WIN32_LEAN_AND_MEAN
#   define NOMINMAX
#   include <windows.h>
#   include <time.h>
#   include <fcntl.h>
#else
#   define ZC_WINDOWS 0
#   include <unistd.h>
#   include <pthread.h>
#   include <semaphore.h>
#   include <sys/stat.h>
#   include <time.h>
#endif

int zc_get_pid();


#ifdef __cplusplus
}
#endif

#endif // ZC_COMPAT_H

// compat.h - implementation
#ifdef ZC_COMPAT_IMPLEMENTATION
#undef ZC_COMPAT_IMPLEMENTATION

int zc_get_pid(void){
#if ZC_WINDOWS
    return (int)GetCurrentProcessId();
#else
    return (int)getpid();
#endif
}

#endif // ZC_COMPAT_IMPLEMENTATION