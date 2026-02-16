#ifndef ZC_PLATFORM_COMPILER_H
#define ZC_PLATFORM_COMPILER_H

#ifdef __cplusplus
/* C++ mode */
#define ZC_AUTO auto ///< Auto type inference.
#define ZC_AUTO_INIT(var, init) auto var = (init)
#define ZC_CAST(T, x) static_cast<T>(x)             ///< Static cast.
#define ZC_REINTERPRET(T, x) reinterpret_cast<T>(x) ///< Reinterpret cast.
#define ZC_EXTERN_C extern "C"                      ///< Extern "C" linkage.
#define ZC_EXTERN_C_BEGIN                                                                          \
    extern "C"                                                                                     \
    {
#define ZC_EXTERN_C_END }
#else
/* C mode */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202300L
#define ZC_AUTO auto ///< C23 standard auto.
#define ZC_AUTO_INIT(var, init) auto var = (init)
#else
#define ZC_AUTO __auto_type ///< GCC/Clang extension.
#define ZC_AUTO_INIT(var, init) __auto_type var = (init)
#endif
#define ZC_CAST(T, x) ((T)(x))        ///< Explicit cast.
#define ZC_REINTERPRET(T, x) ((T)(x)) ///< Reinterpret cast.
#define ZC_EXTERN_C                   ///< Extern "C" (no-op in C).
#define ZC_EXTERN_C_BEGIN
#define ZC_EXTERN_C_END
#endif

#ifdef __TINYC__
/* TCC compatibility */
#undef ZC_AUTO
#undef ZC_AUTO_INIT
#define ZC_AUTO /* Invalid in TCC for raw use */
#define ZC_AUTO_INIT(var, init) __typeof__((init)) var = (init)

#ifndef __builtin_expect
#define __builtin_expect(x, v) (x)
#endif

#ifndef __builtin_unreachable
#define __builtin_unreachable()
#endif
#endif

// Branch prediction hints
#ifndef likely
#define likely(x) __builtin_expect(!!(x), 1)
#endif

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

// Attributes
#ifndef ZC_UNUSED
#ifdef __GNUC__
#define ZC_UNUSED __attribute__((unused))
#else
#define ZC_UNUSED
#endif
#endif

#ifndef ZC_NORETURN
#ifdef __GNUC__
#define ZC_NORETURN __attribute__((noreturn))
#elif defined(_MSC_VER)
#define ZC_NORETURN __declspec(noreturn)
#else
#define ZC_NORETURN
#endif
#endif

#endif // ZC_PLATFORM_COMPILER_H
