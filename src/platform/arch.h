// SPDX-License-Identifier: MIT
#ifndef ZC_PLATFORM_ARCH_H
#define ZC_PLATFORM_ARCH_H

#if defined(_WIN32) || defined(_WIN64)
#define ZC_OS_WINDOWS 1
#else
#define ZC_OS_WINDOWS 0
#endif

#if defined(__linux__) && !defined(__ANDROID__)
#define ZC_OS_LINUX 1
#else
#define ZC_OS_LINUX 0
#endif

#if defined(__APPLE__) && defined(__MACH__)
#define ZC_OS_MACOS 1
#else
#define ZC_OS_MACOS 0
#endif

#if defined(__FreeBSD__)
#define ZC_OS_FREEBSD 1
#else
#define ZC_OS_FREEBSD 0
#endif

#if defined(__OpenBSD__)
#define ZC_OS_OPENBSD 1
#else
#define ZC_OS_OPENBSD 0
#endif

#if defined(__NetBSD__)
#define ZC_OS_NETBSD 1
#else
#define ZC_OS_NETBSD 0
#endif

#if defined(__ANDROID__)
#define ZC_OS_ANDROID 1
#else
#define ZC_OS_ANDROID 0
#endif

#if defined(__EMSCRIPTEN__)
#define ZC_OS_WASM 1
#else
#define ZC_OS_WASM 0
#endif

#if defined(__CYGWIN__)
#define ZC_OS_CYGWIN 1
#else
#define ZC_OS_CYGWIN 0
#endif

/* Grouped checks */
#define ZC_OS_UNIX (ZC_OS_LINUX || ZC_OS_MACOS || ZC_OS_FREEBSD || ZC_OS_OPENBSD || ZC_OS_NETBSD)
#define ZC_OS_BSD (ZC_OS_FREEBSD || ZC_OS_OPENBSD || ZC_OS_NETBSD)
#define ZC_OS_POSIX (ZC_OS_UNIX || ZC_OS_CYGWIN)

/* OS name string (for display/logging) */
#if ZC_OS_WINDOWS
#define ZC_OS_NAME "windows"
#elif ZC_OS_LINUX
#define ZC_OS_NAME "linux"
#elif ZC_OS_MACOS
#define ZC_OS_NAME "macos"
#elif ZC_OS_FREEBSD
#define ZC_OS_NAME "freebsd"
#elif ZC_OS_OPENBSD
#define ZC_OS_NAME "openbsd"
#elif ZC_OS_NETBSD
#define ZC_OS_NAME "netbsd"
#elif ZC_OS_ANDROID
#define ZC_OS_NAME "android"
#elif ZC_OS_WASM
#define ZC_OS_NAME "wasm"
#elif ZC_OS_CYGWIN
#define ZC_OS_NAME "cygwin"
#else
#define ZC_OS_NAME "unknown"
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
#define ZC_ARCH_X64 1
#else
#define ZC_ARCH_X64 0
#endif

#if defined(__i386__) || defined(_M_IX86)
#define ZC_ARCH_X86 1
#else
#define ZC_ARCH_X86 0
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define ZC_ARCH_ARM64 1
#else
#define ZC_ARCH_ARM64 0
#endif

#if defined(__arm__) || defined(_M_ARM)
#define ZC_ARCH_ARM32 1
#else
#define ZC_ARCH_ARM32 0
#endif

#if defined(__riscv)
#define ZC_ARCH_RISCV 1
#if __riscv_xlen == 64
#define ZC_ARCH_RISCV64 1
#define ZC_ARCH_RISCV32 0
#else
#define ZC_ARCH_RISCV64 0
#define ZC_ARCH_RISCV32 1
#endif
#else
#define ZC_ARCH_RISCV 0
#define ZC_ARCH_RISCV64 0
#define ZC_ARCH_RISCV32 0
#endif

#if defined(__mips__) || defined(__mips)
#define ZC_ARCH_MIPS 1
#else
#define ZC_ARCH_MIPS 0
#endif

#if defined(__powerpc__) || defined(__ppc__) || defined(_M_PPC)
#define ZC_ARCH_PPC 1
#else
#define ZC_ARCH_PPC 0
#endif

#if defined(__s390x__)
#define ZC_ARCH_S390X 1
#else
#define ZC_ARCH_S390X 0
#endif

#if defined(__wasm__) || defined(__wasm32__) || defined(__wasm64__)
#define ZC_ARCH_WASM 1
#else
#define ZC_ARCH_WASM 0
#endif

/* Grouped checks */
#define ZC_ARCH_X86_ANY (ZC_ARCH_X64 || ZC_ARCH_X86)
#define ZC_ARCH_ARM_ANY (ZC_ARCH_ARM64 || ZC_ARCH_ARM32)
#define ZC_ARCH_64BIT (ZC_ARCH_X64 || ZC_ARCH_ARM64 || ZC_ARCH_RISCV64 || ZC_ARCH_S390X)
#define ZC_ARCH_32BIT (ZC_ARCH_X86 || ZC_ARCH_ARM32 || ZC_ARCH_RISCV32)

/* Architecture name string (for display/logging) */
#if ZC_ARCH_X64
#define ZC_ARCH_NAME "x86_64"
#elif ZC_ARCH_X86
#define ZC_ARCH_NAME "x86"
#elif ZC_ARCH_ARM64
#define ZC_ARCH_NAME "aarch64"
#elif ZC_ARCH_ARM32
#define ZC_ARCH_NAME "arm"
#elif ZC_ARCH_RISCV64
#define ZC_ARCH_NAME "riscv64"
#elif ZC_ARCH_RISCV32
#define ZC_ARCH_NAME "riscv32"
#elif ZC_ARCH_MIPS
#define ZC_ARCH_NAME "mips"
#elif ZC_ARCH_PPC
#define ZC_ARCH_NAME "powerpc"
#elif ZC_ARCH_S390X
#define ZC_ARCH_NAME "s390x"
#elif ZC_ARCH_WASM
#define ZC_ARCH_NAME "wasm"
#else
#define ZC_ARCH_NAME "unknown"
#endif

#if defined(__SIZEOF_POINTER__)
#define ZC_PTR_SIZE __SIZEOF_POINTER__
#elif ZC_ARCH_64BIT
#define ZC_PTR_SIZE 8
#else
#define ZC_PTR_SIZE 4
#endif

#endif /* ZC_PLATFORM_ARCH_H */
