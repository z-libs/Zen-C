#ifndef ZC_PLATFORM_OS_H
#define ZC_PLATFORM_OS_H

#include "lang.h"

// OS Detection
#ifdef __COSMOPOLITAN__
#include <cosmo.h>
#define z_is_windows() IsWindows()
#else
#ifdef _WIN32
#define z_is_windows() 1
#define ZC_OS_WINDOWS
#else
#define z_is_windows() 0
#define ZC_OS_LINUX /* Assuming Linux/Unix */
#endif
#endif

// System headers
#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#include <io.h>
#ifndef PATH_MAX
#define PATH_MAX 260
#endif
#define realpath(N, R) _fullpath((R), (N), PATH_MAX)
#define F_OK 0
#define access _access
#define getcwd _getcwd
#else
#include <unistd.h>
#include <sys/types.h>
#include <limits.h> /* PATH_MAX */
#endif

// Target architecture
#if defined(__aarch64__)
#define ZC_ARCH_ARM64 1
#elif defined(__arm__)
#define ZC_ARCH_ARM32 1
#elif defined(__x86_64__) || defined(_M_X64)
#define ZC_ARCH_X64 1
#elif defined(__i386__) || defined(_M_IX86)
#define ZC_ARCH_X86 1
#else
#error Add definition for this preprocessors to identify target cpu architecture
#endif

// Path helpers
static inline int z_is_abs_path(const char *p)
{
    if (!p)
    {
        return 0;
    }
    if (p[0] == '/')
    {
        return 1;
    }
#ifdef _WIN32
    if (p[0] == '\\' || (isalpha(p[0]) && p[1] == ':'))
    {
        return 1;
    }
#endif
    return 0;
}

static inline char *z_path_last_sep(const char *path)
{
    char *last_slash = strrchr(path, '/');
#ifdef _WIN32
    char *last_bs = strrchr(path, '\\');
    if (last_bs > last_slash)
    {
        return last_bs;
    }
#endif
    return last_slash;
}

static inline const char *z_get_exe_ext(void)
{
#ifdef _WIN32
    return ".exe";
#else
    return ".bin";
#endif
}

static inline const char *z_get_null_redirect(void)
{
#ifdef _WIN32
    return " > NUL 2>&1";
#else
    return " > /dev/null 2>&1";
#endif
}

static inline const char *z_get_comptime_link_flags(void)
{
#ifdef _WIN32
    return " std/third-party/tre/lib/*.c";
#else
    return "";
#endif
}

static inline const char *z_get_run_prefix(void)
{
#ifdef _WIN32
    return "";
#else
    return "./";
#endif
}

static inline const char *z_get_plugin_ext(void)
{
#ifdef _WIN32
    return ".dll";
#else
    return ".so";
#endif
}

/**
 * @brief Setup terminal (enable ANSI colors on Windows).
 */
void z_setup_terminal(void);

/**
 * @brief Get wall clock time in seconds.
 */
double z_get_time(void);

/**
 * @brief Get monotonic time in seconds (high precision).
 */
double z_get_monotonic_time(void);

/**
 * @brief Get temporary directory path.
 */
const char *z_get_temp_dir(void);

/**
 * @brief Get current process ID.
 */
int z_get_pid(void);

/**
 * @brief Get the path of the current executable.
 */
void z_get_executable_path(char *buffer, size_t size);

/**
 * @brief Check if file descriptor refers to a terminal.
 */
int z_isatty(int fd);

// Console / REPL
void repl_enable_raw_mode(void);
void repl_disable_raw_mode(void);
int repl_read_char(char *c);
int repl_get_window_size(int *rows, int *cols);

// Dynamic Library Loading
void *z_dlopen(const char *path);
void *z_dlsym(void *handle, const char *symbol);
void z_dlclose(void *handle);

// OS Helpers
int z_match_os(const char *os_name);
const char *z_get_system_name(void);
FILE *z_tmpfile(void);

/**
 * @brief Run a command securely without shell interpretation.
 * @param argv NULL-terminated array of arguments.
 * @return Exit code of the process.
 */
int z_run_command(char *const argv[]);

#endif // ZC_PLATFORM_OS_H
