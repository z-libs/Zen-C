/*
 * z_platform.h - Zen-C Runtime Platform Abstraction Layer
 *
 * This header provides cross-platform compatibility for generated C code.
 * It abstracts threading, file descriptors, and other platform-specific APIs.
 */

#ifndef Z_PLATFORM_H
#define Z_PLATFORM_H

#ifdef _WIN32
/* ============ Windows Implementation ============ */
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <basetsd.h>

/* ssize_t for Windows */
typedef SSIZE_T ssize_t;

/* Thread types */
typedef struct {
    HANDLE handle;
    void **result_slot;
} z_thread_t;

typedef struct {
    void *(*func)(void *);
    void *arg;
    void **result_slot;
} z_thread_start_ctx;

static DWORD WINAPI z_thread_start(LPVOID param)
{
    z_thread_start_ctx *ctx = (z_thread_start_ctx *)param;
    void *ret = NULL;
    if (ctx->func)
    {
        ret = ctx->func(ctx->arg);
    }
    if (ctx->result_slot)
    {
        *ctx->result_slot = ret;
    }
    free(ctx);
    return 0;
}

/* Async structure for Windows */
typedef struct {
    z_thread_t thread;
    void *result;
} Async;

/* Create a thread
 * Note: Windows CreateThread expects DWORD WINAPI func(LPVOID),
 * but we accept void* (*)(void*) for compatibility.
 * The caller must ensure the function signature is compatible.
 */
static inline int z_thread_create(z_thread_t *thread, void *(*func)(void *), void *arg)
{
    z_thread_start_ctx *ctx = malloc(sizeof(z_thread_start_ctx));
    void **result_slot = malloc(sizeof(void *));
    if (!ctx || !result_slot)
    {
        free(ctx);
        free(result_slot);
        return -1;
    }
    *result_slot = NULL;
    ctx->func = func;
    ctx->arg = arg;
    ctx->result_slot = result_slot;

    HANDLE handle = CreateThread(NULL, 0, z_thread_start, ctx, 0, NULL);
    if (!handle)
    {
        free(ctx);
        free(result_slot);
        return -1;
    }
    thread->handle = handle;
    thread->result_slot = result_slot;
    return 0;
}

/* Wait for thread completion and get result */
static inline int z_thread_join(z_thread_t thread, void **result)
{
    DWORD wait = WaitForSingleObject(thread.handle, INFINITE);
    if (wait != WAIT_OBJECT_0)
    {
        if (thread.result_slot)
        {
            free(thread.result_slot);
        }
        CloseHandle(thread.handle);
        return -1;
    }
    if (result)
    {
        *result = thread.result_slot ? *thread.result_slot : NULL;
    }
    if (thread.result_slot)
    {
        free(thread.result_slot);
    }
    CloseHandle(thread.handle);
    return 0;
}

/* File descriptor operations */
#define z_dup _dup
#define z_dup2 _dup2
#define z_fileno _fileno
#define z_open _open
#define z_close _close

/* Null device path */
#define Z_NULL_DEVICE "NUL"

/* getline implementation for Windows */
static inline ssize_t z_getline(char **lineptr, size_t *n, FILE *stream)
{
    if (!lineptr || !n || !stream)
        return -1;

    size_t pos = 0;
    int c;

    if (*lineptr == NULL || *n == 0) {
        *n = 128;
        *lineptr = (char *)malloc(*n);
        if (*lineptr == NULL)
            return -1;
    }

    while ((c = fgetc(stream)) != EOF) {
        if (pos + 1 >= *n) {
            size_t new_size = *n * 2;
            char *new_ptr = (char *)realloc(*lineptr, new_size);
            if (new_ptr == NULL)
                return -1;
            *lineptr = new_ptr;
            *n = new_size;
        }
        (*lineptr)[pos++] = (char)c;
        if (c == '\n')
            break;
    }

    if (pos == 0 && c == EOF)
        return -1;

    (*lineptr)[pos] = '\0';
    return (ssize_t)pos;
}

#define getline z_getline

/* Suppress stdout */
static inline int z_suppress_stdout(void)
{
    int saved = _dup(_fileno(stdout));
    int null_fd = _open("NUL", _O_WRONLY);
    _dup2(null_fd, _fileno(stdout));
    _close(null_fd);
    return saved;
}

/* Restore stdout */
static inline void z_restore_stdout(int saved_fd)
{
    fflush(stdout);
    _dup2(saved_fd, _fileno(stdout));
    _close(saved_fd);
}

#else
/* ============ POSIX Implementation (Linux/macOS) ============ */
#include <pthread.h>
#include <unistd.h>
#include <fcntl.h>

/* Thread types */
typedef pthread_t z_thread_t;

/* Async structure for POSIX */
typedef struct {
    z_thread_t thread;
    void *result;
} Async;

/* Create a thread */
static inline int z_thread_create(z_thread_t *thread, void *(*func)(void *), void *arg)
{
    return pthread_create(thread, NULL, func, arg);
}

/* Wait for thread completion and get result */
static inline int z_thread_join(z_thread_t thread, void **result)
{
    return pthread_join(thread, result);
}

/* File descriptor operations */
#define z_dup dup
#define z_dup2 dup2
#define z_fileno fileno
#define z_open open
#define z_close close

/* Null device path */
#define Z_NULL_DEVICE "/dev/null"

/* Suppress stdout */
static inline int z_suppress_stdout(void)
{
    int saved = dup(fileno(stdout));
    int null_fd = open("/dev/null", O_WRONLY);
    dup2(null_fd, fileno(stdout));
    close(null_fd);
    return saved;
}

/* Restore stdout */
static inline void z_restore_stdout(int saved_fd)
{
    fflush(stdout);
    dup2(saved_fd, fileno(stdout));
    close(saved_fd);
}

#endif /* _WIN32 */

#endif /* Z_PLATFORM_H */
