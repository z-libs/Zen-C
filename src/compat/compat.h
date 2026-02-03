///////////////////////////////////////////////////////////////////////
// compat.h - Compatibility layer for different platforms and compilers
///////////////////////////////////////////////////////////////////////
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
#   define ZC_ON_WINDOWS
#   define WIN32_LEAN_AND_MEAN
#   define NOMINMAX
#   include <windows.h>
#   include <winnetwk.h>
#   include <time.h>
#   include <fcntl.h>
#   include <io.h>
#   define strtok_r strtok_s
#else
#   define ZC_ON_POSIX
#   include <unistd.h>
#   include <pthread.h>
#   include <semaphore.h>
#   include <sys/stat.h>
#   include <time.h>
#   include <dirent.h>

#   include <dlfcn.h>
#   include <fcntl.h>
#   include <termios.h>
#endif

/* --- Types --- */
#ifdef ZC_ON_WINDOWS
    typedef HANDLE           zc_thread_t;
    typedef CRITICAL_SECTION zc_mutex_t;
    typedef HANDLE           zc_sem_t;
    typedef HMODULE          zc_dlhandle;

    typedef struct{
        long long tv_sec;
        long   tv_nsec;
    } zc_timespec;
#else
    typedef pthread_t        zc_thread_t;
    typedef pthread_mutex_t  zc_mutex_t;
    typedef sem_t            zc_sem_t;
    typedef void* zc_dlhandle;
    typedef struct timespec zc_timespec;
#endif

typedef void* (*zc_thread_func)(void*);
typedef struct ZCDir ZCDir;
typedef struct ZCDirEnt
{
    char name[256];
} ZCDirEnt;

/* --- Constants --- */
#define ZC_POSIX_PATHSEP '/'
#define ZC_WINDOWS_PATHSEP '\\'

#if defined(ZC_ON_WINDOWS)
#   define ZC_PATHSEP ZC_WINDOWS_PATHSEP
#   define ZC_BINARY_EXT ".exe"
#   define ZC_STDOUT_FILENO 1
#   define ZC_STDERR_FILENO 2
#   define ZC_STDIN_FILENO  0
#   define ZC_O_READONLY  _O_RDONLY
#   define ZC_O_WRONLY    _O_WRONLY
#   define ZC_O_CREAT     _O_CREAT
#   define ZC_O_TRUNC     _O_TRUNC
#   define ZC_O_APPEND    _O_APPEND
#   define ZC_O_BINARY    _O_BINARY
#else
#   define ZC_PATHSEP ZC_POSIX_PATHSEP
#   define ZC_BINARY_EXT ".out"
#   define ZC_STDOUT_FILENO STDOUT_FILENO
#   define ZC_STDERR_FILENO STDERR_FILENO
#   define ZC_STDIN_FILENO  STDIN_FILENO
#   define ZC_O_READONLY  O_RDONLY
#   define ZC_O_WRONLY    O_WRONLY
#   define ZC_O_CREAT     O_CREAT
#   define ZC_O_TRUNC     O_TRUNC
#   define ZC_O_APPEND    O_APPEND
#   define ZC_O_BINARY    0  // No-op on POSIX
#endif

#define ZC_F_OK    0 // F_OK
#define ZC_R_OK    4 // R_OK
#define ZC_W_OK    2 // W_OK
#define ZC_CLOCK_REALTIME 0
#define ZC_CLOCK_MONOTONIC 1


/* --- Networking Types --- */
#ifdef ZC_ON_WINDOWS
    typedef UINT_PTR zc_socket_t;
#   define ZC_INVALID_SOCKET INVALID_SOCKET
#   define ZC_SOCKET_ERROR   SOCKET_ERROR
#else
    typedef int zc_socket_t;
#   define ZC_INVALID_SOCKET -1
#   define ZC_SOCKET_ERROR   -1
#endif

/* --- API Declarations --- */

// Filesystem (UTF-8 Path Support)
int    zc_mkdir(const char* path, int mode);
int    zc_access(const char* path, int mode);
int    zc_unlink(const char* path);
char*  zc_realpath(const char* path, char *resolved_path); // Caller must free()
char*  zc_get_executable_path(void);                       // Caller must free()
bool   zc_is_dir(const char* path);
int    zc_isatty(int fd);
FILE*  zc_popen(const char* command, const char* type);
int    zc_pclose(FILE* stream);
char** zc_split_paths(const char* paths); // Caller must free array and strings
void   zc_free_split_paths(char** paths);
void   zc_normalize_path(char *path);
char*  zc_getcwd(char *buf, size_t size);

bool zc_is_path_absolute(const char* path);

ZCDir *zc_opendir(const char *path);
const ZCDirEnt *zc_readdir(ZCDir *dir);
void zc_closedir(ZCDir *dir);
bool zc_is_dir(const char *path);

// System & Process
void   zc_sleep_ms(unsigned int ms);
int    zc_getpid(void);
int    zc_stricmp(const char* s1, const char* s2);
int    zc_strnicmp(const char* s1, const char* s2, size_t n);
int    zc_strcasecmp(const char *s1, const char *s2);
int    zc_strncasecmp(const char *s1, const char *s2, size_t n);
char*  zc_strdup(const char *s);
int    zc_clock_gettime(int clock_id, zc_timespec* ts);

void*  zc_xcalloc(size_t nmemb, size_t size);
void*  zc_xrealloc(void* ptr, size_t size);

int zc_dup(int fd);
int zc_dup2(int oldfd, int newfd);
int zc_open(const char* pathname, int flags);
int zc_close(int fd);

// Threads & Sync
int    zc_thread_create(zc_thread_t* thread, zc_thread_func func, void* arg);
int    zc_thread_join(zc_thread_t thread);

void   zc_mutex_init(zc_mutex_t* m);
void   zc_mutex_lock(zc_mutex_t* m);
void   zc_mutex_unlock(zc_mutex_t* m);
void   zc_mutex_destroy(zc_mutex_t* m);

int    zc_sem_init(zc_sem_t* s, unsigned int initial_count);
void   zc_sem_wait(zc_sem_t* s);
void   zc_sem_post(zc_sem_t* s);
void   zc_sem_destroy(zc_sem_t* s);

// Dynamic Loading
zc_dlhandle zc_dlopen(const char* path);
void* zc_dlsym(zc_dlhandle handle, const char* symbol);
int           zc_dlclose(zc_dlhandle handle);

/* --- Networking API --- */
int  zc_net_init(void);    // Required for Windows (WSAStartup)
void zc_net_cleanup(void); // Required for Windows (WSACleanup)

zc_socket_t zc_socket_create(int domain, int type, int protocol);
int         zc_socket_connect(zc_socket_t sock, const char* host, int port);
int         zc_socket_send(zc_socket_t sock, const char* buf, int len);
int         zc_socket_recv(zc_socket_t sock, char* buf, int len);
void        zc_socket_close(zc_socket_t sock);
int         zc_socket_set_nonblocking(zc_socket_t sock, bool nonblocking);


#ifdef __cplusplus
}
#endif

#endif // ZC_COMPAT_H

// compat.h - implementation
#ifdef ZC_COMPAT_IMPLEMENTATION
#undef ZC_COMPAT_IMPLEMENTATION

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef ZC_ON_WINDOWS
#   include <io.h>
#   include <process.h>
#   include <direct.h>
#   include <sys/stat.h>
#else
#   include <sys/types.h>
#   include <dlfcn.h>
#   include <time.h>
#   include <errno.h>
#   ifdef __APPLE__
#       include <mach-o/dyld.h>
#   endif
#endif
/* --- Internal Helpers --- */
#ifdef ZC_ON_WINDOWS
static wchar_t* zc_internal_to_w(const char* utf8) {
    if (!utf8) return NULL;
    int len = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, NULL, 0);
    if (len <= 0) return NULL;
    wchar_t* w = (wchar_t*)malloc(len * sizeof(wchar_t));
    if (w) MultiByteToWideChar(CP_UTF8, 0, utf8, -1, w, len);
    return w;
}

static char* zc_internal_from_w(const wchar_t* w) {
    if (!w) return NULL;
    int len = WideCharToMultiByte(CP_UTF8, 0, w, -1, NULL, 0, NULL, NULL);
    if (len <= 0) return NULL;
    char* s = (char*)malloc(len);
    if (s) WideCharToMultiByte(CP_UTF8, 0, w, -1, s, len, NULL, NULL);
    return s;
}

typedef struct { zc_thread_func f; void* a; } zc_win_thread_data;
static unsigned __stdcall zc_win_thread_wrapper(void* p) {
    zc_win_thread_data* d = (zc_win_thread_data*)p;
    d->f(d->a); free(d); return 0;
}
#endif

/* --- API Implementation --- */

int zc_mkdir(const char* path, int mode) {
#ifdef ZC_ON_WINDOWS
    wchar_t* w = zc_internal_to_w(path);
    int res = _wmkdir(w); free(w); return res;
#else
    return mkdir(path, (mode_t)mode);
#endif
}

int zc_access(const char* path, int mode) {
#ifdef ZC_ON_WINDOWS
    wchar_t* w = zc_internal_to_w(path);
    int res = _waccess(w, mode); free(w); return res;
#else
    return access(path, mode);
#endif
}

int zc_unlink(const char* path) {
#ifdef ZC_ON_WINDOWS
    wchar_t* w = zc_internal_to_w(path);
    int res = _wunlink(w); free(w); return res;
#else
    return unlink(path);
#endif
}

bool zc_is_dir(const char* path) {
#ifdef ZC_ON_WINDOWS
    wchar_t* w = zc_internal_to_w(path);
    struct _stat64i32 s;
    bool res = (_wstat(w, &s) == 0) && (s.st_mode & _S_IFDIR);
    free(w); return res;
#else
    struct stat s;
    return (stat(path, &s) == 0) && S_ISDIR(s.st_mode);
#endif
}

char* zc_realpath(const char* path, char* resolved_path) {
    if (!path) return NULL;

#ifdef ZC_ON_WINDOWS
    wchar_t* w = zc_internal_to_w(path);
    if (!w) return NULL;

    // Determine required size for the wide string buffer
    DWORD sz = GetFullPathNameW(w, 0, NULL, NULL);
    if (!sz) {
        free(w);
        return NULL;
    }

    wchar_t* wo = (wchar_t*)malloc(sz * sizeof(wchar_t));
    if (!wo) {
        free(w);
        return NULL;
    }

    if (GetFullPathNameW(w, sz, wo, NULL) == 0) {
        free(w);
        free(wo);
        return NULL;
    }

    // Convert wide result back to UTF-8
    char* res_utf8 = zc_internal_from_w(wo);
    free(w);
    free(wo);

    if (!res_utf8) return NULL;

    if (resolved_path) {
        // If user provided a buffer, copy into it and free our temporary allocation
        // Note: Industry standard assumes resolved_path is at least _MAX_PATH
        strncpy(resolved_path, res_utf8, _MAX_PATH - 1);
        resolved_path[_MAX_PATH - 1] = '\0';
        free(res_utf8);
        return resolved_path;
    }

    // If resolved_path was NULL, return the allocated string (caller must free)
    return res_utf8;

#else
    // POSIX handles the NULL/Buffer logic natively
    return realpath(path, resolved_path);
#endif
}

int zc_isatty(int fd) {
#ifdef ZC_ON_WINDOWS
    return _isatty(fd);
#else
    return isatty(fd);
#endif
}

char* zc_get_executable_path(void) {
#ifdef ZC_ON_WINDOWS
    wchar_t w[MAX_PATH];
    if (GetModuleFileNameW(NULL, w, MAX_PATH) == 0) return NULL;
    return zc_internal_from_w(w);
#elif defined(__APPLE__)
    char p[1024]; uint32_t s = sizeof(p);
    return (_NSGetExecutablePath(p, &s) == 0) ? realpath(p, NULL) : NULL;
#else
    char p[1024]; ssize_t l = readlink("/proc/self/exe", p, sizeof(p)-1);
    if (l == -1) return NULL; p[l] = '\0'; return strdup(p);
#endif
}

char ** zc_split_paths(const char* paths) {
    if (!paths) return NULL;

    // Count number of paths
    int count = 1;
    const char* p = paths;
    while (*p) {
        if (*p == ';' || *p == ':') count++;
        p++;
    }

    char** result = (char**)malloc((count + 1) * sizeof(char*));
    if (!result) return NULL;

    int idx = 0;
    const char* start = paths;
    p = paths;
    while (*p) {
        if (*p == ';' || *p == ':') {
            size_t len = p - start;
            result[idx] = (char*)malloc(len + 1);
            if (result[idx]) {
                strncpy(result[idx], start, len);
                result[idx][len] = '\0';
                //zc_normalize_path(result[idx]);
            }
            idx++;
            start = p + 1;
        }
        p++;
    }
    // Last path
    size_t len = p - start;
    result[idx] = (char*)malloc(len + 1);
    if (result[idx]) {
        strncpy(result[idx], start, len);
        result[idx][len] = '\0';
        //zc_normalize_path(result[idx]);
    }
    idx++;
    result[idx] = NULL; // Null-terminate the array

    return result;
}

void zc_free_split_paths(char** paths) {
    if (!paths) return;
    char** p = paths;
    while (*p) {
        free(*p);
        p++;
    }
    free(paths);
}

// void zc_normalize_path(char *path) {
// #ifdef ZC_ON_WINDOWS
//     for (char *p = path; *p; ++p) {
//         if (*p == ZC_POSIX_PATHSEP) *p = ZC_WINDOWS_PATHSEP;
//     }
// #else
//     for (char *p = path; *p; ++p) {
//         if (*p == ZC_WINDOWS_PATHSEP) *p = ZC_POSIX_PATHSEP;
//     }
// #endif
// }


void zc_normalize_path(char *path) {
    if (!path || !*path) return;

    // 1. Uniform Slashes
    for (char *p = path; *p; p++) if (*p == ZC_WINDOWS_PATHSEP) *p = ZC_POSIX_PATHSEP;

    char *src = path;
    char *dst = path;
    char *base = path; // The point where we stop backtracking

    // 2. Handle Root/Prefix
#ifdef _WIN32
    if (src[0] && src[1] == ':') { // "C:\"
        *dst++ = *src++;
        *dst++ = *src++;
        base = dst; 
        if (*src == ZC_POSIX_PATHSEP) { *dst++ = *src++; base = dst; }
    } else if (src[0] == ZC_POSIX_PATHSEP) { // "\"
        *dst++ = *src++;
        base = dst;
    }
#else
    if (src[0] == ZC_POSIX_PATHSEP) { // "/"
        *dst++ = *src++;
        base = dst;
    }
#endif

    // 3. Process segments
    while (*src) {
        if (*src == ZC_POSIX_PATHSEP) { src++; continue; } // Skip redundant slashes

        char *seg_start = src;
        while (*src && *src != ZC_POSIX_PATHSEP) src++;
        size_t len = src - seg_start;

        if (len == 1 && seg_start[0] == '.') {
            continue; // Skip "."
        } 
        
        if (len == 2 && seg_start[0] == '.' && seg_start[1] == '.') {
            // Backtrack logic
            if (dst > base) {
                // We have a folder to pop
                dst--; // Move before the last separator
                while (dst > base && *(dst - 1) != ZC_POSIX_PATHSEP) dst--;
            } else if (base == path) {
                // Relative path and we are at the very start, 
                // so we must keep the ".."
                if (dst > path) *dst++ = ZC_POSIX_PATHSEP;
                *dst++ = '.'; *dst++ = '.';
            }
            // If it's absolute (base > path) and we are at base, 
            // we just ignore ".." because you can't go above root.
            continue;
        }

        // Normal segment
        if (dst > path && *(dst - 1) != ZC_POSIX_PATHSEP) *dst++ = ZC_POSIX_PATHSEP;
        memmove(dst, seg_start, len);
        dst += len;
    }

    // 4. Final touch
    if (dst == path) {
        *dst++ = '.'; // Empty relative path becomes "."
    }
    *dst = '\0';
}


char* zc_getcwd(char *buf, size_t size) {
#ifdef ZC_ON_WINDOWS
    return _getcwd(buf, (int)size);
#else
    return getcwd(buf, size);
#endif
}

bool zc_is_path_absolute(const char* path) {
    if (!path || !*path) return false;
#ifdef ZC_ON_WINDOWS
    return (strlen(path) >= 2 && path[1] == ':');
#else
    return path[0] == '/';
#endif
}

struct ZCDir
{
#if defined(ZC_ON_WINDOWS)
    HANDLE h_find;
    WIN32_FIND_DATAW find_data;
    int first;
    char pattern[MAX_PATH];
    ZCDirEnt ent;
#else
    DIR *dir;
    ZCDirEnt ent;
#endif
};

ZCDir *zc_opendir(const char *path)
{
#if defined(ZC_ON_WINDOWS)
    ZCDir *d = (ZCDir *)zc_xcalloc(1, sizeof(ZCDir));
    snprintf(d->pattern, sizeof(d->pattern), "%s\\*", path);
    wchar_t wpattern[MAX_PATH];
    MultiByteToWideChar(CP_UTF8, 0, d->pattern, -1, wpattern, MAX_PATH);
    d->h_find = FindFirstFileW(wpattern, &d->find_data);
    d->first = 1;
    if (d->h_find == INVALID_HANDLE_VALUE)
    {
        // Arena: no need to free
        return NULL;
    }
    return d;
#else
    ZCDir *d = (ZCDir *)zc_xcalloc(1, sizeof(ZCDir));
    d->dir = opendir(path);
    if (!d->dir)
    {
        // Arena: no need to free
        return NULL;
    }
    return d;
#endif
}

const ZCDirEnt *zc_readdir(ZCDir *d)
{
    if (!d)
    {
        return NULL;
    }
#if defined(ZC_ON_WINDOWS)
    WIN32_FIND_DATAW *fd = &d->find_data;
    BOOL found;
    if (d->first)
    {
        d->first = 0;
        found = TRUE;
    }
    else
    {
        found = FindNextFileW(d->h_find, fd);
    }
    while (found)
    {
        char name_utf8[256];
        WideCharToMultiByte(CP_UTF8, 0, fd->cFileName, -1, name_utf8, sizeof(name_utf8), NULL, NULL);
        if (strcmp(name_utf8, ".") != 0 && strcmp(name_utf8, "..") != 0)
        {
            strncpy(d->ent.name, name_utf8, sizeof(d->ent.name) - 1);
            d->ent.name[sizeof(d->ent.name) - 1] = '\0';
            return &d->ent;
        }
        found = FindNextFileW(d->h_find, fd);
    }
    return NULL;
#else
    struct dirent *dent;
    while ((dent = readdir(d->dir)) != NULL)
    {
        if (strcmp(dent->d_name, ".") != 0 && strcmp(dent->d_name, "..") != 0)
        {
            strncpy(d->ent.name, dent->d_name, sizeof(d->ent.name) - 1);
            d->ent.name[sizeof(d->ent.name) - 1] = '\0';
            return &d->ent;
        }
    }
    return NULL;
#endif
}

void zc_closedir(ZCDir *d)
{
    if (!d)
    {
        return;
    }
#if defined(ZC_ON_WINDOWS)
    if (d->h_find != INVALID_HANDLE_VALUE)
    {
        FindClose(d->h_find);
    }
    // Arena: no need to free
#else
    if (d->dir)
    {
        closedir(d->dir);
    }
    // Arena: no need to free
#endif
}


FILE* zc_popen(const char* command, const char* type) {
#ifdef ZC_ON_WINDOWS
    return _popen(command, type);
#else
    return popen(command, type);
#endif
}

int zc_pclose(FILE* stream) {
#ifdef ZC_ON_WINDOWS
    return _pclose(stream);
#else
    return pclose(stream);
#endif
}



void zc_sleep_ms(unsigned int ms) {
#ifdef ZC_ON_WINDOWS
    Sleep(ms);
#else
    struct timespec ts = { (time_t)(ms / 1000), (long)((ms % 1000) * 1000000L) };
    nanosleep(&ts, NULL);
#endif
}

// int zc_getpid(void) {
// #ifdef ZC_ON_WINDOWS
//     return (int)_getpid();
// #else
//     return (int)getpid();
// #endif
// }

int zc_thread_create(zc_thread_t* thread, zc_thread_func func, void* arg) {
#ifdef ZC_ON_WINDOWS
    zc_win_thread_data* d = (zc_win_thread_data*)malloc(sizeof(zc_win_thread_data));
    if (!d) return -1;
    d->f = func; d->a = arg;
    uintptr_t h = _beginthreadex(NULL, 0, zc_win_thread_wrapper, d, 0, NULL);
    if (!h) { free(d); return -1; }
    *thread = (HANDLE)h; return 0;
#else
    return pthread_create(thread, NULL, func, arg);
#endif
}

int zc_thread_join(zc_thread_t thread) {
#ifdef ZC_ON_WINDOWS
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread); return 0;
#else
    return pthread_join(thread, NULL);
#endif
}

void zc_mutex_init(zc_mutex_t* m) {
#ifdef ZC_ON_WINDOWS
    InitializeCriticalSection(m);
#else
    pthread_mutex_init(m, NULL);
#endif
}

void zc_mutex_lock(zc_mutex_t* m) {
#ifdef ZC_ON_WINDOWS
    EnterCriticalSection(m);
#else
    pthread_mutex_lock(m);
#endif
}

void zc_mutex_unlock(zc_mutex_t* m) {
#ifdef ZC_ON_WINDOWS
    LeaveCriticalSection(m);
#else
    pthread_mutex_unlock(m);
#endif
}

void zc_mutex_destroy(zc_mutex_t* m) {
#ifdef ZC_ON_WINDOWS
    DeleteCriticalSection(m);
#else
    pthread_mutex_destroy(m);
#endif
}

int zc_sem_init(zc_sem_t* s, unsigned int count) {
#ifdef ZC_ON_WINDOWS
    *s = CreateSemaphoreW(NULL, count, MAXLONG, NULL);
    return *s ? 0 : -1;
#else
    return sem_init(s, 0, count);
#endif
}

void zc_sem_wait(zc_sem_t* s) {
#ifdef ZC_ON_WINDOWS
    WaitForSingleObject(*s, INFINITE);
#else
    sem_wait(s);
#endif
}

void zc_sem_post(zc_sem_t* s) {
#ifdef ZC_ON_WINDOWS
    ReleaseSemaphore(*s, 1, NULL);
#else
    sem_post(s);
#endif
}

void zc_sem_destroy(zc_sem_t* s) {
#ifdef ZC_ON_WINDOWS
    CloseHandle(*s);
#else
    sem_destroy(s);
#endif
}

int zc_stricmp(const char* s1, const char* s2) {
#ifdef ZC_ON_WINDOWS
    return _stricmp(s1, s2);
#else
    return strcasecmp(s1, s2);
#endif
}

int zc_strnicmp(const char* s1, const char* s2, size_t n) {
#ifdef ZC_ON_WINDOWS
    return _strnicmp(s1, s2, n);
#else
    return strncasecmp(s1, s2, n);
#endif
}


int zc_strcasecmp(const char *s1, const char *s2)
{
#ifdef ZC_ON_WINDOWS
    return _stricmp(s1, s2);
#else
    return strcasecmp(s1, s2);
#endif
}

int zc_strncasecmp(const char *s1, const char *s2, size_t n)
{
#ifdef ZC_ON_WINDOWS
    return _strnicmp(s1, s2, n);
#else
    return strncasecmp(s1, s2, n);
#endif
}

char *zc_strdup(const char *s)
{
#ifdef ZC_ON_WINDOWS
    return _strdup(s);
#else
    return strdup(s);
#endif
}

zc_dlhandle zc_dlopen(const char* path) {
#ifdef ZC_ON_WINDOWS
    wchar_t* w = zc_internal_to_w(path);
    HMODULE h = LoadLibraryW(w); free(w); return (zc_dlhandle)h;
#else
    return dlopen(path, RTLD_LAZY);
#endif
}

void* zc_dlsym(zc_dlhandle handle, const char* symbol) {
#ifdef ZC_ON_WINDOWS
    return (void*)GetProcAddress((HMODULE)handle, symbol);
#else
    return dlsym(handle, symbol);
#endif
}


int zc_getpid(void) {
#ifdef ZC_ON_WINDOWS
    return (int)GetCurrentProcessId();
#else
    return (int)getpid();
#endif
}

int zc_clock_gettime(int clock_id, zc_timespec* ts) {
#ifdef ZC_ON_WINDOWS
    if (!ts) return -1;

    if (clock_id == ZC_CLOCK_REALTIME) {
        // System time (Wall clock)
        FILETIME ft;
        GetSystemTimeAsFileTime(&ft);
        
        // Windows FileTime is 100-nanosecond intervals since Jan 1, 1601
        // We convert to Unix Epoch (Jan 1, 1970)
        unsigned long long t = ((unsigned long long)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
        t -= 116444736000000000ULL;
        
        ts->tv_sec = t / 10000000ULL;
        ts->tv_nsec = (long)((t % 10000000ULL) * 100);
        return 0;
    } 
    else if (clock_id == ZC_CLOCK_MONOTONIC) {
        // High-resolution monotonic timer
        static LARGE_INTEGER frequency;
        static int has_freq = 0;
        if (!has_freq) {
            QueryPerformanceFrequency(&frequency);
            has_freq = 1;
        }
        
        LARGE_INTEGER counter;
        QueryPerformanceCounter(&counter);
        
        ts->tv_sec = counter.QuadPart / frequency.QuadPart;
        ts->tv_nsec = (long)((counter.QuadPart % frequency.QuadPart) * 1000000000ULL / frequency.QuadPart);
        return 0;
    }
    
    return -1;
#else
    clockid_t id = (clock_id == ZC_CLOCK_MONOTONIC) ? CLOCK_MONOTONIC : CLOCK_REALTIME;
    return clock_gettime(id, ts);

#endif
}


void*  zc_xcalloc(size_t nmemb, size_t size){
    void* ptr = calloc(nmemb, size);
    if (!ptr){
        fprintf(stderr, "zc_xcalloc: Out of memory\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void*  zc_xrealloc(void* ptr, size_t size){
    void* new_ptr = realloc(ptr, size);
    if (!new_ptr){
        fprintf(stderr, "zc_xrealloc: Out of memory\n");
        exit(EXIT_FAILURE);
    }
    return new_ptr;
}


int zc_dup(int fd) {
#ifdef ZC_ON_WINDOWS
    return _dup(fd);
#else
    return dup(fd);
#endif
}

int zc_dup2(int oldfd, int newfd) {
#ifdef ZC_ON_WINDOWS
    return _dup2(oldfd, newfd);
#else
    return dup2(oldfd, newfd);
#endif
}

int zc_open(const char* pathname, int flags) {
#ifdef ZC_ON_WINDOWS
    return _open(pathname, flags);
#else
    return open(pathname, flags);
#endif
}

int zc_close(int fd) {
#ifdef ZC_ON_WINDOWS
    return _close(fd);
#else
    return close(fd);
#endif
}

int zc_dlclose(zc_dlhandle handle) {
#ifdef ZC_ON_WINDOWS
    return FreeLibrary((HMODULE)handle) ? 0 : -1;
#else
    return dlclose(handle);
#endif
}

/* --- Networking Implementation --- */
#ifdef ZC_ON_WINDOWS
#   include <winsock2.h>
#   include <ws2tcpip.h>
#   pragma comment(lib, "ws2_32.lib") // Auto-link Winsock for MSVC
#else
#   include <sys/socket.h>
#   include <netinet/in.h>
#   include <arpa/inet.h>
#   include <netdb.h>
#   include <fcntl.h>
#endif

int zc_net_init(void) {
#ifdef ZC_ON_WINDOWS
    WSADATA wsa;
    return WSAStartup(MAKEWORD(2, 2), &wsa) == 0 ? 0 : -1;
#else
    return 0; // POSIX doesn't need init
#endif
}

void zc_net_cleanup(void) {
#ifdef ZC_ON_WINDOWS
    WSACleanup();
#endif
}

zc_socket_t zc_socket_create(int domain, int type, int protocol) {
    return socket(domain, type, protocol);
}

int zc_socket_connect(zc_socket_t sock, const char* host, int port) {
    struct addrinfo hints, *res;
    char port_str[10];
    sprintf(port_str, "%d", port);

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC; // Support IPv4 or IPv6
    hints.ai_socktype = SOCK_STREAM;

    if (getaddrinfo(host, port_str, &hints, &res) != 0) return -1;
    
    int result = connect(sock, res->ai_addr, (int)res->ai_addrlen);
    freeaddrinfo(res);
    return result;
}

int zc_socket_send(zc_socket_t sock, const char* buf, int len) {
    return send(sock, buf, len, 0);
}

int zc_socket_recv(zc_socket_t sock, char* buf, int len) {
    return recv(sock, buf, len, 0);
}

void zc_socket_close(zc_socket_t sock) {
#ifdef ZC_ON_WINDOWS
    closesocket(sock);
#else
    close(sock);
#endif
}

int zc_socket_set_nonblocking(zc_socket_t sock, bool nonblocking) {
#ifdef ZC_ON_WINDOWS
    u_long mode = nonblocking ? 1 : 0;
    return ioctlsocket(sock, FIONBIO, &mode);
#else
    int flags = fcntl(sock, F_GETFL, 0);
    if (flags == -1) return -1;
    flags = nonblocking ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK);
    return fcntl(sock, F_SETFL, flags);
#endif
}

#endif // ZC_COMPAT_IMPLEMENTATION

///////////////////////////////////////////////////////////////////////
// compat.h END
///////////////////////////////////////////////////////////////////////