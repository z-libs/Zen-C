#ifndef ZC_COMPAT_H
#define ZC_COMPAT_H

#ifdef _WIN32

#include <stdio.h>
#include <io.h>
#include <basetsd.h>

typedef SSIZE_T ssize_t;

#define RTLD_LAZY 0

typedef void *zc_dl_handle;

zc_dl_handle zc_dlopen(const char *path, int flags);
void *zc_dlsym(zc_dl_handle handle, const char *name);
int zc_dlclose(zc_dl_handle handle);
const char *zc_dlerror(void);

ssize_t zc_getline(char **lineptr, size_t *n, FILE *stream);

#define dup _dup
#define dup2 _dup2
#define strdup _strdup
#define strncasecmp _strnicmp
#define fileno _fileno

#define ZC_PLUGIN_EXT ".dll"
#define ZC_PATH_SEP '\\'
#define ZC_NULL_DEVICE "NUL"

#define popen _popen
#define pclose _pclose

#define isatty _isatty
#define getpid _getpid
#define access _access
#define STDERR_FILENO 2
#define STDOUT_FILENO 1
#define STDIN_FILENO 0
#define R_OK 4
#define W_OK 2
#define X_OK 1 // TODO: _access() on Windows ignores execute checks.
#define F_OK 0

#include <process.h>

const char *zc_get_temp_dir(void);
void zc_seed_random(void);

#else

#include <dlfcn.h>
#include <unistd.h>

typedef void *zc_dl_handle;

#define zc_dlopen dlopen
#define zc_dlsym dlsym
#define zc_dlclose dlclose
#define zc_dlerror dlerror
#define zc_getline getline

#define ZC_PLUGIN_EXT ".so"
#define ZC_PATH_SEP '/'
#define ZC_NULL_DEVICE "/dev/null"

static inline const char *zc_get_temp_dir(void) { return "/tmp/"; }

void zc_seed_random(void);

#endif

#ifdef _WIN32
// TODO: extract_main_body is Windows-only; avoid non-Windows references.
char *extract_main_body(const char *c_file_path);
#endif

#endif
