#ifndef ZC_COMPAT_H
#define ZC_COMPAT_H

#ifdef _WIN32

#include <windows.h>
#include <stdio.h>

#define RTLD_LAZY 0

typedef HMODULE zc_dl_handle;

zc_dl_handle dlopen(const char *path, int flags);
void *dlsym(zc_dl_handle handle, const char *name);
int dlclose(zc_dl_handle handle);
const char *dlerror(void);

ssize_t getline(char **lineptr, size_t *n, FILE *stream);

int dup(int fd);
int dup2(int oldfd, int newfd);

#define ZC_PLUGIN_EXT ".dll"
#define ZC_PATH_SEP '\\'
#define ZC_NULL_DEVICE "NUL"

#define popen _popen
#define pclose _pclose

const char *zc_get_temp_dir(void);

#else

#include <dlfcn.h>
#include <unistd.h>

#define ZC_PLUGIN_EXT ".so"
#define ZC_PATH_SEP '/'
#define ZC_NULL_DEVICE "/dev/null"

static inline const char *zc_get_temp_dir(void) { return "/tmp/"; }

#endif

#ifdef _WIN32
char *extract_main_body(const char *c_file_path);
#endif

#endif
