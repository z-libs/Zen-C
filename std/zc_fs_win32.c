#ifdef _WIN32

#include "zc_fs_platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <windows.h>
#include <io.h>
#include <direct.h>

// typedef needed for Vec<DirEntry*> generation if inferred
typedef struct DirEntry* DirEntryPtr;

void* _z_fs_fopen(char* path, char* mode) { return fopen(path, mode); }
int _z_fs_fclose(void* stream) { return fclose((FILE*)stream); }
size_t _z_fs_fread(void* ptr, size_t size, size_t nmemb, void* stream) { return fread(ptr, size, nmemb, (FILE*)stream); }
size_t _z_fs_fwrite(void* ptr, size_t size, size_t nmemb, void* stream) { return fwrite(ptr, size, nmemb, (FILE*)stream); }
int _z_fs_fseek(void* stream, int64_t offset, int whence) { return fseek((FILE*)stream, (long)offset, whence); }
int64_t _z_fs_ftell(void* stream) { return (int64_t)ftell((FILE*)stream); }

void* _z_fs_malloc(size_t size) { return malloc(size); }
void _z_fs_free(void* ptr) { free(ptr); }

int _z_fs_access(char* pathname, int mode) { return _access(pathname, mode); }
int _z_fs_unlink(char* pathname) { return _unlink(pathname); }
int _z_fs_rmdir(char* pathname) { return _rmdir(pathname); }
int _z_fs_mkdir(char* path) { return _mkdir(path); }

int _z_fs_get_metadata(char* path, uint64_t* size, int* is_dir, int* is_file) {
    struct _stat st;
    if (_stat(path, &st) != 0) return -1;
    *size = (uint64_t)st.st_size;
    *is_dir = (st.st_mode & _S_IFDIR) != 0;
    *is_file = (st.st_mode & _S_IFREG) != 0;
    return 0;
}

typedef struct ZcWinDir {
    HANDLE hFind;
    WIN32_FIND_DATAA data;
    int first_valid;
    char pattern[MAX_PATH]; // "<dir>\*"
} ZcWinDir;

static int zc_is_dot_or_dotdot(const char* name) {
    return (name[0] == '.' && name[1] == '\0') ||
           (name[0] == '.' && name[1] == '.' && name[2] == '\0');
}

void* _z_fs_opendir(char* name) {
    if (!name) return NULL;

    ZcWinDir* d = (ZcWinDir*)malloc(sizeof(ZcWinDir));
    if (!d) return NULL;
    memset(d, 0, sizeof(*d));
    d->hFind = INVALID_HANDLE_VALUE;

    size_t n = strlen(name);
    if (n + 3 >= MAX_PATH) { free(d); return NULL; }

    strcpy(d->pattern, name);
    if (n > 0 && name[n - 1] != '\\' && name[n - 1] != '/') {
        d->pattern[n++] = '\\';
        d->pattern[n] = '\0';
    }
    strcat(d->pattern, "*");

    d->hFind = FindFirstFileA(d->pattern, &d->data);
    if (d->hFind == INVALID_HANDLE_VALUE) {
        free(d);
        return NULL;
    }
    d->first_valid = 1;
    return d;
}

int _z_fs_closedir(void* dir) {
    ZcWinDir* d = (ZcWinDir*)dir;
    if (!d) return 0;
    if (d->hFind != INVALID_HANDLE_VALUE) FindClose(d->hFind);
    free(d);
    return 0;
}

int _z_fs_read_entry(void* dir, char* out_name, int buf_size, int* is_dir) {
    ZcWinDir* d = (ZcWinDir*)dir;
    if (!d || !out_name || buf_size <= 0 || !is_dir) return 0;

    for (;;) {
        WIN32_FIND_DATAA* ent = &d->data;

        if (!d->first_valid) {
            if (!FindNextFileA(d->hFind, &d->data)) return 0;
            ent = &d->data;
        } else {
            d->first_valid = 0;
        }

        if (zc_is_dot_or_dotdot(ent->cFileName)) continue;

        strncpy(out_name, ent->cFileName, (size_t)buf_size - 1);
        out_name[buf_size - 1] = 0;
        *is_dir = (ent->dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
        return 1;
    }
}

#endif // _WIN32
