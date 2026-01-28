#if !def(_WIN32)

#include "zc_fs_platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

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

int _z_fs_access(char* pathname, int mode) { return access(pathname, mode); }
int _z_fs_unlink(char* pathname) { return unlink(pathname); }
int _z_fs_rmdir(char* pathname) { return rmdir(pathname); }

int _z_fs_mkdir(char* path) { return mkdir(path, 0777); }

int _z_fs_get_metadata(char* path, uint64_t* size, int* is_dir, int* is_file) {
    struct stat st;
    if (stat(path, &st) != 0) return -1;
    *size = (uint64_t)st.st_size;
    *is_dir = S_ISDIR(st.st_mode);
    *is_file = S_ISREG(st.st_mode);
    return 0;
}

void* _z_fs_opendir(char* name) { return opendir(name); }
int _z_fs_closedir(void* dir) { return closedir((DIR*)dir); }

// NOTE: This keeps your old behavior (fast) but may be DT_UNKNOWN on some FS.
// If you want correctness, you’ll need to stat(fullpath) which requires the parent path.
int _z_fs_read_entry(void* dir, char* out_name, int buf_size, int* is_dir) {
    struct dirent* ent = readdir((DIR*)dir);
    if (!ent) return 0;

    strncpy(out_name, ent->d_name, (size_t)buf_size - 1);
    out_name[buf_size - 1] = 0;

#ifdef DT_DIR
    *is_dir = (ent->d_type == DT_DIR);
#else
    *is_dir = 0;
#endif
    return 1;
}

#endif // !_WIN32
