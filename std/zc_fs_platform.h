#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// FILE* wrappers
void*   _z_fs_fopen(char* path, char* mode);
int     _z_fs_fclose(void* stream);
size_t  _z_fs_fread(void* ptr, size_t size, size_t nmemb, void* stream);
size_t  _z_fs_fwrite(void* ptr, size_t size, size_t nmemb, void* stream);
int     _z_fs_fseek(void* stream, int64_t offset, int whence);
int64_t _z_fs_ftell(void* stream);

// memory
void* _z_fs_malloc(size_t size);
void  _z_fs_free(void* ptr);

// basic fs ops
int _z_fs_access(char* pathname, int mode);
int _z_fs_unlink(char* pathname);
int _z_fs_rmdir(char* pathname);
int _z_fs_mkdir(char* path);

// metadata
int _z_fs_get_metadata(char* path, uint64_t* size, int* is_dir, int* is_file);

// directory iteration
void* _z_fs_opendir(char* name);
int   _z_fs_closedir(void* dir);
// returns 1 if entry read, 0 if end/no entry
int   _z_fs_read_entry(void* dir, char* out_name, int buf_size, int* is_dir);

#ifdef __cplusplus
}
#endif
