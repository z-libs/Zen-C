#include "path_utils.h"

#if defined(_WIN32)
  #include <windows.h>

  // Use Win32 APIs to canonicalize.
  char *zc_realpath_alloc(const char *path) {
      // First get a full path (absolute + normalized slashes)
      DWORD need = GetFullPathNameA(path, 0, NULL, NULL);
      if (need == 0) return NULL;

      char *buf = (char*)malloc(need);
      if (!buf) return NULL;

      DWORD got = GetFullPathNameA(path, need, buf, NULL);
      if (got == 0 || got >= need) { free(buf); return NULL; }

      // Optional: also resolve to final path (resolves symlinks/junctions) if file exists.
      // If it doesn't exist, keep the GetFullPathNameA result.
      HANDLE h = CreateFileA(buf, 0,
                             FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                             NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
      if (h != INVALID_HANDLE_VALUE) {
          DWORD need2 = GetFinalPathNameByHandleA(h, NULL, 0, FILE_NAME_NORMALIZED);
          if (need2 > 0) {
              char *buf2 = (char*)malloc(need2);
              if (buf2) {
                  DWORD got2 = GetFinalPathNameByHandleA(h, buf2, need2, FILE_NAME_NORMALIZED);
                  if (got2 > 0 && got2 < need2) {
                      CloseHandle(h);
                      free(buf);
                      // Strip leading "\\?\" which Windows sometimes adds
                      const char *p = buf2;
                      if (got2 >= 4 && buf2[0] == '\\' && buf2[1] == '\\' && buf2[2] == '?' && buf2[3] == '\\')
                          p = buf2 + 4;

                      char *out = _strdup(p);
                      free(buf2);
                      return out;
                  }
                  free(buf2);
              }
          }
          CloseHandle(h);
      }

      return buf;
  }

#else
  #include <limits.h>
  #include <unistd.h>

  char *zc_realpath_alloc(const char *path) {
      // POSIX: realpath alloc form is available on many platforms,
      // but the portable approach is to provide our own buffer.
      // Use PATH_MAX when available.
      char tmp[PATH_MAX];
      if (!realpath(path, tmp)) return NULL;
      return strdup(tmp);
  }
#endif
