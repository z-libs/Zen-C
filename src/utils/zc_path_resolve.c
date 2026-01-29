#include "zc_path_resolve.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
  #include <io.h>
  #define zc_access _access
  #define ZC_R_OK 4
#else
  #include <unistd.h>
  #define zc_access access
  #define ZC_R_OK R_OK
#endif

static char *zc_strdup(const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s);
    char *out = (char*)malloc(n + 1);
    if (!out) return NULL;
    memcpy(out, s, n + 1);
    return out;
}

static int zc_is_sep(char c) {
    return c == '/' || c == '\\';
}

// Windows + POSIX absolute detection:
// - POSIX: "/..."
// - Windows: "C:\..." or "C:/..." or "\\server\share" or "//server/share" or "\\?\..."
static int zc_is_abs_path(const char *p) {
    if (!p || !p[0]) return 0;

    // POSIX abs
    if (p[0] == '/') return 1;

    // UNC or extended prefixes
    if ((p[0] == '\\' && p[1] == '\\') || (p[0] == '/' && p[1] == '/')) return 1;

    // Drive letter: C:\ or C:/
    if (((p[0] >= 'A' && p[0] <= 'Z') || (p[0] >= 'a' && p[0] <= 'z')) &&
        p[1] == ':' && zc_is_sep(p[2])) {
        return 1;
    }

    return 0;
}

static int zc_is_explicit_relative(const char *fn) {
    if (!fn || fn[0] != '.') return 0;

    // "./" or ".\"
    if (zc_is_sep(fn[1])) return 1;

    // "../" or "..\"
    if (fn[1] == '.' && zc_is_sep(fn[2])) return 1;

    return 0;
}

static char *zc_dirname_alloc(const char *path) {
    if (!path) return NULL;

    const char *last_fwd = strrchr(path, '/');
    const char *last_bak = strrchr(path, '\\');
    const char *last = last_fwd > last_bak ? last_fwd : last_bak;

    if (!last) {
        // no slash: current directory
        return zc_strdup(".");
    }

    size_t n = (size_t)(last - path);
    char *out = (char*)malloc(n + 1);
    if (!out) return NULL;
    memcpy(out, path, n);
    out[n] = '\0';
    return out;
}

// Join using '/' (Windows APIs and CRT generally accept it fine).
// If you prefer '\\' on Windows, swap join_sep to '\\' under _WIN32.
static char *zc_join_alloc(const char *a, const char *b) {
    if (!a || !b) return NULL;

    size_t na = strlen(a);
    size_t nb = strlen(b);

    char join_sep = '/';

    // a + (sep?) + b + '\0'
    size_t extra = (na > 0 && !zc_is_sep(a[na - 1])) ? 1 : 0;
    char *out = (char*)malloc(na + extra + nb + 1);
    if (!out) return NULL;

    memcpy(out, a, na);
    size_t pos = na;

    if (extra) out[pos++] = join_sep;

    memcpy(out + pos, b, nb);
    out[pos + nb] = '\0';
    return out;
}

char *zc_resolve_import_path_alloc(const char *current_file, const char *import_fn) {
    if (!import_fn) return NULL;

    // Absolute import: just return it
    if (zc_is_abs_path(import_fn)) {
        return zc_strdup(import_fn);
    }

    // If we don't know the current file, can't resolve relative to it
    if (!current_file || !current_file[0]) {
        return zc_strdup(import_fn);
    }

    int explicit_rel = zc_is_explicit_relative(import_fn);

    char *dir = zc_dirname_alloc(current_file);
    if (!dir) return zc_strdup(import_fn);

    char *candidate = zc_join_alloc(dir, import_fn);
    free(dir);

    if (!candidate) return zc_strdup(import_fn);

    // Match your original semantics:
    // - explicit relative: always use candidate
    // - otherwise: only use if exists
    if (explicit_rel || zc_access(candidate, ZC_R_OK) == 0) {
        return candidate;
    }

    free(candidate);
    return zc_strdup(import_fn);
}
