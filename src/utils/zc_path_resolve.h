#pragma once

// Returns a newly-allocated string (caller frees) containing the resolved path,
// or NULL on failure.
//
// Behavior:
// - If import_fn is absolute, returns a copy of import_fn.
// - Otherwise builds "<dir-of-current_file>/<import_fn>".
// - If import_fn starts with "./" or "../" (or .\ / ..\), it returns the joined path
//   (even if file doesn't exist).
// - If it's not explicitly relative, it only returns the joined path if the file exists;
//   otherwise it returns a copy of import_fn.
char *zc_resolve_import_path_alloc(const char *current_file, const char *import_fn);
