#pragma once
#include <stdlib.h>

// Returns a newly allocated canonical/absolute path for `path`.
// Caller must free() it. Returns NULL on failure.
char *zc_realpath_alloc(const char *path);
