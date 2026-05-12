// SPDX-License-Identifier: MIT
#ifndef ZC_PLATFORM_LANG_H
#ifndef ZC_ALLOW_INTERNAL
#error "platform/lang.h is internal to Zen C. Include the appropriate public header instead."
#endif

#define ZC_PLATFORM_LANG_H

// Standard C headers
#include <ctype.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// MSVC polyfills
#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#define strncasecmp _strnicmp
#define strcasecmp _stricmp
#endif

#endif // ZC_PLATFORM_LANG_H
