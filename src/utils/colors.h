#ifndef ZEN_COLORS_H
#ifndef ZC_ALLOW_INTERNAL
#error "utils/colors.h is internal to Zen C. Include the appropriate public header instead."
#endif

#define ZEN_COLORS_H

#include <stdio.h>
#include <stdarg.h>

// ** ANSI COLORS **
#define COLOR_RESET "\033[0m"     ///< Reset color.
#define COLOR_RED "\033[1;31m"    ///< Red color.
#define COLOR_GREEN "\033[1;32m"  ///< Green color.
#define COLOR_YELLOW "\033[1;33m" ///< Yellow color.
#define COLOR_BLUE "\033[1;34m"   ///< Blue color.
#define COLOR_CYAN "\033[1;36m"   ///< Cyan color.
#define COLOR_BOLD "\033[1m"      ///< Bold text.

int zprintf(const char *format, ...);
int zvprintf(const char *format, va_list args);
int zfprintf(FILE *stream, const char *format, ...);
int zvfprintf(FILE *stream, const char *format, va_list args);

#ifndef ZEN_DISABLE_COLORS_WRAPPER
#define printf zprintf
#define fprintf zfprintf
#define vprintf zvprintf
#define vfprintf zvfprintf
#endif

#endif // ZEN_COLORS_H
