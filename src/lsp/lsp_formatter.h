// SPDX-License-Identifier: MIT
#ifndef LSP_FORMATTER_H
#ifndef ZC_ALLOW_INTERNAL
#error "lsp/lsp_formatter.h is internal to Zen C. Include the appropriate public header instead."
#endif

#define LSP_FORMATTER_H

/**
 * @brief Formats Zen C source code.
 *
 * @param src The original source code.
 * @return char* The formatted source code (must be freed by caller).
 */
char *lsp_format_source(const char *src);

#endif
