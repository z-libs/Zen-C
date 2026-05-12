
#ifndef ZC_ALLOW_INTERNAL
#error "zprep.h is internal to Zen C. Include the appropriate public header instead."
#endif

#ifndef ZPREP_H
#define ZPREP_H

// Convenience umbrella header.
// For focused includes, use the specific headers instead:
//   token.h       — Token, Lexer, ZenTokenType
//   arena.h       — xmalloc, zfree, allocation macros
//   compiler.h    — CompilerConfig, ZenCompiler, globals
//   utils/utils.h — Utility function declarations

#include "token.h"
#include "compiler.h"
#include "utils/utils.h"

#include "platform/lang.h"

// ** ANSI COLORS **
#include "utils/colors.h"

// Diagnostics (errors and warnings) are in diagnostics/diagnostics.h
#include "diagnostics/diagnostics.h"

#endif
