// SPDX-License-Identifier: MIT
#ifndef COMPTIME_INTERPRETER_H
#ifndef ZC_ALLOW_INTERNAL
#error                                                                                             \
    "analysis/comptime_interpreter.h is internal to Zen C. Include the appropriate public header instead."
#endif

#define COMPTIME_INTERPRETER_H

#include "../parser/parser.h"

char *interpret_comptime(ParserContext *ctx, ASTNode *body, const char *source_file);

#endif
