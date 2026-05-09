#ifndef COMPTIME_INTERPRETER_H
#define COMPTIME_INTERPRETER_H

#include "../parser/parser.h"

char *interpret_comptime(ParserContext *ctx, ASTNode *body, const char *source_file);

#endif
