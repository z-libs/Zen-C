
#ifndef ZPREP_H
#define ZPREP_H

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ** ANSI COLORS **
#define COLOR_RESET "\033[0m"
#define COLOR_RED "\033[1;31m"
#define COLOR_GREEN "\033[1;32m"
#define COLOR_YELLOW "\033[1;33m"
#define COLOR_BLUE "\033[1;34m"
#define COLOR_CYAN "\033[1;36m"
#define COLOR_BOLD "\033[1m"

// ** MEMORY OVERRIDES (Arena) **
#define free(ptr) ((void)0)
#define malloc(sz) xmalloc(sz)
#define realloc(p, s) xrealloc(p, s)
#define calloc(n, s) xcalloc(n, s)

// ** GLOBAL STATE **
extern char *g_current_filename;

typedef enum
{
    TOK_EOF = 0,
    TOK_IDENT,
    TOK_INT,
    TOK_FLOAT,
    TOK_STRING,
    TOK_FSTRING,
    TOK_CHAR,
    TOK_LPAREN,
    TOK_RPAREN,
    TOK_LBRACE,
    TOK_RBRACE,
    TOK_LBRACKET,
    TOK_RBRACKET,
    TOK_LANGLE,
    TOK_RANGLE,
    TOK_COMMA,
    TOK_COLON,
    TOK_SEMICOLON,
    TOK_OP,
    TOK_AT,
    TOK_DOTDOT,
    TOK_ARROW,
    TOK_PIPE,
    TOK_TEST,
    TOK_ASSERT,
    TOK_SIZEOF,
    TOK_DEFER,
    TOK_AUTOFREE,
    TOK_QUESTION,
    TOK_USE,
    TOK_QQ,
    TOK_QQ_EQ,
    TOK_Q_DOT,
    TOK_DCOLON,
    TOK_TRAIT,
    TOK_IMPL,
    TOK_AND,
    TOK_OR,
    TOK_FOR,
    TOK_COMPTIME,
    TOK_ELLIPSIS,
    TOK_UNION,
    TOK_ASM,
    TOK_VOLATILE,
    TOK_MUT,
    TOK_ASYNC,
    TOK_AWAIT,
    TOK_PREPROC,
    TOK_COMMENT,
    TOK_UNKNOWN
} TokenType;

typedef struct
{
    TokenType type;
    const char *start;
    int len;
    int line;
    int col;
} Token;

typedef struct
{
    const char *src;
    int pos;
    int line;
    int col;
} Lexer;

void lexer_init(Lexer *l, const char *src);
Token lexer_next(Lexer *l);
Token lexer_peek(Lexer *l);
Token lexer_peek2(Lexer *l);

void register_trait(const char *name);
int is_trait(const char *name);

// Arena and memory.
void *xmalloc(size_t size);
void *xrealloc(void *ptr, size_t new_size);
void *xcalloc(size_t n, size_t size);
char *xstrdup(const char *s);

// Error reporting.
void zpanic(const char *fmt, ...);
void zpanic_at(Token t, const char *fmt, ...);

char *load_file(const char *filename);

// ** Buffer Size Constants **
#define MAX_FLAGS_SIZE 1024
#define MAX_PATH_SIZE 1024
#define MAX_PATTERN_SIZE 1024

// ** Build Directives **
extern char g_link_flags[MAX_FLAGS_SIZE];
extern char g_cflags[MAX_FLAGS_SIZE];

struct ParserContext;

void scan_build_directives(struct ParserContext *ctx, const char *src);
int levenshtein(const char *s1, const char *s2);
void zpanic_with_suggestion(Token t, const char *msg, const char *suggestion);

// Specific error types.
void error_undefined_function(Token t, const char *func_name, const char *suggestion);
void error_undefined_variable(Token t, const char *var_name, const char *suggestion);
void error_wrong_arg_count(Token t, const char *func_name, int expected, int got);
void error_undefined_field(Token t, const char *struct_name, const char *field_name,
                           const char *suggestion);
void error_type_expected(Token t, const char *expected, const char *got);
void error_cannot_index(Token t, const char *type_name);

// Warning system.
void zwarn(const char *fmt, ...);
void zwarn_at(Token t, const char *fmt, ...);

// Specific warnings.
void warn_unused_variable(Token t, const char *var_name);
void warn_unused_parameter(Token t, const char *param_name, const char *func_name);
void warn_shadowing(Token t, const char *var_name);
void warn_unreachable_code(Token t);
void warn_implicit_conversion(Token t, const char *from_type, const char *to_type);
void warn_narrowing_conversion(Token t, const char *from_type, const char *to_type);
void warn_missing_return(Token t, const char *func_name);
void warn_comparison_always_true(Token t, const char *reason);
void warn_comparison_always_false(Token t, const char *reason);
void warn_division_by_zero(Token t);
void warn_integer_overflow(Token t, const char *type_name, long long value);
void warn_array_bounds(Token t, int index, int size);
void warn_format_string(Token t, int arg_num, const char *expected, const char *got);
void warn_null_pointer(Token t, const char *expr);

// ** Compiler Config **
typedef struct
{
    char *input_file;
    char *output_file;

    // Modes.
    int mode_run;        // 1 if 'run' command.
    int mode_check;      // 1 if 'check' command or --check.
    int emit_c;          // 1 if --emit-c (keep C file).
    int verbose;         // 1 if --verbose.
    int quiet;           // 1 if --quiet.
    int repl_mode;       // 1 if --repl (internal flag for REPL usage).
    int is_freestanding; // 1 if --freestanding.
    int mode_transpile;  // 1 if 'transpile' command.

    // GCC Flags accumulator.
    char gcc_flags[4096];

    // C Compiler selection (default: gcc)
    char cc[64];
} CompilerConfig;

extern CompilerConfig g_config;
extern char g_link_flags[];
extern char g_cflags[];

struct ParserContext;
void scan_build_directives(struct ParserContext *ctx, const char *src);

#endif
