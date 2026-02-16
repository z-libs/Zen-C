
#ifndef ZPREP_H
#define ZPREP_H

#include "platform/lang.h"
#include "platform/os.h"

// **ZEN VERSION**
#ifndef ZEN_VERSION
#define ZEN_VERSION "0.1.0" ///< Zen-C version.
#endif

// ** ANSI COLORS **
#include "utils/colors.h"

// ** MEMORY OVERRIDES (Arena) **
#define free(ptr) ((void)0)          ///< Free memory.
#define malloc(sz) xmalloc(sz)       ///< Allocate memory.
#define realloc(p, s) xrealloc(p, s) ///< Reallocate memory.
#define calloc(n, s) xcalloc(n, s)   ///< Allocate and zero memory.

// ** GLOBAL STATE **
extern char *g_current_filename; ///< Current filename.

/**
 * @brief Token types for the Lexer.
 */
typedef enum
{
    TOK_EOF = 0,    ///< End of File.
    TOK_IDENT,      ///< Identifier (variable, function name).
    TOK_INT,        ///< Integer literal.
    TOK_FLOAT,      ///< Float literal.
    TOK_STRING,     ///< String literal.
    TOK_FSTRING,    ///< Formatted string literal (f"val is {x}").
    TOK_RAW_STRING, ///< Raw string literal (r"..." - no interpolation).
    TOK_CHAR,       ///< Character literal.
    TOK_LPAREN,     ///< (
    TOK_RPAREN,     ///< )
    TOK_LBRACE,     ///< {
    TOK_RBRACE,     ///< }
    TOK_LBRACKET,   ///< [
    TOK_RBRACKET,   ///< ]
    TOK_LANGLE,     ///< <
    TOK_RANGLE,     ///< >
    TOK_COMMA,      ///< ,
    TOK_COLON,      ///< :
    TOK_SEMICOLON,  ///< ;
    TOK_OP,         ///< General operator (e.g. +, *, /).
    TOK_AT,         ///< @
    TOK_DOTDOT,     ///< ..
    TOK_DOTDOT_EQ,  ///< ..= (inclusive range).
    TOK_DOTDOT_LT,  ///< ..< (exclusive range, explicit).
    TOK_ARROW,      ///< -> or =>
    TOK_PIPE,       ///< |> (pipe operator).
    TOK_TEST,       ///< 'test' keyword.
    TOK_ASSERT,     ///< 'assert' keyword.
    TOK_SIZEOF,     ///< 'sizeof' keyword.
    TOK_DEF,        ///< 'def' keyword.
    TOK_DEFER,      ///< 'defer' keyword.
    TOK_AUTOFREE,   ///< 'autofree' keyword.
    TOK_QUESTION,   ///< ?
    TOK_USE,        ///< 'use' keyword.
    TOK_QQ,         ///< ?? (null coalescing).
    TOK_QQ_EQ,      ///< ??=
    TOK_Q_DOT,      ///< ?. (optional chaining).
    TOK_DCOLON,     ///< ::
    TOK_TRAIT,      ///< 'trait' keyword.
    TOK_IMPL,       ///< 'impl' keyword.
    TOK_AND,        ///< 'and' keyword.
    TOK_OR,         ///< 'or' keyword.
    TOK_FOR,        ///< 'for' keyword.
    TOK_COMPTIME,   ///< 'comptime' keyword.
    TOK_ELLIPSIS,   ///< ...
    TOK_UNION,      ///< 'union' keyword.
    TOK_ASM,        ///< 'asm' keyword.
    TOK_VOLATILE,   ///< 'volatile' keyword.
    TOK_ASYNC,      ///< 'async' keyword.
    TOK_AWAIT,      ///< 'await' keyword.
    TOK_PREPROC,    ///< Preprocessor directive (#...).
    TOK_ALIAS,      ///< 'alias' keyword.
    TOK_COMMENT,    ///< Comment (usually skipped).
    TOK_OPAQUE,     ///< 'opaque' keyword.
    TOK_UNKNOWN     ///< Unknown token.
} ZenTokenType;

/**
 * @brief Represents a source token.
 */
typedef struct
{
    ZenTokenType type; ///< Type of the token.
    const char *start; ///< Pointer to start of token in source buffer.
    int len;           ///< Length of the token text.
    int line;          ///< Line number (1-based).
    int col;           ///< Column number (1-based).
} Token;

/**
 * @brief Lexer state.
 */
typedef struct
{
    const char *src;   ///< Source code buffer.
    int pos;           ///< Current position index.
    int line;          ///< Current line number.
    int col;           ///< Current column number.
    int emit_comments; ///< 1 if comments should be emitted as tokens.
} Lexer;

/**
 * @brief Initialize the lexer.
 */
void lexer_init(Lexer *l, const char *src);

/**
 * @brief Get the next token.
 */
Token lexer_next(Lexer *l);

/**
 * @brief Get the next token without advancing.
 */
Token lexer_peek(Lexer *l);

/**
 * @brief Get the next token without advancing (2 look ahead).
 */
Token lexer_peek2(Lexer *l);

/**
 * @brief Register a trait.
 */
void register_trait(const char *name);

/**
 * @brief Check if a name is a trait.
 */
int is_trait(const char *name);

/**
 * @brief Allocate memory.
 */
void *xmalloc(size_t size);

/**
 * @brief Reallocate memory.
 */
void *xrealloc(void *ptr, size_t new_size);

/**
 * @brief Allocate and zero memory.
 */
void *xcalloc(size_t n, size_t size);

/**
 * @brief Duplicate a string.
 */
char *xstrdup(const char *s);

/**
 * @brief Load a file.
 */
char *load_file(const char *filename);

// ** Buffer Size Constants **
#define MAX_FLAGS_SIZE 1024
#define MAX_PATH_SIZE 1024
#define MAX_PATTERN_SIZE 1024

// ** Build Directives **
extern char g_link_flags[MAX_FLAGS_SIZE];
extern char g_cflags[MAX_FLAGS_SIZE];
extern int g_warning_count;

struct ParserContext;

/**
 * @brief Scan build directives.
 */
void scan_build_directives(struct ParserContext *ctx, const char *src);

/**
 * @brief Calculate Levenshtein distance.
 */
int levenshtein(const char *s1, const char *s2);

// Diagnostics (errors and warnings) are in diagnostics/diagnostics.h
#include "diagnostics/diagnostics.h"

/**
 * @brief Compiler configuration and flags.
 */
typedef struct
{
    char *input_file;      ///< Input source file path.
    char *extra_files[64]; ///< Additional input files.
    int extra_file_count;  ///< Number of extra input files.
    char *c_files[64];     ///< Additional C/C++/OBJ files to be passed directly to backend.
    int c_file_count;      ///< Number of C/C++/OBJ files.
    char *output_file;     ///< Output binary file path.

    // Modes.
    int mode_run;        ///< 1 if 'run' command (compile & execute).
    int mode_check;      ///< 1 if 'check' command (syntax/type check only).
    int emit_c;          ///< 1 if --emit-c (keep generated C file).
    int verbose;         ///< 1 if --verbose.
    int quiet;           ///< 1 if --quiet.
    int no_zen;          ///< 1 if --no-zen (disable zen facts/easter eggs).
    int repl_mode;       ///< 1 if --repl (internal flag for REPL usage).
    int is_freestanding; ///< 1 if --freestanding (no stdlib).
    int mode_transpile;  ///< 1 if 'transpile' command (to C).
    int use_cpp;         ///< 1 if --cpp (emit C++ compatible code).
    int use_cuda;        ///< 1 if --cuda (emit CUDA-compatible code).
    int use_objc;        ///< 1 if --objc (emit Objective-C compatible code).
    int mode_lsp;        ///< 1 if 'lsp' command (Language Server Protocol).
    int json_output;     ///< 1 if --json (emit structured JSON diagnostics).
    int use_typecheck;   ///< 1 if --typecheck (enable manual semantic analysis).

    int keep_comments; ///< 1 if --keep-comments (preserve comments in output).

    // GCC Flags accumulator.
    char gcc_flags[4096]; ///< Flags passed to the backend compiler.

    // C Compiler selection (default: gcc)
    char cc[64]; ///< Backend compiler command (e.g. "gcc", "clang").

    char **c_function_whitelist; ///< List of C functions to suppress warnings for (from zenc.json).
} CompilerConfig;

extern CompilerConfig g_config;
extern char g_link_flags[];
extern char g_cflags[];

struct ParserContext;

/**
 * @brief Scan build directives.
 */
/**
 * @brief Scan build directives.
 */
void scan_build_directives(struct ParserContext *ctx, const char *src);

/**
 * @brief Load all configurations (system, hidden project, visible project).
 */
void load_all_configs(void);

#endif
