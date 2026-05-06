
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
    TOK_NOT,        ///< 'not' keyword.
    TOK_FOR,        ///< 'for' keyword.
    TOK_DO,         ///< 'do' keyword.
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
    const char *file;  ///< Name of file with source code.
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
void clear_registered_traits();

/**
 * @brief Check if a name is a trait.
 */
int is_trait(const char *name);
int is_trait_ptr(const char *name);

/**
 * @brief Allocate memory.
 */
void *xmalloc(size_t size) __attribute__((returns_nonnull));

/**
 * @brief Reallocate memory.
 */
void *xrealloc(void *ptr, size_t new_size) __attribute__((returns_nonnull));

/**
 * @brief Allocate and zero memory.
 */
void *xcalloc(size_t n, size_t size) __attribute__((returns_nonnull));

/**
 * @brief Duplicate a string.
 */
char *xstrdup(const char *s) __attribute__((returns_nonnull));
void arena_reset(void);

/**
 * @brief Resolve a source file path using include paths and root path.
 */
char *z_resolve_path(const char *filename, const char *relative_to);

/**
 * @brief Load a file.
 */
char *load_file(const char *filename);

/**
 * @brief Sanitize file path for C string literals (converts \ to /).
 */
char *sanitize_path_for_c_string(const char *path);

/**
 * @brief Get the basename of a path (strips director).
 */
char *z_basename(const char *path);

/**
 * @brief Strips the extension from a filename.
 */
char *z_strip_ext(const char *filename);

/**
 * @brief Appends a flag to a buffer with space handling and overflow protection.
 */
void append_flag(char *dest, size_t max_size, const char *prefix, const char *val);

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
#include "utils/zvec.h"
ZVEC_GENERATE_IMPL(char *, Str)

/**
 * @brief Compiler configuration and flags.
 */
typedef struct
{
    char *input_file;     ///< Input source file path.
    zvec_Str extra_files; ///< Additional input files.
    zvec_Str c_files;     ///< Additional C/C++/OBJ files to be passed directly to backend.
    char *output_file;    ///< Output binary file path.

    // Modes.
    int mode_run;        ///< 1 if 'run' command (compile & execute).
    int mode_debug;      ///< 1 if `debug` command (emits source mappings and implies mode_run).
    int mode_check;      ///< 1 if 'check' command (syntax/type check only).
    int mode_transpile;  ///< 1 if 'transpile' command (to C).
    int emit_c;          ///< 1 if --emit-c (keep generated C file).
    int verbose;         ///< 1 if --verbose.
    int quiet;           ///< 1 if --quiet.
    int zen_mode;        ///< 1 if --zen (enable zen facts/easter eggs).
    int mode_doc;        ///< 1 if 'doc' command (generate documentation).
    int repl_mode;       ///< 1 if --repl (internal flag for REPL usage).
    int is_freestanding; ///< 1 if --freestanding (no stdlib).
    int use_cpp;         ///< 1 if --cpp (emit C++ compatible code).
    int use_cuda;        ///< 1 if --cuda (emit CUDA-compatible code).
    int use_objc;        ///< 1 if --objc (emit Objective-C compatible code).
    int mode_lsp;        ///< 1 if 'lsp' command (Language Server Protocol).
    int json_output;     ///< 1 if --json (emit structured JSON diagnostics).
    int use_typecheck;   ///< 1 if --check (enable manual semantic analysis).
    int warn_as_errors;  ///< 1 if --warn-errors or -Werror (treat Zen C warnings as errors).
    int no_suppress_warnings; ///< 1 if --no-suppress-warnings (disable default C warning
                              ///< suppressions).
    int warn_pedantic;        ///< 1 if -Wpedantic or --pedantic (show extra diagnostics).
    int misra_mode;           ///< 1 if --misra (emit MISRA C compliant code).
    uint64_t diag_mask;       ///< Bitmask of enabled diagnostics.

    int keep_comments; ///< 1 if --keep-comments (preserve comments in output).
    int recursive_doc; ///< 1 if doc generation should be recursive (default 1).

    // GCC Flags accumulator.
    char gcc_flags[4096]; ///< Flags passed to the backend compiler.

    // C Compiler selection (default: gcc)
    char cc[64]; ///< Backend compiler command (e.g. "gcc", "clang").

    char **c_function_whitelist; ///< List of C functions to suppress warnings for (from zenc.json).
    char **c_type_whitelist;     ///< List of C types to suppress warnings for (from zenc.json).

    // User-defined -D flags tracked for @cfg() evaluation.
    zvec_Str cfg_defines; ///< Define names from -D flags.

    // User-defined -I flags tracked for import resolution.
    zvec_Str include_paths; ///< Include paths for module resolution.

    char *root_path; ///< Detected Zen-C root directory.
    char *input_dir; ///< Directory of the primary input file.
} CompilerConfig;

extern CompilerConfig g_config;

/**
 * @brief Load all configurations (system, hidden project, visible project).
 */
void load_all_configs(void);

#endif
