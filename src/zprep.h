
#ifndef ZPREP_H
#define ZPREP_H

// Phase 1: zprep.h is now a convenience umbrella.
// Individual components are in focused headers:
//   token.h  — Token, Lexer, ZenTokenType, lexer functions
//   arena.h  — xmalloc, zfree, allocation macros

#include "token.h"
#include "arena.h"

#include "platform/lang.h"
#include "platform/os.h"

// **ZEN VERSION**
#ifndef ZEN_VERSION
#define ZEN_VERSION "0.1.0" ///< Zen-C version.
#endif

// ** ANSI COLORS **
#include "utils/colors.h"
#include "utils/zvec.h"
ZVEC_GENERATE_IMPL(char *, Str)

/**
 * @brief Compiler configuration and flags.
 */
typedef struct CompilerConfig
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

// ** GLOBAL STATE **
extern char *g_current_filename; ///< Current filename.

typedef struct ZenCompiler
{
    CompilerConfig config;
    zarena arena; ///< Primary memory arena for the compilation session.
    int error_count;
    int warning_count;
    char link_flags[1024]; // MAX_FLAGS_SIZE
    char cflags[1024];
    double start_time;
} ZenCompiler;

extern ZenCompiler g_compiler;

#define g_config g_compiler.config
#define g_link_flags g_compiler.link_flags
#define g_cflags g_compiler.cflags
#define g_error_count g_compiler.error_count
#define g_warning_count g_compiler.warning_count
#define g_start_time g_compiler.start_time

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

void arena_reset(zarena *a);

/**
 * @brief Resolve a source file path using include paths and root path.
 */
char *z_resolve_path(const char *filename, const char *relative_to, CompilerConfig *cfg);

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

// g_config is now a macro pointing to g_compiler.config

/**
 * @brief Load all configurations (system, hidden project, visible project).
 */
void load_all_configs(CompilerConfig *cfg);

#endif
