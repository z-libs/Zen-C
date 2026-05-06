
#ifndef DIAGNOSTICS_H
#define DIAGNOSTICS_H

#include "zprep.h"

// Forward declaration
struct ParserContext;

// Globals are now managed via g_compiler (see zprep.h)

// ** Core Error Functions **

/**
 * @brief Fatal error (exits).
 */
void zpanic(const char *fmt, ...);

/**
 * @brief Fatal system error (e.g. OOM), prints "Fatal: " prefix.
 */
void zfatal(const char *fmt, ...);

/**
 * @brief Fatal error with token location (exits unless fault-tolerant).
 */
void zpanic_at(Token t, const char *fmt, ...);
void zpanic_at_diag(int diag_id, Token t, const char *fmt, ...);

/**
 * @brief Fatal error with suggestion (exits unless fault-tolerant).
 */
void zpanic_with_suggestion(Token t, const char *msg, const char *suggestion);

/**
 * @brief Fatal error with multiple suggestions/hints (NULL-terminated array).
 */
void zpanic_with_hints(Token t, const char *msg, const char *const *hints);

/**
 * @brief Non-fatal error with token location (does not exit).
 * Used for semantic analysis to report multiple errors.
 */
void zerror_at(Token t, const char *fmt, ...);

/**
 * @brief Non-fatal error with suggestion (does not exit).
 */
void zerror_with_suggestion(Token t, const char *msg, const char *suggestion);

/**
 * @brief Non-fatal error with multiple suggestions/hints (NULL-terminated array).
 */
void zerror_with_hints(Token t, const char *msg, const char *const *hints);

// ** Core Warning Functions **

/**
 * @brief Non-fatal warning.
 */
void zwarn(const char *fmt, ...);

/**
 * @brief Non-fatal warning with token location.
 */
void zwarn_at(Token t, const char *fmt, ...);
void zwarn_at_diag(int diag_id, Token t, const char *fmt, ...);

/**
 * @brief Non-fatal warning with suggestion.
 */
void zwarn_with_suggestion(Token t, const char *msg, const char *suggestion);
void zwarn_with_suggestion_diag(int diag_id, Token t, const char *msg, const char *suggestion);

// ** Specific Error Types **

void error_undefined_function(Token t, const char *func_name, const char *suggestion);
void error_wrong_arg_count(Token t, const char *func_name, int expected, int got);
void error_undefined_field(Token t, const char *struct_name, const char *field_name,
                           const char *suggestion);
void error_type_expected(Token t, const char *expected, const char *got);
void error_cannot_index(Token t, const char *type_name);

// ** Specific Warning Types **

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
void warn_void_main(Token t);
void warn_misra_violation(Token t, const char *msg);

typedef enum
{
    CAT_NONE,
    CAT_INTEROP,
    CAT_PEDANTIC,
    CAT_UNUSED,
    CAT_STYLE,
    CAT_TYPE,
    CAT_SAFETY,
    CAT_LOGIC,
    CAT_CONVERSION
} DiagnosticCategory;

typedef enum
{
    DIAG_NONE,
    // Interop
    DIAG_INTEROP_UNDEF_FUNC, // W100
    // Pedantic
    DIAG_PEDANTIC_STRICT_TYPING, // W300
    // Unused
    DIAG_UNUSED_VAR,      // W200
    DIAG_UNUSED_PARAM,    // W201
    DIAG_STYLE_SHADOWING, // W700
    // Safety
    DIAG_SAFETY_NULL_PTR,         // W400
    DIAG_SAFETY_DIV_ZERO,         // W401
    DIAG_SAFETY_ARRAY_BOUNDS,     // W402
    DIAG_SAFETY_INTEGER_OVERFLOW, // W403
    // Logic
    DIAG_LOGIC_UNREACHABLE,    // W500
    DIAG_LOGIC_ALWAYS_TRUE,    // W501
    DIAG_LOGIC_ALWAYS_FALSE,   // W502
    DIAG_LOGIC_MISSING_RETURN, // W503
    // Conversion
    DIAG_CONVERSION_IMPLICIT,  // W600
    DIAG_CONVERSION_NARROWING, // W601
    // Style
    DIAG_STYLE_FORMAT,           // W701
    DIAG_STYLE_DEPRECATED_VAR,   // W702
    DIAG_STYLE_DEPRECATED_CONST, // W703
    // MISRA
    DIAG_MISRA_VIOLATION,
    // ...
    DIAG_MAX
} DiagnosticID;

int is_diag_enabled(DiagnosticID id);
void zwarn_diag(DiagnosticID id, Token t, const char *msg, const char *hint);
int set_diag_by_name(const char *name, int enabled);

#endif
