
#ifndef ZEN_CONSTANTS_H
#define ZEN_CONSTANTS_H

// Buffer sizes
// Buffer sizes
#define MAX_TYPE_NAME_LEN 256     ///< Max length for type name strings.
#define MAX_FUNC_NAME_LEN 512     ///< Max length for function names.
#define MAX_ERROR_MSG_LEN 1024    ///< Max length for error messages.
#define MAX_MANGLED_NAME_LEN 512  ///< Max length for mangled names (generics).
#define MAX_PATH_LEN 4096         ///< Max length for file paths.

// Type checking helpers
#define IS_INT_TYPE(t) ((t) && strcmp((t), "int") == 0)      ///< Checks if type is "int".
#define IS_BOOL_TYPE(t) ((t) && strcmp((t), "bool") == 0)    ///< Checks if type is "bool".
#define IS_CHAR_TYPE(t) ((t) && strcmp((t), "char") == 0)    ///< Checks if type is "char".
#define IS_VOID_TYPE(t) ((t) && strcmp((t), "void") == 0)    ///< Checks if type is "void".
#define IS_FLOAT_TYPE(t) ((t) && strcmp((t), "float") == 0)  ///< Checks if type is "float".
#define IS_DOUBLE_TYPE(t) ((t) && strcmp((t), "double") == 0)///< Checks if type is "double".
#define IS_USIZE_TYPE(t) ((t) && (strcmp((t), "usize") == 0 || strcmp((t), "size_t") == 0)) ///< Checks if type is "usize" or "size_t".
/**
 * @brief Checks if type is a string type ("string", "char*", "const char*").
 */
#define IS_STRING_TYPE(t)                                                                          \
    ((t) &&                                                                                        \
     (strcmp((t), "string") == 0 || strcmp((t), "char*") == 0 || strcmp((t), "const char*") == 0))

// Composite type checks
/**
 * @brief Checks if type is a basic primitive type.
 */
#define IS_BASIC_TYPE(t)                                                                           \
    ((t) && (IS_INT_TYPE(t) || IS_BOOL_TYPE(t) || IS_CHAR_TYPE(t) || IS_VOID_TYPE(t) ||            \
             IS_FLOAT_TYPE(t) || IS_DOUBLE_TYPE(t) || IS_USIZE_TYPE(t) ||                          \
             strcmp((t), "ssize_t") == 0 || strcmp((t), "__auto_type") == 0))

/**
 * @brief Checks if type is numeric (int, float, double, usize).
 */
#define IS_NUMERIC_TYPE(t)                                                                         \
    ((t) && (IS_INT_TYPE(t) || IS_FLOAT_TYPE(t) || IS_DOUBLE_TYPE(t) || IS_USIZE_TYPE(t)))

// Pointer type check
#define IS_PTR_TYPE(t) ((t) && strchr((t), '*') != NULL) ///< Checks if type string contains '*'.

// Struct prefix check
#define IS_STRUCT_PREFIX(t) ((t) && strncmp((t), "struct ", 7) == 0) ///< Checks if type starts with "struct ".
#define STRIP_STRUCT_PREFIX(t) (IS_STRUCT_PREFIX(t) ? ((t) + 7) : (t)) ///< Returns ptr to name after "struct " prefix.

// Generic type checks
#define IS_OPTION_TYPE(t) ((t) && strncmp((t), "Option_", 7) == 0) ///< Checks if type is Option<T>.
#define IS_RESULT_TYPE(t) ((t) && strncmp((t), "Result_", 7) == 0) ///< Checks if type is Result<T>.
#define IS_VEC_TYPE(t) ((t) && strncmp((t), "Vec_", 4) == 0)       ///< Checks if type is Vec<T>.
#define IS_SLICE_TYPE(t) ((t) && strncmp((t), "Slice_", 6) == 0)   ///< Checks if type is Slice<T>.

#endif // ZEN_CONSTANTS_H
