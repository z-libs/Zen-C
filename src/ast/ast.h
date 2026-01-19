
#ifndef AST_H
#define AST_H

#include "zprep.h"
#include <stdlib.h>

// Forward declarations.
struct ASTNode;
typedef struct ASTNode ASTNode;

// ** Formal Type System **
// Used for Generics, Type Inference, and robust pointer handling.
typedef enum
{
    TYPE_VOID,
    TYPE_BOOL,
    TYPE_CHAR,
    TYPE_STRING,
    TYPE_U0,
    TYPE_I8,
    TYPE_U8,
    TYPE_I16,
    TYPE_U16,
    TYPE_I32,
    TYPE_U32,
    TYPE_I64,
    TYPE_U64,
    TYPE_I128,
    TYPE_U128,
    TYPE_F32,
    TYPE_F64,
    TYPE_INT,
    TYPE_FLOAT,
    TYPE_USIZE,
    TYPE_ISIZE,
    TYPE_BYTE,
    TYPE_RUNE,
    TYPE_UINT,
    TYPE_STRUCT,
    TYPE_ENUM,
    TYPE_POINTER,
    TYPE_ARRAY,
    TYPE_FUNCTION,
    TYPE_GENERIC,
    TYPE_UNKNOWN
} TypeKind;

typedef struct Type
{
    TypeKind kind;
    char *name;         // For STRUCT, GENERIC, ENUM.
    struct Type *inner; // For POINTER, ARRAY.
    struct Type **args; // For GENERIC args.
    int arg_count;
    int is_const;
    int is_explicit_struct; // for example, "struct Foo" vs "Foo"
    union
    {
        int array_size;  // For fixed-size arrays [T; N].
        int is_varargs;  // For function types (...).
        int is_restrict; // For restrict pointers.
        struct
        {
            int has_drop;     // For RAII: does this type implement Drop?
            int has_iterable; // For the for iterator: does the type implement Iterable?
        } traits;
    };
} Type;

// ** AST Node Types **
typedef enum
{
    NODE_ROOT,
    NODE_FUNCTION,
    NODE_BLOCK,
    NODE_RETURN,
    NODE_VAR_DECL,
    NODE_CONST,
    NODE_TYPE_ALIAS,
    NODE_IF,
    NODE_WHILE,
    NODE_FOR,
    NODE_FOR_RANGE,
    NODE_LOOP,
    NODE_REPEAT,
    NODE_UNLESS,
    NODE_GUARD,
    NODE_BREAK,
    NODE_CONTINUE,
    NODE_MATCH,
    NODE_MATCH_CASE,
    NODE_EXPR_BINARY,
    NODE_EXPR_UNARY,
    NODE_EXPR_LITERAL,
    NODE_EXPR_VAR,
    NODE_EXPR_CALL,
    NODE_EXPR_MEMBER,
    NODE_EXPR_INDEX,
    NODE_EXPR_CAST,
    NODE_EXPR_SIZEOF,
    NODE_EXPR_STRUCT_INIT,
    NODE_EXPR_ARRAY_LITERAL,
    NODE_EXPR_SLICE,
    NODE_STRUCT,
    NODE_FIELD,
    NODE_ENUM,
    NODE_ENUM_VARIANT,
    NODE_TRAIT,
    NODE_IMPL,
    NODE_IMPL_TRAIT,
    NODE_INCLUDE,
    NODE_RAW_STMT,
    NODE_TEST,
    NODE_ASSERT,
    NODE_DEFER,
    NODE_DESTRUCT_VAR,
    NODE_TERNARY,
    NODE_ASM,
    NODE_LAMBDA,
    NODE_PLUGIN,
    NODE_GOTO,
    NODE_LABEL,
    NODE_DO_WHILE,
    NODE_TYPEOF,
    NODE_TRY,
    NODE_REFLECTION,
    NODE_AWAIT,
    NODE_REPL_PRINT,
    NODE_CUDA_LAUNCH
} NodeType;

// ** AST Node Structure **
struct ASTNode
{
    NodeType type;
    ASTNode *next;
    int line; // Source line number for debugging.

    // Type information.
    char *resolved_type; // Legacy string representation (for example: "int",
                         // "User*"). > Yes, 'legacy' is a thing, this is the
                         // third iteration > of this project (for now).
    Type *type_info;     // Formal type object (for inference/generics).
    Token token;
    Token definition_token; // For LSP: Location where the symbol used in this
                            // node was defined.

    union
    {
        struct
        {
            ASTNode *children;
        } root;

        struct
        {
            char *name;
            char *args;     // Legacy string args.
            char *ret_type; // Legacy string return type.
            ASTNode *body;
            Type **arg_types;
            char **defaults;
            char **param_names; // Explicit parameter names.
            int arg_count;
            Type *ret_type_info;
            int is_varargs;
            int is_inline;
            int must_use; // @must_use: warn if return value is discarded.
            // GCC attributes
            int noinline;    // @noinline
            int constructor; // @constructor
            int destructor;  // @destructor
            int unused;      // @unused
            int weak;        // @weak
            int is_export;   // @export (visibility default).
            int cold;        // @cold
            int hot;         // @hot
            int noreturn;    // @noreturn
            int pure;        // @pure
            char *section;   // @section("name")
            int is_async;    // async function
            int is_comptime; // @comptime function
            // CUDA qualifiers
            int cuda_global; // @global -> __global__
            int cuda_device; // @device -> __device__
            int cuda_host;   // @host -> __host__
        } func;

        struct
        {
            ASTNode *statements;
        } block;

        struct
        {
            ASTNode *value;
        } ret;

        struct
        {
            char *name;
            char *type_str;
            ASTNode *init_expr;
            Type *type_info;
            int is_autofree;
            int is_static;
        } var_decl;

        struct
        {
            char *name;
            Type *payload;
            int tag_id;
        } variant;

        struct
        {
            char *name;
            ASTNode *variants;
            int is_template;
            char *generic_param;
        } enm;

        struct
        {
            char *alias;
            char *original_type;
        } type_alias;

        struct
        {
            ASTNode *condition;
            ASTNode *then_body;
            ASTNode *else_body;
        } if_stmt;

        struct
        {
            ASTNode *condition;
            ASTNode *body;
            char *loop_label;
        } while_stmt;

        struct
        {
            ASTNode *init;
            ASTNode *condition;
            ASTNode *step;
            ASTNode *body;
            char *loop_label;
        } for_stmt;

        struct
        {
            char *var_name;
            ASTNode *start;
            ASTNode *end;
            char *step;
            int is_inclusive;
            ASTNode *body;
        } for_range;

        struct
        {
            ASTNode *body;
            char *loop_label;
        } loop_stmt;

        struct
        {
            char *count;
            ASTNode *body;
        } repeat_stmt;

        struct
        {
            ASTNode *condition;
            ASTNode *body;
        } unless_stmt;

        struct
        {
            ASTNode *condition;
            ASTNode *body;
        } guard_stmt;

        struct
        {
            ASTNode *condition;
            ASTNode *body;
            char *loop_label;
        } do_while_stmt;

        struct
        {
            ASTNode *expr;
            ASTNode *cases;
        } match_stmt;

        struct
        {
            char *pattern;
            char *binding_name;
            int is_destructuring;
            ASTNode *guard;
            ASTNode *body;
            int is_default;
        } match_case;

        struct
        {
            char *op;
            ASTNode *left;
            ASTNode *right;
        } binary;

        struct
        {
            char *op;
            ASTNode *operand;
        } unary;

        struct
        {
            int type_kind;
            unsigned long long int_val;
            double float_val;
            char *string_val;
        } literal;

        struct
        {
            char *name;
            char *suggestion;
        } var_ref;

        struct
        {
            ASTNode *callee;
            ASTNode *args;
            char **arg_names;
            int arg_count;
        } call;

        struct
        {
            ASTNode *target;
            char *field;
            int is_pointer_access;
        } member;

        struct
        {
            ASTNode *array;
            ASTNode *index;
        } index;

        struct
        {
            ASTNode *array;
            ASTNode *start;
            ASTNode *end;
        } slice;

        struct
        {
            char *target_type;
            ASTNode *expr;
        } cast;

        struct
        {
            char *struct_name;
            ASTNode *fields;
        } struct_init;

        struct
        {
            ASTNode *elements;
            int count;
        } array_literal;

        struct
        {
            char *name;
            ASTNode *fields;
            int is_template;
            char **generic_params;   // Array of generic parameter names (for example, ["K", "V"])
            int generic_param_count; // Number of generic parameters
            char *parent;
            int is_union;
            int is_packed;       // @packed attribute.
            int align;           // @align(N) attribute, 0 = default.
            int is_incomplete;   // Forward declaration (prototype)
            char **used_structs; // Names of structs used/mixed-in
            int used_struct_count;
        } strct;

        struct
        {
            char *name;
            char *type;
            int bit_width;
        } field;

        struct
        {
            char *name;
            ASTNode *methods;
            char **generic_params;
            int generic_param_count;
        } trait;

        struct
        {
            char *struct_name;
            ASTNode *methods;
        } impl;

        struct
        {
            char *trait_name;
            char *target_type;
            ASTNode *methods;
        } impl_trait;

        struct
        {
            char *path;
            int is_system;
        } include;

        struct
        {
            char *content;
            char **used_symbols;
            int used_symbol_count;
        } raw_stmt;

        struct
        {
            char *name;
            ASTNode *body;
        } test_stmt;

        struct
        {
            ASTNode *condition;
            char *message;
        } assert_stmt;

        struct
        {
            ASTNode *stmt;
        } defer_stmt;

        struct
        {
            char *plugin_name;
            char *body;
        } plugin_stmt;

        struct
        {
            char **names;
            int count;
            ASTNode *init_expr;
            int is_struct_destruct;
            char *struct_name;  // "Point" (or NULL if unchecked/inferred).
            char **field_names; // NULL if same as 'names', otherwise mapped.
            int is_guard;
            char *guard_variant; // "Some", "Ok".
            ASTNode *else_block;
        } destruct;

        struct
        {
            ASTNode *cond;
            ASTNode *true_expr;
            ASTNode *false_expr;
        } ternary;

        struct
        {
            char *code;
            int is_volatile;
            char **outputs;
            char **output_modes;
            char **inputs;
            char **clobbers;
            int num_outputs;
            int num_inputs;
            int num_clobbers;
        } asm_stmt;

        struct
        {
            char **param_names;
            char **param_types;
            char *return_type;
            ASTNode *body;
            int num_params;
            int lambda_id;
            int is_expression;
            char **captured_vars;
            char **captured_types;
            int num_captures;
        } lambda;

        struct
        {
            char *target_type;
            ASTNode *expr;
        } size_of;

        struct
        {
            char *label_name;
            ASTNode *goto_expr;
        } goto_stmt;

        struct
        {
            char *label_name;
        } label_stmt;

        struct
        {
            char *target_label;
        } break_stmt;

        struct
        {
            char *target_label;
        } continue_stmt;

        struct
        {
            ASTNode *expr;
        } try_stmt;

        struct
        {
            int kind; // 0=type_name, 1=fields.
            Type *target_type;
        } reflection;

        struct
        {
            ASTNode *expr;
        } repl_print;

        struct
        {
            ASTNode *call;       // The kernel call (NODE_EXPR_CALL)
            ASTNode *grid;       // Grid dimensions expression
            ASTNode *block;      // Block dimensions expression
            ASTNode *shared_mem; // Optional shared memory size (NULL = default)
            ASTNode *stream;     // Optional CUDA stream (NULL = default)
        } cuda_launch;
    };
};

// ** Functions **
ASTNode *ast_create(NodeType type);
void ast_free(ASTNode *node);

Type *type_new(TypeKind kind);
Type *type_new_ptr(Type *inner);
int type_eq(Type *a, Type *b);
int is_integer_type(Type *t);
char *type_to_string(Type *t);
char *type_to_c_string(Type *t);

#endif
