
#ifndef PARSER_H
#define PARSER_H

#include "ast.h"
#include "zprep.h"

// Operator precedence for expression parsing
typedef enum
{
    PREC_NONE,
    PREC_ASSIGNMENT,
    PREC_TERNARY,
    PREC_OR,
    PREC_AND,
    PREC_EQUALITY,
    PREC_COMPARISON,
    PREC_TERM,
    PREC_FACTOR,
    PREC_UNARY,
    PREC_CALL,
    PREC_PRIMARY
} Precedence;

// Main entry points
// Forward declarations
struct ParserContext;
typedef struct ParserContext ParserContext;

ASTNode *parse_program(ParserContext *ctx, Lexer *l);

extern ParserContext *g_parser_ctx;

// Symbol table
typedef struct Symbol
{
    char *name;
    char *type_name;
    Type *type_info;
    int is_mutable;
    int is_used;
    int is_autofree;
    Token decl_token;
    int is_const_value;
    int const_int_val;
    struct Symbol *next;
} Symbol;

typedef struct Scope
{
    Symbol *symbols;
    struct Scope *parent;
} Scope;

// Function registry
typedef struct FuncSig
{
    char *name;
    Token decl_token; // For LSP
    int total_args;
    char **defaults;
    Type **arg_types;
    Type *ret_type;
    int is_varargs;
    int is_async; // Async function flag
    int must_use; // Attribute: warn if return value discarded
    struct FuncSig *next;
} FuncSig;

// Lambda tracking
typedef struct LambdaRef
{
    ASTNode *node;
    struct LambdaRef *next;
} LambdaRef;

typedef struct GenericTemplate
{
    char *name;
    ASTNode *struct_node;
    struct GenericTemplate *next;
} GenericTemplate;

typedef struct GenericFuncTemplate
{
    char *name;
    char *generic_param;
    ASTNode *func_node;
    struct GenericFuncTemplate *next;
} GenericFuncTemplate;

typedef struct GenericImplTemplate
{
    char *struct_name;
    char *generic_param;
    ASTNode *impl_node;
    struct GenericImplTemplate *next;
} GenericImplTemplate;

typedef struct ImportedFile
{
    char *path;
    struct ImportedFile *next;
} ImportedFile;

typedef struct VarMutability
{
    char *name;
    int is_mutable;
    struct VarMutability *next;
} VarMutability;

// Instantiation tracking
typedef struct Instantiation
{
    char *name;
    char *template_name;
    char *concrete_arg;
    ASTNode *struct_node;
    struct Instantiation *next;
} Instantiation;

typedef struct StructRef
{
    ASTNode *node;
    struct StructRef *next;
} StructRef;

typedef struct StructDef
{
    char *name;
    ASTNode *node;
    struct StructDef *next;
} StructDef;

// Type tracking
typedef struct SliceType
{
    char *name;
    struct SliceType *next;
} SliceType;

typedef struct TupleType
{
    char *sig;
    struct TupleType *next;
} TupleType;

// Enum tracking
typedef struct EnumVariantReg
{
    char *enum_name;
    char *variant_name;
    int tag_id;
    struct EnumVariantReg *next;
} EnumVariantReg;

// Deprecated function tracking
typedef struct DeprecatedFunc
{
    char *name;
    char *reason; // Optional reason message
    struct DeprecatedFunc *next;
} DeprecatedFunc;

// Module system
typedef struct Module
{
    char *alias;
    char *path;
    char *base_name;
    int is_c_header;
    struct Module *next;
} Module;

typedef struct SelectiveImport
{
    char *symbol;
    char *alias;
    char *source_module;
    struct SelectiveImport *next;
} SelectiveImport;

// Impl cache
typedef struct ImplReg
{
    char *trait;
    char *strct;
    struct ImplReg *next;
} ImplReg;

// Plugin tracking
typedef struct ImportedPlugin
{
    char *name;  // Original plugin name (for example, "brainfuck")
    char *alias; // Optional alias (for example, "bf"), NULL if no alias
    struct ImportedPlugin *next;
} ImportedPlugin;

struct ParserContext
{
    Scope *current_scope;
    FuncSig *func_registry;

    // Lambdas
    LambdaRef *global_lambdas;
    int lambda_counter;

// Generics
#define MAX_KNOWN_GENERICS 1024
    char *known_generics[MAX_KNOWN_GENERICS];
    int known_generics_count;
    GenericTemplate *templates;
    GenericFuncTemplate *func_templates;
    GenericImplTemplate *impl_templates;

    // Instantiations
    Instantiation *instantiations;
    ASTNode *instantiated_structs;
    ASTNode *instantiated_funcs;

    // Structs/Enums
    StructRef *parsed_structs_list;
    StructRef *parsed_enums_list;
    StructRef *parsed_funcs_list;
    StructRef *parsed_impls_list;
    StructRef *parsed_globals_list;
    StructDef *struct_defs;
    EnumVariantReg *enum_variants;
    ImplReg *registered_impls;

    // Types
    SliceType *used_slices;
    TupleType *used_tuples;

    // Modules/Imports
    Module *modules;
    SelectiveImport *selective_imports;
    char *current_module_prefix;
    ImportedFile *imported_files;
    ImportedPlugin *imported_plugins; // Plugin imports

    // Config/State
    int immutable_by_default;
    char *current_impl_struct;

    // Internal tracking
    VarMutability *var_mutability_table;
    DeprecatedFunc *deprecated_funcs;

    // LSP / Fault Tolerance
    int is_fault_tolerant;
    void *error_callback_data;
    void (*on_error)(void *data, Token t, const char *msg);

    // LSP: Flat symbol list (persists after parsing for LSP queries)
    Symbol *all_symbols;

    // External C interop: suppress undefined warnings for external symbols
    int has_external_includes; // Set when include <...> is used
    char **extern_symbols;     // Explicitly declared extern symbols
    int extern_symbol_count;

    // Codegen state:
    FILE *hoist_out;   // For plugins to hoist code to file scope
    int skip_preamble; // If 1, codegen_node(NODE_ROOT) won't emit preamble
    int is_repl;       // REPL mode flag
};

// Token helpers
char *token_strdup(Token t);
int is_token(Token t, const char *s);
Token expect(Lexer *l, ZTokenType type, const char *msg);
void skip_comments(Lexer *l);
char *consume_until_semicolon(Lexer *l);
char *consume_and_rewrite(ParserContext *ctx, Lexer *l);

// C reserved word warnings
int is_c_reserved_word(const char *name);
void warn_c_reserved_word(Token t, const char *name);

// Symbol table
void enter_scope(ParserContext *ctx);
void exit_scope(ParserContext *ctx);
void add_symbol(ParserContext *ctx, const char *n, const char *t, Type *type_info);
void add_symbol_with_token(ParserContext *ctx, const char *n, const char *t, Type *type_info,
                           Token tok);
Type *find_symbol_type_info(ParserContext *ctx, const char *n);
char *find_symbol_type(ParserContext *ctx, const char *n);
Symbol *find_symbol_entry(ParserContext *ctx, const char *n);
Symbol *find_symbol_in_all(ParserContext *ctx,
                           const char *n); // LSP flat lookup
char *find_similar_symbol(ParserContext *ctx, const char *name);

// Function registry
void register_func(ParserContext *ctx, const char *name, int count, char **defaults,
                   Type **arg_types, Type *ret_type, int is_varargs, int is_async,
                   Token decl_token);
void register_func_template(ParserContext *ctx, const char *name, const char *param, ASTNode *node);
GenericFuncTemplate *find_func_template(ParserContext *ctx, const char *name);

// Generic/template helpers
void register_generic(ParserContext *ctx, char *name);
int is_known_generic(ParserContext *ctx, char *name);
void register_impl_template(ParserContext *ctx, const char *sname, const char *param,
                            ASTNode *node);
void add_to_struct_list(ParserContext *ctx, ASTNode *node);
void add_to_enum_list(ParserContext *ctx, ASTNode *node);
void add_to_func_list(ParserContext *ctx, ASTNode *node);
void add_to_impl_list(ParserContext *ctx, ASTNode *node);
void add_to_global_list(ParserContext *ctx, ASTNode *node);
void register_builtins(ParserContext *ctx);
void add_instantiated_func(ParserContext *ctx, ASTNode *fn);
void instantiate_generic(ParserContext *ctx, const char *name, const char *concrete_type);
char *sanitize_mangled_name(const char *s);
void register_impl(ParserContext *ctx, const char *trait, const char *strct);
int check_impl(ParserContext *ctx, const char *trait, const char *strct);
void register_template(ParserContext *ctx, const char *name, ASTNode *node);
void register_deprecated_func(ParserContext *ctx, const char *name, const char *reason);
DeprecatedFunc *find_deprecated_func(ParserContext *ctx, const char *name);
ASTNode *parse_arrow_lambda_single(ParserContext *ctx, Lexer *l, char *param_name);
ASTNode *parse_arrow_lambda_multi(ParserContext *ctx, Lexer *l, char **param_names, int num_params);

// Utils
char *parse_and_convert_args(ParserContext *ctx, Lexer *l, char ***defaults_out, int *count_out,
                             Type ***types_out, char ***names_out, int *is_varargs_out);
int is_file_imported(ParserContext *ctx, const char *path);
void mark_file_imported(ParserContext *ctx, const char *path);
void register_plugin(ParserContext *ctx, const char *name, const char *alias);
const char *resolve_plugin(ParserContext *ctx, const char *name_or_alias);
void print_type_defs(ParserContext *ctx, FILE *out, ASTNode *nodes);

// String manipulation
char *replace_in_string(const char *src, const char *old_w, const char *new_w);
char *replace_type_str(const char *src, const char *param, const char *concrete,
                       const char *old_struct, const char *new_struct);
Type *replace_type_formal(Type *t, const char *p, const char *c, const char *os, const char *ns);
ASTNode *copy_ast_replacing(ASTNode *n, const char *p, const char *c, const char *os,
                            const char *ns);
char *extract_module_name(const char *path);

// Enum helpers
void register_enum_variant(ParserContext *ctx, const char *ename, const char *vname, int tag);
EnumVariantReg *find_enum_variant(ParserContext *ctx, const char *vname);

// Lambda helpers
void register_lambda(ParserContext *ctx, ASTNode *node);
void analyze_lambda_captures(ParserContext *ctx, ASTNode *lambda);

// Type registration
void register_slice(ParserContext *ctx, const char *type);
void register_tuple(ParserContext *ctx, const char *sig);

// Struct lookup
ASTNode *find_struct_def(ParserContext *ctx, const char *name);
void register_struct_def(ParserContext *ctx, const char *name, ASTNode *node);

// Module system
Module *find_module(ParserContext *ctx, const char *alias);
void register_module(ParserContext *ctx, const char *alias, const char *path);
void register_selective_import(ParserContext *ctx, const char *symbol, const char *alias,
                               const char *source_module);
SelectiveImport *find_selective_import(ParserContext *ctx, const char *name);

// Mutability tracking
void register_var_mutability(ParserContext *ctx, const char *name, int is_mutable);
int is_var_mutable(ParserContext *ctx, const char *name);

// External symbol tracking (C interop)
void register_extern_symbol(ParserContext *ctx, const char *name);
int is_extern_symbol(ParserContext *ctx, const char *name);
int should_suppress_undef_warning(ParserContext *ctx, const char *name);

// Initialization
void init_builtins();

// Expression rewriting
char *rewrite_expr_methods(ParserContext *ctx, char *raw);
char *process_fstring(ParserContext *ctx, const char *content);
char *instantiate_function_template(ParserContext *ctx, const char *name,
                                    const char *concrete_type);
FuncSig *find_func(ParserContext *ctx, const char *name);

Type *parse_type_formal(ParserContext *ctx, Lexer *l);
char *parse_type(ParserContext *ctx, Lexer *l);
Type *parse_type_base(ParserContext *ctx, Lexer *l);

ASTNode *parse_expression(ParserContext *ctx, Lexer *l);
ASTNode *parse_expr_prec(ParserContext *ctx, Lexer *l, Precedence min_prec);
ASTNode *parse_primary(ParserContext *ctx, Lexer *l);
ASTNode *parse_lambda(ParserContext *ctx, Lexer *l);
// parse_arrow_lambda_single/multi already declared above
char *parse_condition_raw(ParserContext *ctx, Lexer *l);
char *parse_array_literal(ParserContext *ctx, Lexer *l, const char *st);
char *parse_tuple_literal(ParserContext *ctx, Lexer *l, const char *tn);
char *parse_embed(ParserContext *ctx, Lexer *l);

ASTNode *parse_macro_call(ParserContext *ctx, Lexer *l, char *name);
ASTNode *parse_statement(ParserContext *ctx, Lexer *l);
ASTNode *parse_block(ParserContext *ctx, Lexer *l);
ASTNode *parse_if(ParserContext *ctx, Lexer *l);
ASTNode *parse_while(ParserContext *ctx, Lexer *l);
ASTNode *parse_for(ParserContext *ctx, Lexer *l);
ASTNode *parse_loop(ParserContext *ctx, Lexer *l);
ASTNode *parse_repeat(ParserContext *ctx, Lexer *l);
ASTNode *parse_unless(ParserContext *ctx, Lexer *l);
ASTNode *parse_guard(ParserContext *ctx, Lexer *l);
ASTNode *parse_match(ParserContext *ctx, Lexer *l);
ASTNode *parse_return(ParserContext *ctx, Lexer *l);

char *process_printf_sugar(ParserContext *ctx, const char *content, int newline,
                           const char *target);
ASTNode *parse_assert(ParserContext *ctx, Lexer *l);
ASTNode *parse_defer(ParserContext *ctx, Lexer *l);
ASTNode *parse_asm(ParserContext *ctx, Lexer *l);
ASTNode *parse_plugin(ParserContext *ctx, Lexer *l);
ASTNode *parse_var_decl(ParserContext *ctx, Lexer *l);
ASTNode *parse_const(ParserContext *ctx, Lexer *l);
ASTNode *parse_type_alias(ParserContext *ctx, Lexer *l);

ASTNode *parse_function(ParserContext *ctx, Lexer *l, int is_async);
ASTNode *parse_struct(ParserContext *ctx, Lexer *l, int is_union);
ASTNode *parse_enum(ParserContext *ctx, Lexer *l);
ASTNode *parse_trait(ParserContext *ctx, Lexer *l);
ASTNode *parse_impl(ParserContext *ctx, Lexer *l);
ASTNode *parse_impl_trait(ParserContext *ctx, Lexer *l);
ASTNode *parse_test(ParserContext *ctx, Lexer *l);
ASTNode *parse_include(ParserContext *ctx, Lexer *l);
ASTNode *parse_import(ParserContext *ctx, Lexer *l);
ASTNode *parse_comptime(ParserContext *ctx, Lexer *l);

char *patch_self_args(const char *args, const char *struct_name);

ASTNode *parse_program_nodes(ParserContext *ctx, Lexer *l);

#endif // PARSER_H
