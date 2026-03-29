#ifndef SYMBOLS_H
#define SYMBOLS_H

#include "ast.h"
#include "zprep.h"

struct FuncSig;
typedef struct FuncSig FuncSig;

/**
 * @brief Kind of symbol in the unified symbol table.
 */
typedef enum
{
    SYM_VARIABLE,  ///< Local or global variable.
    SYM_FUNCTION,  ///< Function or method.
    SYM_STRUCT,    ///< Struct definition.
    SYM_ENUM,      ///< Enum definition.
    SYM_ALIAS,     ///< Type alias (typedef).
    SYM_CONSTANT,  ///< Constant value.
    SYM_MODULE,    ///< Module/Namespace.
    SYM_TRAIT,     ///< Trait definition.
    SYM_PRIMITIVE  ///< Built-in primitive type.
} SymbolKind;

/**
 * @brief Represents a symbol in the unified symbol table.
 */
typedef struct ZenSymbol
{
    char *name;             ///< Symbol name.
    SymbolKind kind;        ///< Kind of symbol.
    char *type_name;        ///< String representation of the type (legacy).
    Type *type_info;        ///< Formal type definition.
    Token decl_token;       ///< Token where the symbol was declared.
    int is_used;            ///< 1 if reference count > 0.
    int is_autofree;        ///< 1 if RAII.
    int is_immutable;       ///< 1 if value cannot be changed.
    int is_static;          ///< 1 if static storage.
    int is_export;          ///< 1 if visible outside module.
    int is_const_value;     ///< 1 if compile-time constant.
    int is_def;             ///< 1 if definition vs declaration.
    int const_int_val;      ///< Integer value if constant.
    int is_moved;           ///< 1 if ownership transferred.
    char *cfg_condition;    ///< Optional @cfg condition.
    
    union
    {
        ASTNode *node;      ///< AST node for the definition (Struct, Enum, Func).
        FuncSig *sig;       ///< Function signature metadata.
        struct {
            char *original_type;
            Type *resolved_type;
        } alias;
        struct {
            int int_val;
            double float_val;
            char *str_val;
        } constant;
        struct {
            char *path;
            char *alias_name;
        } module;
    } data;

    struct ZenSymbol *next; ///< Next symbol in the same scope.
} ZenSymbol;

/**
 * @brief Represents a lexical scope (block).
 */
typedef struct Scope
{
    ZenSymbol *symbols;   ///< Linked list of symbols in this scope.
    struct Scope *parent; ///< Pointer to the parent scope (NULL for global).
    char *name;           ///< Optional name for the scope (e.g. "Module::Func").
    int is_loop;          ///< 1 if this is a loop scope (for break/continue).
} Scope;

// ** Symbol Table Utilities **

/**
 * @brief Creates a new scope as a child of the given parent.
 */
Scope *symbol_scope_create(Scope *parent, const char *name);

/**
 * @brief Frees a scope and all its symbols.
 */
void symbol_scope_free(Scope *s);

/**
 * @brief Adds a symbol to the given scope.
 */
ZenSymbol *symbol_add(Scope *s, const char *name, SymbolKind kind);

/**
 * @brief Look up a symbol by name in the scope hierarchy.
 */
ZenSymbol *symbol_lookup(Scope *s, const char *name);

/**
 * @brief Look up a symbol only in the immediate scope.
 */
ZenSymbol *symbol_lookup_local(Scope *s, const char *name);

/**
 * @brief Look up a symbol of a specific kind.
 */
ZenSymbol *symbol_lookup_kind(Scope *s, const char *name, SymbolKind kind);

#endif // SYMBOLS_H
