
#include "ast.h"
#include "../parser/parser.h"
#include "zprep.h"
#include <stdlib.h>
#include <string.h>

typedef struct TraitReg
{
    char *name;
    struct TraitReg *next;
} TraitReg;

static TraitReg *registered_traits = NULL;

void register_trait(const char *name)
{
    TraitReg *r = xmalloc(sizeof(TraitReg));
    r->name = xstrdup(name);
    r->next = registered_traits;
    registered_traits = r;
}

int is_trait(const char *name)
{
    TraitReg *r = registered_traits;
    while (r)
    {
        if (0 == strcmp(r->name, name))
        {
            return 1;
        }
        r = r->next;
    }
    return 0;
}

ASTNode *ast_create(NodeType type)
{
    ASTNode *node = xmalloc(sizeof(ASTNode));
    memset(node, 0, sizeof(ASTNode));
    node->type = type;
    return node;
}

void ast_free(ASTNode *node)
{
    (void)node;
    return;
}

Type *type_new(TypeKind kind)
{
    Type *t = xmalloc(sizeof(Type));
    memset(t, 0, sizeof(Type));
    t->kind = kind;
    t->name = NULL;
    t->inner = NULL;
    t->args = NULL;
    t->arg_count = 0;
    t->is_const = 0;
    t->array_size = 0;
    t->is_varargs = 0;
    t->is_restrict = 0;
    return t;
}

Type *type_new_ptr(Type *inner)
{
    Type *t = type_new(TYPE_POINTER);
    t->inner = inner;
    return t;
}

int is_char_ptr(Type *t)
{
    // Handle both primitive char* and legacy struct char*.
    if (TYPE_POINTER == t->kind && TYPE_CHAR == t->inner->kind)
    {
        return 1;
    }
    if (TYPE_POINTER == t->kind && TYPE_STRUCT == t->inner->kind &&
        0 == strcmp(t->inner->name, "char"))
    {
        return 1;
    }
    return 0;
}

int is_integer_type(Type *t)
{
    if (!t)
    {
        return 0;
    }

    return (t->kind == TYPE_INT || t->kind == TYPE_CHAR || t->kind == TYPE_BOOL ||
            t->kind == TYPE_I8 || t->kind == TYPE_U8 || t->kind == TYPE_I16 ||
            t->kind == TYPE_U16 || t->kind == TYPE_I32 || t->kind == TYPE_U32 ||
            t->kind == TYPE_I64 || t->kind == TYPE_U64 || t->kind == TYPE_USIZE ||
            t->kind == TYPE_ISIZE || t->kind == TYPE_BYTE || t->kind == TYPE_RUNE ||
            t->kind == TYPE_UINT || t->kind == TYPE_I128 || t->kind == TYPE_U128 ||
            (t->kind == TYPE_STRUCT && t->name &&
             (0 == strcmp(t->name, "int8_t") || 0 == strcmp(t->name, "uint8_t") ||
              0 == strcmp(t->name, "int16_t") || 0 == strcmp(t->name, "uint16_t") ||
              0 == strcmp(t->name, "int32_t") || 0 == strcmp(t->name, "uint32_t") ||
              0 == strcmp(t->name, "int64_t") || 0 == strcmp(t->name, "uint64_t") ||
              0 == strcmp(t->name, "size_t") || 0 == strcmp(t->name, "ssize_t") ||
              0 == strcmp(t->name, "ptrdiff_t"))));
}

int is_float_type(Type *t)
{
    if (!t)
    {
        return 0;
    }

    return (t->kind == TYPE_FLOAT || t->kind == TYPE_F32 || t->kind == TYPE_F64);
}

int type_eq(Type *a, Type *b)
{
    if (!a || !b)
    {
        return 0;
    }

    if (a == b)
    {
        return 1;
    }

    // Lax integer matching (bool == int, char == i8, etc.).
    if (is_integer_type(a) && is_integer_type(b))
    {
        return 1;
    }

    // Lax float matching.
    if (is_float_type(a) && is_float_type(b))
    {
        return 1;
    }

    // String Literal vs char*
    if (a->kind == TYPE_STRING && is_char_ptr(b))
    {
        return 1;
    }

    if (b->kind == TYPE_STRING && is_char_ptr(a))
    {
        return 1;
    }

    if (a->kind != b->kind)
    {
        return 0;
    }

    if (a->kind == TYPE_STRUCT || a->kind == TYPE_GENERIC)
    {
        return 0 == strcmp(a->name, b->name);
    }
    if (a->kind == TYPE_POINTER || a->kind == TYPE_ARRAY)
    {
        return type_eq(a->inner, b->inner);
    }

    return 1;
}

char *type_to_string(Type *t)
{
    if (!t)
    {
        return xstrdup("void");
    }

    switch (t->kind)
    {
    case TYPE_VOID:
        return xstrdup("void");
    case TYPE_BOOL:
        return xstrdup("bool");
    case TYPE_STRING:
        return xstrdup("string");
    case TYPE_CHAR:
        return xstrdup("char");
    case TYPE_I8:
        return xstrdup("int8_t");
    case TYPE_U8:
        return xstrdup("uint8_t");
    case TYPE_I16:
        return xstrdup("int16_t");
    case TYPE_U16:
        return xstrdup("uint16_t");
    case TYPE_I32:
        return xstrdup("int32_t");
    case TYPE_U32:
        return xstrdup("uint32_t");
    case TYPE_I64:
        return xstrdup("int64_t");
    case TYPE_U64:
        return xstrdup("uint64_t");
    case TYPE_F32:
        return xstrdup("float");
    case TYPE_F64:
        return xstrdup("double");
    case TYPE_USIZE:
        return xstrdup("size_t");
    case TYPE_ISIZE:
        return xstrdup("ptrdiff_t");
    case TYPE_BYTE:
        return xstrdup("uint8_t");
    case TYPE_I128:
        return xstrdup("__int128");
    case TYPE_U128:
        return xstrdup("unsigned __int128");
    case TYPE_RUNE:
        return xstrdup("int32_t");
    case TYPE_UINT:
        return xstrdup("unsigned int");
    case TYPE_INT:
        return xstrdup("int");
    case TYPE_FLOAT:
        return xstrdup("float");

    case TYPE_POINTER:
    {
        char *inner = type_to_string(t->inner);
        if (t->is_restrict)
        {
            char *res = xmalloc(strlen(inner) + 16);
            sprintf(res, "%s* __restrict", inner);
            return res;
        }
        else
        {
            char *res = xmalloc(strlen(inner) + 2);
            sprintf(res, "%s*", inner);
            return res;
        }
    }

    case TYPE_ARRAY:
    {
        char *inner = type_to_string(t->inner);

        if (t->array_size > 0)
        {
            char *res = xmalloc(strlen(inner) + 20);
            sprintf(res, "%s[%d]", inner, t->array_size);
            return res;
        }

        char *res = xmalloc(strlen(inner) + 7);
        sprintf(res, "Slice_%s", inner);
        return res;
    }

    case TYPE_FUNCTION:
        if (t->inner)
        {
            free(type_to_string(t->inner));
        }

        return xstrdup("z_closure_T");

    case TYPE_STRUCT:
    case TYPE_GENERIC:
    {
        if (t->arg_count > 0)
        {
            char *base = t->name;
            char *arg = type_to_string(t->args[0]);
            char *clean_arg = sanitize_mangled_name(arg);

            char *res = xmalloc(strlen(base) + strlen(clean_arg) + 2);
            sprintf(res, "%s_%s", base, clean_arg);

            free(arg);
            free(clean_arg);
            return res;
        }
        return xstrdup(t->name);
    }

    default:
        return xstrdup("unknown");
    }
}

// C-compatible type stringifier.
// Strictly uses 'struct T' for explicit structs to support external types.
// Does NOT mangle pointers to 'Ptr'.
char *type_to_c_string(Type *t)
{
    if (!t)
    {
        return xstrdup("void");
    }

    switch (t->kind)
    {
    case TYPE_VOID:
        return xstrdup("void");
    case TYPE_STRUCT:
    {
        // Only prepend 'struct' if explicitly requested (e.g. "struct Foo")
        // otherwise assume it's a typedef/alias (e.g. "Foo").
        if (t->is_explicit_struct)
        {
            char *res = xmalloc(strlen(t->name) + 8);
            sprintf(res, "struct %s", t->name);
            return res;
        }
        else
        {
            return xstrdup(t->name);
        }
    }
    case TYPE_BOOL:
        return xstrdup("bool");
    case TYPE_STRING:
        return xstrdup("string");
    case TYPE_CHAR:
        return xstrdup("char");
    case TYPE_I8:
        return xstrdup("int8_t");
    case TYPE_U8:
        return xstrdup("uint8_t");
    case TYPE_I16:
        return xstrdup("int16_t");
    case TYPE_U16:
        return xstrdup("uint16_t");
    case TYPE_I32:
        return xstrdup("int32_t");
    case TYPE_U32:
        return xstrdup("uint32_t");
    case TYPE_I64:
        return xstrdup("int64_t");
    case TYPE_U64:
        return xstrdup("uint64_t");
    case TYPE_F32:
        return xstrdup("float");
    case TYPE_F64:
        return xstrdup("double");
    case TYPE_USIZE:
        return xstrdup("size_t");
    case TYPE_ISIZE:
        return xstrdup("ptrdiff_t");
    case TYPE_BYTE:
        return xstrdup("uint8_t");
    case TYPE_I128:
        return xstrdup("__int128");
    case TYPE_U128:
        return xstrdup("unsigned __int128");
    case TYPE_RUNE:
        return xstrdup("int32_t");
    case TYPE_UINT:
        return xstrdup("unsigned int");
    case TYPE_INT:
        return xstrdup("int");
    case TYPE_FLOAT:
        return xstrdup("float");

    case TYPE_POINTER:
    {
        char *inner = type_to_c_string(t->inner);
        if (t->is_restrict)
        {
            char *res = xmalloc(strlen(inner) + 16);
            sprintf(res, "%s* __restrict", inner);
            return res;
        }
        else
        {
            char *res = xmalloc(strlen(inner) + 2);
            sprintf(res, "%s*", inner);
            return res;
        }
    }

    case TYPE_ARRAY:
    {
        char *inner = type_to_c_string(t->inner);

        if (t->array_size > 0)
        {
            char *res = xmalloc(strlen(inner) + 20);
            sprintf(res, "%s[%d]", inner, t->array_size);
            return res;
        }

        char *res = xmalloc(strlen(inner) + 7);
        sprintf(res, "Slice_%s", inner);
        return res;
    }

    case TYPE_FUNCTION:
        if (t->inner)
        {
            free(type_to_c_string(t->inner));
        }
        return xstrdup("z_closure_T");

    case TYPE_GENERIC:
        return xstrdup(t->name);

    case TYPE_ENUM:
        return xstrdup(t->name);

    default:
        return xstrdup("unknown");
    }
}
