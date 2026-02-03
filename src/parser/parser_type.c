
#include "../ast/ast.h"
#include "parser.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Type *parse_type_base(ParserContext *ctx, Lexer *l)
{
    Token t = lexer_peek(l);

    if (t.type == TOK_IDENT)
    {
        int explicit_struct = 0;
        // Handle "struct Name" or "enum Name"
        if ((t.len == 6 && strncmp(t.start, "struct", 6) == 0) ||
            (t.len == 4 && strncmp(t.start, "enum", 4) == 0))
        {
            if (strncmp(t.start, "struct", 6) == 0)
            {
                explicit_struct = 1;
            }
            lexer_next(l); // consume keyword
            t = lexer_peek(l);
            if (t.type != TOK_IDENT)
            {
                zpanic_at(t, "Expected identifier after struct/enum");
            }
        }

        lexer_next(l);
        char *name = token_strdup(t);

        // Check for alias
        TypeAlias *alias_node = find_type_alias_node(ctx, name);
        if (alias_node)
        {
            free(name);
            Lexer tmp;
            lexer_init(&tmp, alias_node->original_type);

            if (alias_node->is_opaque)
            {
                Type *underlying = parse_type_formal(ctx, &tmp);
                Type *wrapper = type_new(TYPE_ALIAS);
                wrapper->name = xstrdup(alias_node->alias);
                wrapper->inner = underlying;
                wrapper->alias.is_opaque_alias = 1;
                wrapper->alias.alias_defined_in_file =
                    alias_node->defined_in_file ? xstrdup(alias_node->defined_in_file) : NULL;
                return wrapper;
            }

            return parse_type_formal(ctx, &tmp);
        }

        // Self type alias: Replace "Self" with current impl struct type
        if (strcmp(name, "Self") == 0 && ctx->current_impl_struct)
        {
            name = xstrdup(ctx->current_impl_struct);
        }

        // Handle Namespace :: (A::B -> A_B)
        while (lexer_peek(l).type == TOK_DCOLON)
        {
            lexer_next(l); // eat ::
            Token next = lexer_next(l);
            if (next.type != TOK_IDENT)
            {
                zpanic_at(t, "Expected identifier after ::");
            }

            char *suffix = token_strdup(next);
            char *resolved_suffix = suffix;

            // Map Zen Primitive suffixes to C types to match Generic Instantiation
            if (strcmp(suffix, "I32") == 0)
            {
                resolved_suffix = "int32_t";
            }
            else if (strcmp(suffix, "U32") == 0)
            {
                resolved_suffix = "uint32_t";
            }
            else if (strcmp(suffix, "I8") == 0)
            {
                resolved_suffix = "int8_t";
            }
            else if (strcmp(suffix, "U8") == 0)
            {
                resolved_suffix = "uint8_t";
            }
            else if (strcmp(suffix, "I16") == 0)
            {
                resolved_suffix = "int16_t";
            }
            else if (strcmp(suffix, "U16") == 0)
            {
                resolved_suffix = "uint16_t";
            }
            else if (strcmp(suffix, "I64") == 0)
            {
                resolved_suffix = "int64_t";
            }
            else if (strcmp(suffix, "U64") == 0)
            {
                resolved_suffix = "uint64_t";
            }
            // Lowercase aliases
            else if (strcmp(suffix, "i8") == 0)
            {
                resolved_suffix = "int8_t";
            }
            else if (strcmp(suffix, "u8") == 0)
            {
                resolved_suffix = "uint8_t";
            }
            else if (strcmp(suffix, "i16") == 0)
            {
                resolved_suffix = "int16_t";
            }
            else if (strcmp(suffix, "u16") == 0)
            {
                resolved_suffix = "uint16_t";
            }
            else if (strcmp(suffix, "i32") == 0)
            {
                resolved_suffix = "int32_t";
            }
            else if (strcmp(suffix, "u32") == 0)
            {
                resolved_suffix = "uint32_t";
            }
            else if (strcmp(suffix, "i64") == 0)
            {
                resolved_suffix = "int64_t";
            }
            else if (strcmp(suffix, "u64") == 0)
            {
                resolved_suffix = "uint64_t";
            }
            else if (strcmp(suffix, "usize") == 0)
            {
                resolved_suffix = "size_t";
            }
            else if (strcmp(suffix, "string") == 0)
            {
                resolved_suffix = "char*";
            }

            // Check if 'name' is a module alias (e.g., m::Vector)
            Module *mod = find_module(ctx, name);
            char *merged;
            if (mod)
            {
                // Module-qualified type
                if (mod->is_c_header)
                {
                    // C header: Use type name directly without prefix
                    // To prevent name mangling, we might consider changing
                    // this to also use the prefix.
                    merged = xstrdup(resolved_suffix);
                }
                else
                {
                    // Zen module: Use module base name as prefix
                    merged = xmalloc(strlen(mod->base_name) + strlen(resolved_suffix) + 2);
                    sprintf(merged, "%s_%s", mod->base_name, resolved_suffix);
                }
            }
            else
            {
                // Regular namespace or enum variant
                merged = xmalloc(strlen(name) + strlen(resolved_suffix) + 2);
                sprintf(merged, "%s_%s", name, resolved_suffix);
            }

            free(name);
            if (suffix != resolved_suffix)
            {
                free(suffix); // Only free if we didn't remap
            }
            else
            {
                free(suffix);
            }

            name = merged;
        }

        // Check for Primitives (Base types)
        if (strcmp(name, "U0") == 0)
        {
            free(name);
            return type_new(TYPE_VOID);
        }
        if (strcmp(name, "u0") == 0)
        {
            free(name);
            return type_new(TYPE_VOID);
        }
        if (strcmp(name, "I8") == 0)
        {
            free(name);
            return type_new(TYPE_I8);
        }
        if (strcmp(name, "U8") == 0)
        {
            free(name);
            return type_new(TYPE_U8);
        }
        if (strcmp(name, "I16") == 0)
        {
            free(name);
            return type_new(TYPE_I16);
        }
        if (strcmp(name, "U16") == 0)
        {
            free(name);
            return type_new(TYPE_U16);
        }
        if (strcmp(name, "I32") == 0)
        {
            free(name);
            return type_new(TYPE_I32);
        }
        if (strcmp(name, "U32") == 0)
        {
            free(name);
            return type_new(TYPE_U32);
        }
        if (strcmp(name, "I64") == 0)
        {
            free(name);
            return type_new(TYPE_I64);
        }
        if (strcmp(name, "U64") == 0)
        {
            free(name);
            return type_new(TYPE_U64);
        }
        if (strcmp(name, "F32") == 0)
        {
            free(name);
            return type_new(TYPE_F32);
        }
        if (strcmp(name, "f32") == 0)
        {
            free(name);
            return type_new(TYPE_F32);
        }
        if (strcmp(name, "F64") == 0)
        {
            free(name);
            return type_new(TYPE_F64);
        }
        if (strcmp(name, "f64") == 0)
        {
            free(name);
            return type_new(TYPE_F64);
        }
        if (strcmp(name, "usize") == 0)
        {
            free(name);
            return type_new(TYPE_USIZE);
        }
        if (strcmp(name, "isize") == 0)
        {
            free(name);
            return type_new(TYPE_ISIZE);
        }
        if (strcmp(name, "byte") == 0)
        {
            free(name);
            return type_new(TYPE_BYTE);
        }
        if (strcmp(name, "I128") == 0)
        {
            free(name);
            return type_new(TYPE_I128);
        }
        if (strcmp(name, "U128") == 0)
        {
            free(name);
            return type_new(TYPE_U128);
        }
        if (strcmp(name, "i8") == 0)
        {
            free(name);
            return type_new(TYPE_I8);
        }
        if (strcmp(name, "u8") == 0)
        {
            free(name);
            return type_new(TYPE_U8);
        }
        if (strcmp(name, "i16") == 0)
        {
            free(name);
            return type_new(TYPE_I16);
        }

        // C23 BitInt Support (i42, u256, etc.)
        if ((name[0] == 'i' || name[0] == 'u') && isdigit(name[1]))
        {
            // Verify it is a purely numeric suffix
            int valid = 1;
            for (size_t k = 1; k < strlen(name); k++)
            {
                if (!isdigit(name[k]))
                {
                    valid = 0;
                    break;
                }
            }
            if (valid)
            {
                int width = atoi(name + 1);
                if (width > 0)
                {
                    // Map standard widths to standard types for standard ABI/C compabitility
                    if (name[0] == 'i')
                    {
                        if (width == 8)
                        {
                            free(name);
                            return type_new(TYPE_I8);
                        }
                        if (width == 16)
                        {
                            free(name);
                            return type_new(TYPE_I16);
                        }
                        if (width == 32)
                        {
                            free(name);
                            return type_new(TYPE_I32);
                        }
                        if (width == 64)
                        {
                            free(name);
                            return type_new(TYPE_I64);
                        }
                        if (width == 128)
                        {
                            free(name);
                            return type_new(TYPE_I128);
                        }
                    }
                    else
                    {
                        if (width == 8)
                        {
                            free(name);
                            return type_new(TYPE_U8);
                        }
                        if (width == 16)
                        {
                            free(name);
                            return type_new(TYPE_U16);
                        }
                        if (width == 32)
                        {
                            free(name);
                            return type_new(TYPE_U32);
                        }
                        if (width == 64)
                        {
                            free(name);
                            return type_new(TYPE_U64);
                        }
                        if (width == 128)
                        {
                            free(name);
                            return type_new(TYPE_U128);
                        }
                    }

                    Type *t = type_new(name[0] == 'u' ? TYPE_UBITINT : TYPE_BITINT);
                    t->array_size = width;
                    free(name);
                    return t;
                }
            }
        }
        if (strcmp(name, "u16") == 0)
        {
            free(name);
            return type_new(TYPE_U16);
        }
        if (strcmp(name, "i32") == 0)
        {
            free(name);
            return type_new(TYPE_I32);
        }
        if (strcmp(name, "u32") == 0)
        {
            free(name);
            return type_new(TYPE_U32);
        }
        if (strcmp(name, "i64") == 0)
        {
            free(name);
            return type_new(TYPE_I64);
        }
        if (strcmp(name, "u64") == 0)
        {
            free(name);
            return type_new(TYPE_U64);
        }
        if (strcmp(name, "i128") == 0)
        {
            free(name);
            return type_new(TYPE_I128);
        }
        if (strcmp(name, "u128") == 0)
        {
            free(name);
            return type_new(TYPE_U128);
        }
        if (strcmp(name, "rune") == 0)
        {
            free(name);
            return type_new(TYPE_RUNE);
        }
        if (strcmp(name, "uint") == 0)
        {
            free(name);
            return type_new(TYPE_U32); // Strict uint32_t
        }

        if (strcmp(name, "int") == 0)
        {
            free(name);
            return type_new(TYPE_I32); // Strict int32_t
        }
        if (strcmp(name, "float") == 0)
        {
            free(name);
            return type_new(TYPE_F32);
        }
        if (strcmp(name, "double") == 0)
        {
            free(name);
            return type_new(TYPE_F64);
        }
        if (strcmp(name, "void") == 0)
        {
            free(name);
            return type_new(TYPE_VOID);
        }
        if (strcmp(name, "string") == 0)
        {
            free(name);
            return type_new(TYPE_STRING);
        }
        if (strcmp(name, "bool") == 0)
        {
            free(name);
            return type_new(TYPE_BOOL);
        }
        if (strcmp(name, "char") == 0)
        {
            free(name);
            return type_new(TYPE_CHAR);
        }
        if (strcmp(name, "long") == 0)
        {
            zwarn_at(t, "'long' is treated as portable 'int64_t' in Zen C. Use 'c_long' for "
                        "platform-dependent C long.");
            free(name);
            return type_new(TYPE_I64);
        }
        if (strcmp(name, "short") == 0)
        {
            zwarn_at(t, "'short' is treated as portable 'int16_t' in Zen C. Use 'c_short' for "
                        "platform-dependent C short.");
            free(name);
            return type_new(TYPE_I16);
        }
        if (strcmp(name, "unsigned") == 0)
        {
            zwarn_at(t, "'unsigned' is treated as portable 'uint32_t' in Zen C. Use 'c_uint' for "
                        "platform-dependent C unsigned int.");
            free(name);
            return type_new(TYPE_U32);
        }
        if (strcmp(name, "signed") == 0)
        {
            zwarn_at(t, "'signed' is treated as portable 'int32_t' in Zen C. Use 'c_int' for "
                        "platform-dependent C int.");
            free(name);
            return type_new(TYPE_I32);
        }
        if (strcmp(name, "int8_t") == 0)
        {
            free(name);
            return type_new(TYPE_I8);
        }
        if (strcmp(name, "uint8_t") == 0)
        {
            free(name);
            return type_new(TYPE_U8);
        }
        if (strcmp(name, "int16_t") == 0)
        {
            free(name);
            return type_new(TYPE_I16);
        }
        if (strcmp(name, "uint16_t") == 0)
        {
            free(name);
            return type_new(TYPE_U16);
        }
        if (strcmp(name, "int32_t") == 0)
        {
            free(name);
            return type_new(TYPE_I32);
        }
        if (strcmp(name, "uint32_t") == 0)
        {
            free(name);
            return type_new(TYPE_U32);
        }
        if (strcmp(name, "int64_t") == 0)
        {
            free(name);
            return type_new(TYPE_I64);
        }
        if (strcmp(name, "uint64_t") == 0)
        {
            free(name);
            return type_new(TYPE_U64);
        }
        if (strcmp(name, "size_t") == 0)
        {
            free(name);
            return type_new(TYPE_USIZE);
        }
        if (strcmp(name, "ssize_t") == 0)
        {
            free(name);
            return type_new(TYPE_ISIZE);
        }

        // Portable C Types
        if (strcmp(name, "c_int") == 0)
        {
            free(name);
            return type_new(TYPE_C_INT);
        }
        if (strcmp(name, "c_uint") == 0)
        {
            free(name);
            return type_new(TYPE_C_UINT);
        }
        if (strcmp(name, "c_long") == 0)
        {
            free(name);
            return type_new(TYPE_C_LONG);
        }
        if (strcmp(name, "c_ulong") == 0)
        {
            free(name);
            return type_new(TYPE_C_ULONG);
        }
        if (strcmp(name, "c_short") == 0)
        {
            free(name);
            return type_new(TYPE_C_SHORT);
        }
        if (strcmp(name, "c_ushort") == 0)
        {
            free(name);
            return type_new(TYPE_C_USHORT);
        }
        if (strcmp(name, "c_char") == 0)
        {
            free(name);
            return type_new(TYPE_C_CHAR);
        }
        if (strcmp(name, "c_uchar") == 0)
        {
            free(name);
            return type_new(TYPE_C_UCHAR);
        }

        // Relaxed Type Check: If explicit 'struct Name', trust the user.
        if (explicit_struct)
        {
            Type *ty = type_new(TYPE_STRUCT);
            ty->name = name;
            ty->is_explicit_struct = 1;
        }

        // Selective imports ONLY apply when we're NOT in a module context
        if (!ctx->current_module_prefix)
        {
            SelectiveImport *si = find_selective_import(ctx, name);
            if (si)
            {
                // This is a selectively imported symbol
                // Resolve to the actual struct name which was prefixed during module
                // parsing
                free(name);
                name = xmalloc(strlen(si->source_module) + strlen(si->symbol) + 2);
                sprintf(name, "%s_%s", si->source_module, si->symbol);
            }
        }

        // If we're IN a module and no selective import matched, apply module prefix
        if (ctx->current_module_prefix && !is_known_generic(ctx, name))
        {
            // Auto-prefix struct name if in module context (unless it's a known
            // primitive/generic)
            char *prefixed_name = xmalloc(strlen(ctx->current_module_prefix) + strlen(name) + 2);
            sprintf(prefixed_name, "%s_%s", ctx->current_module_prefix, name);
            free(name);
            name = prefixed_name;
        }

        if (!is_known_generic(ctx, name) && strcmp(name, "Self") != 0)
        {
            register_type_usage(ctx, name, t);
        }

        Type *ty = type_new(TYPE_STRUCT);
        ty->name = name;
        ty->is_explicit_struct = explicit_struct;

        // Handle Generics <T> or <K, V>
        if (lexer_peek(l).type == TOK_LANGLE ||
            (lexer_peek(l).type == TOK_OP && strncmp(lexer_peek(l).start, "<", 1) == 0))
        {
            lexer_next(l); // eat <
            Type *first_arg = parse_type_formal(ctx, l);
            char *first_arg_str = type_to_string(first_arg);

            // Check for multi-arg: <K, V>
            Token next_tok = lexer_peek(l);
            if (next_tok.type == TOK_COMMA)
            {
                // Multi-arg case
                char **args = xmalloc(sizeof(char *) * 8);
                int arg_count = 0;
                args[arg_count++] = xstrdup(first_arg_str);

                while (lexer_peek(l).type == TOK_COMMA)
                {
                    lexer_next(l); // eat ,
                    Type *arg = parse_type_formal(ctx, l);
                    char *arg_str = type_to_string(arg);
                    args = realloc(args, sizeof(char *) * (arg_count + 1));
                    args[arg_count++] = xstrdup(arg_str);
                    free(arg_str);
                }

                // Consume >
                next_tok = lexer_peek(l);
                if (next_tok.type == TOK_RANGLE)
                {
                    lexer_next(l);
                }
                else if (next_tok.type == TOK_OP && next_tok.len == 2 &&
                         strncmp(next_tok.start, ">>", 2) == 0)
                {
                    l->pos += 1;
                    l->col += 1;
                }
                else
                {
                    zpanic_at(t, "Expected > after generic");
                }

                // Call multi-arg instantiation
                int is_generic_dep = 0;
                for (int i = 0; i < arg_count; ++i)
                {
                    for (int k = 0; k < ctx->known_generics_count; ++k)
                    {
                        if (strcmp(args[i], ctx->known_generics[k]) == 0)
                        {
                            is_generic_dep = 1;
                            break;
                        }
                    }
                }

                if (!is_generic_dep)
                {
                    instantiate_generic_multi(ctx, name, args, arg_count, t);
                }

                // Build mangled name
                char mangled[256];
                strcpy(mangled, name);
                for (int i = 0; i < arg_count; i++)
                {
                    char *clean = sanitize_mangled_name(args[i]);
                    strcat(mangled, "_");
                    strcat(mangled, clean);
                    free(clean);
                    free(args[i]);
                }
                free(args);

                free(ty->name);
                ty->name = xstrdup(mangled);
            }
            else
            {
                // Single-arg case - PRESERVE ORIGINAL FLOW EXACTLY
                if (next_tok.type == TOK_RANGLE)
                {
                    lexer_next(l); // Consume >
                }
                else if (next_tok.type == TOK_OP && next_tok.len == 2 &&
                         strncmp(next_tok.start, ">>", 2) == 0)
                {
                    // Split >> into two > tokens
                    l->pos += 1;
                    l->col += 1;
                }
                else
                {
                    zpanic_at(t, "Expected > after generic");
                }

                char *unmangled_arg = type_to_string(first_arg);

                int is_single_dep = 0;
                for (int k = 0; k < ctx->known_generics_count; ++k)
                {
                    if (strcmp(first_arg_str, ctx->known_generics[k]) == 0)
                    {
                        is_single_dep = 1;
                        break;
                    }
                }

                if (!is_single_dep)
                {
                    instantiate_generic(ctx, name, first_arg_str, unmangled_arg, t);
                }
                free(unmangled_arg);

                char *clean_arg = sanitize_mangled_name(first_arg_str);
                char mangled[256];
                sprintf(mangled, "%s_%s", name, clean_arg);
                free(clean_arg);

                free(ty->name);
                ty->name = xstrdup(mangled);
            }

            free(first_arg_str);
            ty->kind = TYPE_STRUCT;
            ty->args = NULL;
            ty->arg_count = 0;
        }
        return ty;
    }

    if (t.type == TOK_LBRACKET)
    {
        lexer_next(l); // eat [
        Type *inner = parse_type_formal(ctx, l);

        // Check for fixed-size array [T; N]
        if (lexer_peek(l).type == TOK_SEMICOLON)
        {
            lexer_next(l); // eat ;
            Token size_tok = lexer_next(l);
            int size = 0;
            if (size_tok.type == TOK_INT)
            {
                size = atoi(size_tok.start);
            }
            else if (size_tok.type == TOK_IDENT)
            {
                // Look up in symbol table for constant propagation
                char *name = token_strdup(size_tok);
                ZenSymbol *sym = find_symbol_entry(ctx, name);
                if (sym && sym->is_const_value)
                {
                    size = sym->const_int_val;
                    sym->is_used = 1; // MARK AS USED
                }
                else
                {
                    zpanic_at(size_tok,
                              "Array size must be a compile-time constant or integer literal");
                }
                free(name);
            }
            else
            {
                zpanic_at(size_tok, "Expected integer for array size");
            }
            if (lexer_next(l).type != TOK_RBRACKET)
            {
                zpanic_at(lexer_peek(l), "Expected ] after array size");
            }

            Type *arr = type_new(TYPE_ARRAY);
            arr->inner = inner;
            arr->array_size = size;
            return arr;
        }

        // Otherwise it's a slice [T]
        if (lexer_next(l).type != TOK_RBRACKET)
        {
            zpanic_at(lexer_peek(l), "Expected ] in type");
        }

        // Register Slice
        char *inner_str = type_to_string(inner);
        register_slice(ctx, inner_str);

        Type *arr = type_new(TYPE_ARRAY);
        arr->inner = inner;
        arr->array_size = 0; // 0 means slice, not fixed-size
        return arr;
    }

    if (t.type == TOK_LPAREN)
    {
        lexer_next(l); // eat (
        char sig[256];
        sig[0] = 0;

        while (1)
        {
            Type *sub = parse_type_formal(ctx, l);
            char *s = type_to_string(sub);
            strcat(sig, s);
            free(s);

            if (lexer_peek(l).type == TOK_COMMA)
            {
                lexer_next(l);
                strcat(sig, "__");
            }
            else
            {
                break;
            }
        }
        if (lexer_next(l).type != TOK_RPAREN)
        {
            zpanic_at(lexer_peek(l), "Expected ) in tuple");
        }

        register_tuple(ctx, sig);

        char *tuple_name = xmalloc(strlen(sig) + 7);
        sprintf(tuple_name, "Tuple_%s", sig);

        Type *ty = type_new(TYPE_STRUCT);
        ty->name = tuple_name;
        return ty;
    }

    // If we have an identifier that wasn't found,
    // assume it is a valid external C type
    // (for example, a struct defined in implementation).
    if (t.type == TOK_IDENT)
    {
        char *fallback = token_strdup(t);
        lexer_next(l);
        Type *ty = type_new(TYPE_STRUCT);
        ty->name = fallback;
        ty->is_explicit_struct = 0;
        return ty;
    }

    return type_new(TYPE_UNKNOWN);
}

Type *parse_type_formal(ParserContext *ctx, Lexer *l)
{
    int is_restrict = 0;
    int is_const = 0;

    if (lexer_peek(l).type == TOK_IDENT)
    {
        if (lexer_peek(l).len == 8 && strncmp(lexer_peek(l).start, "restrict", 8) == 0)
        {
            lexer_next(l); // eat restrict
            is_restrict = 1;
        }
        else if (lexer_peek(l).len == 5 && strncmp(lexer_peek(l).start, "const", 5) == 0)
        {
            lexer_next(l); // eat const
            is_const = 1;
        }
    }

    Type *t = NULL;

    // Example: fn(int, int) -> int
    if (lexer_peek(l).type == TOK_IDENT && strncmp(lexer_peek(l).start, "fn", 2) == 0 &&
        lexer_peek(l).len == 2)
    {
        lexer_next(l); // eat 'fn'

        int star_count = 0;
        while (lexer_peek(l).type == TOK_OP && strncmp(lexer_peek(l).start, "*", 1) == 0)
        {
            lexer_next(l);
            star_count++;
        }

        Type *fn_type = type_new(TYPE_FUNCTION);
        fn_type->is_raw = (star_count > 0);
        fn_type->is_varargs = 0;

        Type *wrapped = fn_type;
        for (int i = 1; i < star_count; i++)
        {
            wrapped = type_new_ptr(wrapped);
        }

        expect(l, TOK_LPAREN, "Expected '(' for function type");

        // Parse Arguments
        fn_type->arg_count = 0;
        fn_type->args = NULL;

        while (lexer_peek(l).type != TOK_RPAREN)
        {
            Type *arg = parse_type_formal(ctx, l);
            fn_type->arg_count++;
            fn_type->args = xrealloc(fn_type->args, sizeof(Type *) * fn_type->arg_count);
            fn_type->args[fn_type->arg_count - 1] = arg;

            if (lexer_peek(l).type == TOK_COMMA)
            {
                lexer_next(l);
            }
            else
            {
                break;
            }
        }
        expect(l, TOK_RPAREN, "Expected ')' after function args");

        // Parse Return Type (-> Type)
        if (lexer_peek(l).type == TOK_ARROW)
        {
            lexer_next(l);                              // eat ->
            fn_type->inner = parse_type_formal(ctx, l); // Return type stored in inner
        }
        else
        {
            fn_type->inner = type_new(TYPE_VOID);
        }

        t = wrapped;
    }
    else
    {
        // Handles: int, Struct, Generic<T>, [Slice], (Tuple)
        t = parse_type_base(ctx, l);
    }

    // Handles: T*, T**, etc.
    while (lexer_peek(l).type == TOK_OP && *lexer_peek(l).start == '*')
    {
        lexer_next(l); // consume '*'
        t = type_new_ptr(t);
    }

    // 4. Handle Array Suffixes (e.g. int[10])
    while (lexer_peek(l).type == TOK_LBRACKET)
    {
        lexer_next(l); // consume '['

        int size = 0;
        if (lexer_peek(l).type == TOK_INT)
        {
            Token t = lexer_peek(l);
            char buffer[64];
            int len = t.len < 63 ? t.len : 63;
            strncpy(buffer, t.start, len);
            buffer[len] = 0;
            size = atoi(buffer);
            lexer_next(l);
        }
        else if (lexer_peek(l).type == TOK_IDENT)
        {
            Token t = lexer_peek(l);
            char *name = token_strdup(t);
            ZenSymbol *sym = find_symbol_entry(ctx, name);
            if (sym && sym->is_const_value)
            {
                size = sym->const_int_val;
                sym->is_used = 1;
            }
            else
            {
                zpanic_at(t, "Array size must be a known compile-time constant integer");
            }
            free(name);
            lexer_next(l);
        }

        expect(l, TOK_RBRACKET, "Expected ']' in array type");

        if (size == 0)
        {
            char *inner_str = type_to_string(t);
            register_slice(ctx, inner_str);
            free(inner_str);
        }

        Type *arr = type_new(TYPE_ARRAY);
        arr->inner = t;
        arr->array_size = size;
        t = arr;
    }

    if (is_restrict)
    {
        t->is_restrict = 1;
    }
    if (is_const)
    {
        t->is_const = 1;
    }
    return t;
}

char *parse_type(ParserContext *ctx, Lexer *l)
{
    Type *t = parse_type_formal(ctx, l);

    return type_to_string(t);
}

char *parse_array_literal(ParserContext *ctx, Lexer *l, const char *st)
{
    (void)ctx; // suppress unused parameter warning
    lexer_next(l);
    size_t cap = 128;
    char *c = xmalloc(cap);
    c[0] = 0;
    int n = 0;

    while (1)
    {
        Token t = lexer_peek(l);
        if (t.type == TOK_RBRACKET)
        {
            lexer_next(l);
            break;
        }
        if (t.type == TOK_COMMA)
        {
            lexer_next(l);
            continue;
        }

        const char *s = l->src + l->pos;
        int d = 0;
        while (1)
        {
            Token it = lexer_peek(l);
            if (it.type == TOK_EOF)
            {
                break;
            }
            if (d == 0 && (it.type == TOK_COMMA || it.type == TOK_RBRACKET))
            {
                break;
            }
            if (it.type == TOK_LBRACKET || it.type == TOK_LPAREN)
            {
                d++;
            }
            if (it.type == TOK_RBRACKET || it.type == TOK_RPAREN)
            {
                d--;
            }
            lexer_next(l);
        }

        int len = (l->src + l->pos) - s;
        if (strlen(c) + len + 5 > cap)
        {
            cap *= 2;
            c = xrealloc(c, cap);
        }
        if (n > 0)
        {
            strcat(c, ", ");
        }
        strncat(c, s, len);
        n++;
    }

    char rt[64];
    if (strncmp(st, "Slice_", 6) == 0)
    {
        strcpy(rt, st + 6);
    }
    else
    {
        strcpy(rt, "int");
    }

    char *o = xmalloc(strlen(c) + 128);
    sprintf(o, "(%s){.data=(%s[]){%s},.len=%d,.cap=%d}", st, rt, c, n, n);
    free(c);
    return o;
}
char *parse_tuple_literal(ParserContext *ctx, Lexer *l, const char *tn)
{
    (void)ctx; // suppress unused parameter warning
    lexer_next(l);
    size_t cap = 128;
    char *c = xmalloc(cap);
    c[0] = 0;

    while (1)
    {
        Token t = lexer_peek(l);
        if (t.type == TOK_RPAREN)
        {
            lexer_next(l);
            break;
        }
        if (t.type == TOK_COMMA)
        {
            lexer_next(l);
            continue;
        }

        const char *s = l->src + l->pos;
        int d = 0;
        while (1)
        {
            Token it = lexer_peek(l);
            if (it.type == TOK_EOF)
            {
                break;
            }
            if (d == 0 && (it.type == TOK_COMMA || it.type == TOK_RPAREN))
            {
                break;
            }
            if (it.type == TOK_LPAREN)
            {
                d++;
            }
            if (it.type == TOK_RPAREN)
            {
                d--;
            }
            lexer_next(l);
        }

        int len = (l->src + l->pos) - s;
        if (strlen(c) + len + 5 > cap)
        {
            cap *= 2;
            c = xrealloc(c, cap);
        }
        if (strlen(c) > 0)
        {
            strcat(c, ", ");
        }
        strncat(c, s, len);
    }

    char *o = xmalloc(strlen(c) + 128);
    sprintf(o, "(%s){%s}", tn, c);
    free(c);
    return o;
}
ASTNode *parse_embed(ParserContext *ctx, Lexer *l)
{
    lexer_next(l);
    Token t = lexer_next(l);
    if (t.type != TOK_STRING)
    {
        zpanic_at(t, "String required");
    }
    char fn[256];
    strncpy(fn, t.start + 1, t.len - 2);
    fn[t.len - 2] = 0;

    // Check for optional "as Type"
    Type *target_type = NULL;
    if (lexer_peek(l).type == TOK_IDENT && lexer_peek(l).len == 2 &&
        strncmp(lexer_peek(l).start, "as", 2) == 0)
    {
        lexer_next(l); // consume 'as'
        target_type = parse_type_formal(ctx, l);
    }

    char *embedpath = NULL;
    if (zc_find_path(fn, &embedpath))
    {
        snprintf(fn, sizeof(fn), "%s", embedpath);
        free(embedpath);
        embedpath = NULL;
    }

    FILE *f = fopen(fn, "rb");
    if (!f)
    {
        zpanic_at(t, "404: %s", fn);
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    rewind(f);
    unsigned char *b = xmalloc(len);
    fread(b, 1, len, f);
    fclose(f);

    size_t oc = len * 6 + 256;
    char *o = xmalloc(oc);

    // Default Type if none
    if (!target_type)
    {
        // Default: Slice_char
        register_slice(ctx, "char");

        Type *slice_type = type_new(TYPE_STRUCT);
        slice_type->name = xstrdup("Slice_char");
        target_type = slice_type;

        sprintf(o, "(Slice_char){.data=(char[]){");
    }
    else
    {
        // Handle specific type
        char *ts = type_to_string(target_type);

        if (target_type->kind == TYPE_ARRAY)
        {
            char *inner_ts = type_to_string(target_type->inner);
            if (target_type->array_size > 0)
            {
                Type *ptr_type = type_new_ptr(target_type->inner); // Reuse inner
                target_type = ptr_type;
                sprintf(o, "(%s[]){", inner_ts);
            }
            else
            {
                // Slice -> Slice_T struct
                register_slice(ctx, inner_ts);
                char slice_name[256];
                sprintf(slice_name, "Slice_%s", inner_ts);
                Type *slice_t = type_new(TYPE_STRUCT);
                slice_t->name = xstrdup(slice_name);
                target_type = slice_t;
                sprintf(o, "(%s){.data=(%s[]){", slice_name, inner_ts);
            }
            free(inner_ts);
        }
        else
        {
            if (strcmp(ts, "string") == 0 || strcmp(ts, "char*") == 0)
            {
                sprintf(o, "(char*)\"");
            }
            else
            {
                sprintf(o, "(%s){", ts);
            }
        }
        free(ts);
    }

    char *p = o + strlen(o);

    // Check if string mode
    int is_string = (target_type && (strcmp(type_to_string(target_type), "string") == 0 ||
                                     strcmp(type_to_string(target_type), "char*") == 0));

    for (int i = 0; i < len; i++)
    {
        if (is_string)
        {
            // Hex escape for string
            p += sprintf(p, "\\x%02X", b[i]);
        }
        else
        {
            p += sprintf(p, "0x%02X,", b[i]);
        }
    }

    if (is_string)
    {
        sprintf(p, "\"");
    }
    else
    {
        int is_slice = (strncmp(o, "(Slice_", 7) == 0);

        if (is_slice)
        {
            sprintf(p, "},.len=%ld,.cap=%ld}", len, len);
        }
        else
        {
            sprintf(p, "}");
        }
    }

    free(b);

    ASTNode *n = ast_create(NODE_RAW_STMT);
    n->raw_stmt.content = o;
    n->type_info = target_type;
    return n;
}
