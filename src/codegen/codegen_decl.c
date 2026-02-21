
#include "../ast/ast.h"
#include "../parser/parser.h"
#include "../zprep.h"
#include "codegen.h"
#include "compat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void emit_freestanding_preamble(FILE *out)
{
    fputs("#include <stddef.h>\n#include <stdint.h>\n#include "
          "<stdbool.h>\n#include <stdarg.h>\n",
          out);
    fputs(ZC_TCC_COMPAT_STR, out);
    fputs("typedef size_t usize;\ntypedef char* string;\n", out);
    fputs("#define U0 void\n#define I8 int8_t\n#define U8 uint8_t\n#define I16 "
          "int16_t\n#define U16 uint16_t\n",
          out);
    fputs("#define I32 int32_t\n#define U32 uint32_t\n#define I64 "
          "int64_t\n#define U64 "
          "uint64_t\n",
          out);
    fputs("#define F32 float\n#define F64 double\n", out);
    fputs("static inline const char* _z_bool_str(_Bool b) { return b ? \"true\" : \"false\"; }\n",
          out);
    fputs("#define _z_str(x) _Generic((x), _Bool: \"%s\", char: \"%c\", "
          "signed char: \"%c\", unsigned char: \"%u\", short: \"%d\", "
          "unsigned short: \"%u\", int: \"%d\", unsigned int: \"%u\", "
          "long: \"%ld\", unsigned long: \"%lu\", long long: \"%lld\", "
          "unsigned long long: \"%llu\", float: \"%f\", double: \"%f\", "
          "char*: \"%s\", const char*: \"%s\", void*: \"%p\")\n",
          out);
    fputs("#define _z_arg(x) _Generic((x), _Bool: _z_bool_str(x), default: (x))\n", out);
    fputs("typedef struct { void *func; void *ctx; } z_closure_T;\n", out);

    // In true freestanding, explicit definitions of z_malloc/etc are removed.
    // The user must implement them if they use features requiring them.
    // Most primitives (integers, pointers) work without them.
}

void emit_preamble(ParserContext *ctx, FILE *out)
{
    if (g_config.is_freestanding)
    {
        emit_freestanding_preamble(out);
        return;
    }
    else
    {
        // Standard hosted preamble.
        fputs("#define _GNU_SOURCE\n", out);
        fputs("#include <stdio.h>\n#include <stdlib.h>\n#include "
              "<stddef.h>\n#include <string.h>\n",
              out);
        fputs("#include <stdarg.h>\n#include <stdint.h>\n#include <stdbool.h>\n", out);
        fputs("#include <unistd.h>\n#include <fcntl.h>\n", out); // POSIX functions

        // C++ compatibility
        if (g_config.use_cpp)
        {
            // For C++: define ZC_AUTO as auto, include compat.h macros inline
            fputs("#define ZC_AUTO auto\n", out);
            fputs("#define ZC_AUTO_INIT(var, init) auto var = (init)\n", out);
            fputs("#define ZC_CAST(T, x) static_cast<T>(x)\n", out);
            // C++ _z_str via overloads
            fputs("inline const char* _z_bool_str(bool b) { return b ? \"true\" : \"false\"; }\n",
                  out);
            fputs("inline const char* _z_str(bool)               { return \"%s\"; }\n", out);
            fputs("inline const char* _z_arg(bool b)             { return _z_bool_str(b); }\n",
                  out);
            fputs("template<typename T> inline T _z_arg(T x)     { return x; }\n", out);
            fputs("inline const char* _z_str(char)               { return \"%c\"; }\n", out);
            fputs("inline const char* _z_str(int)                { return \"%d\"; }\n", out);
            fputs("inline const char* _z_str(unsigned int)       { return \"%u\"; }\n", out);
            fputs("inline const char* _z_str(long)               { return \"%ld\"; }\n", out);
            fputs("inline const char* _z_str(unsigned long)      { return \"%lu\"; }\n", out);
            fputs("inline const char* _z_str(long long)          { return \"%lld\"; }\n", out);
            fputs("inline const char* _z_str(unsigned long long) { return \"%llu\"; }\n", out);
            fputs("inline const char* _z_str(float)              { return \"%f\"; }\n", out);
            fputs("inline const char* _z_str(double)             { return \"%f\"; }\n", out);
            fputs("inline const char* _z_str(char*)              { return \"%s\"; }\n", out);
            fputs("inline const char* _z_str(const char*)        { return \"%s\"; }\n", out);
            fputs("inline const char* _z_str(void*)              { return \"%p\"; }\n", out);
        }
        else
        {
            // C mode
            fputs("#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202300L\n", out);
            fputs("#define ZC_AUTO auto\n", out);
            fputs("#define ZC_AUTO_INIT(var, init) auto var = (init)\n", out);
            fputs("#else\n", out);
            fputs("#define ZC_AUTO __auto_type\n", out);
            fputs("#define ZC_AUTO_INIT(var, init) __auto_type var = (init)\n", out);
            fputs("#endif\n", out);
            fputs("#define ZC_CAST(T, x) ((T)(x))\n", out);
            fputs(ZC_TCC_COMPAT_STR, out);
            fputs("static inline const char* _z_bool_str(_Bool b) { return b ? \"true\" : "
                  "\"false\"; }\n",
                  out);
            fputs(ZC_C_GENERIC_STR, out);
            fputs(ZC_C_ARG_GENERIC_STR, out);
        }

        fputs("typedef size_t usize;\ntypedef char* string;\n", out);
        if (ctx->has_async)
        {
            fputs("#include <pthread.h>\n", out);
            fputs("typedef struct { pthread_t thread; void *result; } Async;\n", out);
        }
        fputs("typedef struct { void *func; void *ctx; } z_closure_T;\n", out);
        fputs("typedef void U0;\ntypedef int8_t I8;\ntypedef uint8_t U8;\ntypedef "
              "int16_t I16;\ntypedef uint16_t U16;\n",
              out);
        fputs("typedef int32_t I32;\ntypedef uint32_t U32;\ntypedef int64_t I64;\ntypedef "
              "uint64_t U64;\n",
              out);
        fputs("#define F32 float\n#define F64 double\n", out);

        // Memory Mapping.
        if (g_config.use_cpp)
        {
            // C++ needs explicit casts for void* conversions
            fputs("#define z_malloc(sz) static_cast<char*>(malloc(sz))\n", out);
            fputs("#define z_realloc(p, sz) static_cast<char*>(realloc(p, sz))\n", out);
        }
        else
        {
            fputs("#define z_malloc malloc\n#define z_realloc realloc\n", out);
        }
        fputs("#define z_free free\n#define z_print printf\n", out);
        fputs("void z_panic(const char* msg) { fprintf(stderr, \"Panic: %s\\n\", "
              "msg); exit(1); }\n",
              out);
        fputs("#if defined(__APPLE__)\n"
              "#define _ZC_SEC __attribute__((used,section(\"__DATA,__zarch\")))\n"
              "#elif defined(_WIN32)\n"
              "#define _ZC_SEC __attribute__((used))\n"
              "#else\n"
              "#define _ZC_SEC __attribute__((used,section(\".note.zarch\")))\n"
              "#endif\n",
              out);
        fputs("static const unsigned char _zc_abi_v1[] _ZC_SEC = {"
              "0x07,0xd5,"
              "0x59,0x30,0x7c,0x7f,0x66,0x75,0x30,0x69,"
              "0x7f,0x65,0x3c,0x30,0x59,0x7c,0x79,0x7e,"
              "0x73,0x71};\n",
              out);

        fputs("void _z_autofree_impl(void *p) { void **pp = (void**)p; if(*pp) { "
              "z_free(*pp); *pp "
              "= NULL; } }\n",
              out);
        fputs("#define assert(cond, ...) if (!(cond)) { fprintf(stderr, "
              "\"Assertion failed: \" "
              "__VA_ARGS__); exit(1); }\n",
              out);

        // C++ compatible readln helper
        if (g_config.use_cpp)
        {
            fputs(
                "string _z_readln_raw() { "
                "size_t cap = 64; size_t len = 0; "
                "char *line = static_cast<char*>(malloc(cap)); "
                "if(!line) return NULL; "
                "int c; "
                "while((c = fgetc(stdin)) != EOF) { "
                "if(c == '\\n') break; "
                "if(len + 1 >= cap) { cap *= 2; char *n = static_cast<char*>(realloc(line, cap)); "
                "if(!n) { free(line); return NULL; } line = n; } "
                "line[len++] = c; } "
                "if(len == 0 && c == EOF) { free(line); return NULL; } "
                "line[len] = 0; return line; }\n",
                out);
        }
        else
        {
            fputs("string _z_readln_raw() { "
                  "size_t cap = 64; size_t len = 0; "
                  "char *line = z_malloc(cap); "
                  "if(!line) return NULL; "
                  "int c; "
                  "while((c = fgetc(stdin)) != EOF) { "
                  "if(c == '\\n') break; "
                  "if(len + 1 >= cap) { cap *= 2; char *n = z_realloc(line, cap); "
                  "if(!n) { z_free(line); return NULL; } line = n; } "
                  "line[len++] = c; } "
                  "if(len == 0 && c == EOF) { z_free(line); return NULL; } "
                  "line[len] = 0; return line; }\n",
                  out);
        }
        fputs("int _z_scan_helper(const char *fmt, ...) { char *l = "
              "_z_readln_raw(); if(!l) return "
              "0; va_list ap; va_start(ap, fmt); int r = vsscanf(l, fmt, ap); "
              "va_end(ap); "
              "z_free(l); return r; }\n",
              out);

        // REPL helpers: suppress/restore stdout.
        fputs("int _z_orig_stdout = -1;\n", out);
        fputs("void _z_suppress_stdout() {\n", out);
        fputs("    fflush(stdout);\n", out);
        fputs("    if (_z_orig_stdout == -1) _z_orig_stdout = dup(STDOUT_FILENO);\n", out);
        fputs("    int nullfd = open(\"/dev/null\", O_WRONLY);\n", out);
        fputs("    dup2(nullfd, STDOUT_FILENO);\n", out);
        fputs("    close(nullfd);\n", out);
        fputs("}\n", out);
        fputs("void _z_restore_stdout() {\n", out);
        fputs("    fflush(stdout);\n", out);
        fputs("    if (_z_orig_stdout != -1) {\n", out);
        fputs("        dup2(_z_orig_stdout, STDOUT_FILENO);\n", out);
        fputs("        close(_z_orig_stdout);\n", out);
        fputs("        _z_orig_stdout = -1;\n", out);
        fputs("    }\n", out);
        fputs("}\n", out);
    }
}

// Emit includes and type aliases (and top-level comments)
void emit_includes_and_aliases(ASTNode *node, FILE *out)
{
    while (node)
    {
        if (node->type == NODE_INCLUDE)
        {
            if (node->include.is_system)
            {
                fprintf(out, "#include <%s>\n", node->include.path);
            }
            else
            {
                fprintf(out, "#include \"%s\"\n", node->include.path);
            }
        }
        else if (node->type == NODE_AST_COMMENT)
        {
            fprintf(out, "%s\n", node->comment.content);
        }
        node = node->next;
    }
}

// Emit type aliases (after struct defs so the aliased types exist)
void emit_type_aliases(ASTNode *node, FILE *out)
{
    while (node)
    {
        if (node->type == NODE_TYPE_ALIAS)
        {
            fprintf(out, "typedef %s %s;\n", node->type_alias.original_type,
                    node->type_alias.alias);
        }
        node = node->next;
    }
}

void emit_global_aliases(ParserContext *ctx, FILE *out)
{
    TypeAlias *ta = ctx->type_aliases;
    while (ta)
    {
        fprintf(out, "typedef %s %s;\n", ta->original_type, ta->alias);
        ta = ta->next;
    }
}

// Emit enum constructor prototypes
void emit_enum_protos(ASTNode *node, FILE *out)
{
    while (node)
    {
        if (node->type == NODE_ENUM && !node->enm.is_template)
        {
            ASTNode *v = node->enm.variants;
            while (v)
            {
                if (v->variant.payload)
                {
                    char *tstr = codegen_type_to_string(v->variant.payload);
                    fprintf(out, "%s %s_%s(%s v);\n", node->enm.name, node->enm.name,
                            v->variant.name, tstr);
                    free(tstr);
                }
                else
                {
                    fprintf(out, "%s %s_%s();\n", node->enm.name, node->enm.name, v->variant.name);
                }
                v = v->next;
            }
        }
        node = node->next;
    }
}

// Emit lambda definitions.
void emit_lambda_defs(ParserContext *ctx, FILE *out)
{
    LambdaRef *cur = ctx->global_lambdas;
    while (cur)
    {
        ASTNode *node = cur->node;
        int saved_defer = defer_count;
        defer_count = 0;

        if (node->lambda.num_captures > 0)
        {
            fprintf(out, "struct Lambda_%d_Ctx {\n", node->lambda.lambda_id);
            for (int i = 0; i < node->lambda.num_captures; i++)
            {
                if (node->lambda.capture_modes && node->lambda.capture_modes[i] == 1)
                {
                    fprintf(out, "    %s* %s;\n", node->lambda.captured_types[i],
                            node->lambda.captured_vars[i]);
                }
                else
                {
                    fprintf(out, "    %s %s;\n", node->lambda.captured_types[i],
                            node->lambda.captured_vars[i]);
                }
            }
            fprintf(out, "};\n");
        }

        char *ret_type_str = node->lambda.return_type;
        if (node->type_info && node->type_info->inner &&
            node->type_info->inner->kind != TYPE_UNKNOWN)
        {
            ret_type_str = type_to_string(node->type_info->inner);
        }

        if (strcmp(ret_type_str, "unknown") == 0)
        {
            fprintf(out, "void* _lambda_%d(void* _ctx", node->lambda.lambda_id);
        }
        else
        {
            fprintf(out, "%s _lambda_%d(void* _ctx", ret_type_str, node->lambda.lambda_id);
        }

        if (node->type_info && node->type_info->inner &&
            node->type_info->inner->kind != TYPE_UNKNOWN)
        {
            free(ret_type_str);
        }

        for (int i = 0; i < node->lambda.num_params; i++)
        {
            char *param_type_str = node->lambda.param_types[i];
            if (node->type_info && node->type_info->args && node->type_info->args[i] &&
                node->type_info->args[i]->kind != TYPE_UNKNOWN)
            {
                param_type_str = type_to_string(node->type_info->args[i]);
            }
            if (strcmp(param_type_str, "unknown") == 0)
            {
                fprintf(out, ", void* %s", node->lambda.param_names[i]);
            }
            else
            {
                fprintf(out, ", %s %s", param_type_str, node->lambda.param_names[i]);
            }
            if (node->type_info && node->type_info->args && node->type_info->args[i] &&
                node->type_info->args[i]->kind != TYPE_UNKNOWN)
            {
                free(param_type_str);
            }
        }
        fprintf(out, ") {\n");

        if (node->lambda.num_captures > 0)
        {
            fprintf(out, "    struct Lambda_%d_Ctx* ctx = (struct Lambda_%d_Ctx*)_ctx;\n",
                    node->lambda.lambda_id, node->lambda.lambda_id);
        }

        g_current_lambda = node;
        if (node->lambda.body && node->lambda.body->type == NODE_BLOCK)
        {
            codegen_walker(ctx, node->lambda.body->block.statements, out);
        }
        g_current_lambda = NULL;

        for (int i = defer_count - 1; i >= 0; i--)
        {
            codegen_node_single(ctx, defer_stack[i], out);
        }

        fprintf(out, "}\n\n");

        defer_count = saved_defer;
        cur = cur->next;
    }
}

// Emit struct and enum definitions.
void emit_struct_defs(ParserContext *ctx, ASTNode *node, FILE *out)
{
    while (node)
    {
        if (node->type == NODE_STRUCT && node->strct.is_template)
        {
            node = node->next;
            continue;
        }
        if (node->type == NODE_ENUM && node->enm.is_template)
        {
            node = node->next;
            continue;
        }
        if (node->type == NODE_STRUCT)
        {
            if (node->strct.is_incomplete)
            {
                // Forward declaration - no body needed (typedef handles it)
                node = node->next;
                continue;
            }

            if (node->strct.is_union)
            {
                fprintf(out, "union %s {", node->strct.name);
            }
            else
            {
                fprintf(out, "struct %s {", node->strct.name);
            }
            fprintf(out, "\n");
            if (node->strct.fields)
            {
                codegen_walker(ctx, node->strct.fields, out);
            }
            else
            {
                // C requires at least one member in a struct.
                fprintf(out, "    char _placeholder;\n");
            }
            fprintf(out, "}");

            if (node->strct.is_packed && node->strct.align)
            {
                fprintf(out, " __attribute__((packed, aligned(%d)))", node->strct.align);
            }
            else if (node->strct.is_packed)
            {
                fprintf(out, " __attribute__((packed))");
            }
            if (node->strct.align)
            {
                fprintf(out, " __attribute__((aligned(%d)))", node->strct.align);
            }
            if (node->strct.is_export)
            {
                fprintf(out, " __attribute__((visibility(\"default\")))");
            }

            if (node->strct.attributes)
            {
                fprintf(out, " __attribute__((");
                Attribute *custom = node->strct.attributes;
                int first = 1;
                while (custom)
                {
                    if (!first)
                    {
                        fprintf(out, ", ");
                    }
                    fprintf(out, "%s", custom->name);
                    if (custom->arg_count > 0)
                    {
                        fprintf(out, "(");
                        for (int i = 0; i < custom->arg_count; i++)
                        {
                            if (i > 0)
                            {
                                fprintf(out, ", ");
                            }
                            fprintf(out, "%s", custom->args[i]);
                        }
                        fprintf(out, ")");
                    }
                    first = 0;
                    custom = custom->next;
                }
                fprintf(out, "))");
            }

            fprintf(out, ";\n\n");
        }
        else if (node->type == NODE_ENUM)
        {
            fprintf(out, "typedef enum { ");
            ASTNode *v = node->enm.variants;
            while (v)
            {
                fprintf(out, "%s_%s_Tag, ", node->enm.name, v->variant.name);
                v = v->next;
            }
            fprintf(out, "} %s_Tag;\n", node->enm.name);
            fprintf(out, "struct %s { %s_Tag tag; union { ", node->enm.name, node->enm.name);
            v = node->enm.variants;
            while (v)
            {
                if (v->variant.payload)
                {
                    char *tstr = codegen_type_to_string(v->variant.payload);
                    fprintf(out, "%s %s; ", tstr, v->variant.name);
                    free(tstr);
                }
                v = v->next;
            }
            fprintf(out, "} data; };\n\n");
            v = node->enm.variants;
            while (v)
            {
                if (v->variant.payload)
                {
                    char *tstr = codegen_type_to_string(v->variant.payload);
                    fprintf(out,
                            "%s %s_%s(%s v) { return (%s){.tag=%s_%s_Tag, "
                            ".data.%s=v}; }\n",
                            node->enm.name, node->enm.name, v->variant.name, tstr, node->enm.name,
                            node->enm.name, v->variant.name, v->variant.name);
                    free(tstr);
                }
                else
                {
                    fprintf(out, "%s %s_%s() { return (%s){.tag=%s_%s_Tag}; }\n", node->enm.name,
                            node->enm.name, v->variant.name, node->enm.name, node->enm.name,
                            v->variant.name);
                }
                v = v->next;
            }
        }
        node = node->next;
    }
}

// Helper to substitute 'Self' with replacement string
static char *substitute_proto_self(const char *type_str, const char *replacement)
{
    if (!type_str)
    {
        return NULL;
    }
    if (strcmp(type_str, "Self") == 0)
    {
        return xstrdup(replacement);
    }
    // Handle pointers (Self* -> replacement*)
    if (strncmp(type_str, "Self", 4) == 0)
    {
        char *rest = (char *)type_str + 4;
        char *buf = xmalloc(strlen(replacement) + strlen(rest) + 1);
        sprintf(buf, "%s%s", replacement, rest);
        return buf;
    }
    return xstrdup(type_str);
}

// Emit trait definitions.
void emit_trait_defs(ASTNode *node, FILE *out)
{
    while (node)
    {
        if (node->type == NODE_TRAIT)
        {
            if (node->trait.generic_param_count > 0)
            {
                node = node->next;
                continue;
            }
            fprintf(out, "typedef struct %s_VTable {\n", node->trait.name);
            ASTNode *m = node->trait.methods;
            while (m)
            {
                char *ret_safe = substitute_proto_self(m->func.ret_type, "void*");
                fprintf(out, "    %s (*%s)(", ret_safe, parse_original_method_name(m->func.name));
                free(ret_safe);

                int has_self = (m->func.args && strstr(m->func.args, "self"));
                if (!has_self)
                {
                    fprintf(out, "void* self");
                }

                if (m->func.args)
                {
                    if (!has_self)
                    {
                        fprintf(out, ", ");
                    }
                    char *args_safe = replace_type_str(m->func.args, "Self", "void*", NULL, NULL);
                    fprintf(out, "%s", args_safe);
                    free(args_safe);
                }
                fprintf(out, ");\n");
                m = m->next;
            }
            fprintf(out, "} %s_VTable;\n", node->trait.name);
            fprintf(out, "typedef struct %s { void *self; %s_VTable *vtable; } %s;\n",
                    node->trait.name, node->trait.name, node->trait.name);

            m = node->trait.methods;
            while (m)
            {
                const char *orig = parse_original_method_name(m->func.name);
                char *ret_sub = substitute_proto_self(m->func.ret_type, node->trait.name);

                fprintf(out, "%s %s__%s(%s* self", ret_sub, node->trait.name, orig,
                        node->trait.name);

                int has_self = (m->func.args && strstr(m->func.args, "self"));
                if (m->func.args)
                {
                    if (has_self)
                    {
                        char *comma = strchr(m->func.args, ',');
                        if (comma)
                        {
                            // Substitute Self -> TraitName in wrapper args
                            char *args_sub =
                                replace_type_str(comma + 1, "Self", node->trait.name, NULL, NULL);
                            fprintf(out, ", %s", args_sub);
                            free(args_sub);
                        }
                    }
                    else
                    {
                        char *args_sub =
                            replace_type_str(m->func.args, "Self", node->trait.name, NULL, NULL);
                        fprintf(out, ", %s", args_sub);
                        free(args_sub);
                    }
                }
                fprintf(out, ") {\n");

                int ret_is_self = (strcmp(m->func.ret_type, "Self") == 0);

                if (ret_is_self)
                {
                    // Special handling: return (Trait){.self = call(), .vtable = self->vtable}
                    fprintf(out, "    void* ret = self->vtable->%s(self->self", orig);
                }
                else
                {
                    fprintf(out, "    return self->vtable->%s(self->self", orig);
                }

                if (m->func.args)
                {
                    char *call_args = extract_call_args(m->func.args);
                    if (has_self)
                    {
                        char *comma = strchr(call_args, ',');
                        if (comma)
                        {
                            fprintf(out, ", %s", comma + 1);
                        }
                    }
                    else
                    {
                        if (strlen(call_args) > 0)
                        {
                            fprintf(out, ", %s", call_args);
                        }
                    }
                    free(call_args);
                }
                fprintf(out, ");\n");

                if (ret_is_self)
                {
                    fprintf(out, "    return (%s){.self = ret, .vtable = self->vtable};\n",
                            node->trait.name);
                }

                fprintf(out, "}\n\n");
                free(ret_sub);

                m = m->next;
            }
            fprintf(out, "\n");
        }
        node = node->next;
    }
}

// Emit global variables
void emit_globals(ParserContext *ctx, ASTNode *node, FILE *out)
{
    while (node)
    {
        if (node->type == NODE_VAR_DECL || node->type == NODE_CONST)
        {
            if (node->type == NODE_CONST)
            {
                fprintf(out, "const ");
            }
            if (node->var_decl.type_str)
            {
                emit_var_decl_type(ctx, out, node->var_decl.type_str, node->var_decl.name);
            }
            else
            {
                char *inferred = NULL;
                if (node->var_decl.init_expr)
                {
                    inferred = infer_type(ctx, node->var_decl.init_expr);
                }

                if (inferred && strcmp(inferred, "__auto_type") != 0)
                {
                    emit_var_decl_type(ctx, out, inferred, node->var_decl.name);
                }
                else
                {
                    emit_auto_type(ctx, node->var_decl.init_expr, node->token, out);
                    fprintf(out, " %s", node->var_decl.name);
                }
            }
            if (node->var_decl.init_expr)
            {
                fprintf(out, " = ");
                codegen_expression(ctx, node->var_decl.init_expr, out);
            }
            fprintf(out, ";\n");
        }
        node = node->next;
    }
}

// Emit function prototypes
void emit_protos(ParserContext *ctx, ASTNode *node, FILE *out)
{
    ASTNode *f = node;
    while (f)
    {
        if (f->type == NODE_FUNCTION)
        {
            if (f->func.is_async)
            {
                fprintf(out, "Async %s(%s);\n", f->func.name, f->func.args);
                // Also emit _impl_ prototype
                if (f->func.ret_type)
                {
                    fprintf(out, "%s _impl_%s(%s);\n", f->func.ret_type, f->func.name,
                            f->func.args);
                }
                else
                {
                    fprintf(out, "void _impl_%s(%s);\n", f->func.name, f->func.args);
                }
            }
            else
            {
                emit_func_signature(ctx, out, f, NULL);
                fprintf(out, ";\n");
            }
        }
        else if (f->type == NODE_IMPL)
        {
            char *sname = f->impl.struct_name;
            if (!sname)
            {
                f = f->next;
                continue;
            }

            char *mangled = replace_string_type(sname);
            ASTNode *def = find_struct_def_codegen(g_parser_ctx, mangled);
            int skip = 0;
            if (def)
            {
                if (def->type == NODE_STRUCT && def->strct.is_template)
                {
                    skip = 1;
                }
                else if (def->type == NODE_ENUM && def->enm.is_template)
                {
                    skip = 1;
                }
            }
            else
            {
                char *buf = strip_template_suffix(sname);
                if (buf)
                {
                    def = find_struct_def_codegen(g_parser_ctx, buf);
                    if (def && def->strct.is_template)
                    {
                        skip = 1;
                    }
                    free(buf);
                }
            }
            if (mangled)
            {
                free(mangled);
            }

            if (skip)
            {
                f = f->next;
                continue;
            }

            ASTNode *m = f->impl.methods;
            while (m)
            {
                if (m->func.generic_params)
                {
                    m = m->next;
                    continue;
                }
                char *fname = m->func.name;
                char *proto = xmalloc(strlen(fname) + strlen(sname) + 2);
                int slen = strlen(sname);
                if (strncmp(fname, sname, slen) == 0 && fname[slen] == '_' &&
                    fname[slen + 1] == '_')
                {
                    strcpy(proto, fname);
                }
                else
                {
                    sprintf(proto, "%s__%s", sname, fname);
                }

                if (m->func.is_async)
                {
                    fprintf(out, "Async %s(%s);\n", proto, m->func.args);
                }
                else
                {
                    emit_func_signature(ctx, out, m, proto);
                    fprintf(out, ";\n");
                }

                free(proto);
                m = m->next;
            }
        }
        else if (f->type == NODE_IMPL_TRAIT)
        {
            char *sname = f->impl_trait.target_type;
            if (!sname)
            {
                f = f->next;
                continue;
            }

            char *mangled = replace_string_type(sname);
            ASTNode *def = find_struct_def_codegen(g_parser_ctx, mangled);
            int skip = 0;
            if (def)
            {
                if (def->strct.is_template)
                {
                    skip = 1;
                }
            }
            else
            {
                char *buf = strip_template_suffix(sname);
                if (buf)
                {
                    def = find_struct_def_codegen(g_parser_ctx, buf);
                    if (def && def->strct.is_template)
                    {
                        skip = 1;
                    }
                    free(buf);
                }
            }
            if (mangled)
            {
                free(mangled);
            }

            if (skip)
            {
                f = f->next;
                continue;
            }

            ASTNode *m = f->impl_trait.methods;
            while (m)
            {
                if (m->func.generic_params)
                {
                    m = m->next;
                    continue;
                }
                if (m->func.is_async)
                {
                    fprintf(out, "Async %s(%s);\n", m->func.name, m->func.args);
                }
                else
                {
                    fprintf(out, "%s %s(%s);\n", m->func.ret_type, m->func.name, m->func.args);
                }
                m = m->next;
            }
        }
        f = f->next;
    }
}

// Emit VTable instances for trait implementations.
void emit_impl_vtables(ParserContext *ctx, FILE *out)
{
    StructRef *ref = ctx->parsed_impls_list;
    struct
    {
        char *trait;
        char *strct;
    } emitted[1024];
    int count = 0;

    while (ref)
    {
        ASTNode *node = ref->node;
        if (node && node->type == NODE_IMPL_TRAIT)
        {
            char *trait = node->impl_trait.trait_name;

            // Filter generic traits (VTables for them are not emitted)
            int is_generic_trait = 0;
            StructRef *search = ctx->parsed_globals_list;
            while (search)
            {
                if (search->node && search->node->type == NODE_TRAIT &&
                    strcmp(search->node->trait.name, trait) == 0)
                {
                    if (search->node->trait.generic_param_count > 0)
                    {
                        is_generic_trait = 1;
                    }
                    break;
                }
                search = search->next;
            }
            if (is_generic_trait)
            {
                ref = ref->next;
                continue;
            }

            char *strct = node->impl_trait.target_type;

            // Filter templates
            char *mangled = replace_string_type(strct);
            ASTNode *def = find_struct_def_codegen(ctx, mangled);
            int skip = 0;
            if (def)
            {
                if (def->type == NODE_STRUCT && def->strct.is_template)
                {
                    skip = 1;
                }
                else if (def->type == NODE_ENUM && def->enm.is_template)
                {
                    skip = 1;
                }
            }
            else
            {
                char *buf = strip_template_suffix(strct);
                if (buf)
                {
                    def = find_struct_def_codegen(ctx, buf);
                    if (def && def->strct.is_template)
                    {
                        skip = 1;
                    }
                    free(buf);
                }
            }
            if (mangled)
            {
                free(mangled);
            }
            if (skip)
            {
                ref = ref->next;
                continue;
            }

            // Check duplication
            int dup = 0;
            for (int i = 0; i < count; i++)
            {
                if (strcmp(emitted[i].trait, trait) == 0 && strcmp(emitted[i].strct, strct) == 0)
                {
                    dup = 1;
                    break;
                }
            }
            if (dup)
            {
                ref = ref->next;
                continue;
            }

            emitted[count].trait = trait;
            emitted[count].strct = strct;
            count++;

            if (0 == strcmp(trait, "Copy"))
            {
                // Marker trait, no runtime vtable needed
                ref = ref->next;
                continue;
            }

            fprintf(out, "%s_VTable %s_%s_VTable = {", trait, strct, trait);

            ASTNode *m = node->impl_trait.methods;
            while (m)
            {
                // Calculate expected prefix: Struct__Trait_
                char prefix[256];
                sprintf(prefix, "%s__%s_", strct, trait);
                const char *orig = m->func.name;
                if (strncmp(orig, prefix, strlen(prefix)) == 0)
                {
                    orig += strlen(prefix);
                }
                else
                {
                    // Fallback if mangling schema differs (shouldn't happen)
                    orig = parse_original_method_name(m->func.name);
                }

                fprintf(out, ".%s = (__typeof__(((%s_VTable*)0)->%s))%s__%s_%s", orig, trait, orig,
                        strct, trait, orig);
                if (m->next)
                {
                    fprintf(out, ", ");
                }
                m = m->next;
            }
            fprintf(out, "};\n");
        }
        ref = ref->next;
    }
}

// Emit test functions and runner. Returns number of tests emitted.
int emit_tests_and_runner(ParserContext *ctx, ASTNode *node, FILE *out)
{
    ASTNode *cur = node;
    int test_count = 0;
    while (cur)
    {
        if (cur->type == NODE_TEST)
        {
            fprintf(out, "static void _z_test_%d() {\n", test_count);
            int saved = defer_count;
            codegen_walker(ctx, cur->test_stmt.body, out);
            // Run defers
            for (int i = defer_count - 1; i >= saved; i--)
            {
                codegen_node_single(ctx, defer_stack[i], out);
            }
            defer_count = saved;
            fprintf(out, "}\n");
            test_count++;
        }
        cur = cur->next;
    }
    if (test_count > 0)
    {
        fprintf(out, "\nvoid _z_run_tests() {\n");
        for (int i = 0; i < test_count; i++)
        {
            fprintf(out, "    _z_test_%d();\n", i);
        }
        fprintf(out, "}\n\n");
    }
    return test_count;
}

// Emit type definitions-
void print_type_defs(ParserContext *ctx, FILE *out, ASTNode *nodes)
{
    if (!g_config.is_freestanding)
    {
        fprintf(out, "typedef char* string;\n");

        fprintf(out, "typedef struct { void **data; int len; int cap; } Vec;\n");
        fprintf(out, "#define Vec_new() (Vec){.data=0, .len=0, .cap=0}\n");

        if (g_config.use_cpp)
        {
            fprintf(out,
                    "void _z_vec_push(Vec *v, void *item) { if(v->len >= v->cap) { "
                    "v->cap = v->cap?v->cap*2:8; "
                    "v->data = static_cast<void**>(realloc(v->data, v->cap * sizeof(void*))); } "
                    "v->data[v->len++] = item; }\n");
            fprintf(out, "static inline Vec _z_make_vec(int count, ...) { Vec v = {0}; v.cap = "
                         "count > 8 ? "
                         "count : 8; v.data = static_cast<void**>(malloc(v.cap * sizeof(void*))); "
                         "v.len = 0; va_list "
                         "args; "
                         "va_start(args, count); for(int i=0; i<count; i++) { v.data[v.len++] = "
                         "va_arg(args, void*); } va_end(args); return v; }\n");
        }
        else
        {
            fprintf(out, "void _z_vec_push(Vec *v, void *item) { if(v->len >= v->cap) { "
                         "v->cap = v->cap?v->cap*2:8; "
                         "v->data = z_realloc(v->data, v->cap * sizeof(void*)); } "
                         "v->data[v->len++] = item; }\n");
            fprintf(out, "static inline Vec _z_make_vec(int count, ...) { Vec v = {0}; v.cap = "
                         "count > 8 ? "
                         "count : 8; v.data = z_malloc(v.cap * sizeof(void*)); v.len = 0; va_list "
                         "args; "
                         "va_start(args, count); for(int i=0; i<count; i++) { v.data[v.len++] = "
                         "va_arg(args, void*); } va_end(args); return v; }\n");
        }
        fprintf(out, "#define Vec_push(v, i) _z_vec_push(&(v), (void*)(long)(i))\n");

        fprintf(out, "#define _z_check_bounds(index, limit) ({ ZC_AUTO_INIT(_i, index); if(_i < 0 "
                     "|| _i >= (limit)) { fprintf(stderr, \"Index out of bounds: %%ld (limit "
                     "%%d)\\n\", (long)_i, (int)(limit)); exit(1); } _i; })\n");
    }
    else
    {
        // We might need to change this later. So TODO.
        fprintf(out, "#define _z_check_bounds(index, limit) ({ ZC_AUTO _i = "
                     "(index); if(_i < 0 "
                     "|| _i >= (limit)) { z_panic(\"index out of bounds\"); } _i; })\n");
    }

    SliceType *c = ctx->used_slices;
    while (c)
    {
        fprintf(out,
                "typedef struct Slice_%s Slice_%s;\nstruct Slice_%s { %s *data; "
                "int len; int cap; };\n",
                c->name, c->name, c->name, c->name);
        c = c->next;
    }

    TupleType *t = ctx->used_tuples;
    while (t)
    {
        fprintf(out, "typedef struct Tuple_%s Tuple_%s;\nstruct Tuple_%s { ", t->sig, t->sig,
                t->sig);
        char *s = xstrdup(t->sig);
        char *current = s;
        char *next_sep = strstr(current, "__");
        int i = 0;
        while (current)
        {
            if (next_sep)
            {
                *next_sep = 0;
                fprintf(out, "%s v%d; ", current, i++);
                current = next_sep + 2;
                next_sep = strstr(current, "__");
            }
            else
            {
                fprintf(out, "%s v%d; ", current, i++);
                break;
            }
        }
        free(s);
        fprintf(out, "};\n");
        t = t->next;
    }
    fprintf(out, "\n");

    // FIRST: Emit typedefs for ALL structs and enums in the current compilation
    // unit (local definitions)
    ASTNode *local = nodes;
    while (local)
    {
        if (local->type == NODE_STRUCT && !local->strct.is_template)
        {
            const char *keyword = local->strct.is_union ? "union" : "struct";
            fprintf(out, "typedef %s %s %s;\n", keyword, local->strct.name, local->strct.name);
        }
        if (local->type == NODE_ENUM && !local->enm.is_template)
        {
            fprintf(out, "typedef struct %s %s;\n", local->enm.name, local->enm.name);
        }
        local = local->next;
    }
}
