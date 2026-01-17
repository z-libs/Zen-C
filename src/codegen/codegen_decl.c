
#include "../ast/ast.h"
#include "../parser/parser.h"
#include "../zprep.h"
#include "codegen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Emit C preamble with standard includes and type definitions.
void emit_preamble(ParserContext *ctx, FILE *out)
{
    if (g_config.is_freestanding)
    {
        // Freestanding preamble.
        // It actually needs more work, but yk.
        fputs("#include <stddef.h>\n#include <stdint.h>\n#include "
              "<stdbool.h>\n#include <stdarg.h>\n",
              out);
        fputs("#ifdef __TINYC__\n#define __auto_type __typeof__\n#endif\n", out);
        fputs("typedef size_t usize;\ntypedef char* string;\n", out);
        fputs("#define U0 void\n#define I8 int8_t\n#define U8 uint8_t\n#define I16 "
              "int16_t\n#define U16 uint16_t\n",
              out);
        fputs("#define I32 int32_t\n#define U32 uint32_t\n#define I64 "
              "int64_t\n#define U64 "
              "uint64_t\n",
              out);
        fputs("#define F32 float\n#define F64 double\n", out);
        fputs("#define _z_str(x) _Generic((x), _Bool: \"%d\", char: \"%c\", "
              "signed char: \"%c\", unsigned char: \"%u\", short: \"%d\", "
              "unsigned short: \"%u\", int: \"%d\", unsigned int: \"%u\", "
              "long: \"%ld\", unsigned long: \"%lu\", long long: \"%lld\", "
              "unsigned long long: \"%llu\", float: \"%f\", double: \"%f\", "
              "char*: \"%s\", void*: \"%p\")\n",
              out);
        fputs("typedef struct { void *func; void *ctx; } z_closure_T;\n", out);

        fputs("__attribute__((weak)) void* z_malloc(usize sz) { return NULL; }\n", out);
        fputs("__attribute__((weak)) void* z_realloc(void* ptr, usize sz) { return "
              "NULL; }\n",
              out);
        fputs("__attribute__((weak)) void z_free(void* ptr) { }\n", out);
        fputs("__attribute__((weak)) void z_print(const char* fmt, ...) { }\n", out);
        fputs("__attribute__((weak)) void z_panic(const char* msg) { while(1); }\n", out);
    }
    else
    {
        // Standard hosted preamble.
        fputs("#include <stdio.h>\n#include <stdlib.h>\n#include "
              "<stddef.h>\n#include <string.h>\n",
              out);
        fputs("#include <stdarg.h>\n#include <stdint.h>\n#include <stdbool.h>\n", out);
        fputs("#include \"z_platform.h\"\n", out); // Cross-platform abstraction
        fputs("#ifdef __TINYC__\n#define __auto_type __typeof__\n#endif\n", out);
        fputs("typedef size_t usize;\ntypedef char* string;\n", out);
        if (ctx->has_async)
        {
            fputs("#include <pthread.h>\n", out);
            // Async typedef is already defined in z_platform.h
        }
        fputs("typedef struct { void *func; void *ctx; } z_closure_T;\n", out);
        fputs("#define U0 void\n#define I8 int8_t\n#define U8 uint8_t\n#define I16 "
              "int16_t\n#define U16 uint16_t\n",
              out);
        fputs("#define I32 int32_t\n#define U32 uint32_t\n#define I64 "
              "int64_t\n#define U64 "
              "uint64_t\n",
              out);
        fputs("#define F32 float\n#define F64 double\n", out);
        fputs("#define _z_str(x) _Generic((x), _Bool: \"%d\", char: \"%c\", "
              "signed char: \"%c\", unsigned char: \"%u\", short: \"%d\", "
              "unsigned short: \"%u\", int: \"%d\", unsigned int: \"%u\", "
              "long: \"%ld\", unsigned long: \"%lu\", long long: \"%lld\", "
              "unsigned long long: \"%llu\", float: \"%f\", double: \"%f\", "
              "char*: \"%s\", void*: \"%p\")\n",
              out);

        // Memory Mapping.
        fputs("#define z_malloc malloc\n#define z_realloc realloc\n#define z_free "
              "free\n#define "
              "z_print printf\n",
              out);
        fputs("void z_panic(const char* msg) { fprintf(stderr, \"Panic: %s\\n\", "
              "msg); exit(1); }\n",
              out);

        fputs("void _z_autofree_impl(void *p) { void **pp = (void**)p; if(*pp) { "
              "z_free(*pp); *pp "
              "= NULL; } }\n",
              out);
        fputs("#define assert(cond, ...) if (!(cond)) { fprintf(stderr, "
              "\"Assertion failed: \" "
              "__VA_ARGS__); exit(1); }\n",
              out);
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
        fputs("int _z_scan_helper(const char *fmt, ...) { char *l = "
              "_z_readln_raw(); if(!l) return "
              "0; va_list ap; va_start(ap, fmt); int r = vsscanf(l, fmt, ap); "
              "va_end(ap); "
              "z_free(l); return r; }\n",
              out);

        // REPL helpers: suppress/restore stdout (using z_platform.h functions).
        fputs("int _z_orig_stdout = -1;\n", out);
        fputs("void _z_suppress_stdout() {\n", out);
        fputs("    if (_z_orig_stdout == -1) _z_orig_stdout = z_suppress_stdout();\n", out);
        fputs("}\n", out);
        fputs("void _z_restore_stdout() {\n", out);
        fputs("    if (_z_orig_stdout != -1) {\n", out);
        fputs("        z_restore_stdout(_z_orig_stdout);\n", out);
        fputs("        _z_orig_stdout = -1;\n", out);
        fputs("    }\n", out);
        fputs("}\n", out);
    }
}

// Emit includes and type aliases.
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
        else if (node->type == NODE_TYPE_ALIAS)
        {
            fprintf(out, "typedef %s %s;\n", node->type_alias.original_type,
                    node->type_alias.alias);
        }
        node = node->next;
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
                fprintf(out, "    %s %s;\n", node->lambda.captured_types[i],
                        node->lambda.captured_vars[i]);
            }
            fprintf(out, "};\n");
        }

        fprintf(out, "%s _lambda_%d(void* _ctx", node->lambda.return_type, node->lambda.lambda_id);

        for (int i = 0; i < node->lambda.num_params; i++)
        {
            fprintf(out, ", %s %s", node->lambda.param_types[i], node->lambda.param_names[i]);
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
            else if (node->strct.align)
            {
                fprintf(out, " __attribute__((aligned(%d)))", node->strct.align);
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

// Emit trait definitions.
void emit_trait_defs(ASTNode *node, FILE *out)
{
    while (node)
    {
        if (node->type == NODE_TRAIT)
        {
            fprintf(out, "typedef struct %s_VTable {\n", node->trait.name);
            ASTNode *m = node->trait.methods;
            while (m)
            {
                fprintf(out, "    %s (*%s)(", m->func.ret_type,
                        parse_original_method_name(m->func.name));
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
                    fprintf(out, "%s", m->func.args);
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
                fprintf(out, "%s %s__%s(%s* self", m->func.ret_type, node->trait.name, orig,
                        node->trait.name);

                int has_self = (m->func.args && strstr(m->func.args, "self"));
                if (m->func.args)
                {
                    if (has_self)
                    {
                        char *comma = strchr(m->func.args, ',');
                        if (comma)
                        {
                            fprintf(out, ", %s", comma + 1);
                        }
                    }
                    else
                    {
                        fprintf(out, ", %s", m->func.args);
                    }
                }
                fprintf(out, ") {\n");

                fprintf(out, "    return self->vtable->%s(self->self", orig);

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
                fprintf(out, ");\n}\n");

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
void emit_protos(ASTNode *node, FILE *out)
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
                emit_func_signature(out, f, NULL);
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
                char *lt = strchr(sname, '<');
                if (lt)
                {
                    int len = lt - sname;
                    char *buf = xmalloc(len + 1);
                    strncpy(buf, sname, len);
                    buf[len] = 0;
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
                    emit_func_signature(out, m, proto);
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
                char *lt = strchr(sname, '<');
                if (lt)
                {
                    int len = lt - sname;
                    char *buf = xmalloc(len + 1);
                    strncpy(buf, sname, len);
                    buf[len] = 0;
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

            if (strcmp(f->impl_trait.trait_name, "Drop") == 0)
            {
                fprintf(out, "void %s__Drop_glue(%s *self);\n", sname, sname);
            }

            ASTNode *m = f->impl_trait.methods;
            while (m)
            {
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
            // RAII: Emit glue prototype
            if (strcmp(f->impl_trait.trait_name, "Drop") == 0)
            {
                char *tname = f->impl_trait.target_type;
                fprintf(out, "void %s_Drop_glue(%s *self);\n", tname, tname);
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
                char *lt = strchr(strct, '<');
                if (lt)
                {
                    int len = lt - strct;
                    char *buf = xmalloc(len + 1);
                    strncpy(buf, strct, len);
                    buf[len] = 0;
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

            fprintf(out, "%s_VTable %s_%s_VTable = {", trait, strct, trait);

            ASTNode *m = node->impl_trait.methods;
            while (m)
            {
                const char *orig = parse_original_method_name(m->func.name);
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

// Emit test functions and runner
void emit_tests_and_runner(ParserContext *ctx, ASTNode *node, FILE *out)
{
    ASTNode *cur = node;
    int test_count = 0;
    while (cur)
    {
        if (cur->type == NODE_TEST)
        {
            fprintf(out, "static void _z_test_%d() {\n", test_count);
            codegen_walker(ctx, cur->test_stmt.body, out);
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
    else
    {
        fprintf(out, "void _z_run_tests() {}\n");
    }
}

// Emit type definitions-
void print_type_defs(ParserContext *ctx, FILE *out, ASTNode *nodes)
{
    if (!g_config.is_freestanding)
    {
        fprintf(out, "typedef char* string;\n");
    }

    fprintf(out, "typedef struct { void **data; int len; int cap; } Vec;\n");
    fprintf(out, "#define Vec_new() (Vec){.data=0, .len=0, .cap=0}\n");
    fprintf(out, "void _z_vec_push(Vec *v, void *item) { if(v->len >= v->cap) { "
                 "v->cap = v->cap?v->cap*2:8; "
                 "v->data = z_realloc(v->data, v->cap * sizeof(void*)); } "
                 "v->data[v->len++] = item; }\n");
    fprintf(out, "#define Vec_push(v, i) _z_vec_push(&(v), (void*)(long)(i))\n");
    fprintf(out, "static inline Vec _z_make_vec(int count, ...) { Vec v = {0}; v.cap = "
                 "count > 8 ? "
                 "count : 8; v.data = z_malloc(v.cap * sizeof(void*)); v.len = 0; va_list "
                 "args; "
                 "va_start(args, count); for(int i=0; i<count; i++) { v.data[v.len++] = "
                 "va_arg(args, void*); } va_end(args); return v; }\n");

    if (g_config.is_freestanding)
    {
        fprintf(out, "#define _z_check_bounds(index, limit) ({ __auto_type _i = "
                     "(index); if(_i < 0 "
                     "|| _i >= (limit)) { z_panic(\"index out of bounds\"); } _i; })\n");
    }
    else
    {
        fprintf(out, "#define _z_check_bounds(index, limit) ({ __auto_type _i = "
                     "(index); if(_i < 0 "
                     "|| _i >= (limit)) { fprintf(stderr, \"Index out of bounds: "
                     "%%ld (limit "
                     "%%d)\\n\", (long)_i, (int)(limit)); exit(1); } _i; })\n");
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
        char *p = strtok(s, "_");
        int i = 0;
        while (p)
        {
            fprintf(out, "%s v%d; ", p, i++);
            p = strtok(NULL, "_");
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

    // THEN: Emit typedefs for instantiated generics
    Instantiation *i = ctx->instantiations;
    while (i)
    {
        if (i->struct_node->type == NODE_RAW_STMT)
        {
            fprintf(out, "%s\n", i->struct_node->raw_stmt.content);
        }
        else
        {
            fprintf(out, "typedef struct %s %s;\n", i->struct_node->strct.name,
                    i->struct_node->strct.name);
            codegen_node(ctx, i->struct_node, out);
        }
        i = i->next;
    }

    StructRef *sr = ctx->parsed_structs_list;
    while (sr)
    {
        if (sr->node && sr->node->type == NODE_STRUCT && !sr->node->strct.is_template)
        {
            const char *keyword = sr->node->strct.is_union ? "union" : "struct";
            fprintf(out, "typedef %s %s %s;\n", keyword, sr->node->strct.name,
                    sr->node->strct.name);
        }

        if (sr->node && sr->node->type == NODE_ENUM && !sr->node->enm.is_template)
        {
            fprintf(out, "typedef struct %s %s;\n", sr->node->enm.name, sr->node->enm.name);
        }
        sr = sr->next;
    }

    // Also check instantiated_structs list.
    ASTNode *inst_s = ctx->instantiated_structs;
    while (inst_s)
    {
        if (inst_s->type == NODE_STRUCT && !inst_s->strct.is_template)
        {
            const char *keyword = inst_s->strct.is_union ? "union" : "struct";
            fprintf(out, "typedef %s %s %s;\n", keyword, inst_s->strct.name, inst_s->strct.name);
        }

        if (inst_s->type == NODE_ENUM && !inst_s->enm.is_template)
        {
            fprintf(out, "typedef struct %s %s;\n", inst_s->enm.name, inst_s->enm.name);
        }
        inst_s = inst_s->next;
    }
}
