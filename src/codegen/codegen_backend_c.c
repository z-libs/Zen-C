#include "codegen_backend.h"
#include "codegen.h"
#include "../parser/parser.h"
#include <string.h>

#define MAX_BACKENDS 8

static const CodegenBackend *registered_backends[MAX_BACKENDS];
static int backend_count = 0;

static const CodegenBackend c_backend = {
    .name = "c",
    .extension = ".c",
    .emit_program = codegen_c_program,
    .emit_preamble = emit_preamble,
};

void codegen_register_backend(const CodegenBackend *backend)
{
    if (!backend || !backend->name || backend_count >= MAX_BACKENDS)
    {
        return;
    }
    registered_backends[backend_count++] = backend;
}

const CodegenBackend *codegen_get_backend(const char *name)
{
    if (!name)
    {
        return &c_backend;
    }
    for (int i = 0; i < backend_count; i++)
    {
        if (strcmp(registered_backends[i]->name, name) == 0)
        {
            return registered_backends[i];
        }
    }
    return &c_backend;
}

void codegen_init_backends(void)
{
    if (backend_count > 0)
    {
        return;
    }
    codegen_register_backend(&c_backend);
}

void codegen_node(ParserContext *ctx, ASTNode *root)
{
    const CodegenBackend *be = codegen_get_backend(ctx->config->backend_name);
    if (be && be->emit_program)
    {
        be->emit_program(ctx, root);
    }
}
