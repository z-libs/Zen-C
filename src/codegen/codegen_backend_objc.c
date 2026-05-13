// SPDX-License-Identifier: MIT
#include "codegen_backend.h"
#include "codegen.h"

static const CodegenBackend objc_backend = {
    .name = "objc",
    .extension = ".m",
    .emit_program = codegen_c_program,
    .emit_preamble = emit_preamble,
    .needs_cc = 1,
};

void codegen_register_objc_backend(void)
{
    codegen_register_backend(&objc_backend);
}
