// SPDX-License-Identifier: MIT
#include "codegen_backend.h"
#include "codegen.h"

static const CodegenBackend cpp_backend = {
    .name = "cpp",
    .extension = ".cpp",
    .emit_program = codegen_c_program,
    .emit_preamble = emit_preamble,
    .needs_cc = 1,
};

void codegen_register_cpp_backend(void)
{
    codegen_register_backend(&cpp_backend);
}
