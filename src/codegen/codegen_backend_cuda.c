// SPDX-License-Identifier: MIT
#include "codegen_backend.h"
#include "codegen.h"

static const CodegenBackend cuda_backend = {
    .name = "cuda",
    .extension = ".cu",
    .emit_program = codegen_c_program,
    .emit_preamble = emit_preamble,
    .needs_cc = 1,
};

void codegen_register_cuda_backend(void)
{
    codegen_register_backend(&cuda_backend);
}
