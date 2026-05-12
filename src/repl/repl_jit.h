/**
#ifndef ZC_ALLOW_INTERNAL
#error "repl/repl_jit.h is internal to Zen C. Include the appropriate public header instead."
#endif

 * @file repl_jit.h
 * @brief JIT execution module using LibTCC.
 */

#ifndef REPL_JIT_H
#define REPL_JIT_H

#if ZC_HAS_JIT
#include <libtcc.h>
#endif

typedef struct CompilerConfig CompilerConfig;

/**
 * @brief Executes C code in-process using LibTCC.
 *
 * @param c_code The C source code string to compile and run.
 * @return int 0 on success, non-zero on error.
 */
int repl_jit_execute(const char *c_code, CompilerConfig *cfg);

#endif /* REPL_JIT_H */
