// SPDX-License-Identifier: MIT

#ifndef ZC_ALLOW_INTERNAL
#error "driver/driver.h is internal to Zen C. Include the appropriate public header instead."
#endif

#ifndef DRIVER_H
#define DRIVER_H

#include "../compiler.h"

/**
 * @brief Orchestrates the compilation process based on the provided configuration.
 *
 * @param compiler The compiler instance to use.
 * @return int 0 on success, non-zero on failure.
 */
int driver_run(ZenCompiler *compiler);

/**
 * @brief Performs the actual compilation steps (parse, analyze, codegen).
 *
 * @param compiler The compiler instance.
 * @return int 0 on success, non-zero on failure.
 */
int driver_compile(ZenCompiler *compiler);

/**
 * @brief Handles the transpilation process.
 *
 * @param compiler The compiler instance.
 * @return int 0 on success, non-zero on failure.
 */
int driver_transpile(ZenCompiler *compiler);

#endif // DRIVER_H
