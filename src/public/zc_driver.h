#ifndef ZC_DRIVER_H
#define ZC_DRIVER_H

/**
 * @file zc_driver.h
 * @brief Public API for libzc-driver: orchestration of the full pipeline.
 */

struct ZenCompiler;

/**
 * @brief Run the full compilation pipeline (parse → analyze → codegen → C compiler).
 *
 * @param compiler Initialized compiler instance with config set.
 * @return 0 on success, non-zero on failure.
 */
int driver_run(struct ZenCompiler *compiler);

/**
 * @brief Run the compilation steps only (parse → analyze → codegen), no C compiler.
 *
 * @param compiler Initialized compiler instance.
 * @return 0 on success, non-zero on failure.
 */
int driver_compile(struct ZenCompiler *compiler);

#endif // ZC_DRIVER_H
