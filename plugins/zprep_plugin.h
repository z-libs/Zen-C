
#ifndef ZPREP_PLUGIN_H
#define ZPREP_PLUGIN_H

#include <stddef.h>
#include <stdio.h>

/**
 * @brief Host API provided to plugins.
 * 
 * Plugins use this structure to interact with the compiler/codegen environment.
 */
typedef struct
{
    // Context Information (Where are we?).
    const char *filename;   ///< Current file name being processed.
    int current_line;       ///< Current line number.
    FILE *out;              ///< Inline output stream (replaces the macro call).
    FILE *hoist_out;        ///< Hoisted output stream (writes to file scope/header).
} ZApi;

/**
 * @brief The Plugin Function Signature.
 * 
 * Plugins must implement a function with this signature to handle transpilation.
 * 
 * @param input_body The raw text content inside the plugin call.
 * @param api Pointer to the host API structure.
 */
typedef void (*ZPluginTranspileFn)(const char *input_body, const ZApi *api);

/**
 * @brief Plugin definition structure.
 */
typedef struct
{
    char name[32];          ///< Name of the plugin.
    ZPluginTranspileFn fn;  ///< Pointer to the transpilation function.
} ZPlugin;

/**
 * @brief Signature for the plugin entry point.
 * 
 * Dynamic libraries must export a function named `z_init` matching this signature.
 */
typedef ZPlugin *(*ZPluginInitFn)(void);

#endif
