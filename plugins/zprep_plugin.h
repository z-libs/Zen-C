
#ifndef ZPREP_PLUGIN_H
#define ZPREP_PLUGIN_H

#include <stddef.h>
#include <stdio.h>

// The Host provides this API to the Plugin.
typedef struct
{
    // Context Information (Where are we?).
    const char *filename;
    int current_line;
    FILE *out;       // Inline output (expression context)
    FILE *hoist_out; // Hoisted output (file scope / top level)
} ZApi;

// The Plugin Function Signature.
// Returns void. All output is done via 'api'.
typedef void (*ZPluginTranspileFn)(const char *input_body, const ZApi *api);

typedef struct
{
    char name[32];
    ZPluginTranspileFn fn;
} ZPlugin;

typedef ZPlugin *(*ZPluginInitFn)(void);

#ifdef _WIN32
#define ZC_PLUGIN_API __declspec(dllexport)
#else
#define ZC_PLUGIN_API
#endif

#endif
