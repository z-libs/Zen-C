# Zen C Plugin System Guide

Zen C offers a plugin system that allows you to extend the language syntax and transpilation process. Plugins act as compile-time hooks that can parse arbitrary text blocks and generate C code.

## Quick Start

### Using a Plugin

To use a plugin, import it using the `import plugin` syntax:

```zc
import plugin "regex" as re
// or simple: import plugin "regex" (uses "regex" as identifier)

fn main() {
    let valid = re! { ^[a-z]+$ };
}
```

The syntax `alias! { ... }` invokes the plugin. The content inside the braces is passed as a raw string to the plugin's transpiler function.

## Creating a Plugin

Plugins are written in C and compiled into shared objects (or built-in to the compiler).

### API Reference (`zprep_plugin.h`)

The core API is simple. A plugin exposes a `ZPlugin` struct containing its name and a transpiler function.

```c
typedef struct {
    // -> Context information
    const char *filename;   // Current file being compiled
    int current_line;       // Line number of the invocation
    FILE *out;              // Output stream (injects code at call site)
    FILE *hoist_out;        // Hoisted output (injects code at file scope)
} ZApi;

// -> The transpiler function
// input_body: The raw text inside the plugin block { ... }
// api: Access to transpiler context and output streams
typedef void (*ZPluginTranspileFn)(const char *input_body, const ZApi *api);

typedef struct {
    char name[32];
    ZPluginTranspileFn fn;
} ZPlugin;
```

### Writing a Brainfuck Plugin

Let's implement a plugin that compiles Brainfuck code directly into C logic at compile time.

#### 1. Include the API

```c
#include "zprep_plugin.h"
```

#### 2. Implement the transpiler

The transpiler function reads the Brainfuck source code and writes equivalent C code to `api->out`.

```c
void bf_transpile(const char *input_body, const ZApi *api)
{
    FILE *out = api->out;
    
    // Initialize tape and pointer in a local block.
    fprintf(out, "{\n");
    fprintf(out, "    static unsigned char tape[30000] = {0};\n");
    fprintf(out, "    unsigned char *ptr = tape;\n");

    const char *c = input_body;
    while (*c)
    {
        switch (*c)
        {
        case '>': fprintf(out, "    ++ptr;\n"); break;
        case '<': fprintf(out, "    --ptr;\n"); break;
        case '+': fprintf(out, "    ++*ptr;\n"); break;
        case '-': fprintf(out, "    --*ptr;\n"); break;
        case '.': fprintf(out, "    putchar(*ptr);\n"); break;
        case ',': fprintf(out, "    *ptr = getchar();\n"); break;
        case '[': fprintf(out, "    while (*ptr) {\n"); break;
        case ']': fprintf(out, "    }\n"); break;
        }
        c++;
    }
    fprintf(out, "}\n");
}
```

#### 3. Register the Plugin

For dynamic plugins, you must export a `z_plugin_init` function that returns a pointer to your `ZPlugin` struct.

```c
ZPlugin brainfuck_plugin = {
    .name = "brainfuck",
    .fn = bf_transpile
};

// Entry point for the dynamic loader
ZPlugin *z_plugin_init(void)
{
    return &brainfuck_plugin;
}
```

### Building the Plugin

Compile your plugin as a shared object (`.so`). You'll need access to `zprep_plugin.h`.

```bash
# If building from source repository:
gcc -shared -fPIC -o brainfuck.so brainfuck.c -I./plugins

# If Zen C is installed system-wide:
gcc -shared -fPIC -o brainfuck.so brainfuck.c -I/usr/local/include/zenc
```

### Using the Plugin

You can import plugins using relative paths or system-wide lookups:

```zc
// Relative import (finds .so relative to this file)
import plugin "./brainfuck.so" as bf

// System import (finds .so in CWD or library path)
import plugin "brainfuck.so" as bf2

fn main() {
    bf! {
        ++++++++++[>+++++++>++++++++++>+++<<<-]>++.
    }
}
```

### Output example

When you write:

```zc
bf! {
    ++++++++++[>+++++++>++++++++++>+++<<<-]>++.
}
```

The plugin generates:

```c
{
    static unsigned char tape[30000] = {0};
    unsigned char *ptr = tape;
    ++*ptr;
    ++*ptr;
    // ... (logic for + and [ )
    while (*ptr) {
        ++ptr;
        // ...
    }
    ptr++;
    // ...
    putchar(*ptr);
}
```

This C code is then compiled by the backend, resulting in a highly optimized native binary from your DSL!
