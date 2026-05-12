// SPDX-License-Identifier: MIT
#include "plugin_manager.h"
#include <string.h>
#ifdef ZC_STATIC_PLUGINS
extern ZPlugin *z_plugin_init_befunge(void);
extern ZPlugin *z_plugin_init_brainfuck(void);
extern ZPlugin *z_plugin_init_forth(void);
extern ZPlugin *z_plugin_init_lisp(void);
extern ZPlugin *z_plugin_init_sql(void);
ZPlugin *zptr_get_static_plugin(const char *name)
{
    if (strcmp(name, "befunge") == 0 || strcmp(name, "plugins/befunge") == 0)
    {
        return z_plugin_init_befunge();
    }
    if (strcmp(name, "brainfuck") == 0 || strcmp(name, "plugins/brainfuck") == 0)
    {
        return z_plugin_init_brainfuck();
    }
    if (strcmp(name, "forth") == 0 || strcmp(name, "plugins/forth") == 0)
    {
        return z_plugin_init_forth();
    }
    if (strcmp(name, "lisp") == 0 || strcmp(name, "plugins/lisp") == 0)
    {
        return z_plugin_init_lisp();
    }
    if (strcmp(name, "sql") == 0 || strcmp(name, "plugins/sql") == 0)
    {
        return z_plugin_init_sql();
    }
    return NULL;
}
#else
ZPlugin *zptr_get_static_plugin(const char *name)
{
    (void)name;
    return NULL;
}
#endif
