
#include "plugin_manager.h"
#include "compat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef zc_dl_handle PluginHandle;

typedef struct PluginNode
{
    ZPlugin *plugin;
    PluginHandle handle;
    struct PluginNode *next;
} PluginNode;

static PluginNode *head = NULL;

void zptr_plugin_mgr_init(void)
{
    head = NULL;
}

void zptr_register_plugin(ZPlugin *plugin)
{
    if (!plugin)
        return;

    if (zptr_find_plugin(plugin->name))
        return;

    PluginNode *node = malloc(sizeof(PluginNode));
    node->plugin = plugin;
    node->handle = NULL;
    node->next = head;
    head = node;
}

int zptr_load_plugin(const char *path)
{
    PluginHandle handle = zc_dlopen(path, RTLD_LAZY);
    if (!handle)
    {
        fprintf(stderr, "Failed to load plugin '%s': %s\n", path, zc_dlerror());
        return 0;
    }

    ZPluginInitFn init_fn = (ZPluginInitFn)zc_dlsym(handle, "z_plugin_init");
    if (!init_fn)
    {
        fprintf(stderr, "Plugin '%s' missing 'z_plugin_init' symbol\n", path);
        zc_dlclose(handle);
        return 0;
    }

    ZPlugin *plugin = init_fn();
    if (!plugin)
    {
        fprintf(stderr, "Plugin '%s' init returned NULL\n", path);
        zc_dlclose(handle);
        return 0;
    }

    PluginNode *node = malloc(sizeof(PluginNode));
    node->plugin = plugin;
    node->handle = handle;
    node->next = head;
    head = node;

    return 1;
}

ZPlugin *zptr_find_plugin(const char *name)
{
    PluginNode *curr = head;
    while (curr)
    {
        if (strcmp(curr->plugin->name, name) == 0)
            return curr->plugin;
        curr = curr->next;
    }
    return NULL;
}

void zptr_plugin_mgr_cleanup(void)
{
    PluginNode *curr = head;
    while (curr)
    {
        PluginNode *next = curr->next;
        if (curr->handle)
            zc_dlclose(curr->handle);
        free(curr);
        curr = next;
    }
    head = NULL;
}
