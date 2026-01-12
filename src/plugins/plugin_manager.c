#include "plugin_manager.h"
#ifndef _WIN32
#include <dlfcn.h>
#else
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Linked list node for plugins.
typedef struct PluginNode
{
    ZPlugin *plugin;
    void *handle; // dlopen handle (NULL for built-ins).
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
    {
        return;
    }

    if (zptr_find_plugin(plugin->name))
    {
        return;
    }

    PluginNode *node = malloc(sizeof(PluginNode));
    node->plugin = plugin;
    node->handle = NULL;
    node->next = head;
    head = node;
}

ZPlugin *zptr_load_plugin(const char *path)
{
#ifndef _WIN32
    void *handle = dlopen(path, RTLD_LAZY);
#else
    void *handle = LoadLibraryA(path);
#endif
    if (!handle)
    {
        return NULL;
    }

    ZPluginInitFn init_fn = (ZPluginInitFn)dlsym(handle, "z_plugin_init");
    if (!init_fn)
    {
        fprintf(stderr, "Plugin '%s' missing 'z_plugin_init' symbol\n", path);
        dlclose(handle);
        return NULL;
    }

    ZPlugin *plugin = init_fn();
    if (!plugin)
    {
        fprintf(stderr, "Plugin '%s' init returned NULL\n", path);
        dlclose(handle);
        return NULL;
    }

    // Register
    PluginNode *node = malloc(sizeof(PluginNode));
    node->plugin = plugin;
    node->handle = handle;
    node->next = head;
    head = node;

    return plugin;
}

ZPlugin *zptr_find_plugin(const char *name)
{
    PluginNode *curr = head;
    while (curr)
    {
        if (strcmp(curr->plugin->name, name) == 0)
        {
            return curr->plugin;
        }
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
        {
            dlclose(curr->handle);
        }
        free(curr);
        curr = next;
    }
    head = NULL;
}
