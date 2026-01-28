#include "plugin_manager.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
    #include <windows.h>

    typedef HMODULE ZcPluginHandle;

    static ZcPluginHandle zc_dlopen(const char *path) {
        return LoadLibraryA(path);
    }

    static void *zc_dlsym(ZcPluginHandle handle, const char *symbol) {
        return (void *)GetProcAddress(handle, symbol);
    }

    static void zc_dlclose(ZcPluginHandle handle) {
        if (handle) FreeLibrary(handle);
    }

#else
    #include <dlfcn.h>

    typedef void *ZcPluginHandle;

    static ZcPluginHandle zc_dlopen(const char *path) {
        return dlopen(path, RTLD_LAZY);
    }

    static void *zc_dlsym(ZcPluginHandle handle, const char *symbol) {
        return dlsym(handle, symbol);
    }

    static void zc_dlclose(ZcPluginHandle handle) {
        if (handle) dlclose(handle);
    }
#endif

// Linked list node for plugins.
typedef struct PluginNode
{
    ZPlugin *plugin;
    ZcPluginHandle handle; // dynamic library handle (NULL for built-ins).
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

    PluginNode *node = (PluginNode *)malloc(sizeof(PluginNode));
    if (!node) return;

    node->plugin = plugin;
    node->handle = (ZcPluginHandle)0;
    node->next = head;
    head = node;
}

ZPlugin *zptr_load_plugin(const char *path)
{
    ZcPluginHandle handle = zc_dlopen(path);
    if (!handle)
    {
#if defined(_WIN32)
        // Optional: nicer error
        // fprintf(stderr, "Failed to load plugin '%s' (LoadLibraryA failed)\n", path);
#else
        // fprintf(stderr, "Failed to load plugin '%s': %s\n", path, dlerror());
#endif
        return NULL;
    }

    ZPluginInitFn init_fn = (ZPluginInitFn)zc_dlsym(handle, "z_plugin_init");
    if (!init_fn)
    {
        fprintf(stderr, "Plugin '%s' missing 'z_plugin_init' symbol\n", path);
        zc_dlclose(handle);
        return NULL;
    }

    ZPlugin *plugin = init_fn();
    if (!plugin)
    {
        fprintf(stderr, "Plugin '%s' init returned NULL\n", path);
        zc_dlclose(handle);
        return NULL;
    }

    // Register
    PluginNode *node = (PluginNode *)malloc(sizeof(PluginNode));
    if (!node) {
        zc_dlclose(handle);
        return NULL;
    }

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
            zc_dlclose(curr->handle);
        }
        free(curr);
        curr = next;
    }
    head = NULL;
}
