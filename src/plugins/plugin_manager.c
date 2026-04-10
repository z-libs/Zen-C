#include "plugin_manager.h"

#include "../constants.h"
#include "../diagnostics/diagnostics.h"
#include <stdarg.h>
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

// Diagnostic wrappers for plugins
static void plugin_error(const ZApi *api, const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    char msg[MAX_ERROR_MSG_LEN];
    vsnprintf(msg, sizeof(msg), fmt, args);
    va_end(args);

    Token t = {0};
    t.file = api->filename;
    t.line = api->current_line;
    zerror_at(t, "%s", msg);
}

static void plugin_warn(const ZApi *api, const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    char msg[MAX_ERROR_MSG_LEN];
    vsnprintf(msg, sizeof(msg), fmt, args);
    va_end(args);

    Token t = {0};
    t.file = api->filename;
    t.line = api->current_line;
    zwarn_at(t, "%s", msg);
}

static void plugin_note(const ZApi *api, const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    char msg[MAX_ERROR_MSG_LEN];
    vsnprintf(msg, sizeof(msg), fmt, args);
    va_end(args);

    // Using printf for nnote if we don't have a znote_at
    printf(COLOR_BOLD COLOR_CYAN "  note: " COLOR_RESET "%s:%d: %s\n", api->filename,
           api->current_line, msg);
}

void zptr_init_api(ZApi *api, const char *filename, int line, FILE *out, FILE *hoist_out)
{
    if (!api)
    {
        return;
    }

    api->api_version = ZEN_PLUGIN_API_VERSION;
    api->filename = filename ? filename : "input.zc";
    api->current_line = line;
    api->out = out;
    api->hoist_out = hoist_out;

    api->error = plugin_error;
    api->warn = plugin_warn;
    api->note = plugin_note;

    api->config.is_debug = g_config.mode_debug;
    api->config.verbose = g_config.verbose;
    api->config.target = ZC_OS_NAME;
    api->config.cc = g_config.cc;
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
    void *handle = z_dlopen(path);
    if (!handle)
    {
        return NULL;
    }

    ZPluginInitFn init_fn = (ZPluginInitFn)z_dlsym(handle, "z_plugin_init");
    if (!init_fn)
    {
        fprintf(stderr, "Plugin '%s' missing 'z_plugin_init' symbol\n", path);
        z_dlclose(handle);
        return NULL;
    }

    ZPlugin *plugin = init_fn();
    if (!plugin)
    {
        fprintf(stderr, "Plugin '%s' init returned NULL\n", path);
        z_dlclose(handle);
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
            z_dlclose(curr->handle);
        }
        free(curr);
        curr = next;
    }
    head = NULL;
}
