
#include "plugin_manager.h"
#include "../compat/compat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#endif

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

ZPlugin *zptr_load_plugin(const char *path)
{
    PluginHandle handle = zc_dlopen(path, RTLD_LAZY);
    if (!handle)
    {
        fprintf(stderr, "Failed to load plugin '%s': %s\n", path, zc_dlerror());
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

static char *get_exe_dir(void)
{
    static char path[1024];
#ifdef _WIN32
    DWORD len = GetModuleFileNameA(NULL, path, sizeof(path));
    if (len == 0 || len >= sizeof(path))
        return NULL;
#else
    ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len < 0)
    {
        len = readlink("/proc/curproc/file", path, sizeof(path) - 1);
    }
    if (len < 0)
        return NULL;
    path[len] = '\0';
#endif
    char *last_sep = strrchr(path, ZC_PATH_SEP);
    if (last_sep)
        *last_sep = '\0';
    return path;
}

static void load_plugins_from_path(const char *plugin_dir)
{
#ifdef _WIN32
    char search_path[1024];
    snprintf(search_path, sizeof(search_path), "%s\\*%s", plugin_dir, ZC_PLUGIN_EXT);

    WIN32_FIND_DATAA fd;
    HANDLE hFind = FindFirstFileA(search_path, &fd);
    if (hFind == INVALID_HANDLE_VALUE)
        return;

    do
    {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
        {
            char full_path[1024];
            snprintf(full_path, sizeof(full_path), "%s\\%s", plugin_dir, fd.cFileName);
            zptr_load_plugin(full_path);
        }
    } while (FindNextFileA(hFind, &fd));
    FindClose(hFind);
#else
    DIR *dir = opendir(plugin_dir);
    if (!dir)
        return;

    struct dirent *entry;
    size_t ext_len = strlen(ZC_PLUGIN_EXT);
    while ((entry = readdir(dir)) != NULL)
    {
        size_t name_len = strlen(entry->d_name);
        if (name_len > ext_len &&
            strcmp(entry->d_name + name_len - ext_len, ZC_PLUGIN_EXT) == 0)
        {
            char full_path[1024];
            snprintf(full_path, sizeof(full_path), "%s/%s", plugin_dir, entry->d_name);
            zptr_load_plugin(full_path);
        }
    }
    closedir(dir);
#endif
}

void zptr_load_plugins_from_dir(void)
{
    char plugin_dir[1024];
    char *exe_dir = get_exe_dir();

    // 1. Load from exe_dir/plugins/ (build directory or portable install)
    if (exe_dir)
    {
        snprintf(plugin_dir, sizeof(plugin_dir), "%s%cplugins", exe_dir, ZC_PATH_SEP);
        load_plugins_from_path(plugin_dir);
    }

    // 2. Load from system install path
#ifdef ZC_PLUGIN_INSTALL_PATH
    if (strlen(ZC_PLUGIN_INSTALL_PATH) > 0)
    {
        load_plugins_from_path(ZC_PLUGIN_INSTALL_PATH);
    }
#endif

    // 3. Fallback: current directory plugins/
    load_plugins_from_path("plugins");
}
