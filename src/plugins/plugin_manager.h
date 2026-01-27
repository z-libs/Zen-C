#ifndef PLUGIN_MANAGER_H
#define PLUGIN_MANAGER_H

#include "../../plugins/zprep_plugin.h"

// Initialize the plugin system.
/**
 * @brief Initialize the plugin system.
 */
void zptr_plugin_mgr_init(void);

/**
 * @brief Register a plugin directly (for built-in plugins).
 * @param plugin The plugin to register.
 */
void zptr_register_plugin(ZPlugin *plugin);

/**
 * @brief Load a plugin from a shared object file (.so).
 * 
 * @param path Path to the shared object file.
 * @return ZPlugin* Pointer to the loaded plugin on success, NULL on failure.
 */
ZPlugin *zptr_load_plugin(const char *path);

/**
 * @brief Find a registered plugin by name.
 * @param name The name of the plugin.
 * @return ZPlugin* Pointer to the plugin or NULL if not found.
 */
ZPlugin *zptr_find_plugin(const char *name);

/**
 * @brief Cleanup the plugin system and free resources.
 */
void zptr_plugin_mgr_cleanup(void);

#endif
