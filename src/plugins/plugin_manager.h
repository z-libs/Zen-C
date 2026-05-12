#ifndef PLUGIN_MANAGER_H
#ifndef ZC_ALLOW_INTERNAL
#error                                                                                             \
    "plugins/plugin_manager.h is internal to Zen C. Include the appropriate public header instead."
#endif

#define PLUGIN_MANAGER_H

#include "../../plugins/zprep_plugin.h"

typedef struct CompilerConfig CompilerConfig;

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
 * @brief Initialize a ZApi structure with host-provided context.
 */
void zptr_init_api(ZApi *api, const char *filename, int line, CompilerConfig *cfg);

/**
 * @brief Cleanup the plugin system and free resources.
 */
void zptr_plugin_mgr_cleanup(void);

/**
 * @brief Unload a plugin by name.
 * @param name The name of the plugin to unload.
 * @return int 1 on success, 0 if not found.
 */
int zptr_unload_plugin(const char *name);

/**
 * @brief Get a static built-in plugin by name.
 * @param name The name of the plugin.
 * @return ZPlugin* Pointer to the plugin or NULL if not found.
 */
ZPlugin *zptr_get_static_plugin(const char *name);

#endif
