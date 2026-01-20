#ifndef PLUGIN_MANAGER_H
#define PLUGIN_MANAGER_H

#include "../../plugins/zprep_plugin.h"

// Initialize the plugin system.
void zptr_plugin_mgr_init(void);

// Register a plugin directly (for built-ins).
void zptr_register_plugin(ZPlugin *plugin);

// Load a plugin from a shared object file (.so).
// Returns ZPlugin pointer on success, NULL on failure.
ZPlugin *zptr_load_plugin(const char *path);

// Find a registered plugin by name.
ZPlugin *zptr_find_plugin(const char *name);

// Cleanup.
void zptr_plugin_mgr_cleanup(void);

// Auto-load plugins from exe_dir/plugins directory.
void zptr_load_plugins_from_dir(void);

#endif
