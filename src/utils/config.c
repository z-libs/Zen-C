#include "zprep.h"
#include "cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Helper to append strings to the global whitelist
static void append_whitelist(cJSON *c_funcs)
{
    if (!cJSON_IsArray(c_funcs))
    {
        return;
    }

    int new_count = cJSON_GetArraySize(c_funcs);
    if (new_count == 0)
    {
        return;
    }

    int current_count = 0;
    if (g_config.c_function_whitelist)
    {
        char **ptr = g_config.c_function_whitelist;
        while (*ptr)
        {
            current_count++;
            ptr++;
        }
    }

    size_t new_size = sizeof(char *) * (current_count + new_count + 1);
    g_config.c_function_whitelist = xrealloc(g_config.c_function_whitelist, new_size);

    int added = 0;
    cJSON *item = NULL;
    cJSON_ArrayForEach(item, c_funcs)
    {
        if (cJSON_IsString(item) && item->valuestring)
        {
            g_config.c_function_whitelist[current_count + added] = xstrdup(item->valuestring);
            added++;
        }
    }
    g_config.c_function_whitelist[current_count + added] = NULL;
}

static int load_config_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f)
    {
        return 0;
    }

    fseek(f, 0, SEEK_END);
    long length = ftell(f);
    fseek(f, 0, SEEK_SET);

    // Use standard malloc/free for temporary file buffer to avoid arena pollution
    char *data = malloc(length + 1);
    if (!data)
    {
        fclose(f);
        return 0;
    }

    fread(data, 1, length, f);
    data[length] = '\0';
    fclose(f);

    cJSON *json = cJSON_Parse(data);
    free(data);

    if (json)
    {
        cJSON *c_funcs = cJSON_GetObjectItemCaseSensitive(json, "c_functions");
        if (cJSON_IsArray(c_funcs))
        {
            append_whitelist(c_funcs);
        }
        cJSON_Delete(json);
        return 1;
    }
    return 0;
}

void load_all_configs(void)
{
    // 1. System-wide config
    int loaded_system = 0;
    char *root = getenv("ZC_ROOT");
    if (root)
    {
        char path[1024];
        snprintf(path, sizeof(path), "%s/zenc.json", root);
        if (load_config_file(path))
        {
            loaded_system = 1;
        }
    }

#ifdef ZEN_SHARE_DIR
    if (!loaded_system)
    {
        char system_path[1024];
        snprintf(system_path, sizeof(system_path), "%s/zenc.json", ZEN_SHARE_DIR);
        load_config_file(system_path);
    }
#endif

    // 2. Hidden project config
    load_config_file(".zenc.json");

    // 3. Visible project config (legacy/override)
    load_config_file("zenc.json");
}
