
#include "json_rpc.h"
#include "../constants.h"
#include "zprep.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Simple Main Loop for LSP.
int lsp_main(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    fprintf(stderr, "zls: Zen Language Server starting...\n");
    g_config.mode_lsp = 1;
    g_config.json_output = 1;

    // Initialize root path from executable to find std/
    char self_path[MAX_PATH_LEN];
    void z_get_executable_path(char *buf, size_t size);
    z_get_executable_path(self_path, sizeof(self_path));
    if (self_path[0])
    {
        g_config.root_path = xstrdup(self_path);
    }

    while (1)
    {
        // Read headers
        int content_len = 0;
        char line[MAX_MANGLED_NAME_LEN];
        while (fgets(line, sizeof(line), stdin))
        {
            if (0 == strcmp(line, "\r\n"))
            {
                break; // End of headers
            }
            if (0 == strncmp(line, "Content-Length: ", 16))
            {
                content_len = atoi(line + 16);
            }
        }

        if (content_len <= 0)
        {
            if (feof(stdin))
            {
                break;
            }
            continue;
        }

        if (content_len > 10 * 1024 * 1024)
        {
            fprintf(stderr, "zls: Content-Length too large (%d)\n", content_len);
            break;
        }

        // Read body.
        char *body = malloc(content_len + 1);
        if (fread(body, 1, content_len, stdin) != (size_t)content_len)
        {
            fprintf(stderr, "zls: Error reading body\n");
            zfree(body);
            break;
        }
        body[content_len] = 0;

        // Process JSON-RPC.
        fprintf(stderr, "zls: Received: %s\n", body);
        handle_request(body);

        zfree(body);
    }

    return 0;
}
