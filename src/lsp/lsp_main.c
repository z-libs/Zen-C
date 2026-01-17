
#include "json_rpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <unistd.h>
#endif

// Simple Main Loop for LSP.
int lsp_main(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    fprintf(stderr, "zls: Zen Language Server starting...\n");

    while (1)
    {
        // Read headers
        int content_len = 0;
        char line[512];
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
            // Maybe EOF or error?
            if (feof(stdin))
            {
                break;
            }
            continue; // Wait for more (yeah we gotta work on this).
        }

        // Read body.
        char *body = malloc(content_len + 1);
        if (fread(body, 1, content_len, stdin) != (size_t)content_len)
        {
            fprintf(stderr, "zls: Error reading body\n");
            free(body);
            break;
        }
        body[content_len] = 0;

        // Process JSON-RPC.
        fprintf(stderr, "zls: Received: %s\n", body);
        handle_request(body);

        free(body);
    }

    return 0;
}
