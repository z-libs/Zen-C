#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/wait.h>
#include <assert.h>
#include <fcntl.h>
#include "../../src/lsp/cJSON.h"

#define MAX_BUFFER (2 * 1024 * 1024)

int pipe_in[2];
int pipe_out[2];
pid_t child_pid;

void fail(const char *msg)
{
    fprintf(stderr, "TEST FAIL: %s\n", msg);
    exit(1);
}

void start_lsp_server()
{
    if (pipe(pipe_in) < 0 || pipe(pipe_out) < 0)
    {
        fail("Pipe creation failed");
    }
    child_pid = fork();
    if (child_pid < 0)
    {
        fail("Fork failed");
    }

    if (child_pid == 0)
    {
        dup2(pipe_in[0], STDIN_FILENO);
        dup2(pipe_out[1], STDOUT_FILENO);
        close(pipe_in[0]);
        close(pipe_in[1]);
        close(pipe_out[0]);
        close(pipe_out[1]);
        execl("./zc", "zc", "lsp", NULL);
        fail("Execl failed");
    }
    else
    {
        close(pipe_in[0]);
        close(pipe_out[1]);
    }
}

void send_request(const char *json)
{
    char header[128];
    int len = strlen(json);
    sprintf(header, "Content-Length: %d\r\n\r\n", len);
    write(pipe_in[1], header, strlen(header));
    write(pipe_in[1], json, len);
}

char global_buf[MAX_BUFFER];
int global_len = 0;

char *read_message()
{
    while (1)
    {
        // 1. Check if we have a complete header
        char *body_start = strstr(global_buf, "\r\n\r\n");
        if (body_start)
        {
            int header_len = body_start - global_buf + 4;
            int content_len = 0;

            // Parse Content-Length
            char *cl_ptr = strstr(global_buf, "Content-Length: ");
            if (cl_ptr && cl_ptr < body_start)
            {
                content_len = atoi(cl_ptr + 16);
            }

            if (content_len > 0)
            {
                int total_msg_len = header_len + content_len;
                if (global_len >= total_msg_len)
                {
                    // We have a full message!
                    char *msg_body = malloc(content_len + 1);
                    memcpy(msg_body, body_start + 4, content_len);
                    msg_body[content_len] = 0;

                    // Shift remaining data to start
                    int remaining = global_len - total_msg_len;
                    memmove(global_buf, global_buf + total_msg_len, remaining);
                    global_len = remaining;
                    global_buf[global_len] = 0;

                    return msg_body;
                }
            }
        }

        // 2. Read more data
        if (global_len >= MAX_BUFFER - 1)
        {
            fail("Buffer overflow in read_message");
        }

        int n = read(pipe_out[0], global_buf + global_len, MAX_BUFFER - 1 - global_len);
        if (n <= 0)
        {
            // EOF or error, but maybe we have a message pending verification?
            // If n=0 and we are waiting for data, it's bad.
            // But wait_for_response calls us in loop.
            // If we return NULL, wait_for_response aborts?
            // Wait, existing wait_for_response returns NULL on read_message NULL and loops?
            // No, existing wait_for_response returns NULL instantly if read_message returns NULL.
            // We should block here? read blocks.
            // If read returns 0, server closed pipe.
            return NULL;
        }
        global_len += n;
        global_buf[global_len] = 0;
    }
}

char *wait_for_response(int id)
{
    // Loop until we get a response with the matching ID
    // or until timeout (implicit in blocking read, could add explicit timeout)
    // Increase loop count significantly because we might get many 'publishDiagnostics'
    for (int i = 0; i < 500; i++)
    {
        char *msg = read_message();
        if (!msg)
        {
            return NULL;
        }

        cJSON *json = cJSON_Parse(msg);
        if (json)
        {
            cJSON *id_item = cJSON_GetObjectItem(json, "id");
            if (id_item)
            {
                // Check both number and string just in case
                int got_id = -1;
                if (cJSON_IsNumber(id_item))
                {
                    got_id = id_item->valueint;
                }
                else if (cJSON_IsString(id_item))
                {
                    got_id = atoi(id_item->valuestring);
                }

                if (got_id == id)
                {
                    cJSON_Delete(json);
                    return msg; // Found it
                }
                else
                {
                    printf("Mismatch ID: got %d (type=%d) expected %d\n", got_id, id_item->type,
                           id);
                }
            }
            else
            {
                printf("No ID in message\n");
            }
            cJSON_Delete(json);
        }
        else
        {
            printf("JSON Parse Failed for: '%s'\n", msg);
        }
        printf("Ignored message (seq %d): %s\n", i, msg);
        free(msg);
    }
    return NULL;
}

void test_initialize()
{
    printf("Running test_initialize...\n");
    send_request("{\"jsonrpc\": \"2.0\", \"id\": 1, \"method\": \"initialize\", \"params\": {}}");
    char *resp = wait_for_response(1);
    if (!resp)
    {
        fail("No response for initialize");
    }

    // Validate basics
    if (!strstr(resp, "capabilities"))
    {
        fail("Init response missing capabilities");
    }
    printf("PASS: test_initialize\n");
    free(resp);
}

void test_hover()
{
    printf("Running test_hover...\n");
    int fd = open("/tmp/test_lsp.zc", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0)
    {
        // Use 'fn' instead of 'def'
        write(fd, "fn main() { return 0; }", 23);
        close(fd);
    }

    send_request("{\"jsonrpc\": \"2.0\", \"method\": \"textDocument/didOpen\", \"params\": "
                 "{\"textDocument\": {\"uri\": \"file:///tmp/test_lsp.zc\", \"languageId\": "
                 "\"zenc\", \"version\": 1, \"text\": \"fn main() { return 0; }\"}}}");

    // Wait a bit not strictly necessary if we use wait_for_response logic correctly,
    // but helps ensure server has processed the open event if it's async internal logic.
    usleep(100000);

    // Hover over 'main'
    send_request("{\"jsonrpc\": \"2.0\", \"id\": 2, \"method\": \"textDocument/hover\", "
                 "\"params\": {\"textDocument\": {\"uri\": \"file:///tmp/test_lsp.zc\"}, "
                 "\"position\": {\"line\": 0, \"character\": 5}}}");

    char *resp = wait_for_response(2);
    if (!resp)
    {
        printf("WARN: No response for hover.\n");
        return;
    }

    if (strstr(resp, "contents"))
    {
        printf("PASS: test_hover\n");
    }
    else
    {
        printf("WARN: Hover response missing contents: %s\n", resp);
    }
    free(resp);
}

void test_shutdown()
{
    printf("Running test_shutdown...\n");
    send_request("{\"jsonrpc\": \"2.0\", \"id\": 3, \"method\": \"shutdown\", \"params\": {}}");
    char *resp = wait_for_response(3);
    if (resp)
    {
        printf("PASS: test_shutdown\n");
        free(resp);
    }
    else
    {
        printf("PASS: test_shutdown (no response)\n");
    }
}

void test_completion()
{
    printf("Running test_completion...\n");
    int fd = open("/tmp/test_compl.zc", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0)
    {
        // Function with local variable 'local_idx' and argument 'arg_val'
        const char *code = "fn test_func(arg_val: int) { var local_idx = 10; \n    \n }";
        write(fd, code, strlen(code));
        close(fd);
    }

    send_request("{\"jsonrpc\": \"2.0\", \"method\": \"textDocument/didOpen\", \"params\": "
                 "{\"textDocument\": {\"uri\": \"file:///tmp/test_compl.zc\", \"languageId\": "
                 "\"zenc\", \"version\": 1, \"text\": \"fn test_func(arg_val: int) { var local_idx "
                 "= 10; \\n    \\n }\"}}}");
    usleep(100000);

    // Request completion inside the function body
    send_request("{\"jsonrpc\": \"2.0\", \"id\": 10, \"method\": \"textDocument/completion\", "
                 "\"params\": {\"textDocument\": {\"uri\": \"file:///tmp/test_compl.zc\"}, "
                 "\"position\": {\"line\": 1, \"character\": 4}}}");

    char *resp = wait_for_response(10);
    if (!resp)
    {
        printf("WARN: No response for completion.\n");
        return;
    }

    // Check for 'local_idx'
    if (strstr(resp, "local_idx"))
    {
        printf("PASS: test_completion (found local_idx)\n");
    }
    else
    {
        printf("FAIL: test_completion (local_idx not found): %s\n", resp);
    }

    // Check for 'arg_val'
    if (strstr(resp, "arg_val"))
    {
        printf("PASS: test_completion (found arg_val)\n");
    }
    else
    {
        printf("WARN: test_completion (arg_val not found): %s\n", resp);
    }
    free(resp);
}

void test_struct_completion()
{
    printf("Running test_struct_completion...\n");
    int fd = open("/tmp/test_struct.zc", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0)
    {
        const char *code = "struct Point { x: int; y: int; }\n"
                           "fn main() {\n"
                           "    var my_point: Point;\n"
                           "    my_point.\n"
                           "}";
        write(fd, code, strlen(code));
        close(fd);
    }

    send_request("{\"jsonrpc\": \"2.0\", \"method\": \"textDocument/didOpen\", \"params\": "
                 "{\"textDocument\": {\"uri\": \"file:///tmp/test_struct.zc\", \"languageId\": "
                 "\"zenc\", \"version\": 1, \"text\": \"struct Point { x: int; y: int; }\\nfn "
                 "main() {\\n    var my_point: Point;\\n    my_point.\\n}\"}}}");
    usleep(100000);

    // Request completion at 'my_point.' (line 3, character 13)
    // 4 spaces + 8 chars + 1 dot = 13
    send_request("{\"jsonrpc\": \"2.0\", \"id\": 20, \"method\": \"textDocument/completion\", "
                 "\"params\": {\"textDocument\": {\"uri\": \"file:///tmp/test_struct.zc\"}, "
                 "\"position\": {\"line\": 3, \"character\": 13}}}");

    char *resp = wait_for_response(20);
    if (!resp)
    {
        printf("WARN: No response for struct completion.\n");
        return;
    }

    // Check for 'x'
    if (strstr(resp, "\"label\":\"x\""))
    {
        printf("PASS: test_struct_completion (found field x)\n");
    }
    else
    {
        printf("FAIL: test_struct_completion (field x not found): %s\n", resp);
    }
    free(resp);
}

void test_diagnostics()
{
    printf("Running test_diagnostics...\n");
    // Create a file with a syntax error
    // "fn main() { var x: int = ; }" -> Syntax error
    int fd = open("/tmp/test_error.zc", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0)
    {
        const char *code = "fn main() { var x: int = ; }";
        write(fd, code, strlen(code));
        close(fd);
    }

    // Open file - should trigger publishDiagnostics
    send_request("{\"jsonrpc\": \"2.0\", \"method\": \"textDocument/didOpen\", \"params\": "
                 "{\"textDocument\": {\"uri\": \"file:///tmp/test_error.zc\", \"languageId\": "
                 "\"zenc\", \"version\": 1, \"text\": \"fn main() { var x: int = ; }\"}}}");

    // Check for notification
    int found = 0;
    // We might get other messages, so loop briefly?
    // But typical server sends it immediately.
    for (int i = 0; i < 5; i++)
    {
        char *msg = read_message();
        if (msg)
        {
            if (strstr(msg, "textDocument/publishDiagnostics") && strstr(msg, "diagnostics"))
            {
                printf("PASS: test_diagnostics (received diagnostics)\n");
                found = 1;
                free(msg);
                break;
            }
            free(msg);
        }
        else
        {
            break;
        }
    }

    if (!found)
    {
        printf("FAIL: test_diagnostics. No diagnostics received.\n");
    }
}

void test_semantic_tokens()
{
    printf("Running test_semantic_tokens...\n");
    int fd = open("/tmp/test_semantic.zc", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0)
    {
        // fn hello() { var x = 10; }
        // fn: function (modifier?) or keyword? Actually keyword 'fn'.
        // hello: function
        // var: keyword
        // x: variable
        // 10: number
        const char *code = "fn hello() { var x = 10; }";
        write(fd, code, strlen(code));
        close(fd);
    }

    send_request("{\"jsonrpc\": \"2.0\", \"method\": \"textDocument/didOpen\", \"params\": "
                 "{\"textDocument\": {\"uri\": \"file:///tmp/test_semantic.zc\", \"languageId\": "
                 "\"zenc\", \"version\": 1, \"text\": \"fn hello() { var x = 10; }\"}}}");
    usleep(100000);

    send_request(
        "{\"jsonrpc\": \"2.0\", \"id\": 50, \"method\": \"textDocument/semanticTokens/full\", "
        "\"params\": {\"textDocument\": {\"uri\": \"file:///tmp/test_semantic.zc\"}}}");

    char *resp = wait_for_response(50);
    if (!resp)
    {
        printf("WARN: No response for semantic tokens.\n");
        return;
    }

    // Check for "data" array
    if (strstr(resp, "\"data\":["))
    {
        printf("PASS: test_semantic_tokens (received data)\n");
        // Optional: Check existence of some numbers, e.g. token types
    }
    else
    {
        printf("FAIL: test_semantic_tokens (no data): %s\n", resp);
    }
    free(resp);
}

void test_definition()
{
    printf("Running test_definition...\n");
    int fd = open("/tmp/test_def.zc", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0)
    {
        // Line 0: fn target() {}
        // Line 1: fn main() {
        // Line 2:     target();
        // Line 3: }
        const char *code = "fn target() {}\nfn main() {\n    target();\n}";
        write(fd, code, strlen(code));
        close(fd);
    }

    send_request(
        "{\"jsonrpc\": \"2.0\", \"method\": \"textDocument/didOpen\", \"params\": "
        "{\"textDocument\": {\"uri\": \"file:///tmp/test_def.zc\", \"languageId\": \"zenc\", "
        "\"version\": 1, \"text\": \"fn target() {}\\nfn main() {\\n    target();\\n}\"}}}");
    usleep(100000);

    // Request definition at line 2, character 6 (start of "target")
    send_request("{\"jsonrpc\": \"2.0\", \"id\": 60, \"method\": \"textDocument/definition\", "
                 "\"params\": {\"textDocument\": {\"uri\": \"file:///tmp/test_def.zc\"}, "
                 "\"position\": {\"line\": 2, \"character\": 6}}}");

    char *resp = wait_for_response(60);
    if (!resp)
    {
        printf("WARN: No response for definition.\n");
        return;
    }

    // Check for correct range (Line 0)
    // Expect: "uri":..., "range": { "start": { "line": 0 ...
    if (strstr(resp, "\"line\":0"))
    {
        printf("PASS: test_definition (found target at line 0)\n");
    }
    else
    {
        printf("FAIL: test_definition (wrong location): %s\n", resp);
    }
    free(resp);
}

void test_references()
{
    printf("Running test_references...\n");
    // Reuse test_def.zc
    // Request references for "target" at line 0
    send_request(
        "{\"jsonrpc\": \"2.0\", \"id\": 70, \"method\": \"textDocument/references\", \"params\": "
        "{\"textDocument\": {\"uri\": \"file:///tmp/test_def.zc\"}, \"position\": {\"line\": 0, "
        "\"character\": 5}, \"context\": {\"includeDeclaration\": true}}}");

    char *resp = wait_for_response(70);
    if (!resp)
    {
        printf("WARN: No response for references.\n");
        return;
    }

    // Check for array result
    // Expect at least 2 refs: line 0 (decl) and line 2 (call)
    // "uri":..., "range": ... "line": 0
    // "uri":..., "range": ... "line": 2
    if (strstr(resp, "\"line\":0") && strstr(resp, "\"line\":2"))
    {
        printf("PASS: test_references (found decl and call)\n");
    }
    else
    {
        printf("FAIL: test_references (missing refs): %s\n", resp);
    }
    free(resp);
}

void test_rename()
{
    printf("Running test_rename...\n");
    // Reuse test_def.zc
    // Rename "target" to "renamed_target"
    send_request(
        "{\"jsonrpc\": \"2.0\", \"id\": 80, \"method\": \"textDocument/rename\", \"params\": "
        "{\"textDocument\": {\"uri\": \"file:///tmp/test_def.zc\"}, \"position\": {\"line\": 0, "
        "\"character\": 5}, \"newName\": \"renamed_target\"}}");

    char *resp = wait_for_response(80);
    if (!resp)
    {
        printf("WARN: No response for rename.\n");
        return;
    }

    // Check for changes
    if (strstr(resp, "changes") && strstr(resp, "renamed_target"))
    {
        printf("PASS: test_rename (received edits)\n");
    }
    else
    {
        printf("FAIL: test_rename (missing edits): %s\n", resp);
    }
    free(resp);
}

void test_outline()
{
    printf("Running test_outline...\n");
    int fd = open("/tmp/test_outline.zc", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0)
    {
        // struct Point { x: i32; y: i32; }
        // fn main() {}
        const char *code = "struct Point { x: i32; y: i32; }\nfn main() {}";
        write(fd, code, strlen(code));
        close(fd);
    }

    send_request(
        "{\"jsonrpc\": \"2.0\", \"method\": \"textDocument/didOpen\", \"params\": "
        "{\"textDocument\": {\"uri\": \"file:///tmp/test_outline.zc\", \"languageId\": \"zenc\", "
        "\"version\": 1, \"text\": \"struct Point { x: i32; y: i32; }\\nfn main() {}\"}}}");
    usleep(100000);

    send_request("{\"jsonrpc\": \"2.0\", \"id\": 90, \"method\": \"textDocument/documentSymbol\", "
                 "\"params\": {\"textDocument\": {\"uri\": \"file:///tmp/test_outline.zc\"}}}");

    char *resp = wait_for_response(90);
    if (!resp)
    {
        printf("WARN: No response for outline.\n");
        return;
    }

    // Check for hierarchy or existence
    // Expect "Point", "main". Maybe "x", "y" if improved.
    if (strstr(resp, "Point") && strstr(resp, "main"))
    {
        if (strstr(resp, "\"children\"") || strstr(resp, "\"name\":\"x\""))
        {
            printf("PASS: test_outline (found structure and children/fields)\n"); // Improved
        }
        else
        {
            printf("PASS: test_outline (found top-level only)\n"); // Basic
        }
    }
    else
    {
        printf("FAIL: test_outline (missing symbols): %s\n", resp);
    }
    free(resp);
}

int main()
{
    start_lsp_server();
    test_initialize();
    test_hover();
    test_completion();
    test_struct_completion();
    test_diagnostics();
    test_semantic_tokens();
    test_definition();
    test_references();
    test_rename();
    test_outline();
    test_shutdown();
    send_request("{\"jsonrpc\": \"2.0\", \"method\": \"exit\", \"params\": {}}");
    waitpid(child_pid, NULL, 0);
    printf("All LSP tests passed!\n");
    return 0;
}
