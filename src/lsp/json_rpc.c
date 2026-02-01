#include "json_rpc.h"
#include "cJSON.h"
#include "lsp_project.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void lsp_check_file(const char *uri, const char *src, int id);
void lsp_goto_definition(const char *uri, int line, int col, int id);
void lsp_hover(const char *uri, int line, int col, int id);
void lsp_completion(const char *uri, int line, int col, int id);
void lsp_document_symbol(const char *uri, int id);
void lsp_references(const char *uri, int line, int col, int id);
void lsp_signature_help(const char *uri, int line, int col, int id);

// Helper to extract textDocument params
static void get_params(cJSON *root, char **uri, int *line, int *col)
{
    cJSON *params = cJSON_GetObjectItem(root, "params");
    if (!params)
    {
        return;
    }

    cJSON *doc = cJSON_GetObjectItem(params, "textDocument");
    if (doc)
    {
        cJSON *u = cJSON_GetObjectItem(doc, "uri");
        if (u && u->valuestring)
        {
            *uri = strdup(u->valuestring);
        }
    }

    cJSON *pos = cJSON_GetObjectItem(params, "position");
    if (pos)
    {
        cJSON *l = cJSON_GetObjectItem(pos, "line");
        cJSON *c = cJSON_GetObjectItem(pos, "character");
        if (l)
        {
            *line = l->valueint;
        }
        if (c)
        {
            *col = c->valueint;
        }
    }
}

void send_lsp_message(const char *response)
{
	fprintf(stdout, "Content-Length: %zu\r\n\r\n%s", strlen(response), response);
	fflush(stdout);
}

void handle_initialize(cJSON *json, int id)
{
	cJSON *params = cJSON_GetObjectItem(json, "params");
	char *root = NULL;
	if (params)
	{
		cJSON *rp = cJSON_GetObjectItem(params, "rootPath");
		if (rp && rp->valuestring)
		{
			root = strdup(rp->valuestring);
		}
		else
		{
			cJSON *ru = cJSON_GetObjectItem(params, "rootUri");
			if (ru && ru->valuestring)
			{
				root = strdup(ru->valuestring);
			}
		}
	}

	if (root && strncmp(root, "file://", 7) == 0)
	{
		char *clean = strdup(root + 7);
		free(root);
		root = clean;
	}

	lsp_project_init(root ? root : ".");
	if (root)
	{
		free(root);
	}

	char response[1024];

	snprintf(response, 1024, 
		"{\"jsonrpc\":\"2.0\",\"id\":%d,\"result\":{"
		"\"serverInfo\":{\"name\":\"ZenC LS\",\"version\": \"1.0.0\"},"
		"\"capabilities\":{\"textDocumentSync\":{\"openClose\":true,\"change\":1},"
		"\"definitionProvider\":true,\"hoverProvider\":true,"
		"\"referencesProvider\":true,\"documentSymbolProvider\":true,"
		"\"signatureHelpProvider\":{\"triggerCharacters\":[\"(\"]},"
		"\"completionProvider\":{"
		"\"triggerCharacters\":[\".\"]}}}}"
		, id);

	send_lsp_message(response);
}

void handle_request(const char *json_str)
{
    cJSON *json = cJSON_Parse(json_str);
    if (!json)
    {
        return;
    }

    int id = 0;
    cJSON *id_item = cJSON_GetObjectItem(json, "id");
    if (id_item)
    {
        id = id_item->valueint;
    }

    cJSON *method_item = cJSON_GetObjectItem(json, "method");
    if (!method_item || !method_item->valuestring)
    {
        cJSON_Delete(json);
        return;
    }
    char *method = method_item->valuestring;

    if (strcmp(method, "initialize") == 0)
    {
        handle_initialize(json, id);
    }
    else if (strcmp(method, "textDocument/didOpen") == 0 ||
             strcmp(method, "textDocument/didChange") == 0)
    {
        cJSON *params = cJSON_GetObjectItem(json, "params");
        if (params)
        {
            cJSON *doc = cJSON_GetObjectItem(params, "textDocument");
            if (doc)
            {
                cJSON *uri = cJSON_GetObjectItem(doc, "uri");
                cJSON *text = cJSON_GetObjectItem(doc, "text");
                // For didChange, text is inside contentChanges
                if (!text && strcmp(method, "textDocument/didChange") == 0)
                {
                    cJSON *changes = cJSON_GetObjectItem(params, "contentChanges");
                    if (changes && cJSON_GetArraySize(changes) > 0)
                    {
                        cJSON *change = cJSON_GetArrayItem(changes, 0);
                        text = cJSON_GetObjectItem(change, "text");
                    }
                }

                if (uri && uri->valuestring && text && text->valuestring)
                {
                    lsp_check_file(uri->valuestring, text->valuestring, id);
                }
            }
        }
    }
    else if (strcmp(method, "textDocument/definition") == 0)
    {
        char *uri = NULL;
        int line = 0, col = 0;
        get_params(json, &uri, &line, &col);
        if (uri)
        {
            lsp_goto_definition(uri, line, col, id);
            free(uri);
        }
    }
    else if (strcmp(method, "textDocument/hover") == 0)
    {
        char *uri = NULL;
        int line = 0, col = 0;
        get_params(json, &uri, &line, &col);
        if (uri)
        {
            lsp_hover(uri, line, col, id);
            free(uri);
        }
    }
    else if (strcmp(method, "textDocument/completion") == 0)
    {
        char *uri = NULL;
        int line = 0, col = 0;
        get_params(json, &uri, &line, &col);
        if (uri)
        {
            lsp_completion(uri, line, col, id);
            free(uri);
        }
    }
    else if (strcmp(method, "textDocument/documentSymbol") == 0)
    {
        char *uri = NULL;
        int line = 0, col = 0; // Unused for outline
        get_params(json, &uri, &line, &col);
        if (uri)
        {
            lsp_document_symbol(uri, id);
            free(uri);
        }
    }
    else if (strcmp(method, "textDocument/references") == 0)
    {
        char *uri = NULL;
        int line = 0, col = 0;
        get_params(json, &uri, &line, &col);
        if (uri)
        {
            lsp_references(uri, line, col, id);
            free(uri);
        }
    }
    else if (strcmp(method, "textDocument/signatureHelp") == 0)
    {
        char *uri = NULL;
        int line = 0, col = 0;
        get_params(json, &uri, &line, &col);
        if (uri)
        {
            lsp_signature_help(uri, line, col, id);
            free(uri);
        }
    }

    cJSON_Delete(json);
}
