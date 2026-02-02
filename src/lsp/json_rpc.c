#include "json_rpc.h"
#include "cJSON.h"
#include "lsp_project.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

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

void send_lsp_message_json(const cJSON *json)
{
    char *body;
    size_t len;

    assert(json);

    body = cJSON_PrintUnformatted(json);

    len = strlen(body);

    fprintf(stdout, "Content-Length: %zu\r\n\r\n", len);
    fwrite(body, 1, len, stdout);
    fflush(stdout);

    free(body);
}

cJSON *create_response(const cJSON *id_item,
                       const cJSON *result,
                       const cJSON *error)
{
    cJSON *res;

    assert(id_item || (error && !id_item));
    assert((result && !error) || (!result && error));

    res = cJSON_CreateObject();
    if (!res) {
        return NULL;
    }

    cJSON_AddStringToObject(res, "jsonrpc", "2.0");

    cJSON_AddItemToObject(res, "id", cJSON_Duplicate(id_item, 1));

    if (result)
	{
        cJSON_AddItemToObject(res, "result", cJSON_Duplicate(result, 1));
    }
    else {
        cJSON_AddItemToObject(res, "error", cJSON_Duplicate(error, 1));
    }

    return res;
}

void handle_initialize(const cJSON *json, const cJSON *id_item)
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

	cJSON *result = cJSON_CreateObject();

    cJSON *serverInfo = cJSON_AddObjectToObject(result, "serverInfo");
    cJSON_AddStringToObject(serverInfo, "name", "ZenC LS");
    cJSON_AddStringToObject(serverInfo, "version", "1.0.0");

    // server capabilities
    cJSON *caps = cJSON_AddObjectToObject(result, "capabilities");

    cJSON *sync = cJSON_AddObjectToObject(caps, "textDocumentSync");
    cJSON_AddBoolToObject(sync, "openClose", true);
    cJSON_AddNumberToObject(sync, "change", 1);

    cJSON_AddBoolToObject(caps, "definitionProvider", true);
    cJSON_AddBoolToObject(caps, "hoverProvider", true);
    cJSON_AddBoolToObject(caps, "referencesProvider", true);
    cJSON_AddBoolToObject(caps, "documentSymbolProvider", true);

    cJSON *sig = cJSON_AddObjectToObject(caps, "signatureHelpProvider");
    cJSON *sig_trig = cJSON_AddArrayToObject(sig, "triggerCharacters");
    cJSON_AddItemToArray(sig_trig, cJSON_CreateString("("));

    cJSON *comp = cJSON_AddObjectToObject(caps, "completionProvider");
    cJSON *comp_trig = cJSON_AddArrayToObject(comp, "triggerCharacters");
    cJSON_AddItemToArray(comp_trig, cJSON_CreateString("."));

    cJSON *response = create_response(id_item, result, NULL);
    send_lsp_message_json(response);

    cJSON_Delete(result);
    cJSON_Delete(response);
}

void handle_shutdown(cJSON *id_item)
{
    assert(id_item);

    fprintf(stderr, "zls: shutdown received\n");

    // lsp_state.shutdown = 1;
	// TODO: after shutdown every request except exit gonna send JSONRPC_INVALID_REQUEST

    cJSON *result = cJSON_CreateNull();

    cJSON *response = create_response(id_item, result, NULL);

    send_lsp_message_json(response);

    cJSON_Delete(result);
    cJSON_Delete(response);
}

void handle_exit(void)
{
    fprintf(stderr, "zls: exit received\n");
    // TODO: add the lsp clean here
	
    // TODO: exit 0 if shutdown is call before exit else exit 1 
    exit(0);
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
	if (id_item && !(cJSON_IsNumber(id_item) || cJSON_IsString(id_item)))
	{
		invalid_request(NULL);
		cJSON_Delete(json);
		return;
	}
	if (id_item)
	{
		// FIXME: not always int but can be string too
		id = id_item->valueint;
	}

    cJSON *method_item = cJSON_GetObjectItem(json, "method");
    if (!method_item || !cJSON_IsString(method_item))
    {
		invalid_request(id_item);
        cJSON_Delete(json);
        return;
    }
    char *method = method_item->valuestring;

    if (strcmp(method, "initialize") == 0)
    {
        handle_initialize(json, id_item);
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
    else if (strcmp(method, "shutdown") == 0)
	{
		handle_shutdown(id_item);
	}
    else if (strcmp(method, "exit") == 0)
	{
		handle_exit();
	}
    else 
	{
		method_not_found(id_item);
	}

    cJSON_Delete(json);
}
