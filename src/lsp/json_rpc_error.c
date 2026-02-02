
#include "json_rpc.h"
#include <assert.h>

cJSON *create_error(const jsonrpc_error_t err_code,
                       const char *err_msg)
{
    cJSON *res;

    assert(err_msg);

    res = cJSON_CreateObject();
	cJSON_AddNumberToObject(res, "code", err_code);
	cJSON_AddStringToObject(res, "message", err_msg);

    return res;
}

void send_error(const cJSON *id_item, const jsonrpc_error_t err_code, const char *err_msg)
{
	cJSON *err = create_error(err_code, err_msg);
	cJSON *res = create_response(id_item, NULL, err);

	send_lsp_message_json(res);

	cJSON_Delete(err);
	cJSON_Delete(res);
}

const char *jsonrpc_error_message(jsonrpc_error_t code)
{
    switch (code) {
        case JSONRPC_PARSE_ERROR: return "Parse error";
        case JSONRPC_INVALID_REQUEST: return "Invalid Request";
        case JSONRPC_METHOD_NOT_FOUND: return "Method not found";
        case JSONRPC_INVALID_PARAMS: return "Invalid params";
        case JSONRPC_INTERNAL_ERROR: return "Internal error";
        default: return "Unknown error";
    }
}


void method_not_found(const cJSON *id_item)
{
	const jsonrpc_error_t err_code = JSONRPC_METHOD_NOT_FOUND;
	const char *msg = jsonrpc_error_message(err_code);

	send_error(id_item, err_code, msg);
}

void parse_error(const cJSON *id_item)
{
	const jsonrpc_error_t err_code = JSONRPC_PARSE_ERROR; 
	const char *msg = jsonrpc_error_message(err_code);

	send_error(id_item, err_code, msg);
}

void invalid_request(const cJSON *id_item)
{
	const jsonrpc_error_t err_code = JSONRPC_INVALID_REQUEST; 
	const char *msg = jsonrpc_error_message(err_code);

	send_error(id_item, err_code, msg);
}

void invalid_params(const cJSON *id_item)
{
	const jsonrpc_error_t err_code = JSONRPC_INVALID_PARAMS;
	const char *msg = jsonrpc_error_message(err_code);

	send_error(id_item, err_code, msg);
}

void internal_error(const cJSON *id_item)
{
	const jsonrpc_error_t err_code = JSONRPC_INTERNAL_ERROR; 
	const char *msg = jsonrpc_error_message(err_code);

	send_error(id_item, err_code, msg);
}