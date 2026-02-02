
#ifndef JSON_RPC_H
#define JSON_RPC_H

#include "cJSON.h"

typedef enum jsonrpc_error_t
{
    JSONRPC_PARSE_ERROR     = -32700,
    JSONRPC_INVALID_REQUEST = -32600,
    JSONRPC_METHOD_NOT_FOUND= -32601,
    JSONRPC_INVALID_PARAMS  = -32602,
    JSONRPC_INTERNAL_ERROR  = -32603,
} jsonrpc_error_t;


/**
 * @brief Handle a raw JSON-RPC request string.
 *
 * Parses the request, routes it to the appropriate handler (initialize, textDocument/didChange,
 * etc.), and sends back the response to stdout.
 *
 * @param json_str Null-terminated JSON request string.
 */
void handle_request(const char *json_str);

void method_not_found(const cJSON *id_item);
void internal_error(const cJSON *id_item);
void parse_error(const cJSON *id_item);
void invalid_params(const cJSON *id_item);
void invalid_request(const cJSON *id_item);

cJSON *create_response(const cJSON *id_item,
                       const cJSON *result,
                       const cJSON *error);

void send_lsp_message_json(const cJSON *json);


#endif
