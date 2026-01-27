
#ifndef JSON_RPC_H
#define JSON_RPC_H

/**
 * @brief Handle a raw JSON-RPC request string.
 * 
 * Parses the request, routes it to the appropriate handler (initialize, textDocument/didChange, etc.),
 * and sends back the response to stdout.
 * 
 * @param json_str Null-terminated JSON request string.
 */
void handle_request(const char *json_str);

#endif
