# Zen-C HTTP Module

A simple, ergonomic HTTP client and server library for Zen-C with native support for `@derive(ToJson)` serialization.

## Dependencies

- **Server**: `libmicrohttpd` - Install via `apt install libmicrohttpd-dev` (Debian/Ubuntu) or `brew install libmicrohttpd` (macOS)
- **Client**: `libcurl` - Install via `apt install libcurl4-openssl-dev` (Debian/Ubuntu) or `brew install curl` (macOS)

## Quick Start

### HTTP Server

```zenc
//> link: -lmicrohttpd

import "std/http/server.zc"

// Define response struct with automatic JSON serialization
@derive(ToJson)
struct Message {
    message: char*;
    status: int;
}

fn handle_hello(req: Request) -> Response {
    let msg = Message { message: "Hello World!", status: 200 };
    return Response::json_value(msg.to_json());
}

fn main() {
    let server = HttpServer::new(8080);
    server.get("/hello", handle_hello);
    server.run();
    server.free();
}
```

### HTTP Client

```zenc
//> link: -lcurl

import "std/http/client.zc"
import "std/json.zc"

@derive(ToJson)
struct CreateUser {
    name: char*;
    email: char*;
}

fn main() {
    let client = HttpClient::new();

    // Simple GET
    let resp = client.get("https://api.example.com/users");
    if resp.ok() {
        "{resp.body}";
    }
    resp.free();

    // POST with @derive(ToJson)
    let user = CreateUser { name: "John", email: "john@example.com" };
    let json = user.to_json().stringify();
    let resp2 = client.post_json("https://api.example.com/users", json);
    free(json);
    resp2.free();
}
```

## Server API Reference

### HttpServer

| Method | Description |
|--------|-------------|
| `HttpServer::new(port: int)` | Create a new server on the specified port |
| `server.get(path, handler)` | Register a GET handler |
| `server.post(path, handler)` | Register a POST handler |
| `server.put(path, handler)` | Register a PUT handler |
| `server.delete(path, handler)` | Register a DELETE handler |
| `server.patch(path, handler)` | Register a PATCH handler |
| `server.run()` | Start the server (blocking) |
| `server.free()` | Clean up resources |

### Request

| Method | Description |
|--------|-------------|
| `req.method()` | Get HTTP method as string ("GET", "POST", etc.) |
| `req.path()` | Get request path |
| `req.body()` | Get request body (for POST/PUT/PATCH) |
| `req.param(name)` | Get URL parameter (e.g., `:id` from `/users/:id`) |
| `req.header(name)` | Get request header value |

### Response Builders

| Method | Description |
|--------|-------------|
| `Response::text(body)` | Plain text response (200 OK) |
| `Response::html(body)` | HTML response (200 OK) |
| `Response::json(body)` | JSON response from string (200 OK) |
| `Response::json_value(json)` | JSON response from `JsonValue` (200 OK) |
| `Response::json_value_status(json, status)` | JSON response with custom status |
| `Response::not_found()` | 404 Not Found response |
| `Response::bad_request()` | 400 Bad Request response |
| `Response::error()` | 500 Internal Server Error response |
| `Response::status(resp, code)` | Set status code on existing response |

### HTTP Status Constants

```zenc
HTTP_OK              = 200
HTTP_CREATED         = 201
HTTP_NO_CONTENT      = 204
HTTP_BAD_REQUEST     = 400
HTTP_UNAUTHORIZED    = 401
HTTP_FORBIDDEN       = 403
HTTP_NOT_FOUND       = 404
HTTP_METHOD_NOT_ALLOWED = 405
HTTP_INTERNAL_ERROR  = 500
```

## Client API Reference

### HttpClient

| Method | Description |
|--------|-------------|
| `HttpClient::new()` | Create a new client with default settings |
| `client.get(url)` | Make a GET request |
| `client.post(url, body)` | Make a POST request with form data |
| `client.post_json(url, json)` | Make a POST request with JSON body |
| `client.put(url, body)` | Make a PUT request |
| `client.patch(url, body)` | Make a PATCH request |
| `client.delete(url)` | Make a DELETE request |
| `client.head(url)` | Make a HEAD request |
| `client.set_timeout(ms)` | Set request timeout in milliseconds |
| `client.set_follow_redirects(bool)` | Enable/disable redirect following |
| `client.set_verify_ssl(bool)` | Enable/disable SSL verification |
| `client.set_user_agent(ua)` | Set custom User-Agent header |

### HttpResponse

| Field/Method | Description |
|--------------|-------------|
| `resp.status_code` | HTTP status code (int) |
| `resp.body` | Response body (char*) |
| `resp.body_len` | Body length in bytes |
| `resp.headers` | Response headers (char*) |
| `resp.error` | Error message if request failed (char*) |
| `resp.ok()` | Returns true if status is 2xx and no error |
| `resp.free()` | Free response resources |

### Convenience Functions

```zenc
// Quick one-liner requests
let resp = http_get("https://example.com/api");
let resp = http_post("https://example.com/api", "data=value");
let resp = http_post_json("https://example.com/api", json_string);
```

## Using @derive(ToJson) with HTTP

The HTTP module integrates seamlessly with Zen-C's `@derive(ToJson)` for automatic JSON serialization:

```zenc
@derive(ToJson)
struct User {
    id: int;
    name: char*;
    active: bool;
}

// Server: Return struct as JSON
fn get_user(req: Request) -> Response {
    let user = User { id: 1, name: "Alice", active: true };
    return Response::json_value(user.to_json());
}

// Client: Send struct as JSON
fn create_user() {
    let user = User { id: 0, name: "Bob", active: true };
    let json_val = user.to_json();
    let json_str = json_val.stringify();

    let client = HttpClient::new();
    let resp = client.post_json("https://api.example.com/users", json_str);

    free(json_str);
    json_val.free();
    resp.free();
}
```

## URL Parameters

The server supports URL parameters using the `:param` syntax:

```zenc
fn get_user(req: Request) -> Response {
    let id = req.param("id");      // From /users/:id
    let name = req.param("name");  // From /users/:id/:name
    // ...
}

server.get("/users/:id", get_user);
server.get("/users/:id/:name", get_user);
```

## Error Handling

### Server-side

```zenc
fn handle_create(req: Request) -> Response {
    let body = req.body();
    if body == NULL {
        return Response::bad_request();
    }

    let result = JsonValue::parse(body);
    if result.is_err() {
        return Response::json_value_status(
            ErrorResponse { error: "Invalid JSON" }.to_json(),
            HTTP_BAD_REQUEST
        );
    }
    // ...
}
```

### Client-side

```zenc
let resp = client.get("https://example.com/api");

if resp.ok() {
    "Success: {resp.body}";
} else if resp.error != NULL {
    "Request failed: {resp.error}";
} else {
    "HTTP error: {resp.status_code}";
}

resp.free();
```

## Examples

See the `examples/networking/http/` directory for complete examples:

- `server_example.zc` - Simple REST API server
- `rest_api.zc` - Full CRUD REST API with data store
- `client_example.zc` - HTTP client with various request types

## Running Examples

```bash
# Build and run server
zc examples/networking/http/server_example.zc -o server_example
./server_example

# In another terminal, test with curl
curl http://localhost:8080/api/hello

# Build and run client
zc examples/networking/http/client_example.zc -o client_example
./client_example
```
