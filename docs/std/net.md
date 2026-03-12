# Standard Library: Networking (`std/net/`)

The `std/net` module provides a comprehensive networking stack including TCP, UDP, DNS, and HTTP.

## Usage

```zc
import "std/net/tcp.zc"  // TcpStream, TcpListener
import "std/net/udp.zc"  // UdpSocket
import "std/net/http.zc" // HTTP Client/Server
import "std/net/dns.zc"  // DNS Resolution
import "std/net/url.zc"  // URL Parsing
import "std/net/websocket.zc" // WebSocket
```

## WebSocket (`std/net/websocket.zc`)

### Type `WebSocket`

Provides server-side WebSocket handshake and framing.

| Method | Signature | Description |
| :--- | :--- | :--- |
| **handshake** | `WebSocket::handshake(stream: TcpStream, key: String) -> Result<WebSocket>` | Performs the server-side handshake. |
| **recv** | `recv(self) -> Result<String>` | Receives a text frame (handles unmasking). |
| **send** | `send(self, msg: String) -> Result<int>` | Sends a text frame. |

## HTTP Client & Server (`std/net/http.zc`)

### Type `Server`

A simple multithreaded-capable HTTP server.

| Method | Signature | Description |
| :--- | :--- | :--- |
| **new** | `Server::new(port: int, handler: fn(Request*, Response*)) -> Server` | Creates a new HTTP server. |
| **start** | `start(self)` | Starts the server listening loop. |

### Client `fetch`

| Method | Signature | Description |
| :--- | :--- | :--- |
| **fetch** | `fetch(url: String) -> Response` | Performs a GET request. |

## URL Parsing (`std/net/url.zc`)

### Type `Url`

| Method | Signature | Description |
| :--- | :--- | :--- |
| **parse** | `Url::parse(s: String) -> Option<Url>` | Parses a URL string. |

## TCP (`std/net/tcp.zc`)

### Type `TcpListener`

| Method | Signature | Description |
| :--- | :--- | :--- |
| **bind** | `TcpListener::bind(host: char*, port: int) -> Result<TcpListener>` | Binds to a local address. |
| **accept** | `accept(self) -> Result<TcpStream>` | Accepts a new connection. |

### Type `TcpStream`

| Method | Signature | Description |
| :--- | :--- | :--- |
| **connect** | `TcpStream::connect(host: char*, port: int) -> Result<TcpStream>` | Connects to a remote host. |
| **read** | `read(self, buf: char*, len: usize) -> Result<usize>` | Reads from the stream. |
| **write** | `write(self, buf: u8*, len: usize) -> Result<usize>` | Writes to the stream. |

## UDP (`std/net/udp.zc`)

### Type `UdpSocket`

| Method | Signature | Description |
| :--- | :--- | :--- |
| **bind** | `UdpSocket::bind(host: char*, port: int) -> Result<UdpSocket>` | Binds to a local address. |
| **recv_from** | `recv_from(self, buf: char*, len: usize) -> Result<UdpRecvResult>` | Receives data and sender info. |
| **send_to** | `send_to(self, buf: char*, len: usize, host: char*, port: int) -> Result<usize>` | Sends data to a specific destination. |

## DNS (`std/net/dns.zc`)

### Type `Dns`

| Method | Signature | Description |
| :--- | :--- | :--- |
| **resolve** | `Dns::resolve(host: char*) -> Result<String>` | Resolves a hostname to an IP address. |
