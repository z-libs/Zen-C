#ifdef _WIN32
#include "zc_net_platform.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>

static int g_wsa_inited = 0;

int _z_net_init(void) {
    if (g_wsa_inited) return 0;
    WSADATA wsa;
    int r = WSAStartup(MAKEWORD(2,2), &wsa);
    if (r != 0) return -1;
    g_wsa_inited = 1;
    return 0;
}

int _z_socket(int domain, int type, int proto) {
    if (_z_net_init() != 0) return -1;
    SOCKET s = socket(domain, type, proto);
    if (s == INVALID_SOCKET) return -1;
    return (int)(uintptr_t)s;
}

int _z_net_close(int fd) {
    // sockets are SOCKET on Windows; treat fd as intptr-sized
    SOCKET s = (SOCKET)(uintptr_t)fd;
    return closesocket(s);
}

static int zc_set_reuseaddr(SOCKET s) {
    BOOL opt = TRUE;
    return setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
}

int _z_net_bind(int fd, char *host, int port) {
    if (_z_net_init() != 0) return -4;

    SOCKET s = (SOCKET)(uintptr_t)fd;

    struct sockaddr_in addr;
    ZeroMemory(&addr, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((u_short)port);
    if (InetPton(AF_INET, host, &addr.sin_addr) != 1) return -1;

    zc_set_reuseaddr(s);

    if (bind(s, (struct sockaddr*)&addr, (int)sizeof(addr)) == SOCKET_ERROR) return -2;
    if (listen(s, 10) == SOCKET_ERROR) return -3;
    return 0;
}

int _z_net_connect(int fd, char *host, int port) {
    if (_z_net_init() != 0) return -3;

    SOCKET s = (SOCKET)(uintptr_t)fd;

    struct sockaddr_in addr;
    ZeroMemory(&addr, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((u_short)port);
    if (InetPton(AF_INET, host, &addr.sin_addr) != 1) return -1;

    if (connect(s, (struct sockaddr*)&addr, (int)sizeof(addr)) == SOCKET_ERROR) return -2;
    return 0;
}

int _z_net_accept(int fd) {
    SOCKET s = (SOCKET)(uintptr_t)fd;
    SOCKET c = accept(s, NULL, NULL);
    if (c == INVALID_SOCKET) return -1;
    return (int)(uintptr_t)c;
}

int64_t _z_net_read(int fd, char *buf, size_t n) {
    SOCKET s = (SOCKET)(uintptr_t)fd;
    int r = recv(s, buf, (int)n, 0);
    return (int64_t)r;
}

int64_t _z_net_write(int fd, char *buf, size_t n) {
    SOCKET s = (SOCKET)(uintptr_t)fd;
    int r = send(s, buf, (int)n, 0);
    return (int64_t)r;
}

#endif
