#ifndef _WIN32

#include "zc_net_platform.h"

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#include <errno.h>
#include <string.h>
#include <stdio.h>

int _z_net_init(void) { return 0; }

int _z_socket(int domain, int type, int proto) {
    return socket(domain, type, proto);
}

int _z_net_last_error_code(void) {
    return errno;
}

int _z_net_last_error_message(char *out, int cap) {
    if (!out || cap <= 0) return 0;

    int code = errno;
    out[0] = 0;

    // strerror_r has two variants (XSI returns int, GNU returns char*)
    #if defined(__GLIBC__) && defined(_GNU_SOURCE)
        char *msg = strerror_r(code, out, (size_t)cap);
        if (msg != out) {
            // GNU may return pointer to static string
            strncpy(out, msg, (size_t)cap - 1);
            out[cap - 1] = 0;
        }
    #else
        // XSI/POSIX variant
        if (strerror_r(code, out, (size_t)cap) != 0) {
            snprintf(out, (size_t)cap, "errno %d", code);
        }
    #endif

    return (int)strlen(out);
}


int _z_net_close(int fd) {
    return close(fd);
}

int _z_net_bind(int fd, char *host, int port) {
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) <= 0) return -1;

    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) return -2;
    if (listen(fd, 10) < 0) return -3;
    return 0;
}

int _z_net_connect(int fd, char *host, int port) {
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) <= 0) return -1;

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) return -2;
    return 0;
}

int _z_net_accept(int fd) {
    return accept(fd, 0, 0);
}

int64_t _z_net_read(int fd, char *buf, size_t n) {
    ssize_t r = read(fd, (void*)buf, n);
    return (int64_t)r;
}

int64_t _z_net_write(int fd, char *buf, size_t n) {
    ssize_t w = write(fd, (const void*)buf, n);
    return (int64_t)w;
}

#endif
