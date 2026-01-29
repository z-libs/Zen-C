#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Return values match your current conventions:
// _z_net_bind/connect: 0 on success, negative on error
// _z_net_accept: new fd or negative on error
// _z_net_read/write: bytes or negative on error
int     _z_net_init(void);
int     _z_socket(int domain, int type, int proto);

int     _z_net_bind(int fd, char *host, int port);
int     _z_net_connect(int fd, char *host, int port);
int     _z_net_accept(int fd);
int64_t _z_net_read(int fd, char *buf, size_t n);
int64_t _z_net_write(int fd, char *buf, size_t n);

// Close socket cross-platform (POSIX close, Win32 closesocket)
int     _z_net_close(int fd);

// Windows needs WSAStartup once. On POSIX it's a no-op.
// Safe to call multiple times.
int     _z_net_init(void);

// --- Error reporting ---
int     _z_net_last_error_code(void);
// Returns number of bytes written (excluding null terminator). Always null-terminates if cap>0.
int     _z_net_last_error_message(char *out, int cap);

#ifdef __cplusplus
}
#endif