
#include "os.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <process.h>
#else
#include <unistd.h>
#include <time.h>
#endif

void z_setup_terminal(void)
{
#ifdef _WIN32
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut == INVALID_HANDLE_VALUE)
    {
        return;
    }
    DWORD dwMode = 0;
    if (!GetConsoleMode(hOut, &dwMode))
    {
        return;
    }
    dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    SetConsoleMode(hOut, dwMode);

    HANDLE hErr = GetStdHandle(STD_ERROR_HANDLE);
    if (hErr == INVALID_HANDLE_VALUE)
    {
        return;
    }
    if (!GetConsoleMode(hErr, &dwMode))
    {
        return;
    }
    dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    SetConsoleMode(hErr, dwMode);
#endif
}

double z_get_monotonic_time(void)
{
#ifdef _WIN32
    static LARGE_INTEGER freq;
    static int init = 0;
    if (!init)
    {
        QueryPerformanceFrequency(&freq);
        init = 1;
    }
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
#endif
}

double z_get_time(void)
{
#ifdef _WIN32
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec + ts.tv_nsec / 1e9;
#else
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
#endif
}

#define MAX_PATH_SIZE 1024

const char *z_get_temp_dir(void)
{
#ifdef _WIN32
    static char tmp[MAX_PATH_SIZE] = {0};
    if (tmp[0])
    {
        return tmp;
    }

    if (GetTempPathA(sizeof(tmp), tmp))
    {
        // Remove trailing backslash if present
        int len = strlen(tmp);
        if (len > 0 && tmp[len - 1] == '\\')
        {
            tmp[len - 1] = 0;
        }
        return tmp;
    }
    return "C:\\Windows\\Temp";
#else
    return "/tmp";
#endif
}

int z_get_pid(void)
{
#ifdef _WIN32
    return _getpid();
#else
    return getpid();
#endif
}

void z_get_executable_path(char *buffer, size_t size)
{
    memset(buffer, 0, size);
#ifdef _WIN32
    GetModuleFileNameA(NULL, buffer, (DWORD)size);
#elif defined(__linux__)
    ssize_t len = readlink("/proc/self/exe", buffer, size - 1);
    if (len != -1)
    {
        buffer[len] = '\0';
    }
#elif defined(__APPLE__)
    // _NSGetExecutablePath usually needs <mach-o/dyld.h>
    // Fallback or leave empty
#else
    // Fallback
#endif
}

int z_isatty(int fd)
{
#ifdef _WIN32
    return _isatty(fd);
#else
    return isatty(fd);
#endif
}
