#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
//#include <unistd.h>
#include "compat/compat.h"
//#include <libc/cosmo.h>

int __real_main(int argc, char **argv);

static bool streq(const char *a, const char *b)
{
    return a && b && !strcmp(a, b);
}

static bool has_cc_flag(int argc, char **argv)
{
    for (int i = 1; i < argc; ++i)
    {
        const char *a = argv[i];
        if (a && streq(a, "--cc"))
        {
            return true;
        }
    }
    return false;
}

static const char *get_cc_value(int argc, char **argv)
{
    for (int i = 1; i < argc; ++i)
    {
        const char *a = argv[i];

        if (a && streq(a, "--cc"))
        {
            if (i + 1 < argc)
            {
                return argv[i + 1];
            }
            return NULL;
        }
    }
    return NULL;
}

static bool cc_is_cosmocc(int argc, char **argv)
{
    const char *cc = get_cc_value(argc, argv);
    if (!cc)
    {
        return false;
    }
    const char *p = strstr(cc, "cosmocc");
    return p != NULL;
}

static bool is_cmd(int argc, char **argv, const char *cmd)
{
    // usage: zc [command] [options] <file.zc>
    return argc > 1 && streq(argv[1], cmd);
}

static bool has_o_flag(int argc, char **argv)
{
    for (int i = 1; i < argc; ++i)
    {
        const char *a = argv[i];
        if (!a)
        {
            continue;
        }
        if (streq(a, "-o"))
        {
            return true;
        }
        if (!strncmp(a, "-o", 2) && a[2])
        {
            return true;
        }
    }
    return false;
}

static const char *get_out_value(int argc, char **argv)
{
    for (int i = 1; i < argc; ++i)
    {
        const char *a = argv[i];
        if (!a)
        {
            continue;
        }

        if (streq(a, "-o"))
        {
            if (i + 1 < argc)
            {
                return argv[i + 1];
            }
            return NULL;
        }
        if (!strncmp(a, "-o", 2) && a[2])
        {
            return a + 2;
        }
    }
    return NULL;
}

static void inject_default_out(int *pargc, char ***pargv, const char *out)
{
    int argc = *pargc;
    char **argv = *pargv;

    if (!(is_cmd(argc, argv, "run") || is_cmd(argc, argv, "build")))
    {
        return;
    }
    if (has_o_flag(argc, argv))
    {
        return;
    }

    char **newv = (char **)malloc((size_t)(argc + 2 + 1) * sizeof(char *));
    if (!newv)
    {
        return;
    }

    int k = 0;
    newv[k++] = argv[0];
    newv[k++] = argv[1];
    newv[k++] = (char *)"-o";
    newv[k++] = (char *)out;

    for (int i = 2; i < argc; ++i)
    {
        newv[k++] = argv[i];
    }
    newv[k] = NULL;

    *pargc = k;
    *pargv = newv;
}

static void unlink_if_exists(const char *path)
{
    if (!path || !*path)
    {
        return;
    }
    int old = errno;
    if (unlink(path) == -1)
    {
        if (errno == ENOENT)
        {
            errno = old;
        }
    }
    else
    {
        errno = old;
    }
}

static void cleanup_cosmocc_out(const char *out)
{
    char buf[4096];

    if (snprintf(buf, sizeof(buf), "%s.dbg", out) < (int)sizeof(buf))
    {
        unlink_if_exists(buf);
    }

    size_t n = strlen(out);
    if (n > 4 && !strcmp(out + (n - 4), ".com"))
    {
        char base[4096];
        if (n - 4 < sizeof(base))
        {
            memcpy(base, out, n - 4);
            base[n - 4] = 0;

            if (snprintf(buf, sizeof(buf), "%s.aarch64.elf", base) < (int)sizeof(buf))
            {
                unlink_if_exists(buf);
            }
        }
    }
}

static bool out_exists(const char *out)
{
    char buf[4096];
    if (snprintf(buf, sizeof(buf), "%s.dbg", out) < (int)sizeof(buf))
    {
        if (access(buf, F_OK) != -1)
        {
            return true;
        }
    }

    char base[4096];
    size_t n = strlen(out);
    if (n > 4 && !strcmp(out + (n - 4), ".com"))
    {
        if (n - 4 < sizeof(base))
        {
            memcpy(base, out, n - 4);
            base[n - 4] = 0;
            if (snprintf(buf, sizeof(buf), "%s.aarch64.elf", base) < (int)sizeof(buf))
            {
                if (access(buf, F_OK) != -1)
                {
                    return true;
                }
            }
        }
    }
    return false;
}

static void ensure_env(const char *key, const char *val)
{
    const char *cur = getenv(key);
    if (cur && *cur)
    {
        return;
    }
    setenv(key, val, 0);
}

int __wrap_main(int argc, char **argv)
{
    ensure_env("ZC_ROOT", "/zip");

    int newargc = cosmo_args("/zip/.args", &argv);
    if (newargc != -1)
    {
        argc = newargc;
    }

    inject_default_out(&argc, &argv, "a.out.com");

    const char *out = get_out_value(argc, argv);

    bool do_cleanup = is_cmd(argc, argv, "run") && cc_is_cosmocc(argc, argv) && !out_exists(out);

    int rc = __real_main(argc, argv);

    if (do_cleanup)
    {
        cleanup_cosmocc_out(out);
    }

    return rc;
}
