#ifndef _WIN32

#include "compat.h"
#include <time.h>
#include <unistd.h>
#include <stdlib.h>

void zc_seed_random(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    srand((unsigned int)(ts.tv_nsec ^ getpid()));
}

#endif
