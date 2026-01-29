#ifndef ZENC_LOCAL_MATH_H
#define ZENC_LOCAL_MATH_H

#ifdef __cplusplus
extern "C" {
#endif

// Minimal interface needed by your Zen-C snippet.
double sin(double x);

// Provide abs too, so Zen-C's `extern fn abs(int)->int` can link.
int abs(int x);

#ifdef __cplusplus
}
#endif

#endif // ZENC_LOCAL_MATH_H
