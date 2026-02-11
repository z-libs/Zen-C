#ifndef TEST_FFI_HELPER_H
#define TEST_FFI_HELPER_H

#include "test_ffi_nested.h"

int ffi_add(int a, int b);
double ffi_multiply(double a, double b);
void ffi_noop(void);
typedef struct FfiPoint
{
    int x;
    int y;
} FfiPoint;

typedef union FfiValue
{
    int i;
    double d;
} FfiValue;

int ffi_add(int a, int b)
{
    return a + b;
}
double ffi_multiply(double a, double b)
{
    return a * b;
}
void ffi_noop(void)
{
}

#endif
