
#ifndef TEST_FFI_NESTED_H
#define TEST_FFI_NESTED_H

int ffi_nested_add(int a, int b);
typedef struct FfiNested
{
    int val;
} FfiNested;
int ffi_nested_add(int a, int b)
{
    return a + b;
}

#endif
