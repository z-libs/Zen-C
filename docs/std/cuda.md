# Standard Library: CUDA (`std/cuda.zc`)

The `std/cuda` module provides helper functions and types for CUDA interoperability, simplifying memory management, synchronization, and device queries.

> [!NOTE]
> This module requires compiling with the `--cuda` flag.

## Usage

```zc
import "std/cuda.zc"

fn main() {
    let dev_ptr = cuda_alloc<float>(1024);
    defer cuda_free(dev_ptr);
    
    cuda_sync();
}
```

## Struct Definition

```zc
struct CudaDeviceProp {
    name: String;
    total_global_mem: usize;
    multi_processor_count: int;
    major: int;
    minor: int;
    max_threads_per_block: int;
    warp_size: int;
}
```

## Methods

### Memory Management

| Method | Signature | Description |
| :--- | :--- | :--- |
| **cuda_alloc** | `cuda_alloc<T>(n: usize) -> T*` | Allocates device memory for `n` elements of type `T`. |
| **cuda_free** | `cuda_free(ptr: void*)` | Frees device memory. |
| **cuda_copy_to_device** | `cuda_copy_to_device(dst: void*, src: void*, bytes: usize)` | Copies data from host (CPU) to device (GPU). |
| **cuda_copy_to_host** | `cuda_copy_to_host(dst: void*, src: void*, bytes: usize)` | Copies data from device (GPU) to host (CPU). |
| **cuda_copy_device** | `cuda_copy_device(dst: void*, src: void*, bytes: usize)` | Copies data from device to device. |
| **cuda_zero** | `cuda_zero(ptr: void*, bytes: usize)` | Sets device memory to zero. |

### Synchronization

| Method | Signature | Description |
| :--- | :--- | :--- |
| **cuda_sync** | `cuda_sync()` | Synchronizes the device (blocks until all previous CUDA calls complete). |

### Device Information

| Method | Signature | Description |
| :--- | :--- | :--- |
| **cuda_device_count** | `cuda_device_count() -> int` | Returns the number of available CUDA devices. |
| **cuda_set_device** | `cuda_set_device(id: int)` | Sets the active CUDA device. |
| **cuda_device_properties** | `cuda_device_properties(device_id: int) -> CudaDeviceProp` | Returns properties of the specified device. |

### Device Functions (Kernel Only)

These functions are marked `@device` and should only be called from within a kernel (`@global`) or device functions.

| Function | Signature | Description |
| :--- | :--- | :--- |
| **thread_id** | `thread_id() -> int` | Global thread index. |
| **block_id** | `block_id() -> int` | Block index (`blockIdx.x`). |
| **local_id** | `local_id() -> int` | Local thread index (`threadIdx.x`). |
| **block_size** | `block_size() -> int` | Block dimension (`blockDim.x`). |
| **grid_size** | `grid_size() -> int` | Grid dimension (`gridDim.x`). |
