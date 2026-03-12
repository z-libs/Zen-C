# Standard Library: Memory (`std/mem.zc`)

The `std/mem` module provides core memory management utilities and standard traits for manual control and lifecycle management.

## Usage

```zc
import "std/mem.zc"

fn main() {
    let ptr = alloc<int>();
    mem_zero(ptr, 1);
    free(ptr);
}
```

## Methods

### Allocation

| Method | Signature | Description |
| :--- | :--- | :--- |
| **alloc** | `alloc<T>() -> T*` | Allocates memory for a single instance of `T`. |
| **zalloc** | `zalloc<T>() -> T*` | Allocates zero-initialized memory for `T`. |
| **alloc_n** | `alloc_n<T>(n: usize) -> T*` | Allocates memory for `n` instances of `T`. |

### Operations

| Method | Signature | Description |
| :--- | :--- | :--- |
| **mem_zero** | `mem_zero<T>(ptr: T*, count: usize)` | Sets memory to zero. |
| **mem_copy** | `mem_copy<T>(dst: T*, src: T*, count: usize)` | Copies memory buffer. |
| **swap** | `swap<T>(a: T*, b: T*)` | Swaps values at two locations. |

## Traits

| Trait | Method | Signature | Description |
| :--- | :--- | :--- | :--- |
| **Drop** | **drop** | `drop(self)` | Destructor run when object goes out of scope. |
| **Clone** | **clone** | `clone(self) -> Self` | Manual deep copy of resource-owning types. |
| **Copy** | *(Marker)* | N/A | Opt-in to implicit copying instead of moves. |

## Struct Definition: `Box<T>`

A simple smart pointer wrapper for heap-allocated memory.

```zc
struct Box<T> {
    ptr: T*;
}
```

### `Box` Methods

| Method | Signature | Description |
| :--- | :--- | :--- |
| **new** | `Box::new() -> Box<T>` | Allocates a new heap instance. |
| **from_ptr** | `Box::from_ptr(p: T*) -> Box<T>` | Creates a Box taking ownership of `p`. |
| **get** | `get(self) -> T*` | Returns the raw pointer. |
| **free** | `free(self)` | Frees the underlying memory manually. |
