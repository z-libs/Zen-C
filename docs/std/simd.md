# Standard Library: SIMD (`std/simd.zc`)

Zen C provides native SIMD (Single Instruction, Multiple Data) vector types that compile to hardware-optimized vector extensions.

## Usage

```zc
import "std/simd.zc"

fn main() {
    let a = f32x4{1.0, 2.0, 3.0, 4.0};
    let b = f32x4{v: 2.0}; // Broadcast
    let c = a + b;   // {3.0, 4.0, 5.0, 6.0}
}
```

## Defining Vector Types

Use the `@vector(N)` attribute on a single-field struct:

```zc
@vector(4)
struct f32x4 {
    v: f32;
}
```

## Predefined Types

The standard library provides 128-bit and 256-bit vector types.

### 128-bit Vectors (SSE/NEON)

| Type | Element | Lanes | Bytes |
| :--- | :--- | :--- | :--- |
| `f32x4` | `f32` | 4 | 16 |
| `f64x2` | `f64` | 2 | 16 |
| `i32x4` | `i32` | 4 | 16 |
| `u32x4` | `u32` | 4 | 16 |

### 256-bit Vectors (AVX)

| Type | Element | Lanes | Bytes |
| :--- | :--- | :--- | :--- |
| `f32x8` | `f32` | 8 | 32 |
| `f64x4` | `f64` | 4 | 32 |

## Operations

Operations work **element-wise** on every lane simultaneously.

| Category | Description |
| :--- | :--- |
| **Arithmetic** | `+`, `-`, `*`, `/` |
| **Bitwise** | `&`, `|`, `^` |
| **Access** | `v[i]` for individual lane access. |
