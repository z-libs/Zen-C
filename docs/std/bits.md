# Standard Library: Bits (`std/bits.zc`)

The `std/bits` module provides low-level bitwise operations implemented via compiler intrinsics.

## Usage

```zc
import "std/bits.zc"

fn main() {
    let a: u32 = 0x80000001;
    let rotated = Bits::rotl32(a, 1); 
}
```

## Methods

### Bitwise Rotation

| Method | Signature | Description |
| :--- | :--- | :--- |
| **rotl8** | `Bits::rotl8(n: u8, c: u8) -> u8` | Rotates `n` left by `c` bits. |
| **rotr8** | `Bits::rotr8(n: u8, c: u8) -> u8` | Rotates `n` right by `c` bits. |
| **rotl16** | `Bits::rotl16(n: u16, c: u16) -> u16` | Rotates `n` left by `c` bits. |
| **rotr16** | `Bits::rotr16(n: u16, c: u16) -> u16` | Rotates `n` right by `c` bits. |
| **rotl32** | `Bits::rotl32(n: u32, c: u32) -> u32` | Rotates `n` left by `c` bits. |
| **rotr32** | `Bits::rotr32(n: u32, c: u32) -> u32` | Rotates `n` right by `c` bits. |
| **rotl64** | `Bits::rotl64(n: u64, c: u64) -> u64` | Rotates `n` left by `c` bits. |
| **rotr64** | `Bits::rotr64(n: u64, c: u64) -> u64` | Rotates `n` right by `c` bits. |
| **rotl128** | `Bits::rotl128(n: u128, c: u128) -> u128` | Rotates `n` left by `c` bits. |
| **rotr128** | `Bits::rotr128(n: u128, c: u128) -> u128` | Rotates `n` right by `c` bits. |

### Population Count

| Method | Signature | Description |
| :--- | :--- | :--- |
| **popcount8** | `Bits::popcount8(n: u8) -> u8` | Returns number of 1s in `n`. |
| **popcount16** | `Bits::popcount16(n: u16) -> u16` | Returns number of 1s in `n`. |
| **popcount32** | `Bits::popcount32(n: u32) -> u32` | Returns number of 1s in `n`. |
| **popcount64** | `Bits::popcount64(n: u64) -> u64` | Returns number of 1s in `n`. |
| **popcount128** | `Bits::popcount128(n: u128) -> u128` | Returns number of 1s in `n`. |

### Count Leading/Trailing Zeros

| Method | Signature | Description |
| :--- | :--- | :--- |
| **clz8** | `Bits::clz8(n: u8) -> u8` | Returns leading zero bits. |
| **ctz8** | `Bits::ctz8(n: u8) -> u8` | Returns trailing zero bits. |
| **clz64** | `Bits::clz64(n: u64) -> u64` | Returns leading zero bits in `u64`. |
| **ctz64** | `Bits::ctz64(n: u64) -> u64` | Returns trailing zero bits in `u64`. |

### Byte Swap & Bit Reversal

| Method | Signature | Description |
| :--- | :--- | :--- |
| **bswap16** | `Bits::bswap16(n: u16) -> u16` | Reverses byte order. |
| **bswap32** | `Bits::bswap32(n: u32) -> u32` | Reverses byte order in `u32`. |
| **reverse_bits8** | `Bits::reverse_bits8(n: u8) -> u8` | Reverses bit order. |
| **reverse_bits32** | `Bits::reverse_bits32(n: u32) -> u32` | Reverses bit order in `u32`. |
