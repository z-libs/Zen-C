# Standard Library: BigInt (`std/bigint.zc`)

`BigInt` provides arbitrary-precision integer arithmetic for Zen-C.

## Usage

```zc
import "std/bigint.zc"

fn main() {
    let a = BigInt::from_int(1_000_000_000_000_000);
    let b = BigInt::from_int(2_000_000_000_000_000);
    let sum = a + b; 
    println "Sum: {sum}";
} // sum, a, and b are freed automatically here
```

## Struct Definition

```zc
struct BigInt {
    digits: Vec<u8>*
}
```

## Methods

### Construction

| Method | Signature | Description |
| :--- | :--- | :--- |
| **new** | `BigInt::new() -> BigInt` | Creates a new `BigInt` initialized to 0. |
| **from_int** | `BigInt::from_int(val_in: u64) -> BigInt` | Creates a new `BigInt` from a `u64`. |

### Modification

| Method | Signature | Description |
| :--- | :--- | :--- |
| **add_in_place** | `add_in_place(self, other: BigInt)` | Adds `other` to `self` in place. |

### Utility

| Method | Signature | Description |
| :--- | :--- | :--- |
| **clone** | `clone(self) -> BigInt` | Returns a deep copy of the BigInt. |

## Operators

| Operator | Method | Description |
| :--- | :--- | :--- |
| `+` | **add** | Returns a new `BigInt` containing the sum. |
| `{}` | **to_string** | Converts to string for interpolation. |

## Memory Management

| Method | Signature | Description |
| :--- | :--- | :--- |
| **free** | `free_mem(self)` | Manually frees underlying memory. |
| **Trait** | `impl Drop for BigInt` | Automatically calls `free_mem()` when out of scope. |
