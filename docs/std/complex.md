# Complex Number Library (`std/complex.zc`)

The `std/complex.zc` library provides the `Complex` struct and essential mathematical operations for working with complex numbers in Zen C.

## Complex Struct

The `Complex` struct represents a complex number with a real and imaginary part, both stored as double-precision floating-point numbers.

```rust
struct Complex {
    real: double;
    imag: double;
}
```

### Instantiation
```rust
let c = Complex::new(3.0, 4.0);
```

## Methods and Operators

Zen C supports operator overloading natively for `Complex` through the following methods:

| Operator | Method | Description |
| :--- | :--- | :--- |
| `+` | **add** | Adds two complex numbers. |
| `-` | **sub** | Subtracts one complex number from another. |
| `*` | **mul** | Multiplies two complex numbers. |
| `/` | **div** | Divides one complex number by another. |
| `==` | **eq** | Checks if two complex numbers are strictly equal. |
| `!=` | **neq** | Checks if two complex numbers are NOT strictly equal. |

## Properties

| Method | Signature | Description |
| :--- | :--- | :--- |
| **magnitude** | `magnitude(self) -> double` | Returns the magnitude (absolute value) of the complex number: `sqrt(real^2 + imag^2)`. |
| **phase** | `phase(self) -> double` | Returns the phase (angle) in radians: `atan2(imag, real)`. |

## String Formatting
`Complex` implements `to_string(self) -> String`, allowing for direct and intuitive string interpolation:
```rust
let c = Complex::new(3.0, 4.0);
println "Value: {c}"; // Value: 3.000000 + 4.000000i
```
