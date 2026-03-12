# Standard Library: Complex Number Library (`std/complex.zc`)

The `std/complex.zc` library provides the `Complex` struct and essential mathematical operations for working with complex numbers in Zen C.

## Usage

```zc
import "std/complex.zc"

fn main() {
    let c = Complex::new(3.0, 4.0);
    println "Value: {c}"; // Value: 3.000000 + 4.000000i
}
```

## Struct Definition

```zc
struct Complex {
    real: double;
    imag: double;
}
```

## Methods

### Construction

| Method | Signature | Description |
| :--- | :--- | :--- |
| **new** | `Complex::new(real: double, imag: double) -> Complex` | Creates a new complex number. |

### Access & Query

| Method | Signature | Description |
| :--- | :--- | :--- |
| **magnitude** | `magnitude(self) -> double` | Returns the magnitude (absolute value): `sqrt(real^2 + imag^2)`. |
| **phase** | `phase(self) -> double` | Returns the phase (angle) in radians: `atan2(imag, real)`. |

### Operators

Zen C supports operator overloading natively for `Complex` through the following methods:

| Operator | Method | Description |
| :--- | :--- | :--- |
| `+` | **add** | Adds two complex numbers. |
| `-` | **sub** | Subtracts one complex number from another. |
| `*` | **mul** | Multiplies two complex numbers. |
| `/` | **div** | Divides one complex number by another. |
| `==` | **eq** | Checks if two complex numbers are strictly equal. |
| `!=` | **neq** | Checks if two complex numbers are NOT strictly equal. |
| `{}` | **to_string** | Allows for direct string interpolation. |
