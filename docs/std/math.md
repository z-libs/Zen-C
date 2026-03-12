# Standard Library: Math (`std/math.zc`)

The `Math` module provides a core set of standard mathematical constants and functions wrapped around the POSIX `<math.h>` standard library.

## Usage

```zc
import "std/math.zc"

fn main() {
    let radius = 5.0;
    let area = Math::PI() * Math::pow(radius, 2.0);
    println "Area of circle: {area}";
}
```

## Constants

All constants are functions returning a `double`.

| Constant | Description |
| :--- | :--- |
| **Math::PI()** | Archimedes' constant. |
| **Math::E()** | Euler's number. |

## Methods

### Arithmetic

| Method | Signature | Description |
| :--- | :--- | :--- |
| **abs** | `abs(x: double) -> double` | Returns the absolute value of `x`. |
| **sqrt** | `sqrt(x: double) -> double` | Returns the square root of `x`. |
| **pow** | `pow(base: double, exp: double) -> double` | Returns `base` raised to the power of `exp`. |
| **exp** | `exp(x: double) -> double` | Returns `e` raised to the power of `x`. |
| **log** | `log(x: double) -> double` | Returns the natural logarithm (base-e) of `x`. |
| **log10** | `log10(x: double) -> double` | Returns the base-10 logarithm of `x`. |

### Trigonometry

| Method | Signature | Description |
| :--- | :--- | :--- |
| **sin** | `sin(x: double) -> double` | Returns the sine of `x` (radians). |
| **cos** | `cos(x: double) -> double` | Returns the cosine of `x` (radians). |
| **tan** | `tan(x: double) -> double` | Returns the tangent of `x` (radians). |
| **asin** | `asin(x: double) -> double` | Returns the arcsine of `x`. |
| **acos** | `acos(x: double) -> double` | Returns the arccosine of `x`. |
| **atan** | `atan(x: double) -> double` | Returns the arctangent of `x`. |
| **atan2** | `atan2(y: double, x: double) -> double` | Returns the arctangent of `y/x`. |

### Rounding & Remainder

| Method | Signature | Description |
| :--- | :--- | :--- |
| **ceil** | `ceil(x: double) -> double` | Rounds up to the nearest integer. |
| **floor** | `floor(x: double) -> double` | Rounds down to the nearest integer. |
| **round** | `round(x: double) -> double` | Rounds to the closest integer. |
| **mod** | `mod(x: double, y: double) -> double` | Computes the floating-point remainder of `x / y`. |

### Min / Max

| Method | Signature | Description |
| :--- | :--- | :--- |
| **max** | `max(a: double, b: double) -> double` | Returns the larger of two values. |
| **min** | `min(a: double, b: double) -> double` | Returns the smaller of two values. |
