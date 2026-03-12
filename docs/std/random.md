# Standard Library: Random (`std/random.zc`)

The `std/random` module provides an idiomatic, object-oriented pseudo-random number generator (PRNG) wrapper around POSIX `<stdlib.h>` functions.

## Usage

```zc
import "std/random.zc"

fn main() {
    // Automatically seeds the generator with the current time
    let rng = Random::new();

    // Generate random integers
    let bounded = rng.next_int_range(1, 100); // 1 to 100 inclusive
    
    println "Rolled: {bounded}";
}
```

## Struct Definition

```zc
struct Random {
    seed: U32;
}
```

## Methods

### Initialization

| Method | Signature | Description |
| :--- | :--- | :--- |
| **new** | `Random::new() -> Random` | Creates a new random generator seeded with current system time. |
| **from_seed** | `Random::from_seed(seed: U32) -> Random` | Creates a new random generator using a specific seed. |

### Generation

| Method | Signature | Description |
| :--- | :--- | :--- |
| **next_int** | `next_int(self) -> int` | Returns a random integer in raw range `[0, RAND_MAX]`. |
| **next_int_range** | `next_int_range(self, min: int, max: int) -> int` | Returns a random integer in range `[min, max]` inclusive. |
| **next_double** | `next_double(self) -> double` | Returns a random floating-point number in range `[0.0, 1.0)`. |
| **next_bool** | `next_bool(self) -> bool` | Returns a random boolean. |
