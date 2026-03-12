# Standard Library: Time (`std/time.zc`)

The `std/time` module provides functionality for measuring time and sleeping.

## Usage

```zc
import "std/time.zc"

fn main() {
    let start = Time::now();
    Time::sleep(Duration::from_secs(1));
    let end = Time::now();
    
    println "Elapsed: {end - start} ms";
}
```

## Struct Definitions

### `Duration`

Represents a span of time.

```zc
struct Duration {
    // Internal milliseconds count
}
```

### `Time`

Utilities for system time.

```zc
struct Time {
    // Static methods
}
```

## Methods

### `Duration` Methods

| Method | Signature | Description |
| :--- | :--- | :--- |
| **from_ms** | `Duration::from_ms(ms: U64) -> Duration` | Creates a duration from milliseconds. |
| **from_secs** | `Duration::from_secs(s: U64) -> Duration` | Creates a duration from seconds. |

### `Time` Static Methods

| Method | Signature | Description |
| :--- | :--- | :--- |
| **now** | `Time::now() -> U64` | Returns current system time in milliseconds since epoch. |
| **sleep** | `Time::sleep(d: Duration)` | Sleeps for the specified duration. |
| **sleep_ms** | `Time::sleep_ms(ms: U64)` | Sleeps for the specified number of milliseconds. |
