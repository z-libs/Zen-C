# Standard Library: Result (`std/result.zc`)

`Result<T>` is the standard type for error handling. It represents either success (`Ok`) containing a value, or failure (`Err`) containing a string error message.

## Usage

```zc
import "std/result.zc"

fn divide(a: int, b: int) -> Result<int> {
    if (b == 0) {
        return Result<int>::Err("Division by zero");
    }
    return Result<int>::Ok(a / b);
}

fn main() {
    let res = divide(10, 2);
    if (res.is_ok()) {
        println "Result: {res.unwrap()}";
    } else {
        println "Error: {res.err}";
    }
}
```

## Structure

```zc
struct Result<T> {
    is_ok: bool;
    val: T;
    err: char*;
}
```

## Methods

### Construction

| Method | Signature | Description |
| :--- | :--- | :--- |
| **Ok** | `Result<T>::Ok(v: T) -> Result<T>` | Creates a success result containing `v`. |
| **Err** | `Result<T>::Err(e: char*) -> Result<T>` | Creates an error result with message `e`. |

### Query / Extraction

| Method | Signature | Description |
| :--- | :--- | :--- |
| **is_ok** | `is_ok(self) -> bool` | Returns `true` if the result is `Ok`. |
| **is_err** | `is_err(self) -> bool` | Returns `true` if the result is `Err`. |
| **unwrap** | `unwrap(self) -> T` | Returns the contained value. Panics if `Err`. |
| **unwrap_ref** | `unwrap_ref(self) -> T*` | Returns a pointer to the contained value. Panics if `Err`. |
| **expect** | `expect(self, msg: char*) -> T` | Returns the contained value or panics with `msg` if `Err`. |
| **forget** | `forget(self)` | Zeroes out memory without destructors. |
