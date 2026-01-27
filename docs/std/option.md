# Standard Library: Option (`std/option.zc`)

`Option<T>` represents an optional value: every `Option` is either `Some` and contains a value, or `None`, and does not.

## Usage

```zc
import "std/option.zc"

fn main() {
    let val = Option<int>::Some(10);
    
    if (val.is_some()) {
        println "{val.unwrap()}";
    }
    
    let nothing = Option<int>::None();
    println "{nothing.unwrap_or(0)}"; // Prints 0
}
```

## Structure

```zc
struct Option<T> {
    is_some: bool;
    val: T;
}
```

## Methods

### Construction

| Method | Signature | Description |
| :--- | :--- | :--- |
| **Some** | `Option<T>::Some(v: T) -> Option<T>` | Creates a `Some` option containing `v`. |
| **None** | `Option<T>::None() -> Option<T>` | Creates a `None` option. |

### Query / Extraction

| Method | Signature | Description |
| :--- | :--- | :--- |
| **is_some** | `is_some(self) -> bool` | Returns `true` if the option is a `Some` value. |
| **is_none** | `is_none(self) -> bool` | Returns `true` if the option is a `None` value. |
| **unwrap** | `unwrap(self) -> T` | Returns the contained value. Panics if `None`. |
| **unwrap_ref** | `unwrap_ref(self) -> T*` | Returns a pointer to the contained value. Panics if `None`. |
| **unwrap_or** | `unwrap_or(self, def: T) -> T` | Returns the contained value or `def` if `None`. |
| **expect** | `expect(self, msg: char*) -> T` | Returns the contained value or panics with `msg` if `None`. |
| **or_else** | `or_else(self, other: Option<T>) -> Option<T>` | Returns the option if it contains a value, otherwise returns `other`. |
| **forget** | `forget(self)` | Zeroes out memory without destructors (if applicable). |
