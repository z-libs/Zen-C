# Standard Library: Option (`std/option.zc`)

`Option<T>` represents an optional value: every `Option` is either `Some` and contains a value, or `None`.

## Usage

```zc
import "std/option.zc"

fn main() {
    let val = Option<int>::Some(10);
    
    if (val.is_some()) {
        println "{val.unwrap()}";
    }
}
```

## Struct Definition

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

### Query & Extraction

| Method | Signature | Description |
| :--- | :--- | :--- |
| **is_some** | `is_some(self) -> bool` | Returns `true` if the option is `Some`. |
| **is_none** | `is_none(self) -> bool` | Returns `true` if the option is `None`. |
| **unwrap** | `unwrap(self) -> T` | Returns the contained value. Panics if `None`. |
| **unwrap_ref** | `unwrap_ref(self) -> T*` | Returns a pointer to the value. Panics if `None`. |
| **unwrap_or** | `unwrap_or(self, def: T) -> T` | Returns the contained value or `def`. |
| **expect** | `expect(self, msg: char*) -> T` | Returns the value or panics with `msg`. |
| **or_else** | `or_else(self, other: Option<T>) -> Option<T>` | Returns the option if `Some`, otherwise returns `other`. |

## Memory Management

| Method | Signature | Description |
| :--- | :--- | :--- |
| **forget** | `forget(self)` | Zeroes out memory without calling destructors. |
