# Standard Library: Set (`std/set.zc`)

The `std/set` module provides a Generic Hash Set `Set<T>`.

## Usage

```zc
import "std/set.zc"

fn main() {
    let s = Set<int>::new();
    s.add(10);
    s.add(20);
    
    if (s.contains(10)) {
        println "Set contains 10";
    }
} // s is freed automatically here
```

## Struct Definition

```zc
struct Set<T> {
    // Internal implementation details
}
```

## Methods

### Construction

| Method | Signature | Description |
| :--- | :--- | :--- |
| **new** | `Set<T>::new() -> Set<T>` | Creates a new, empty set. |

### Modification

| Method | Signature | Description |
| :--- | :--- | :--- |
| **add** | `add(self, val: T) -> bool` | Adds a value to the set. Returns `true` if the value was added, `false` if it was already present. |
| **remove** | `remove(self, val: T) -> bool` | Removes a value from the set. Returns `true` if present and removed. |
| **clear** | `clear(self)` | Removes all elements from the set. |

### Access & Query

| Method | Signature | Description |
| :--- | :--- | :--- |
| **contains** | `contains(self, val: T) -> bool` | Returns `true` if the set contains the value. |
| **length** | `length(self) -> usize` | Returns the number of elements in the set. |
| **is_empty** | `is_empty(self) -> bool` | Returns `true` if the set is empty. |

## Memory Management

| Method | Signature | Description |
| :--- | :--- | :--- |
| **free** | `free(self)` | Manually frees the set memory. |
| **Trait** | `impl Drop for Set` | Automatically calls `free()` when the set goes out of scope. |
