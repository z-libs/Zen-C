# Standard Library: Slice (`std/slice.zc`)

`Slice<T>` is a lightweight, non-owning view into a contiguous sequence of elements, useful for working with fixed-size arrays.

## Usage

```zc
import "std/slice.zc"

fn main() {
    let arr: int[5] = [1, 2, 3, 4, 5];
    
    // Direct iteration (auto-imports std/slice.zc)
    for val in arr {
        println "{val}";
    }
}
```

## Struct Definition

```zc
struct Slice<T> {
    data: T*;
    len: usize;
}
```

## Methods

### Construction

| Method | Signature | Description |
| :--- | :--- | :--- |
| **from_array** | `Slice<T>::from_array(arr: T*, len: usize) -> Slice<T>` | Creates a slice view over an array. |
| **new** | `Slice<T>::new(data: T*, len: usize) -> Slice<T>` | Alias for `from_array`. |

### Iteration

| Method | Signature | Description |
| :--- | :--- | :--- |
| **iterator** | `iterator(self) -> SliceIter<T>` | Returns an iterator for `for-in` loops. |

### Access & Query

| Method | Signature | Description |
| :--- | :--- | :--- |
| **length** | `length(self) -> usize` | Returns the number of elements. |
| **is_empty** | `is_empty(self) -> bool` | Returns true if length is 0. |
| **get** | `get(self, idx: usize) -> Option<T>` | Returns the element at index, or None if out of bounds. |
| **at** | `at(self, idx: usize) -> Option<T>` | Alias for `get`. |

## Notes

- `Slice<T>` does not own its data — it's just a view.
- No memory management needed (no `free()` method).
- **Auto-import**: `std/slice.zc` is automatically imported when using `for val in arr` on a fixed-size array.
