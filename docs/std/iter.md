# Standard Library: Iterator (`std/iter.zc`)

The `std/iter` module provides traits for defining custom iterators compatible with Zen C's `for-in` loop syntax.

## Usage

```zc
import "std/iter.zc"

fn main() {
    // Assuming my_collection implements Iterable<T>
    for item in my_collection {
        // ...
    }
}
```

## Traits

### `Iterator<T>`

An interface for advancing through a sequence.

```zc
trait Iterator<T> {
    fn next(self) -> Option<T>;
}
```

| Method | Signature | Description |
| :--- | :--- | :--- |
| **next** | `next(self) -> Option<T>` | Returns `Some(item)` if there is a next item, or `None` if iteration is complete. |

### `Iterable<T>`

An interface for types that can produce an `Iterator`.

```zc
trait Iterable<T> {
    fn iterator(self) -> Iterator<T>;
}
```

| Method | Signature | Description |
| :--- | :--- | :--- |
| **iterator** | `iterator(self) -> Iterator<T>` | Creates and returns an iterator for the collection. |
