# Standard Library: Queue (`std/queue.zc`)

`Queue<T>` is a generic First-In-First-Out (FIFO) queue implemented as a **Ring Buffer (Circular Buffer)**.

## Usage

```zc
import "std/queue.zc"

fn main() {
    let q = Queue<int>::new();
    
    q.push(1);
    q.push(2);
    q.push(3);
    
    // Pop returns an Option<T>
    if (q.pop().is_some()) {
        println "Popped: {q.pop().unwrap()}"; // 1
    }
}
```

## Implementation Details

- **Ring Buffer**: Uses a circular buffer with `head` and `tail` indices.
- **Performance**:
    - `push`: **Amortized O(1)** (resizes when full).
    - `pop`: **O(1)** (advances head index).
    - `clone`: **O(N)**.
- **Safety**: Safe handling of memory wrapping and resizing.

## Structure

```zc
struct Queue<T> {
    data: T*;
    cap: usize;
    head: usize;
    tail: usize;
    count: usize;
}
```

## Methods

### Construction

| Method | Signature | Description |
| :--- | :--- | :--- |
| **new** | `Queue<T>::new() -> Queue<T>` | Creates a new, empty queue. |
| **clone** | `clone(self) -> Queue<T>` | Creates a deep copy of the queue. |

### Modification

| Method | Signature | Description |
| :--- | :--- | :--- |
| **push** | `push(self, value: T)` | Adds an element to the back of the queue. |
| **pop** | `pop(self) -> Option<T>` | Removes and returns the element at the front. Returns `None` if empty. |
| **clear** | `clear(self)` | Removes all items from the queue. |

### Access / Query

| Method | Signature | Description |
| :--- | :--- | :--- |
| **length** | `length(self) -> usize` | Returns the number of items. |
| **is_empty** | `is_empty(self) -> bool` | Returns `true` if the queue is empty. |

### Memory Management

| Method | Signature | Description |
| :--- | :--- | :--- |
| **free** | `free(self)` | Frees the internal buffer. |
| **Trait** | `impl Drop for Queue` | Automatically calls `free()` when out of scope. |
