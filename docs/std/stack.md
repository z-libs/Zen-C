# Standard Library: Stack (`std/stack.zc`)

The `std/stack` module provides a LIFO (Last-In, First-Out) stack data structure.

## Usage

```zc
import "std/stack.zc"

fn main() {
    let s = Stack<int>::new();
    s.push(10);
    s.push(20);
    
    let top = s.pop(); // Some(20)
} // s is freed automatically here
```

## Struct Definition

```zc
struct Stack<T> {
    // Internal implementation details
}
```

## Methods

### Construction

| Method | Signature | Description |
| :--- | :--- | :--- |
| **new** | `Stack<T>::new() -> Stack<T>` | Creates a new, empty stack. |
| **clone** | `clone(self) -> Stack<T>` | Creates a deep copy of the stack. |

### Modification

| Method | Signature | Description |
| :--- | :--- | :--- |
| **push** | `push(self, value: T)` | Pushes a value onto the top of the stack. |
| **pop** | `pop(self) -> Option<T>` | Removes and returns the top element of the stack. Returns `None` if empty. |
| **clear** | `clear(self)` | Removes all elements from the stack. |

### Access & Query

| Method | Signature | Description |
| :--- | :--- | :--- |
| **length** | `length(self) -> usize` | Returns the number of elements in the stack. |
| **is_empty** | `is_empty(self) -> bool` | Returns `true` if the stack contains no elements. |

## Memory Management

| Method | Signature | Description |
| :--- | :--- | :--- |
| **free** | `free(self)` | Manually frees the stack memory. |
| **Trait** | `impl Drop for Stack` | Automatically calls `free()` when the stack goes out of scope. |
