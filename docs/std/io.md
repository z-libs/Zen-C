# Standard Library: IO (`std/io.zc`)

The `std/io` module provides standard input/output functionality, including printing to stdout and reading from stdin.

## Usage

```zc
import "std/io.zc"

fn main() {
    // Printing
    io.println("Hello %s", "World");
    
    // Formatting strings
    autofree let s = io.format_new("Value: %d", 42);
    
    // Reading input
    io.print("Enter name: ");
    autofree let name = io.readln();
}
```

## Functions

### Output

| Function | Signature | Description |
| :--- | :--- | :--- |
| **print** | `print(fmt: char*, ...) -> int` | Prints formatted output to stdout. |
| **println** | `println(fmt: char*, ...) -> int` | Prints formatted output to stdout with a newline. |

### Formatting

| Function | Signature | Description |
| :--- | :--- | :--- |
| **format** | `format(fmt: char*, ...) -> char*` | Formats string into a static buffer (returns pointer). **Not thread-safe**. |
| **format_into** | `format_into(buf: char*, size: usize, fmt: char*, ...) -> int` | Formats string into a user-provided buffer. |
| **format_new** | `format_new(fmt: char*, ...) -> char*` | Formats string into a newly allocated buffer (caller must free). |

### Input

| Function | Signature | Description |
| :--- | :--- | :--- |
| **readln** | `readln() -> char*` | Reads a line from stdin. Returns heap-allocated string (caller must free) or `NULL` on EOF/error. |
