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

## Methods

### Output

| Method | Signature | Description |
| :--- | :--- | :--- |
| **print** | `io::print(fmt: char*, ...) -> int` | Prints formatted output to stdout. |
| **println** | `io::println(fmt: char*, ...) -> int` | Prints formatted output to stdout with a newline. |

### Formatting

| Method | Signature | Description |
| :--- | :--- | :--- |
| **format** | `io::format(fmt: char*, ...) -> char*` | Formats string into a static buffer. **Not thread-safe**. |
| **format_into** | `io::format_into(buf: char*, size: usize, fmt: char*, ...) -> int` | Formats string into a user-provided buffer. |
| **format_new** | `io::format_new(fmt: char*, ...) -> char*` | Formats string into a newly allocated buffer (caller must free). |

### Input

| Method | Signature | Description |
| :--- | :--- | :--- |
| **readln** | `io::readln() -> char*` | Reads a line from stdin (caller must free). |
| **read_rune** | `io::read_rune() -> rune` | Reads a single UTF-8 character from stdin. |
