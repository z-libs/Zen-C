# Standard Library: Path (`std/path.zc`)

The `Path` module provides cross-platform file path manipulation utilities.

## Usage

```zc
import "std/path.zc"

fn main() {
    let p = Path::new("/home/user");
    let full_path = p.join("docs/file.txt");
    
    println "Full path: {full_path.c_str()}";
} // full_path and p are freed automatically here
```

## Struct Definition

```zc
struct Path {
    str: String;
}
```

## Methods

### Construction

| Method | Signature | Description |
| :--- | :--- | :--- |
| **new** | `Path::new(s: char*) -> Path` | Creates a new Path from a C string. |
| **from_string** | `Path::from_string(s: String) -> Path` | Creates a Path taking ownership of a String. |
| **clone** | `clone(self) -> Path` | Creates a deep copy of the Path. |

### Manipulation

| Method | Signature | Description |
| :--- | :--- | :--- |
| **join** | `join(self, other: char*) -> Path` | Joins path with `other` using correct separator. |

### Parsing

| Method | Signature | Description |
| :--- | :--- | :--- |
| **extension** | `extension(self) -> Option<String>` | Returns the file extension (without the dot). |
| **file_name** | `file_name(self) -> Option<String>` | Returns the last component of the path. |
| **parent** | `parent(self) -> Option<Path>` | Returns the path without its final component. |

### Access

| Method | Signature | Description |
| :--- | :--- | :--- |
| **c_str** | `c_str(self) -> char*` | Returns a pointer to the underlying C string. |

## Memory Management

| Method | Signature | Description |
| :--- | :--- | :--- |
| **free** | `free(self)` | Manually frees the path memory. |
| **Trait** | `impl Drop for Path` | Automatically calls `free()` when out of scope. |
