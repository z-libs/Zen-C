# Standard Library: Path (`std/path.zc`)

`Path` provides cross-platform file path manipulation utilities. It wraps a `String` and handles separators (slash vs backslash) intelligently, though currently biased towards Unix defaults pending full Windows support integration.

## Usage

```zc
import "std/path.zc"

fn main() {
    let p = Path::new("/home/user");
    let full_path = p.join("docs/file.txt");
    
    println "Full path: {full_path.c_str()}";
    
    if (full_path.extension().is_some()) {
        println "Extension: {full_path.extension().unwrap()}";
    }
}
```

## Structure

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

### Manipulation

| Method | Signature | Description |
| :--- | :--- | :--- |
| **join** | `join(self, other: char*) -> Path` | Joins the current path with `other` using the correct separator. Returns a new Path. |
| **clone** | `clone(self) -> Path` | Creates a deep copy of the Path. |

### Parsing

| Method | Signature | Description |
| :--- | :--- | :--- |
| **extension** | `extension(self) -> Option<String>` | Returns the file extension (without the dot), or `None` if no extension. |
| **file_name** | `file_name(self) -> Option<String>` | Returns the last component of the path. |
| **parent** | `parent(self) -> Option<Path>` | Returns the path without its final component. |

### Access

| Method | Signature | Description |
| :--- | :--- | :--- |
| **c_str** | `c_str(self) -> char*` | Returns a pointer to the underlying C string. |
