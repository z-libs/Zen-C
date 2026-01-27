# Standard Library: String (`std/string.zc`)

`String` is a growable, heap-allocated string type. It wraps a `Vec<char>` and ensures null-termination for C compatibility.

## Usage

```zc
import "std/string.zc"

fn main() {
    let s = String::from("Hello");
    s.append(String::from(" World"));
    
    println "{s}"; // Prints "Hello World"
    
    if (s.starts_with("Hello")) {
        // ...
    }
} // s is freed automatically
```

## Structure

```zc
struct String {
    vec: Vec<char>;
}
```

## Methods

### Construction

| Method | Signature | Description |
| :--- | :--- | :--- |
| **new** | `String::new(s: char*) -> String` | Creates a new String from a C string primitive. |
| **from** | `String::from(s: char*) -> String` | Alias for `new`. |

### Modification

| Method | Signature | Description |
| :--- | :--- | :--- |
| **append** | `append(self, other: String*)` | Appends another string to this one. |
| **add** | `add(self, other: String*) -> String` | Concatenates this string and another into a new String. |

### Access & Query

| Method | Signature | Description |
| :--- | :--- | :--- |
| **c_str** | `c_str(self) -> char*` | Returns the underlying C string pointer. |
| **length** | `length(self) -> usize` | Returns the length of the string (excluding null terminator). |
| **is_empty** | `is_empty(self) -> bool` | Returns true if length is 0. |
| **starts_with** | `starts_with(self, prefix: char*) -> bool` | Checks if the string starts with the given prefix. |
| **ends_with** | `ends_with(self, suffix: char*) -> bool` | Checks if the string ends with the given suffix. |
| **contains** | `contains(self, target: char) -> bool` | Checks if the string contains the given character. |
| **find** | `find(self, target: char) -> Option<usize>` | Returns the index of the first occurrence of `target`. |
| **substring** | `substring(self, start: usize, len: usize) -> String` | Returns a new String containing the specified substring. |

### UTF-8 Support

These methods handle UTF-8 character boundaries correctly, contrasting with the byte-oriented methods above.

| Method | Signature | Description |
| :--- | :--- | :--- |
| **utf8_len** | `utf8_len(self) -> usize` | Returns the number of Unicode code points (characters). |
| **utf8_at** | `utf8_at(self, idx: usize) -> String` | Returns the character at the specified *character index* as a new String. |
| **utf8_substr** | `utf8_substr(self, start_idx: usize, num_chars: usize) -> String` | Returns a substring based on character indices and character count. |


### Transformations

| Method | Signature | Description |
| :--- | :--- | :--- |
| **split** | `split(self, delim: char) -> Vec<String>` | Splits the string into a vector of substrings separated by `delim`. |
| **trim** | `trim(self) -> String` | Returns a new string with leading and trailing whitespace removed. |
| **replace** | `replace(self, target: char*, replacement: char*) -> String` | Returns a new string with all occurrences of `target` replaced by `replacement`. |

### Comparison

| Method | Signature | Description |
| :--- | :--- | :--- |
| **eq** | `eq(self, other: String*) -> bool` | Returns true if the strings are equal content-wise. |

### Memory Management

| Method | Signature | Description |
| :--- | :--- | :--- |
| **free** | `free(self)` | Frees the string memory. |
| **destroy** | `destroy(self)` | Alias for `free`. |
| **forget** | `forget(self)` | Prevents automatic freeing (useful for transferring ownership). |
