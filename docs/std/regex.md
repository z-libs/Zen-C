# Standard Library: Regex (`std/regex.zc`)

The `std/regex` module providing regular expression support based on POSIX `regex.h`.

## Usage

```zc
import "std/regex.zc"

fn main() {
    if regex_match("^[a-z]+$", "hello") {
        println "Matches!";
    }
    
    let re = Regex::compile("\\d+");
    let count = re.count("123 abc 456");
    re.destroy();
}
```

## Struct Definitions

### `Regex`

Represents a compiled regular expression.

```zc
struct Regex {
    // Internal handles
}
```

### `Match`

Represents a successful regex match.

```zc
struct Match {
    text: char*;
    start: int;
    len: int;
}
```

## Methods

### Regex Construction

| Method | Signature | Description |
| :--- | :--- | :--- |
| **compile** | `Regex::compile(pattern: char*) -> Regex` | Compiles a regex pattern with default flags. |
| **compile_with_flags** | `Regex::compile_with_flags(pattern: char*, flags: int) -> Regex` | Compiles with custom POSIX flags. |
| **destroy** | `destroy(self)` | Frees the compiled regex. |

### Matching & Search

| Method | Signature | Description |
| :--- | :--- | :--- |
| **match** | `match(self, text: char*) -> bool` | Returns true if the pattern matches anywhere in `text`. |
| **find** | `find(self, text: char*) -> Option<Match>` | Returns the first match including position and length. |
| **count** | `count(self, text: char*) -> int` | Returns the number of non-overlapping matches. |
| **split** | `split(self, text: char*) -> Vec<String>` | Splits the text by the pattern. |

### Match Access

| Method | Signature | Description |
| :--- | :--- | :--- |
| **as_string** | `as_string(self) -> char*` | Returns a pointer to the start of the match. |
| **end** | `end(self) -> int` | Returns the index after the last character of the match. |

### Static Helper Functions

| Method | Signature | Description |
| :--- | :--- | :--- |
| **regex_match** | `regex_match(pattern: char*, text: char*) -> bool` | Quick check for a match. |
| **regex_find** | `regex_find(pattern: char*, text: char*) -> Option<Match>` | Find first match. |
| **regex_count** | `regex_count(pattern: char*, text: char*) -> int` | Count all matches. |
| **regex_split** | `regex_split(pattern: char*, text: char*) -> Vec<String>` | Split text by pattern. |
