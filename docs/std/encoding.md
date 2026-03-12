# Standard Library: Encoding (`std/encoding/`)

The `std/encoding` module provides data encoding and decoding utilities.

## Base64 (`std/encoding/base64.zc`)

Base64 encoding implementation (RFC 4648).

### Usage

```zc
import "std/encoding/base64.zc"

fn main() {
    let data = "Hello";
    let encoded = Base64::encode((u8*)data, 5);
    // encoded is "SGVsbG8="
}
```

### Methods

| Method | Signature | Description |
| :--- | :--- | :--- |
| **encode** | `Base64::encode(data: u8*, len: usize) -> char*` | Encodes data into a Base64 string. |
| **decode** | `Base64::decode(s: char*) -> Vec<u8>` | Decodes a Base64 string into raw bytes. |
