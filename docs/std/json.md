# Standard Library: JSON (`std/json.zc`)

The `std/json` module provides a DOM-style JSON parser and builder implementation.

## Usage

```zc
import "std/json.zc"

fn main() {
    let obj = JsonValue::object();
    obj.set("name", JsonValue::string("Alice"));
    obj.set("age", JsonValue::number(30.0));
    
    let json_str = obj.to_string();
    println "{json_str.c_str()}";  // {"name":"Alice","age":30}
    
    json_str.free();
    obj.free();
}
```

## Struct Definition

```zc
struct JsonValue {
    // Internal implementation details
}
```

## Methods

### Construction

| Method | Signature | Description |
| :--- | :--- | :--- |
| **null** | `JsonValue::null() -> JsonValue` | Creates a null value. |
| **bool** | `JsonValue::bool(b: bool) -> JsonValue` | Creates a boolean value. |
| **number** | `JsonValue::number(n: double) -> JsonValue` | Creates a numeric value. |
| **string** | `JsonValue::string(s: char*) -> JsonValue` | Creates a string value. |
| **array** | `JsonValue::array() -> JsonValue` | Creates an empty array. |
| **object** | `JsonValue::object() -> JsonValue` | Creates an empty object. |

### Parsing

| Method | Signature | Description |
| :--- | :--- | :--- |
| **parse** | `JsonValue::parse(json: char*) -> Result<JsonValue*>` | Parses a JSON string into a heap-allocated `JsonValue` tree. |

### Accessors

| Method | Signature | Description |
| :--- | :--- | :--- |
| **is_null** | `is_null(self) -> bool` | Checks if value is null. |
| **is_bool** | `is_bool(self) -> bool` | Checks if value is a boolean. |
| **is_number** | `is_number(self) -> bool` | Checks if value is a number. |
| **is_string** | `is_string(self) -> bool` | Checks if value is a string. |
| **is_array** | `is_array(self) -> bool` | Checks if value is an array. |
| **is_object** | `is_object(self) -> bool` | Checks if value is an object. |
| **as_string** | `as_string(self) -> Option<char*>` | Returns string content if applicable. |
| **as_int** | `as_int(self) -> Option<int>` | Returns integer value if applicable. |
| **as_float** | `as_float(self) -> Option<double>` | Returns float value if applicable. |
| **as_bool** | `as_bool(self) -> Option<bool>` | Returns boolean value if applicable. |

### Modification

| Method | Signature | Description |
| :--- | :--- | :--- |
| **push** | `push(self, val: JsonValue)` | Appends a value to an array. |
| **set** | `set(self, key: char*, val: JsonValue)` | Sets a key-value pair in an object. |
| **get** | `get(self, key: char*) -> Option<JsonValue*>` | Retrieves a value by key. |
| **at** | `at(self, index: usize) -> Option<JsonValue*>` | Retrieves a value by index. |

### Serialization

| Method | Signature | Description |
| :--- | :--- | :--- |
| **to_string** | `to_string(self) -> String` | Serializes to a string representation. |

## Memory Management

| Method | Signature | Description |
| :--- | :--- | :--- |
| **free** | `free(self)` | Recursively frees the JSON value and all its children. |
