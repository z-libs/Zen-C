# JSON (`std/json.zc`)

The `std/json` module provides a DOM-style JSON parser and builder.

## Usage

```zc
import "std/json.zc"
```

## Types

### Struct `JsonValue`

Represents a node in a JSON document.

#### Creation Methods

- **`fn null() -> JsonValue`**, **`fn null_ptr() -> JsonValue*`**
- **`fn bool(b: bool) -> JsonValue`**, **`fn bool_ptr(b: bool) -> JsonValue*`**
- **`fn number(n: double) -> JsonValue`**, **`fn number_ptr(n: double) -> JsonValue*`**
- **`fn string(s: char*) -> JsonValue`**, **`fn string_ptr(s: char*) -> JsonValue*`**
- **`fn array() -> JsonValue`**, **`fn array_ptr() -> JsonValue*`**
- **`fn object() -> JsonValue`**, **`fn object_ptr() -> JsonValue*`**

#### Parsing

- **`fn parse(json: char*) -> Result<JsonValue*>`**
  Parses a JSON string into a heap-allocated `JsonValue` tree.

#### Accessors

- **`fn is_null(self) -> bool`**, **`is_bool`**, **`is_number`**, **`is_string`**, **`is_array`**, **`is_object`**
  Check the type of the value.

- **`fn as_string(self) -> Option<char*>`**
  Returns `Some(string)` if the value is a string, `None` otherwise.
- **`fn as_int(self) -> Option<int>`**
- **`fn as_float(self) -> Option<double>`**
- **`fn as_bool(self) -> Option<bool>`**

#### Object/Array Operations

- **`fn push(self, val: JsonValue)`**
  Appends a value to an array.
- **`fn set(self, key: char*, val: JsonValue)`**
  Sets a key-value pair in an object.

- **`fn get(self, key: char*) -> Option<JsonValue*>`**
  Retrieves a value from an object by key.
- **`fn at(self, index: usize) -> Option<JsonValue*>`**
  Retrieves a value from an array by index.

#### Serialization

- **`fn to_string(self) -> String`**
  Serializes the JSON value to a string representation.
  
- **`fn stringify(self, buf: String*)`**
  Internal recursive serialization method that appends to a string buffer.
  
**Example:**
```zc
let obj = JsonValue::object();
obj.set("name", JsonValue::string("Alice"));
obj.set("age", JsonValue::number(30.0));

let json_str = obj.to_string();
println "{json_str.c_str()}";  // {"name":"Alice","age":30}
json_str.free();
obj.free();
```

**Features:**
- Proper escaping of special characters: `\"`, `\\`, `\n`, `\t`, `\r`, `\b`, `\f`
- Numbers formatted with `%.15g` for precision
- Recursive serialization for nested objects and arrays
- Round-trip compatible with `parse()`

#### Memory Management

- **`fn free(self)`**
  Recursively frees the JSON value and all its children.
