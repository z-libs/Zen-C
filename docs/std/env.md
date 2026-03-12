# Standard Library: Env (`std/env.zc`)

The `std/env` module provides access to process environment variables.

## Usage

```zc
import "std/env.zc"

fn main() {
    Env::set("HELLO", "world");

    let hello = Env::get("HELLO");

    if (hello.is_some()) {
        println "Hello {hello.unwrap()}";
    }
}
```

## Enum Definition

```zc
enum EnvRes {
    ERR,
    OK,
}
```

## Methods

### Access & Query

| Method | Signature | Description |
| :--- | :--- | :--- |
| **get** | `Env::get(name: char*) -> Option<char*>` | Retrieves an environment variable as a borrowed C string (no allocation). |
| **get_dup** | `Env::get_dup(name: char*) -> Option<String>` | Retrieves an environment variable as a heap-allocated `String`. |

### Modification

| Method | Signature | Description |
| :--- | :--- | :--- |
| **set** | `Env::set(name: char*, value: char*) -> EnvRes` | Sets an environment variable. |
| **unset** | `Env::unset(name: char*) -> EnvRes` | Unsets an environment variable. |
