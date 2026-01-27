# Standard Library: Map (`std/map.zc`)

`Map<V>` is a generic hash map implementation mapping string keys to values of type `V`.

## Usage

```zc
import "std/map.zc"

fn main() {
    let m = Map<int>::new();
    
    m.put("one", 1);
    m.put("two", 2);
    
    if (m.contains("one")) {
        let val = m.get("one");
        println "{val.unwrap()}";
    }
    
    m.remove("two");
} // m is freed automatically (but values are NOT freed automatically if they are pointers)
```

## Structure

```zc
struct Map<V> {
    keys: char**;
    vals: V*;
    // ... internal fields
}
```

## Methods

### Construction

| Method | Signature | Description |
| :--- | :--- | :--- |
| **new** | `Map<V>::new() -> Map<V>` | Creates a new, empty map. |

### Iteration

You can iterate over the map's key-value pairs using a `for` loop.

```zc
let m = Map<int>::new();
m.put("a", 1);

for entry in m {
    println "Key: {entry.key}, Val: {entry.val}";
}
```

The iterator yields a `MapEntry<V>` struct:
```zc
struct MapEntry<V> {
    key: char*;
    val: V;
}
```

### Modification

| Method | Signature | Description |
| :--- | :--- | :--- |
| **put** | `put(self, key: char*, val: V)` | Inserts or updates a key-value pair. |
| **remove** | `remove(self, key: char*)` | Removes a key and its value from the map. |
| **free** | `free(self)` | Frees the map's internal storage. **Note**: This does not free the values if they are pointers/objects. |

### Access

| Method | Signature | Description |
| :--- | :--- | :--- |
| **get** | `get(self, key: char*) -> Option<V>` | Retrieves the value associated with the key. |
| **contains** | `contains(self, key: char*) -> bool` | Returns true if the key exists. |
| **length** | `length(self) -> usize` | Returns the number of items in the map. |
| **is_empty** | `is_empty(self) -> bool` | Returns true if the map is empty. |
| **capacity** | `capacity(self) -> usize` | Returns the current capacity of the map. |


### Iteration Helpers
 
You can use index-based access if needed:

| Method | Signature | Description |
| :--- | :--- | :--- |
| **is_slot_occupied** | `is_slot_occupied(self, idx: usize) -> bool` | Checks if a raw slot index is occupied. |
| **key_at** | `key_at(self, idx: usize) -> char*` | Gets key at raw slot index. |
| **val_at** | `val_at(self, idx: usize) -> V` | Gets value at raw slot index. |
