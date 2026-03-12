# Standard Library: Thread (`std/thread.zc`)

The `std/thread` module provides primitives for multithreading and synchronization.

## Usage

```zc
import "std/thread.zc"

fn worker() {
    println "Hello from thread!";
}

fn main() {
    let t = Thread::spawn(worker).unwrap();
    t.join();
}
```

## Struct Definitions

### `Thread`

Represents a handle to a spawned thread.

```zc
struct Thread {
    // Internal handle
}
```

### `Mutex`

A mutual exclusion primitive for protecting shared data.

```zc
struct Mutex {
    // Internal sync primitives
}
```

## Methods

### `Thread` Methods

| Method | Signature | Description |
| :--- | :--- | :--- |
| **spawn** | `Thread::spawn(func: fn()) -> Result<Thread>` | Spawns a new thread executing `func`. |
| **join** | `join(self) -> Result<bool>` | Blocks current thread until the spawned thread finishes. |

### `Mutex` Methods

| Method | Signature | Description |
| :--- | :--- | :--- |
| **new** | `Mutex::new() -> Mutex` | Creates a new mutex. |
| **lock** | `lock(self)` | Acquires the lock (blocking). |
| **unlock** | `unlock(self)` | Releases the lock. |
| **free** | `free(self)` | Destroys the mutex and frees associated resources. |

### Utility Functions

| Method | Signature | Description |
| :--- | :--- | :--- |
| **sleep_ms** | `thread::sleep_ms(ms: int)` | Sleeps the current thread. |
