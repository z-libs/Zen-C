
<div align="center">

# Zen C

**Modern Ergonomics. Zero Overhead. Pure C.**

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![Version](https://img.shields.io/badge/version-0.1.0-orange)]()
[![Platform](https://img.shields.io/badge/platform-linux-lightgrey)]()

*Write like a high-level language, run like C.*

</div>

---

## Overview

**Zen C** is a modern systems programming language that compiles to human-readable `GNU C`/`C11`. It provides a rich feature set including type inference, pattern matching, generics, traits, async/await, and manual memory management with RAII capabilities, all while maintaining 100% C ABI compatibility.

---

## Quick Start

### Installation

```bash
git clone https://github.com/z-libs/Zen-C.git
cd Zen-C
make
sudo make install
```

### Usage

```bash
# Compile and run
zc run hello.zc

# Build executable
zc build hello.zc -o hello

# Interactive Shell
zc repl
```

---

## Language Reference

### 1. Variables and Constants

Zen C uses type inference by default.

```zc
var x = 42;                 // Inferred as int
const PI = 3.14159;         // Compile-time constant
var explicit: float = 1.0;  // Explicit type
```

#### Mutability
By default, variables are immutable. You can enable **Mutable by Default** mode using a directive.

```zc
var x = 10;
// x = 20; // Error: x is immutable

var mut y = 10;
y = 20;    // OK
```

### 2. Primitive Types

| Type | C Equivalent | Description |
|:---|:---|:---|
| `int`, `uint` | `int`, `unsigned int` | Platform standard integer |
| `I8` .. `I128` or `i8` .. `i128` | `int8_t` .. `__int128_t` | Signed fixed-width integers |
| `U8` .. `U128` or `u8` .. `u128` | `uint8_t` .. `__uint128_t` | Unsigned fixed-width integers |
| `isize`, `usize` | `ptrdiff_t`, `size_t` | Pointer-sized integers |
| `byte` | `uint8_t` | Alias for U8 |
| `F32`, `F64` or `f32`, `f64`  | `float`, `double` | Floating point numbers |
| `bool` | `bool` | `true` or `false` |
| `char` | `char` | Single character |
| `string` | `char*` | C-string (null-terminated) |
| `U0`, `u0`, `void` | `void` | Empty type |

### 3. Aggregate Types

#### Arrays
Fixed-size arrays with value semantics.
```zc
var ints: int[5] = {1, 2, 3, 4, 5};
var zeros: [int; 5]; // Zero-initialized
```

#### Tuples
Group multiple values together.
```zc
var pair = (1, "Hello");
var x = pair.0;
var s = pair.1;
```

#### Structs
Data structures with optional bitfields.
```zc
struct Point {
    x: int;
    y: int;
}

// Struct initialization
var p = Point { x: 10, y: 20 };

// Bitfields
struct Flags {
    valid: U8 : 1;
    mode:  U8 : 3;
}
```

#### Enums
Tagged unions (Sum types) capable of holding data.
```zc
enum Shape {
    Circle(float),      // Holds radius
    Rect(float, float), // Holds width, height
    Point               // No data
}
```

#### Unions
Standard C unions (unsafe access).
```zc
union Data {
    i: int;
    f: float;
}
```

### 4. Functions & Lambdas

#### Functions
```zc
fn add(a: int, b: int) -> int {
    return a + b;
}

// Named arguments supported in calls
add(a: 10, b: 20);
```

#### Lambdas (Closures)
Anonymous functions that can capture their environment.
```zc
var factor = 2;
var double = x -> x * factor;  // Arrow syntax
var full = fn(x: int) -> int { return x * factor; }; // Block syntax
```

### 5. Control Flow

#### Conditionals
```zc
if x > 10 {
    print("Large");
} else if x > 5 {
    print("Medium");
} else {
    print("Small");
}

// Ternary
var y = if x > 10 ? 1 : 0;
```

#### Pattern Matching
Powerful alternative to `switch`.
```zc
match val {
    1 => print("One"),
    2 | 3 => print("Two or Three"),
    4..10 => print("Range"),
    _ => print("Other")
}

// Destructuring Enums
match shape {
    Circle(r) => print(f"Radius: {r}"),
    Rect(w, h) => print(f"Area: {w*h}"),
    Point => print("Point")
}
```

#### Loops
```zc
// Range
for i in 0..10 { ... }
for i in 0..10 step 2 { ... }

// Iterator/Collection
for item in vec { ... }

// While
while x < 10 { ... }

// Infinite with label
outer: loop {
    if done { break outer; }
}

// Repeat
repeat 5 { ... }
```

#### Advanced Control
```zc
// Guard: Execute else and return if condition is false
guard ptr != NULL else { return; }

// Unless: If not true
unless is_valid { return; }
```

### 6. Operators

| Operator | Description | Function Mapping |
|:---|:---|:---|
| `+`, `-`, `*`, `/`, `%` | Arithmetic | `add`, `sub`, `mul`, `div`, `rem` |
| `==`, `!=`, `<`, `>` | Comparison | `eq`, `neq`, `lt`, `gt` |
| `[]` | Indexing | `get`, `set` |
| `??` | Null Coalescing (`val ?? default`) | - |
| `??=` | Null Assignment (`val ??= init`) | - |
| `?.` | Safe Navigation (`ptr?.field`) | - |
| `?` | Try Operator (`res?` returns error if present) | - |

### 7. Memory Management

Zen C allows manual memory management with ergonomic aids.

#### Defer
Execute code when the current scope exits.
```zc
var f = fopen("file.txt", "r");
defer fclose(f);
```

#### Autofree
Automatically free the variable when scope exits.
```zc
autofree var types = malloc(1024);
```

#### RAII / Drop Trait
Implement `Drop` to run cleanup logic automatically.
```zc
impl Drop for MyStruct {
    fn drop(mut self) {
        free(self.data);
    }
}
```

### 8. Object Oriented Programming

#### Methods
Define methods on types using `impl`.
```zc
impl Point {
    // Static method (constructor convention)
    fn new(x: int, y: int) -> Point {
        return Point{x: x, y: y};
    }

    // Instance method
    fn dist(self) -> float {
        return sqrt(self.x * self.x + self.y * self.y);
    }
}
```

#### Traits
Define shared behavior.
```zc
trait Drawable {
    fn draw(self);
}

impl Drawable for Circle {
    fn draw(self) { ... }
}
```

#### Composition
Use `use` to mixin fields from another struct.
```zc
struct Entity { id: int; }
struct Player {
    use Entity; // Adds 'id' field
    name: string;
}
```

### 9. Generics

Type-safe templates for Structs and Functions.

```zc
// Generic Struct
struct Box<T> {
    item: T;
}

// Generic Function
fn identity<T>(val: T) -> T {
    return val;
}
```

### 10. Concurrency (Async/Await)

Built on pthreads.

```zc
async fn fetch_data() -> string {
    // Runs in background
    return "Data";
}

fn main() {
    var future = fetch_data();
    var result = await future;
}
```

### 11. Metaprogramming

#### Comptime
Run code at compile-time to generate source or print messages.
```zc
comptime {
    print("Compiling...");
}
```

#### Embed
Embed files as byte arrays.
```zc
var png = embed "assets/logo.png";
```

#### Plugins
Import compiler plugins to extend syntax.
```zc
import plugin "regex"
var re = regex! { ^[a-z]+$ };
```

#### Generic C Macros
Pass preprocessor macros through to C.
```zc
#define MAX_BUFFER 1024
```

### 12. Attributes

Decorate functions and structs to modify compiler behavior.

| Attribute | Scope | Description |
|:---|:---|:---|
| `@must_use` | Fn | Warn if return value is ignored. |
| `@deprecated("msg")` | Fn/Struct | Warn on usage with message. |
| `@inline` | Fn | Hint compiler to inline. |
| `@noinline` | Fn | Prevent inlining. |
| `@packed` | Struct | Remove padding between fields. |
| `@align(N)` | Struct | Force alignment to N bytes. |
| `@constructor` | Fn | Run before main. |
| `@destructor` | Fn | Run after main exits. |
| `@unused` | Fn/Var | Suppress unused variable warnings. |
| `@weak` | Fn | Weak symbol linkage. |
| `@section("name")` | Fn | Place code in specific section. |
| `@noreturn` | Fn | Function does not return (e.g. exit). |
| `@derived(...)` | Struct | Auto-implement traits (e.g. `Debug`). |

### 13. Inline Assembly

Zen C provides first-class support for inline assembly, transpiling directly to GCC-style extended `asm`.

#### Basic Usage
Write raw assembly within `asm` blocks. Strings are concatenated automatically.
```zc
asm {
    "nop"
    "mfence"
}
```

#### Volatile
Prevent the compiler from optimizing away assembly that has side effects.
```zc
asm volatile {
    "rdtsc"
}
```

#### Named Constraints
Zen C simplifies the complex GCC constraint syntax with named bindings.

```zc
// Syntax: : out(var) : in(var) : clobber(reg)
// Uses {var} placeholder syntax for readability

fn add(a: int, b: int) -> int {
    var result: int;
    asm {
        "add {result}, {a}, {b}"
        : out(result)
        : in(a), in(b)
        : clobber("cc")
    }
    return result;
}
```

| Type | Syntax | GCC Equivalent |
|:---|:---|:---|
| **Output** | `: out(var)` | `"=r"(var)` |
| **Input** | `: in(var)` | `"r"(var)` |
| **Clobber** | `: clobber("rax")` | `"rax"` |
| **Memory** | `: clobber("memory")` | `"memory"` |

> **Note:** When using Intel syntax (via `-masm=intel`), you must ensure your build is configured correctly (for example, `//> cflags: -masm=intel`). TCC does not support Intel syntax assembly.

### 14. Build Directives

Zen C supports special comments at the top of your source file to configure the build process without needing a complex build system or Makefile.

| Directive | Arguments | Description |
|:---|:---|:---|
| `//> link:` | `-lfoo` or `path/to/lib.a` | Link against a library or object file. |
| `//> lib:` | `path/to/libs` | Add a library search path (`-L`). |
| `//> include:` | `path/to/headers` | Add an include search path (`-I`). |
| `//> cflags:` | `-Wall -O3` | Pass arbitrary flags to the C compiler. |
| `//> define:` | `MACRO` or `KEY=VAL` | Define a preprocessor macro (`-D`). |
| `//> pkg-config:` | `gtk+-3.0` | Run `pkg-config` and append `--cflags` and `--libs`. |
| `//> shell:` | `command` | Execute a shell command during the build. |
| `//> get:` | `http://url/file` | Download a file if specific file does not exist. |
| `//> immutable-by-default` | None | Make variables immutable unless declared `mut`. |

#### Examples

```zc
//> include: ./include
//> lib: ./libs
//> link: -lraylib -lm
//> cflags: -Ofast
//> pkg-config: gtk+-3.0

import "raylib.h"

fn main() { ... }
```

---

## Compiler Support & Compatibility

Zen C is designed to work with most C11 compilers. Some features rely on GNU C extensions, but these often work in other compilers. Use the `--cc` flag to switch backends.

```bash
zc run app.zc --cc clang
zc run app.zc --cc zig
```

### Test Suite Status

| Compiler | Pass Rate | Supported Features | Known Limitations |
|:---|:---:|:---|:---|
| **GCC** | **100%** | All Features | None. |
| **Clang** | **100%** | All Features | None. |
| **Zig** | **100%** | All Features | None. Uses `zig cc` as a drop-in C compiler. |
| **TCC** | **~70%** | Basic Syntax, Generics, Traits | No `__auto_type`, No Intel ASM, No Nested Functions. |

> **Recommendation:** Use **GCC**, **Clang**, or **Zig** for production builds. TCC is excellent for rapid prototyping due to its compilation speed but misses some advanced C extensions Zen C relies on for full feature support.

### Building with Zig

Zig's `zig cc` command provides a drop-in replacement for GCC/Clang with excellent cross-compilation support. To use Zig:

```bash
# Compile and run a Zen C program with Zig
zc run app.zc --cc zig

# Build the Zen C compiler itself with Zig
make zig
```

---

## Contributing

We welcome contributions! Whether it's fixing bugs, adding documentation, or proposing new features.

### How to Contribute
1.  **Fork the Repository**: standard GitHub workflow.
2.  **Create a Feature Branch**: `git checkout -b feature/NewThing`.
3.  **Code Guidelines**:
    *   Follow the existing C style.
    *   Ensure all tests pass: `make test`.
    *   Add new tests for your feature in `tests/`.
4.  **Submit a Pull Request**: Describe your changes clearly.

### Running Tests
The test suite is your best friend.

```bash
# Run all tests (GCC)
make test

# Run specific test
./zc run tests/test_match.zc

# Run with different compiler
./tests/run_tests.sh --cc clang
./tests/run_tests.sh --cc zig
./tests/run_tests.sh --cc tcc
```

### Extending the Compiler
*   **Parser**: `src/parser/` - Recursive descent parser.
*   **Codegen**: `src/codegen/` - Transpiler logic (Zen C -> GNU C/C11).
*   **Standard Library**: `std/` - Written in Zen C itself.
