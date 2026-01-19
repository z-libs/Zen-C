
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

## Community

Join the discussion, share demos, ask questions, or report bugs in the official Zen C Discord server!

- Discord: [Join here](https://discord.com/invite/q6wEsCmkJP)

---

## Index

- [Overview](#overview)
- [Community](#community)
- [Quick Start](#quick-start)
    - [Installation](#installation)
    - [Usage](#usage)
    - [Environment Variables](#environment-variables)
- [Language Reference](#language-reference)
    - [1. Variables and Constants](#1-variables-and-constants)
    - [2. Primitive Types](#2-primitive-types)
    - [3. Aggregate Types](#3-aggregate-types)
        - [Arrays](#arrays)
        - [Tuples](#tuples)
        - [Structs](#structs)
        - [Enums](#enums)
        - [Unions](#unions)
        - [Type Aliases](#type-aliases)
    - [4. Functions & Lambdas](#4-functions--lambdas)
        - [Functions](#functions)
        - [Lambdas (Closures)](#lambdas-closures)
    - [5. Control Flow](#5-control-flow)
        - [Conditionals](#conditionals)
        - [Pattern Matching](#pattern-matching)
        - [Loops](#loops)
        - [Advanced Control](#advanced-control)
    - [6. Operators](#6-operators)
        - [Overloadable Operators](#overloadable-operators)
        - [Syntactic Sugar](#syntactic-sugar)
    - [7. Printing and String Interpolation](#7-printing-and-string-interpolation)
        - [Keywords](#keywords)
        - [Shorthands](#shorthands)
        - [String Interpolation (F-strings)](#string-interpolation-f-strings)
        - [Input Prompts (`?`)](#input-prompts-)
    - [8. Memory Management](#8-memory-management)
        - [Defer](#defer)
        - [Autofree](#autofree)
        - [RAII / Drop Trait](#raii--drop-trait)
    - [9. Object Oriented Programming](#9-object-oriented-programming)
        - [Methods](#methods)
        - [Traits](#traits)
        - [Standard Traits](#standard-traits)
        - [Composition](#composition)
    - [10. Generics](#10-generics)
    - [11. Concurrency (Async/Await)](#11-concurrency-asyncawait)
    - [12. Metaprogramming](#12-metaprogramming)
        - [Comptime](#comptime)
        - [Embed](#embed)
        - [Plugins](#plugins)
        - [Generic C Macros](#generic-c-macros)
    - [13. Attributes](#13-attributes)
    - [14. Inline Assembly](#14-inline-assembly)
        - [Basic Usage](#basic-usage)
        - [Volatile](#volatile)
        - [Named Constraints](#named-constraints)
    - [15. Build Directives](#15-build-directives)
- [Compiler Support & Compatibility](#compiler-support--compatibility)
    - [Test Suite Status](#test-suite-status)
    - [Building with Zig](#building-with-zig)
    - [C++ Interop](#c-interop)
    - [CUDA Interop](#cuda-interop)
- [Contributing](#contributing)

---

## Quick Start

### Installation

```bash
git clone https://github.com/z-libs/Zen-C.git
cd Zen-C
make
sudo make install
```

#### CMake (cross-platform, including Windows)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --config Debug --prefix .
```

Note: On Windows, you can build the compiler with MSVC, but you still need GCC/MinGW to compile generated C code from `zc` programs at the moment.

### Usage

```bash
# Compile and run
zc run hello.zc

# Build executable
zc build hello.zc -o hello

# Interactive Shell
zc repl
```

### Environment Variables

You can set `ZC_ROOT` to specify the location of the Standard Library (standard imports like `import "std/vector.zc"`). This allows you to run `zc` from any directory.

```bash
export ZC_ROOT=/path/to/Zen-C
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

#### Type Aliases
Create a new name for an existing type.
```zc
alias ID = int;
alias PointMap = Map<string, Point>;
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
var y = x > 10 ? 1 : 0;
```

#### Pattern Matching
Powerful alternative to `switch`.
```zc
match val {
    1 => print("One"),
    2 || 3 => print("Two or Three"),   // OR with ||
    4 or 5 => print("Four or Five"),   // OR with 'or'
    6, 7, 8 => print("Six to Eight"),  // OR with comma
    10..15 => print("10 to 14"),       // Exclusive range (Legacy)
    10..<15 => print("10 to 14"),      // Exclusive range (Explicit)
    20..=25 => print("20 to 25"),      // Inclusive range
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
for i in 0..10 { ... }      // Exclusive (0 to 9)
for i in 0..<10 { ... }     // Exclusive (Explicit)
for i in 0..=10 { ... }     // Inclusive (0 to 10)
for i in 0..10 step 2 { ... }

// Iterator (Vec, Array, or custom Iterable)
for item in collection { ... }

// While
while x < 10 { ... }

// Infinite with label
outer: loop {
    if done { break outer; }
}

// Repeat N times
for _ in 0..5 { ... }
```

#### Advanced Control
```zc
// Guard: Execute else and return if condition is false
guard ptr != NULL else { return; }

// Unless: If not true
unless is_valid { return; }
```

### 6. Operators

Zen C supports operator overloading for user-defined structs by implementing specific method names.

#### Overloadable Operators

| Category | Operator | Method Name |
|:---|:---|:---|
| **Arithmetic** | `+`, `-`, `*`, `/`, `%` | `add`, `sub`, `mul`, `div`, `rem` |
| **Comparison** | `==`, `!=` | `eq`, `neq` |
| | `<`, `>`, `<=`, `>=` | `lt`, `gt`, `le`, `ge` |
| **Bitwise** | `&`, `\|`, `^` | `bitand`, `bitor`, `bitxor` |
| | `<<`, `>>` | `shl`, `shr` |
| **Unary** | `-` | `neg` |
| | `!` | `not` |
| | `~` | `bitnot` |
| **Index** | `a[i]` | `get(a, i)` |
| | `a[i] = v` | `set(a, i, v)` |

**Example:**
```zc
impl Point {
    fn add(self, other: Point) -> Point {
        return Point{x: self.x + other.x, y: self.y + other.y};
    }
}

var p3 = p1 + p2; // Calls p1.add(p2)
```

#### Syntactic Sugar

These operators are built-in language features and cannot be overloaded directly.

| Operator | Name | Description |
|:---|:---|:---|
| `\|>` | Pipeline | `x \|> f(y)` desugars to `f(x, y)` |
| `??` | Null Coalescing | `val ?? default` returns `default` if `val` is NULL (pointers) |
| `??=` | Null Assignment | `val ??= init` assigns if `val` is NULL |
| `?.` | Safe Navigation | `ptr?.field` accesses field only if `ptr` is not NULL |
| `?` | Try Operator | `res?` returns error if present (Result/Option types) |

### 7. Printing and String Interpolation

Zen C provides versatile options for printing to the console, including keywords and concise shorthands.

#### Keywords

- `print "text"`: Prints to `stdout` without a trailing newline.
- `println "text"`: Prints to `stdout` with a trailing newline.
- `eprint "text"`: Prints to `stderr` without a trailing newline.
- `eprintln "text"`: Prints to `stderr` with a trailing newline.

#### Shorthands

Zen C allows you to use string literals directly as statements for quick printing:

- `"Hello World"`: Equivalent to `println "Hello World"`. (Implicitly adds newline)
- `"Hello World"..`: Equivalent to `print "Hello World"`. (No trailing newline)
- `!"Error"`: Equivalent to `eprintln "Error"`. (Output to stderr)
- `!"Error"..`: Equivalent to `eprint "Error"`. (Output to stderr, no newline)

#### String Interpolation (F-strings)

You can embed expressions directly into string literals using `{}` syntax. This works with all printing methods and string shorthands.

```zc
var x = 42;
var name = "Zen";
println "Value: {x}, Name: {name}";
"Value: {x}, Name: {name}"; // shorthand println
```

#### Input Prompts (`?`)

Zen C supports a shorthand for prompting user input using the `?` prefix.

- `? "Prompt text"`: Prints the prompt (without newline) and waits for input (reads a line).
- `? "Enter age: " (age)`: Prints prompt and scans input into the variable `age`.
    - Format specifiers are automatically inferred based on variable type.

```c
var age: int;
? "How old are you? " (age);
println "You are {age} years old.";
```

### 8. Memory Management

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

### 9. Object Oriented Programming

#### Methods
Define methods on types using `impl`.
```zc
impl Point {
    // Static method (constructor convention)
    fn new(x: int, y: int) -> Self {
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
struct Circle { radius: f32; }

trait Drawable {
    fn draw(self);
}

impl Drawable for Circle {
    fn draw(self) { ... }
}

var circle = Circle{};
var drawable: Drawable = &circle;
```

#### Standard Traits
Zen C includes standard traits that integrate with language syntax.

**Iterable**

Implement `Iterable<T>` to enable `for-in` loops for your custom types.

```zc
import "std/iter.zc"

// Define an Iterator
struct MyIter {
    curr: int;
    stop: int;
}

impl MyIter {
    fn next(self) -> Option<int> {
        if self.curr < self.stop {
            self.curr += 1;
            return Option<int>::Some(self.curr - 1);
        }
        return Option<int>::None();
    }
}

// Implement Iterable
impl MyRange {
    fn iterator(self) -> MyIter {
        return MyIter{curr: self.start, stop: self.end};
    }
}

// Use in Loop
for i in my_range {
    println "{i}";
}
```

**Drop**

Implement `Drop` to define a destructor that runs when the object goes out of scope (RAII).

```zc
import "std/mem.zc"

struct Resource {
    ptr: void*;
}

impl Drop for Resource {
    fn drop(self) {
        if self.ptr != NULL {
            free(self.ptr);
        }
    }
}
```

#### Composition
Use `use` to embed other structs. You can either mix them in (flatten fields) or name them (nest fields).

```zc
struct Entity { id: int; }

struct Player {
    // Mixin (Unnamed): Flattens fields
    use Entity;  // Adds 'id' to Player directly
    name: string;
}

struct Match {
    // Composition (Named): Nests fields
    use p1: Player; // Accessed via match.p1
    use p2: Player; // Accessed via match.p2
}
```

### 10. Generics

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

// Multi-parameter Generics
struct Pair<K, V> {
    key: K;
    value: V;
}
```

### 11. Concurrency (Async/Await)

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

### 12. Metaprogramming

#### Comptime
Run code at compile-time to generate source or print messages.
```zc
comptime {
    // Generate code at compile-time (written to stdout)
    println "var build_date = \"2024-01-01\";";
}

println "Build Date: {build_date}";
```

#### Embed
Embed files as specified types.
```zc
// Default (Slice_char)
var data = embed "assets/logo.png";

// Typed Embed
var text = embed "shader.glsl" as string;    // Embbed as C-string
var rom  = embed "bios.bin" as u8[1024];     // Embed as fixed array
var wav  = embed "sound.wav" as u8[];        // Embed as Slice_u8
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

### 13. Attributes

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

### 14. Inline Assembly

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

### 15. Build Directives

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

### C++ Interop

Zen C can generate C++-compatible code with the `--cpp` flag, allowing seamless integration with C++ libraries.

```bash
# Direct compilation with g++
zc app.zc --cpp

# Or transpile for manual build
zc transpile app.zc --cpp
g++ out.c my_cpp_lib.o -o app
```

#### Using C++ in Zen C

Include C++ headers and use raw blocks for C++ code:

```zc
include <vector>
include <iostream>

raw {
    std::vector<int> make_vec(int a, int b) {
        return {a, b};
    }
}

fn main() {
    var v = make_vec(1, 2);
    raw { std::cout << "Size: " << v.size() << std::endl; }
}
```

> **Note:** The `--cpp` flag switches the backend to `g++` and emits C++-compatible code (uses `auto` instead of `__auto_type`, function overloads instead of `_Generic`, and explicit casts for `void*`).

#### CUDA Interop

Zen C supports GPU programming by transpiling to **CUDA C++**. This allows you to leverage powerful C++ features (templates, constexpr) within your kernels while maintaining Zen C's ergonomic syntax.

```bash
# Direct compilation with nvcc
zc run app.zc --cuda

# Or transpile for manual build
zc transpile app.zc --cuda -o app.cu
nvcc app.cu -o app
```

#### CUDA-Specific Attributes

| Attribute | CUDA Equivalent | Description |
|:---|:---|:---|
| `@global` | `__global__` | Kernel function (runs on GPU, called from host) |
| `@device` | `__device__` | Device function (runs on GPU, called from GPU) |
| `@host` | `__host__` | Host function (explicit CPU-only) |

#### Kernel Launch Syntax

Zen C provides a clean `launch` statement for invoking CUDA kernels:

```zc
launch kernel_name(args) with {
    grid: num_blocks,
    block: threads_per_block,
    shared_mem: 1024,  // Optional
    stream: my_stream   // Optional
};
```

This transpiles to: `kernel_name<<<grid, block, shared, stream>>>(args);`

#### Writing CUDA Kernels

Use Zen C function syntax with `@global` and the `launch` statement:

```zc
import "std/cuda.zc"

@global
fn add_kernel(a: float*, b: float*, c: float*, n: int) {
    var i = thread_id();
    if i < n {
        c[i] = a[i] + b[i];
    }
}

fn main() {
    const N = 1024;
    var d_a = cuda_alloc<float>(N);
    var d_b = cuda_alloc<float>(N); 
    var d_c = cuda_alloc<float>(N);
    defer cuda_free(d_a);
    defer cuda_free(d_b);
    defer cuda_free(d_c);

    // ... init data ...
    
    launch add_kernel(d_a, d_b, d_c, N) with {
        grid: (N + 255) / 256,
        block: 256
    };
    
    cuda_sync();
}
```

#### Standard Library (`std/cuda.zc`)
Zen C provides a standard library for common CUDA operations to reduce `raw` blocks:

```zc
import "std/cuda.zc"

// Memory management
var d_ptr = cuda_alloc<float>(1024);
cuda_copy_to_device(d_ptr, h_ptr, 1024 * sizeof(float));
defer cuda_free(d_ptr);

// Synchronization
cuda_sync();

// Thread Indexing (use inside kernels)
var i = thread_id(); // Global index
var bid = block_id();
var tid = local_id();
```


> **Note:** The `--cuda` flag sets `nvcc` as the compiler and implies `--cpp` mode. Requires the NVIDIA CUDA Toolkit.

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
