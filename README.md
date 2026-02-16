<div align="center">
  <p>
    <a href="README.md">English</a> •
    <a href="translations/README_RU.md">Русский</a> •
    <a href="translations/README_ZH_CN.md">简体中文</a> •
    <a href="translations/README_ZH_TW.md">繁體中文</a> •
    <a href="translations/README_ES.md">Español</a> •
    <a href="translations/README_IT.md">Italiano</a> •
    <a href="translations/README_PT_BR.md">Português Brasileiro</a>
  </p>
</div>

<div align="center">
  <h1>Zen C</h1>
  <h3>Modern Ergonomics. Zero Overhead. Pure C.</h3>
  <br>
  <p>
    <a href="#"><img src="https://img.shields.io/badge/build-passing-brightgreen" alt="Build Status"></a>
    <a href="#"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
    <a href="#"><img src="https://img.shields.io/github/v/release/z-libs/Zen-C?label=version&color=orange" alt="Version"></a>
    <a href="#"><img src="https://img.shields.io/badge/platform-linux-lightgrey" alt="Platform"></a>
  </p>
  <p><em>Write like a high-level language, run like C.</em></p>
</div>

<hr>

<div align="center">
  <p>
    <b><a href="#overview">Overview</a></b> •
    <b><a href="#community">Community</a></b> •
    <b><a href="#quick-start">Quick Start</a></b> •
    <b><a href="#language-reference">Language Reference</a></b> •
    <b><a href="#standard-library">Standard Library</a></b> •
    <b><a href="#tooling">Toolchain</a></b>
  </p>
</div>

---

## Overview

**Zen C** is a modern systems programming language that compiles to human-readable `GNU C`/`C11`. It provides a rich feature set including type inference, pattern matching, generics, traits, async/await, and manual memory management with RAII capabilities, all while maintaining 100% C ABI compatibility.

## Community

Join the discussion, share demos, ask questions, or report bugs in the official Zen C Discord server!

- Discord: [Join here](https://discord.com/invite/q6wEsCmkJP)

## Showcase

Check out these projects built with Zen C:

- **[ZC-pong-3ds](https://github.com/5quirre1/ZC-pong-3ds)**: A Pong clone for the Nintendo 3DS.
- **[zen-c-parin](https://github.com/Kapendev/zen-c-parin)**: A basic example using Zen C with Parin.
- **[almond](https://git.sr.ht/~leanghok/almond)**: A minimal web browser written in Zen C.

---

## Index

<table align="center">
  <tr>
    <th width="50%">General</th>
    <th width="50%">Language Reference</th>
  </tr>
  <tr>
    <td valign="top">
      <ul>
        <li><a href="#overview">Overview</a></li>
        <li><a href="#community">Community</a></li>
        <li><a href="#quick-start">Quick Start</a></li>
        <li><a href="#standard-library">Standard Library</a></li>
        <li><a href="#tooling">Tooling</a></li>
        <li><a href="#compiler-support--compatibility">Compiler Support & Compatibility</a></li>
        <li><a href="#contributing">Contributing</a></li>
        <li><a href="#attributions">Attributions</a></li>
      </ul>
    </td>
    <td valign="top">
      <ul>
        <li><a href="#1-variables-and-constants">1. Variables & Constants</a></li>
        <li><a href="#2-primitive-types">2. Primitive Types</a></li>
        <li><a href="#3-aggregate-types">3. Aggregate Types</a></li>
        <li><a href="#4-functions--lambdas">4. Functions & Lambdas</a></li>
        <li><a href="#5-control-flow">5. Control Flow</a></li>
        <li><a href="#6-operators">6. Operators</a></li>
        <li><a href="#7-printing-and-string-interpolation">7. Printing & Interpolation</a></li>
        <li><a href="#8-memory-management">8. Memory Management</a></li>
        <li><a href="#9-object-oriented-programming">9. Object Oriented Programming</a></li>
        <li><a href="#10-generics">10. Generics</a></li>
        <li><a href="#11-concurrency-asyncawait">11. Concurrency</a></li>
        <li><a href="#12-metaprogramming">12. Metaprogramming</a></li>
        <li><a href="#13-attributes">13. Attributes</a></li>
        <li><a href="#14-inline-assembly">14. Inline Assembly</a></li>
        <li><a href="#15-build-directives">15. Build Directives</a></li>
        <li><a href="#16-keywords">16. Keywords</a></li>
        <li><a href="#17-c-interoperability">17. C Interoperability</a></li>
      </ul>
    </td>
  </tr>
</table>

---

## Quick Start

### Installation

```bash
git clone https://github.com/z-libs/Zen-C.git
cd Zen-C
make clean # remove old build files
make
sudo make install
```

### Windows

You can build natively on Windows using the provided batch script with GCC (MinGW) or MSVC:

```cmd
build.bat
```

Alternatively, you can use `make` if you have a Unix-like environment (MSYS2, Cygwin, git-bash).

### Portable Build (APE)

Zen C can be compiled as an **Actually Portable Executable (APE)** using [Cosmopolitan Libc](https://github.com/jart/cosmopolitan). This produces a single binary (`.com`) that runs natively on Linux, macOS, Windows, FreeBSD, OpenBSD, and NetBSD on both x86_64 and aarch64 architectures.

**Prerequisites:**
- `cosmocc` toolchain (must be in your PATH)

**Build & Install:**
```bash
make ape
sudo env "PATH=$PATH" make install-ape
```

**Artifacts:**
- `out/bin/zc.com`: The portable Zen-C compiler. Includes the standard library embedded within the executable.
- `out/bin/zc-boot.com`: A self-contained bootstrap installer for setting up new Zen-C projects.

**Usage:**
```bash
# Run on any supported OS
./out/bin/zc.com build hello.zc -o hello
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

### Environment Variables

You can set `ZC_ROOT` to specify the location of the Standard Library (standard imports like `import "std/vec.zc"`). This allows you to run `zc` from any directory.

```bash
export ZC_ROOT=/path/to/Zen-C
```

---

## Language Reference

### 1. Variables and Constants

Zen C distinguishes between compile-time constants and runtime variables.

#### Manifest Constants (`def`)
Values that exist only at compile-time (folded into code). Use these for array sizes, fixed configuration, and magic numbers.

```zc
def MAX_SIZE = 1024;
let buffer: char[MAX_SIZE]; // Valid array size
```

#### Variables (`let`)
Storage locations in memory. Can be mutable or read-only (`const`).

```zc
let x = 10;             // Mutable
x = 20;                 // OK

let y: const int = 10;  // Read-only (Type qualified)
// y = 20;              // Error: cannot assign to const
```

> [!TIP]
> **Type Inference**: Zen C automatically infers types for initialized variables. It compiles to C23 `auto` on supported compilers, or GCC's `__auto_type` extension otherwise.

### 2. Primitive Types

| Type | C Equivalent | Description |
|:---|:---|:---|
| `int`, `uint` | `int32_t`, `uint32_t` | 32-bit signed/unsigned integer |
| `c_char`, `c_uchar` | `char`, `unsigned char` | C char / unsigned char (Interop) |
| `c_short`, `c_ushort` | `short`, `unsigned short` | C short / unsigned short (Interop) |
| `c_int`, `c_uint` | `int`, `unsigned int` | C int / unsigned int (Interop) |
| `c_long`, `c_ulong` | `long`, `unsigned long` | C long / unsigned long (Interop) |
| `c_long_long`, `c_ulong_long` | `long long`, `unsigned long long` | C long long / unsigned long long (Interop) |
| `I8` .. `I128` or `i8` .. `i128` | `int8_t` .. `__int128_t` | Signed fixed-width integers |
| `U8` .. `U128` or `u8` .. `u128` | `uint8_t` .. `__uint128_t` | Unsigned fixed-width integers |
| `isize`, `usize` | `ptrdiff_t`, `size_t` | Pointer-sized integers |
| `byte` | `uint8_t` | Alias for U8 |
| `F32`, `F64` or `f32`, `f64`  | `float`, `double` | Floating point numbers |
| `bool` | `bool` | `true` or `false` |
| `char` | `char` | Single character |
| `string` | `char*` | C-string (null-terminated) |
| `U0`, `u0`, `void` | `void` | Empty type |
| `iN` (for example, `i256`) | `_BitInt(N)` | Arbitrary bit-width signed integer (C23) |
| `uN` (for example, `u42`) | `unsigned _BitInt(N)` | Arbitrary bit-width unsigned integer (C23) |

#### Literals
- **Integers**: Decimal (`123`), Hex (`0xFF`), Octal (`0o755`), Binary (`0b1011`).
  - *Note*: Numbers with leading zeros are treated as decimal (`0123` is `123`), unlike C.
- **Floats**: Standard (`3.14`), Scientific (`1e-5`, `1.2E3`).

> [!IMPORTANT]
> **Best Practices for Portable Code**
>
> - Use **Portable Types** (`int`, `uint`, `i64`, `u8`, etc.) for all pure Zen C logic. `int` is guaranteed to be 32-bit signed on all architectures.
> - Use **C Interop Types** (`c_int`, `c_char`, `c_long`, ``c_ulong``, ``c_long_long``, ``c_ulong_long``) **only** when interacting with C libraries (FFI). Their size varies by platform and C compiler (e.g. `c_long` size differs between Windows and Linux).
> - Use `isize` and `usize` for array indexing and memory pointer arithmetic.

### 3. Aggregate Types

#### Arrays
Fixed-size arrays with value semantics.
```zc
def SIZE = 5;
let ints: int[SIZE] = [1, 2, 3, 4, 5];
let zeros: [int; SIZE]; // Zero-initialized
```

#### Tuples
Group multiple values together, access elements by index.
```zc
let pair = (1, "Hello");
let x = pair.0;  // 1
let s = pair.1;  // "Hello"
```

**Multiple Return Values**

Functions can return tuples to provide multiple results:
```zc
fn add_and_subtract(a: int, b: int) -> (int, int) {
    return (a + b, a - b);
}

let result = add_and_subtract(3, 2);
let sum = result.0;   // 5
let diff = result.1;  // 1
```

**Destructuring**

Tuples can be destructured directly into variables:
```zc
let (sum, diff) = add_and_subtract(3, 2);
// sum = 5, diff = 1
```

Typed tuple destructuring allows explicit type annotations:
```zc
let (a: string, b: u8) = ("hello", 42);
let (x, y: i32) = (1, 2);  // Mixed: x inferred, y explicit
```

#### Structs
Data structures with optional bitfields.
```zc
struct Point {
    x: int;
    y: int;
}

// Struct initialization
let p = Point { x: 10, y: 20 };

// Bitfields
struct Flags {
    valid: U8 : 1;
    mode:  U8 : 3;
}
```

> [!NOTE]
> Structs use [Move Semantics](#move-semantics--copy-safety) by default. Fields can be accessed via `.` even on pointers (Auto-Dereference).

#### Opaque Structs
You can define a struct as `opaque` to restrict access to its fields to the defining module only, while still allowing the struct to be allocated on the stack (size is known).

```zc
// In user.zc
opaque struct User {
    id: int;
    name: string;
}

fn new_user(name: string) -> User {
    return User{id: 1, name: name}; // OK: Inside module
}

// In main.zc
import "user.zc";

fn main() {
    let u = new_user("Alice");
    // let id = u.id; // Error: Cannot access private field 'id'
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

#### Opaque Type Aliases
You can define a type alias as `opaque` to create a new type that is distinct from its underlying type outside of the defining module. This provides strong encapsulation and type safety without the runtime overhead of a wrapper struct.

```zc
// In library.zc
opaque alias Handle = int;

fn make_handle(v: int) -> Handle {
    return v; // Implicit conversion allowed inside module
}

// In main.zc
import "library.zc";

fn main() {
    let h: Handle = make_handle(42);
    // let i: int = h; // Error: Type validation failed
    // let h2: Handle = 10; // Error: Type validation failed
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

> **Note**: Named arguments must strictly follow the defined parameter order. `add(b: 20, a: 10)` is invalid.

#### Const Arguments
Function arguments can be marked as `const` to enforce read-only semantics. This is a type qualifier, not a manifest constant.

```zc
fn print_val(v: const int) {
    // v = 10; // Error: Cannot assign to const variable
    println "{v}";
}
```

#### Default Arguments
Functions can define default values for trailing arguments. These can be literals, expressions, or valid Zen C code (like struct constructors).
```zc
// Simple default value
fn increment(val: int, amount: int = 1) -> int {
    return val + amount;
}

// Expression default value (evaluated at call site)
fn offset(val: int, pad: int = 10 * 2) -> int {
    return val + pad;
}

// Struct default value
struct Config { debug: bool; }
fn init(cfg: Config = Config { debug: true }) {
    if cfg.debug { println "Debug Mode"; }
}

fn main() {
    increment(10);      // 11
    offset(5);          // 25
    init();             // Prints "Debug Mode"
}
```

#### Lambdas (Closures)
Anonymous functions that can capture their environment.
```zc
let factor = 2;
let doubler = x -> x * factor;  // Arrow syntax
let full = fn(x: int) -> int { return x * factor; }; // Block syntax

// Capture by Reference (Block Syntax)
let val = 10;
let modify = fn[&]() { val += 1; }; 
modify(); // val is now 11

// Capture by Reference (Arrow Syntax)
let modify_arrow = [&] x -> val += x;
modify_arrow(5); // val is now 16

// Capture by Reference (Arrow Syntax with Multiple Arguments)
let sum_into = [&] (a, b) -> val += (a + b);
sum_into(2, 2); // val is now 20

// Capture by Value (Default)
let original = 100;
let implicit = x -> original + x;       // Implicit capture by value (no brackets)
let explicit = [=] x -> original + x;   // Explicit capture by value
// let fail = x -> original += x;       // Error: cannot assign to captured value

```

#### Raw Function Pointers
Zen C supports raw C function pointers using the `fn*` syntax. This allows seamless interop with C libraries that expect function pointers without closure overhead.
```zc
// Function taking a raw function pointer
fn set_callback(cb: fn*(int)) {
    cb(42);
}

// Function returning a raw function pointer
fn get_callback() -> fn*(int) {
    return my_handler;
}

// Pointers to function pointers are supported (fn**)
let pptr: fn**(int) = &ptr;
```

#### Variadic Functions
Functions can accept a variable number of arguments using `...` and the `va_list` type.
```zc
fn log(lvl: int, fmt: char*, ...) {
    let ap: va_list;
    va_start(ap, fmt);
    vprintf(fmt, ap); // Use C stdio
    va_end(ap);
}
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
let y = x > 10 ? 1 : 0;

// If-Expression (for complex conditions)
let category = if (x > 100) { "huge" } else if (x > 10) { "large" } else { "small" };
```

#### Pattern Matching
Powerful alternative to `switch`.
```zc
match val {
    1         => { print "One" },
    2 || 3    => { print "Two or Three" },    // OR with ||
    4 or 5    => { print "Four or Five" },    // OR with 'or'
    6, 7, 8   => { print "Six to Eight" },    // OR with comma
    10 .. 15  => { print "10 to 14" },        // Exclusive range (Legacy)
    10 ..< 15 => { print "10 to 14" },        // Exclusive range (Explicit)
    20 ..= 25 => { print "20 to 25" },        // Inclusive range
    _         => { print "Other" },
}

// Destructuring Enums
match shape {
    Shape::Circle(r)   => { println "Radius: {r}" },
    Shape::Rect(w, h)  => { println "Area: {w*h}" },
    Shape::Point       => { println "Point" },
}
```

#### Reference Binding
To inspect a value without taking ownership (moving it), use the `ref` keyword in the pattern. This is essential for types that implement Move Semantics (like `Option`, `Result`, non-Copy structs).

```zc
let opt = Some(NonCopyVal{...});
match opt {
    Some(ref x) => {
        // 'x' is a pointer to the value inside 'opt'
        // 'opt' is NOT moved/consumed here
        println "{x.field}"; 
    },
    None => {}
}
```

#### Loops
```zc
// Range
for i in 0..10 { ... }      // Exclusive (0 to 9)
for i in 0..<10 { ... }     // Exclusive (Explicit)
for i in 0..=10 { ... }     // Inclusive (0 to 10)
for i in 0..10 step 2 { ... }

// Iterator (Vec or custom Iterable)
for item in vec { ... }

// Iterate over fixed-size arrays directly
let arr: int[5] = [1, 2, 3, 4, 5];
for val in arr {
    // val is int
    println "{val}";
}

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

> **Note on String Equality**:
> - `string == string` performs **value comparison** (equivalent to `strcmp`).
> - `char* == char*` performs **pointer comparison** (checks memory addresses).
> - Mixed comparisons (e.g. `string == char*`) default to **pointer comparison**.

**Example:**
```zc
impl Point {
    fn add(self, other: Point) -> Point {
        return Point{x: self.x + other.x, y: self.y + other.y};
    }
}

let p3 = p1 + p2; // Calls p1.add(p2)
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

**Auto-Dereference**:
Pointer field access (`ptr.field`) and method calls (`ptr.method()`) automatically dereference the pointer, equivalent to `(*ptr).field`.

### 7. Printing and String Interpolation

Zen C provides versatile options for printing to the console, including keywords and concise shorthands.

#### Keywords

| Keyword | Description |
|:---|:---|
| `print "..."` | Prints to `stdout` without a trailing newline. |
| `println "..."` | Prints to `stdout` **with** a trailing newline. |
| `eprint "..."` | Prints to `stderr` without a trailing newline. |
| `eprintln "..."` | Prints to `stderr` **with** a trailing newline. |

#### Shorthands

Zen C allows you to use string literals directly as statements for quick printing:

| Syntax | Equivalent | Description |
|:---|:---|:---|
| `"Hz"` | `println "Hz"` | Prints to `stdout` with newline. |
| `"Hz"..` | `print "Hz"` | Prints to `stdout` without newline. |
| `!"Err"` | `eprintln "Err"` | Prints to `stderr` with newline. |
| `!"Err"..` | `eprint "Err"` | Prints to `stderr` without newline. |

#### String Interpolation (F-strings)

You can embed expressions directly into string literals using `{}` syntax. This works with all printing methods and string shorthands.

```zc
let x = 42;
let name = "Zen";
println "Value: {x}, Name: {name}";
"Value: {x}, Name: {name}"; // shorthand println
```

**Escaping Braces**: Use `{{` to produce a literal `{` and `}}` for a literal `}`:

```zc
let json = "JSON: {{\"key\": \"value\"}}";
// Output: JSON: {"key": "value"}
```

#### Input Prompts (`?`)

Zen C supports a shorthand for prompting user input using the `?` prefix.

- `? "Prompt text"`: Prints the prompt (without newline) and waits for input (reads a line).
- `? "Enter age: " (age)`: Prints prompt and scans input into the variable `age`.
    - Format specifiers are automatically inferred based on variable type.

```zc
let age: int;
? "How old are you? " (age);
println "You are {age} years old.";
```

### 8. Memory Management

Zen C allows manual memory management with ergonomic aids.

#### Defer
Execute code when the current scope exits. Defer statements are executed in LIFO (last-in, first-out) order.
```zc
let f = fopen("file.txt", "r");
defer fclose(f);
```

> [!WARNING]
> To prevent undefined behavior, control flow statements (`return`, `break`, `continue`, `goto`) are **not allowed** inside a `defer` block.

#### Autofree
Automatically free the variable when scope exits.
```zc
autofree let types = malloc(1024);
```

#### Resource Semantics (Move by Default)
Zen C treats types with destructors (like `File`, `Vec`, or malloc'd pointers) as **Resources**. To prevent double-free errors, resources cannot be implicitly duplicated.

- **Move by Default**: Assigning a resource variable transfers ownership. The original variable becomes invalid (Moved).
- **Copy Types**: Types without destructors may opt-in to `Copy` behavior, making assignment a duplication.

**Diagnostics & Philosophy**:
If you see an error "Use of moved value", the compiler is telling you: *"This type owns a resource (like memory or a handle) and blindly copying it is unsafe."*

> [!NOTE]
> **Contrast:** Unlike C/C++, Zen C does not implicitly duplicate resource-owning values.

**Function Arguments**:
Passing a value to a function follows the same rules as assignment: resources are moved unless passed by reference.

```zc
fn process(r: Resource) { ... } // 'r' is moved into function
fn peek(r: Resource*) { ... }   // 'r' is borrowed (reference)
```

**Explicit Cloning**:
If you *do* want two copies of a resource, make it explicit:

```zc
let b = a.clone(); // Calls the 'clone' method from the Clone trait
```

**Opt-in Copy (Value Types)**:
For small types without destructors:

```zc
struct Point { x: int; y: int; }
impl Copy for Point {} // Opt-in to implicit duplication

fn main() {
    let p1 = Point { x: 1, y: 2 };
    let p2 = p1; // Copied. p1 stays valid.
}
```

#### RAII / Drop Trait
Implement `Drop` to run cleanup logic automatically.
```zc
impl Drop for MyStruct {
    fn drop(self) {
        self.free();
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

**Self Shorthand**: In methods with a `self` parameter, you can use `.field` as shorthand for `self.field`:
```zc
impl Point {
    fn dist(self) -> float {
        return sqrt(.x * .x + .y * .y);  // Equivalent to self.x, self.y
    }
}
```

#### Primitive Methods
Zen C allows you to define methods on primitive types (like `int`, `bool`, etc.) using the same `impl` syntax.

```zc
impl int {
    fn abs(self) -> int {
        return *self < 0 ? -(*self) : *self;
    }
}

let x = -10;
let y = x.abs(); // 10
let z = (-5).abs(); // 5 (Literals supported)
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

let circle = Circle{};
let drawable: Drawable = &circle;
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

> [!IMPORTANT]
> **Note:** If a variable is moved, `drop` is NOT called on the original variable. It adheres to [Resource Semantics](#resource-semantics-move-by-default).

**Copy**

Marker trait to opt-in to `Copy` behavior (implicit duplication) instead of Move semantics. Used via `@derive(Copy)`.

> [!CAUTION]
> **Rule:** Types that implement `Copy` must not define a destructor (`Drop`).

```zc
@derive(Copy)
struct Point { x: int; y: int; }

fn main() {
    let p1 = Point{x: 1, y: 2};
    let p2 = p1; // Copied! p1 remains valid.
}
```

**Clone**

Implement `Clone` to allow explicit duplication of resource-owning types.

```zc
import "std/mem.zc"

struct MyBox { val: int; }

impl Clone for MyBox {
    fn clone(self) -> MyBox {
        return MyBox{val: self.val};
    }
}

fn main() {
    let b1 = MyBox{val: 42};
    let b2 = b1.clone(); // Explicit copy
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
    let future = fetch_data();
    let result = await future;
}
```

### 12. Metaprogramming

#### Comptime
Run code at compile-time to generate source or print messages.
```zc
comptime {
    // Generate code at compile-time (written to stdout)
    println "let build_date = \"2024-01-01\";";
}

println "Build Date: {build_date}";
```

<details>
<summary><b>Helper Functions</b></summary>

Special functions available inside `comptime` blocks for code generation and diagnostics:

<table>
<tr>
<th>Function</th>
<th>Description</th>
</tr>
<tr>
<td><code>yield(str)</code></td>
<td>Explicitly emit generated code (alternative to <code>printf</code>)</td>
</tr>
<tr>
<td><code>code(str)</code></td>
<td>Alias for <code>yield()</code> - clearer intent for code generation</td>
</tr>
<tr>
<td><code>compile_error(msg)</code></td>
<td>Halt compilation with a fatal error message</td>
</tr>
<tr>
<td><code>compile_warn(msg)</code></td>
<td>Emit a compile-time warning (allows compilation to continue)</td>
</tr>
</table>

**Example:**
```zc
comptime {
    compile_warn("Generating optimized code...");
    
    let ENABLE_FEATURE = 1;
    if (ENABLE_FEATURE == 0) {
        compile_error("Feature must be enabled!");
    }
    
    // Use code() with raw strings for clean generation
    code(r"let FEATURE_ENABLED = 1;");
}
```
</details>

<details>
<summary><b>Build Metadata</b></summary>

Access compiler build information at compile-time:

<table>
<tr>
<th>Constant</th>
<th>Type</th>
<th>Description</th>
</tr>
<tr>
<td><code>__COMPTIME_TARGET__</code></td>
<td>string</td>
<td>Platform: <code>"linux"</code>, <code>"windows"</code>, or <code>"macos"</code></td>
</tr>
<tr>
<td><code>__COMPTIME_FILE__</code></td>
<td>string</td>
<td>Current source filename being compiled</td>
</tr>
</table>

**Example:**
```zc
comptime {
    // Platform-specific code generation
    println "let PLATFORM = \"{__COMPTIME_TARGET__}\";";
}

println "Running on: {PLATFORM}";
```
</details>

> [!TIP]
> Use raw strings (`r"..."`) in comptime to avoid escaping braces: `code(r"fn test() { return 42; }")`. Otherwise, use `{{` and `}}` to escape braces inside regular strings.


#### Embed
Embed files as specified types.
```zc
// Default (Slice_char)
let data = embed "assets/logo.png";

// Typed Embed
let text = embed "shader.glsl" as string;    // Embbed as C-string
let rom  = embed "bios.bin" as u8[1024];     // Embed as fixed array
let wav  = embed "sound.wav" as u8[];        // Embed as Slice_u8
```

#### Plugins
Import compiler plugins to extend syntax.
```zc
import plugin "regex"
let re = regex! { ^[a-z]+$ };
```

#### Generic C Macros
Pass preprocessor macros through to C.

> **Tip**: For simple constants, use `def` instead. Use `#define` when you need C-preprocessor macros or conditional compilation flags.

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
| `@pure` | Fn | Function has no side effects (optimization hint). |
| `@cold` | Fn | Function is unlikely to be executed (branch prediction hint). |
| `@hot` | Fn | Function is frequently executed (optimization hint). |
| `@export` | Fn/Struct | Export symbol (visibility default). |
| `@global` | Fn | CUDA: Kernel entry point (`__global__`). |
| `@device` | Fn | CUDA: Device function (`__device__`). |
| `@host` | Fn | CUDA: Host function (`__host__`). |
| `@comptime` | Fn | Helper function available for compile-time execution. |
| `@derive(...)` | Struct | Auto-implement traits. Supports `Debug`, `Eq` (Smart Derive), `Copy`, `Clone`. |
| `@ctype("type")` | Fn Param | Overrides generated C type for a parameter. |
| `@<custom>` | Any | Passes generic attributes to C (e.g. `@flatten`, `@alias("name")`). |

#### Custom Attributes

Zen C supports a powerful **Custom Attribute** system that allows you to use any GCC/Clang `__attribute__` directly in your code. Any attribute that is not explicitly recognized by the Zen C compiler is treated as a generic attribute and passed through to the generated C code.

This provides access to advanced compiler features, optimizations, and linker directives without needing explicit support in the language core.

#### Syntax Mapping
Zen C attributes are mapped directly to C attributes:
- `@name` → `__attribute__((name))`
- `@name(args)` → `__attribute__((name(args)))`
- `@name("string")` → `__attribute__((name("string")))`

#### Smart Derives

Zen C provides "Smart Derives" that respect Move Semantics:

- **`@derive(Eq)`**: Generates an equality method that takes arguments by reference (`fn eq(self, other: T*)`).
    - When comparing two non-Copy structs (`a == b`), the compiler automatically passes `b` by reference (`&b`) to avoid moving it.
    - Recursive equality checks on fields also prefer pointer access to prevent ownership transfer.

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
// Syntax: : out(variable) : in(variable) : clobber(reg)
// Uses {variable} placeholder syntax for readability

fn add_five(x: int) -> int {
    let result: int;
    asm {
        "mov {x}, {result}"
        "add $5, {result}"
        : out(result)
        : in(x)
        : clobber("cc")
    }
    return result;
}
```

| Type | Syntax | GCC Equivalent |
|:---|:---|:---|
| **Output** | `: out(variable)` | `"=r"(variable)` |
| **Input** | `: in(variable)` | `"r"(variable)` |
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
| `//> framework:` | `Cocoa` | Link against a macOS framework. |
| `//> cflags:` | `-Wall -O3` | Pass arbitrary flags to the C compiler. |
| `//> define:` | `MACRO` or `KEY=VAL` | Define a preprocessor macro (`-D`). |
| `//> pkg-config:` | `gtk+-3.0` | Run `pkg-config` and append `--cflags` and `--libs`. |
| `//> shell:` | `command` | Execute a shell command during the build. |
| `//> get:` | `http://url/file` | Download a file if specific file does not exist. |

#### Features

**1. OS Guarding**
Prefix directives with an OS name to apply them only on specific platforms.
Supported prefixes: `linux:`, `windows:`, `macos:` (or `darwin:`).

```zc
//> linux: link: -lm
//> windows: link: -lws2_32
//> macos: framework: Cocoa
```

**2. Environment Variable Expansion**
Use `${VAR}` syntax to expand environment variables in your directives.

```zc
//> include: ${HOME}/mylib/include
//> lib: ${ZC_ROOT}/std
```

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

### 16. Keywords

The following keywords are reserved in Zen C.

#### Declarations
`alias`, `def`, `enum`, `fn`, `impl`, `import`, `let`, `module`, `opaque`, `struct`, `trait`, `union`, `use`

#### Control Flow
`async`, `await`, `break`, `catch`, `continue`, `defer`, `else`, `for`, `goto`, `guard`, `if`, `loop`, `match`, `return`, `try`, `unless`, `while`

#### Special
`asm`, `assert`, `autofree`, `comptime`, `const`, `embed`, `launch`, `ref`, `sizeof`, `static`, `test`, `volatile`

#### Constants
`true`, `false`, `null`

#### C Reserved
The following identifiers are reserved because they are keywords in C11:
`auto`, `case`, `char`, `default`, `do`, `double`, `extern`, `float`, `inline`, `int`, `long`, `register`, `restrict`, `short`, `signed`, `switch`, `typedef`, `unsigned`, `void`, `_Atomic`, `_Bool`, `_Complex`, `_Generic`, `_Imaginary`, `_Noreturn`, `_Static_assert`, `_Thread_local`

#### Operators
`and`, `or`

### 17. C Interoperability

Zen C offers two ways to interact with C code: **Trusted Imports** (Convenient) and **Explicit FFI** (Safe/Precise).

#### Method 1: Trusted Imports (Convenient)

You can import a C header directly using the `import` keyword with the `.h` extension. This treats the header as a module and assumes all symbols accessed through it exist.

```zc
//> link: -lm
import "math.h" as c_math;

fn main() {
    // Compiler trusts correctness; emits 'cos(...)' directly
    let x = c_math::cos(3.14159);
}
```

> **Pros**: Zero boilerplate. Access everything in the header immediately.
> **Cons**: No type safety from Zen C (errors caught by C compiler later).

#### Method 2: Explicit FFI (Safe)

For strict type checking or when you don't want to include the text of a header, use `extern fn`.

```zc
include <stdio.h> // Emits #include <stdio.h> in generated C

// Define strict signature
extern fn printf(fmt: char*, ...) -> c_int;

fn main() {
    printf("Hello FFI: %d\n", 42); // Type checked by Zen C
}
```

> **Pros**: Zen C ensures types match.
> **Cons**: Requires manual declaration of functions.

#### `import` vs `include`

- **`import "file.h"`**: Registers the header as a named module. Enables implicit access to symbols (for example, `file::function()`).
- **`include <file.h>`**: Purely emits `#include <file.h>` in the generated C code. Does not introduce any symbols to the Zen C compiler; you must use `extern fn` to access them.


---

## Standard Library

Zen C includes a standard library (`std`) covering essential functionality.

[Browse the Standard Library Documentation](docs/std/README.md)

### Key Modules

<details>
<summary>Click to see all Standard Library modules</summary>

| Module | Description | Docs |
| :--- | :--- | :--- |
| **`std/vec.zc`** | Growable dynamic array `Vec<T>`. | [Docs](docs/std/vec.md) |
| **`std/string.zc`** | Heap-allocated `String` type with UTF-8 support. | [Docs](docs/std/string.md) |
| **`std/queue.zc`** | FIFO queue (Ring Buffer). | [Docs](docs/std/queue.md) |
| **`std/map.zc`** | Generic Hash Map `Map<V>`. | [Docs](docs/std/map.md) |
| **`std/fs.zc`** | File system operations. | [Docs](docs/std/fs.md) |
| **`std/io.zc`** | Standard Input/Output (`print`/`println`). | [Docs](docs/std/io.md) |
| **`std/option.zc`** | Optional values (`Some`/`None`). | [Docs](docs/std/option.md) |
| **`std/result.zc`** | Error handling (`Ok`/`Err`). | [Docs](docs/std/result.md) |
| **`std/path.zc`** | Cross-platform path manipulation. | [Docs](docs/std/path.md) |
| **`std/env.zc`** | Process environment variables. | [Docs](docs/std/env.md) |
| **`std/net/`** | TCP, UDP, HTTP, DNS, URL. | [Docs](docs/std/net.md) |
| **`std/thread.zc`** | Threads and Synchronization. | [Docs](docs/std/thread.md) |
| **`std/time.zc`** | Time measurement and sleep. | [Docs](docs/std/time.md) |
| **`std/json.zc`** | JSON parsing and serialization. | [Docs](docs/std/json.md) |
| **`std/stack.zc`** | LIFO Stack `Stack<T>`. | [Docs](docs/std/stack.md) |
| **`std/set.zc`** | Generic Hash Set `Set<T>`. | [Docs](docs/std/set.md) |
| **`std/process.zc`** | Process execution and management. | [Docs](docs/std/process.md) |
| **`std/regex.zc`** | Regular Expressions (TRE based). | [Docs](docs/std/regex.md) |

</details>

---

## Tooling

Zen C provides a built-in Language Server and REPL to enhance the development experience.

### Language Server (LSP)

The Zen C Language Server (LSP) supports standard LSP features for editor integration, providing:

*   **Go to Definition**
*   **Find References**
*   **Hover Information**
*   **Completion** (Function/Struct names, Dot-completion for methods/fields)
*   **Document Symbols** (Outline)
*   **Signature Help**
*   **Diagnostics** (Syntax/Semantic errors)

To start the language server (typically configured in your editor's LSP settings):

```bash
zc lsp
```

It communicates via standard I/O (JSON-RPC 2.0).

### REPL

The Read-Eval-Print Loop allows you to experiment with Zen C code interactively.

```bash
zc repl
```

#### Features

*   **Interactive Coding**: Type expressions or statements for immediate evaluation.
*   **Persistent History**: Commands are saved to `~/.zprep_history`.
*   **Startup Script**: Auto-loads commands from `~/.zprep_init.zc`.

#### Commands

| Command | Description |
|:---|:---|
| `:help` | Show available commands. |
| `:reset` | Clear current session history (variables/functions). |
| `:vars` | Show active variables. |
| `:funcs` | Show user-defined functions. |
| `:structs` | Show user-defined structs. |
| `:imports` | Show active imports. |
| `:history` | Show session input history. |
| `:type <expr>` | Show the type of an expression. |
| `:c <stmt>` | Show the generated C code for a statement. |
| `:time <expr>` | Benchmark an expression (runs 1000 iterations). |
| `:edit [n]` | Edit command `n` (default: last) in `$EDITOR`. |
| `:save <file>` | Save the current session to a `.zc` file. |
| `:load <file>` | Load and execute a `.zc` file into the session. |
| `:watch <expr>` | Watch an expression (re-evaluated after every entry). |
| `:unwatch <n>` | Remove a watch. |
| `:undo` | Remove the last command from the session. |
| `:delete <n>` | Remove command at index `n`. |
| `:clear` | Clear the screen. |
| `:quit` | Exit the REPL. |
| `! <cmd>` | Run a shell command (e.g. `!ls`). |

---


## Compiler Support & Compatibility

Zen C is designed to work with most C11 compilers. Some features rely on GNU C extensions, but these often work in other compilers. Use the `--cc` flag to switch backends.

```bash
zc run app.zc --cc clang
zc run app.zc --cc zig
```

### Test Suite Status

<details>
<summary>Click to view Compiler Support details</summary>

| Compiler | Pass Rate | Supported Features | Known Limitations |
|:---|:---:|:---|:---|
| **GCC** | **100% (Full)** | All Features | None. |
| **Clang** | **100% (Full)** | All Features | None. |
| **Zig** | **100% (Full)** | All Features | None. Uses `zig cc` as a drop-in C compiler. |
| **TCC** | **98% (High)** | Structs, Generics, Traits, Pattern Matching | No Intel ASM, No `__attribute__((constructor))`. |

</details>

> [!WARNING]
> **COMPILER BUILD WARNING:** While **Zig CC** works excellently as a backend for your Zen C programs, building the *Zen C compiler itself* with it may verify but produce an unstable binary that fails tests. We recommend building the compiler with **GCC** or **Clang** and using Zig only as a backend for your operational code.

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
    let v = make_vec(1, 2);
    raw { std::cout << "Size: " << v.size() << std::endl; }
}
```

> [!NOTE]
> The `--cpp` flag switches the backend to `g++` and emits C++-compatible code (uses `auto` instead of `__auto_type`, function overloads instead of `_Generic`, and explicit casts for `void*`).

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
    let i = thread_id();
    if i < n {
        c[i] = a[i] + b[i];
    }
}

fn main() {
    def N = 1024;
    let d_a = cuda_alloc<float>(N);
    let d_b = cuda_alloc<float>(N); 
    let d_c = cuda_alloc<float>(N);
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
let d_ptr = cuda_alloc<float>(1024);
cuda_copy_to_device(d_ptr, h_ptr, 1024 * sizeof(float));
defer cuda_free(d_ptr);

// Synchronization
cuda_sync();

// Thread Indexing (use inside kernels)
let i = thread_id(); // Global index
let bid = block_id();
let tid = local_id();
```


> [!NOTE]
> **Note:** The `--cuda` flag sets `nvcc` as the compiler and implies `--cpp` mode. Requires the NVIDIA CUDA Toolkit.

### C23 Support

Zen C supports modern C23 features when using a compatible backend compiler (GCC 14+, Clang 14+, TCC (partial)).

- **`auto`**: Zen C automatically maps type inference to standard C23 `auto` if `__STDC_VERSION__ >= 202300L`.
- **`_BitInt(N)`**: Use `iN` and `uN` types (e.g., `i256`, `u12`, `i24`) to access C23 arbitrary-width integers.

### Objective-C Interop

Zen C can compile to Objective-C (`.m`) using the `--objc` flag, allowing you to use Objective-C frameworks (like Cocoa/Foundation) and syntax.

```bash
# Compile with clang (or gcc/gnustep)
zc app.zc --objc --cc clang
```

#### Using Objective-C in Zen C

Use `include` for headers and `raw` blocks for Objective-C syntax (`@interface`, `[...]`, `@""`).

```zc
//> macos: framework: Foundation
//> linux: cflags: -fconstant-string-class=NSConstantString -D_NATIVE_OBJC_EXCEPTIONS
//> linux: link: -lgnustep-base -lobjc

include <Foundation/Foundation.h>

fn main() {
    raw {
        NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
        NSLog(@"Hello from Objective-C!");
        [pool drain];
    }
    println "Zen C works too!";
}
```

> [!NOTE]
> **Note:** Zen C string interpolation works with Objective-C objects (`id`) by calling `debugDescription` or `description`.

---

## Contributing
 
 We welcome contributions! Whether it's fixing bugs, adding documentation, or proposing new features.
 
 Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on how to contribute, run tests, and submit pull requests.

---
 
 ## Security
 
 For security reporting instructions, please see [SECURITY.md](SECURITY.md).
 
 ---
 
 ## Attributions

This project uses third-party libraries. Full license texts can be found in the `LICENSES/` directory.

*   **[cJSON](https://github.com/DaveGamble/cJSON)** (MIT License): Used for JSON parsing and generation in the Language Server.
*   **[zc-ape](https://github.com/OEvgeny/zc-ape)** (MIT License): The original Actually Portable Executable port of Zen-C by [Eugene Olonov](https://github.com/OEvgeny).
*   **[Cosmopolitan Libc](https://github.com/jart/cosmopolitan)** (ISC License): The foundational library that makes APE possible.
*   **[TRE](https://github.com/laurikari/tre)** (BSD License): Used for the regular expression engine in the standard library.

---

<div align="center">
  <p>
    Copyright © 2026 Zen C Programming Language.<br>
    Start your journey today.
  </p>
  <p>
    <a href="https://discord.com/invite/q6wEsCmkJP">Discord</a> •
    <a href="https://github.com/z-libs/Zen-C">GitHub</a> •
    <a href="CONTRIBUTING.md">Contribute</a>
  </p>
</div>
