# Zenckly — Visual Block Programming for Zen-C

Zenckly is a visual block-based programming environment for the 
[Zen-C] language. It uses [Blockly](https://developers.google.com/blockly) to let you drag-and-drop blocks that generate valid Zen-C code, which can then be compiled and run via the Electron desktop app.

## Installation

### Prerequisites

- [Node.js](https://nodejs.org/) (v18 or later)
- [npm](https://www.npmjs.com/) (comes with Node.js)
- [Zen-C compiler (`zc`)]

### Setup

```bash
cd zenckly
npm install
```

### Run the Electron App

```bash
npm start
```

This opens the Zenckly desktop window. You can drag blocks, see generated Zen-C code in the right panel, and compile/run directly.

### Browser-Only Mode

You can also open `index.html` directly in a browser. Compile and Run buttons won't execute code, but you can still build block programs and use **Copy Code** to paste the generated Zen-C into a `.zc` file, then compile manually:

```bash
zc build program.zc -o program
./program
```

## Project Structure

```
zenckly/
  index.html            — Main HTML page
  electron-main.js      — Electron main process (compile/run IPC)
  preload.js            — Electron preload bridge
  package.json          — Node/Electron dependencies
  css/
    zenckly.css         — Dark theme styles
  js/
    blockly.min.js      — Blockly library
    toolbox.js          — Block toolbox categories
    zenckly.js          — App logic, examples, workspace init
    generators/
      zenc_generator.js — ZenC code generator base
      variables.js      — Variable & type blocks (1-18)
      math.js           — Math & arithmetic blocks (19-34)
      logic.js          — Logic & control flow blocks (35-48)
      functions.js      — Function blocks (49-60)
      structs.js        — Struct, enum, trait blocks (61-74)
      memory.js         — Memory & pointer blocks (75-84)
      text.js           — Text, print, I/O blocks (85-96)
      errors.js         — Error handling blocks (97-104)
      files.js          — File I/O & system blocks (105-114)
      build.js          — Build directive blocks (169-174)
      advanced.js       — Advanced blocks (method calls, Vec, guard, def, hex, packed struct)
```

## Toolbar

| Button | Description |
|---|---|
| **Compile** | Writes generated code to a temp file and runs `zc build` (Electron only) |
| **Run** | Compiles and then executes the binary, streaming output to the output panel (Electron only) |
| **Save** | Saves the current workspace to browser localStorage |
| **Load** | Restores a previously saved workspace |
| **Examples** | Opens the examples modal with pre-built block programs |
| **Copy Code** | Copies the generated Zen-C code to clipboard |

## Built-in Examples

Open **Examples** from the toolbar to load these:

| Example | Description |
|---|---|
| Hello World | Minimal `fn main` with `println` |
| Fibonacci | Recursive function, if/else, function calls |
| Struct Example | Struct definition, init, field access with string interpolation |
| File I/O | `File::open`, `write_string`, `close` using static/method call blocks |
| Variables & Math | `let`, `let` (inferred), booleans, arithmetic, assignment |
| Logic & Loops | `if/else`, `for..in` range, `while`, compound assignment |
| Functions & Calls | Function definition, return, call (statement & expression), defer |
| Structs & Methods | Struct, impl, method with `self`, field access |
| Memory & Pointers | `&` address-of, `*` dereference, `alloc`, `free` |
| Vec & Collections | `Vec::new`, push, pop, len, `for..in` loop over vec |
| Match & Cases | `match` expression with multiple `case` arms and default `_` |
| Advanced Blocks | `def`, hex numbers, `guard`, static calls, method calls |

---

## Block Reference

Every block, the toolbox category it belongs to, and the Zen-C code it generates.

### Variables (Toolbox: Variables)

| Block | Description | Generated Code |
|---|---|---|
| `let` | Declare a typed variable | `let x: int = 42;` |
| `let` (inferred) | Declare with inferred type | `let x = 42;` |
| `const` | Compile-time constant | `const MAX: int = 100;` |
| `static` | Static variable | `static counter: int = 0;` |
| Number | Number literal | `42` |
| Boolean | `true` / `false` | `true` |
| Char | Character literal | `'a'` |
| String | String literal (supports `{var}` interpolation) | `"hello {name}"` |
| Variable Get | Reference a variable by name | `x` |
| Assign | Assign a value | `x = 10;` |
| Compound Assign | `+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `\|=`, `^=` | `x += 5;` |
| Array Declare | Fixed-size array | `let arr: int[10];` |
| Array Literal | Array value | `[1, 2, 3]` |
| Array Access | Read element at index | `arr[0]` |
| Array Set | Write element at index | `arr[0] = 99;` |
| Cast | Type cast | `(float)x` |
| Sizeof | Size of a type | `sizeof(int)` |
| Typeof | Type of a value | `typeof(x)` |

**Example — Variables & Math:**
```
fn main() {
    let x: int = 42;
    let name = "Zen-C";
    let flag: bool = true;
    let sum = x + 10;
    x = x * 2;
    println "x = {x}, sum = {sum}, name = {name}, flag = {flag}";
}
```

---

### Math (Toolbox: Math)

| Block | Description | Generated Code |
|---|---|---|
| Arithmetic | `+`, `-`, `*`, `/`, `%` | `a + b` |
| Compare | `==`, `!=`, `<`, `>`, `<=`, `>=` | `a > b` |
| Negate | Unary minus | `-x` |
| Math Constant | PI, E, TAU, INF | `3.14159265358979` |
| Single-arg | `abs`, `sqrt`, `floor`, `ceil`, `round` | `@abs(x)` |
| Min / Max | Two-argument min or max | `@min(a, b)` |
| Trig | `sin`, `cos`, `tan`, `asin`, `acos`, `atan` | `@sin(x)` |
| Power | Exponentiation | `@pow(base, exp)` |
| Log | `ln`, `log2`, `log10` | `@log(x)` |
| Bitwise | `&`, `\|`, `^`, `<<`, `>>` | `a & b` |
| Bitwise NOT | Bitwise complement | `~x` |
| Random Int | Random integer in range | `random_int(0, 100)` |
| Random Float | Random float 0.0 to 1.0 | `random_float()` |
| Modulo | Modulo operator | `a % b` |
| Inc / Dec | Increment or decrement | `x += 1;` / `x -= 1;` |
| Clamp | Clamp between min and max | `@clamp(x, 0, 100)` |

---

### Logic (Toolbox: Logic)

| Block | Description | Generated Code |
|---|---|---|
| if | Conditional | `if x > 0 { ... }` |
| if / else | Conditional with else | `if x > 0 { ... } else { ... }` |
| else if | Chained condition | `else if x > 0 { ... }` |
| and / or | Logical operators | `a and b` / `a or b` |
| not | Logical NOT | `!x` |
| match | Match expression (like switch) | `match x { ... }` |
| match case | A single match arm | `1 => { ... },` |

**Example — Match:**
```
fn main() {
    let day: int = 3;
    match day {
        1 => {
            println "Monday";
        },
        2 => {
            println "Tuesday";
        },
        3 => {
            println "Wednesday";
        },
        _ => {
            println "Weekend!";
        },
    }
}
```

---

### Loops (Toolbox: Loops)

| Block | Description | Generated Code |
|---|---|---|
| while | While loop | `while x > 0 { ... }` |
| for | C-style for loop | `for (let i = 0; i < 10; i += 1) { ... }` |
| for..in | Range-based for loop | `for i in 0..10 { ... }` |
| for each | Iterate over a collection | `for item in collection { ... }` |
| loop forever | Infinite loop | `loop { ... }` |
| break | Break out of a loop | `break;` |
| continue | Skip to next iteration | `continue;` |

**Example — Logic & Loops:**
```
fn main() {
    let x: int = 15;
    if x > 10 {
        println "{x} is big";
    } else {
        println "{x} is small";
    }
    for i in 0..5 {
        print "{i} ";
    }
    println "";
    while x > 0 {
        x -= 3;
    }
    println "x after loop: {x}";
}
```

---

### Functions (Toolbox: Functions)

| Block | Description | Generated Code |
|---|---|---|
| fn main() | Main entry point | `fn main() { ... }` |
| fn | Function definition | `fn add(a: int, b: int) -> int { ... }` |
| pub fn | Public function | `pub fn myFunc() { ... }` |
| return | Return a value | `return x;` |
| return (void) | Return from void function | `return;` |
| call | Function call (statement) | `greet("World");` |
| call (expression) | Function call (returns value) | `add(10, 20)` |
| lambda | Anonymous function | `fn(x: int) -> x * 2` |
| defer | Defer until scope exit | `defer { ... }` |
| extern fn | External function declaration | `extern fn printf(fmt: *const u8, ...);` |
| comptime | Compile-time execution | `comptime { ... }` |
| param | Function parameter | `x: int` |

**Example — Functions & Calls:**
```
fn add(a: int, b: int) -> int {
    return a + b;
}

fn greet(name: char*) {
    println "Hello, {name}!";
}

fn main() {
    defer {
        println "Done!";
    }
    let result = add(10, 20);
    println "10 + 20 = {result}";
    greet("World");
}
```

---

### Structs & Types (Toolbox: Structs)

| Block | Description | Generated Code |
|---|---|---|
| struct | Define a struct | `struct Point { ... }` |
| struct field | A field inside a struct | `x: int;` |
| struct init | Create struct instance | `Point{ x: 10, y: 20 }` |
| field access | Read a field | `p.x` |
| field set | Write a field | `p.x = 99;` |
| impl | Implement methods | `impl Point { ... }` |
| method | Method definition (with `self`) | `fn sum(self) -> int { ... }` |
| self | Reference to self | `self` |
| trait | Define a trait | `trait Drawable { ... }` |
| enum | Define an enum | `enum Color { ... }` |
| enum variant | Enum variant | `Red,` |
| enum variant (value) | Enum variant with explicit value | `Code = 0,` |
| type alias | Type alias | `type MyInt = int;` |
| generic type | Generic type expression | `Vec<int>` |

**Example — Structs & Methods:**
```
struct Point {
    x: int;
    y: int;
}

impl Point {
    fn sum(self) -> int {
        return self.x + self.y;
    }
}

fn main() {
    let p = Point{ x: 10, y: 20 };
    println "Point({p.x}, {p.y}), sum = {p.sum()}";
}
```

---

### Memory (Toolbox: Memory)

| Block | Description | Generated Code |
|---|---|---|
| alloc | Heap allocate | `alloc<int>()` or `alloc_n<int>(10)` |
| free | Free heap memory | `free(ptr);` |
| & (address of) | Get address of a variable | `&x` |
| * (dereference) | Dereference a pointer | `(*ptr)` |
| pointer declare | Declare a pointer variable | `let ptr: int* = NULL;` |
| null | Null pointer literal | `NULL` |
| undefined | Undefined value | `undefined` |
| memcpy | Copy memory | `memcpy(dest, src, size);` |
| memset | Set memory | `memset(dest, 0, size);` |
| pointer slice | Slice from pointer | `ptr[0..10]` |
| autofree | Auto-freeing variable | `autofree let x = value;` |

**Example — Memory & Pointers:**
```
import "std/mem.zc"

fn main() {
    let x: int = 42;
    let ptr = &x;
    println "value of x via pointer: {(*ptr)}";
    let heap_val = alloc<int>();
    (*heap_val) = 99;
    println "heap value: {(*heap_val)}";
    free(heap_val);
    println "memory freed!";
}
```

---

### Text & I/O (Toolbox: Text)

| Block | Description | Generated Code |
|---|---|---|
| println | Print line to stdout | `println "hello";` |
| print | Print without newline | `print "hello";` |
| println (format) | Formatted print | `println "x = {}";` |
| string length | Length of a string | `str.len` |
| concat | String concatenation | `a ++ b` |
| string compare | String equality | `a == b` / `a != b` |
| slice | Substring | `str[0..5]` |
| char at | Character at index | `str[0]` |
| to_string | Convert to string | `to_string(42)` |
| parse_int / parse_float | Parse number from string | `parse_int(str)` |
| raw code | Insert raw Zen-C code | *(literal text)* |
| comment | Add a comment | `// my comment` |

> **Note:** `println` and `print` require a string literal. If you connect a non-string value (like a variable or number), the generator automatically wraps it in interpolation: `println "{x}";`

---

### Errors (Toolbox: Errors)

| Block | Description | Generated Code |
|---|---|---|
| Ok | Wrap value in Ok | `Ok(value)` |
| Err | Wrap value in Err | `Err("error")` |
| Some | Wrap value in Some | `Some(value)` |
| None | No value | `None` |
| unwrap | Unwrap Result/Option | `result.unwrap()` |
| unwrap_or | Unwrap with default | `result.unwrap_or(0)` |
| try | Try operator (propagate errors) | `result?` |
| if let | Pattern match unwrap | `if (expr) \|value\| { ... } else { ... }` |

---

### Files (Toolbox: Files)

| Block | Description | Generated Code |
|---|---|---|
| fopen | Open a file | `fopen("file.txt", "r")` |
| fclose | Close a file | `fclose(file);` |
| fread | Read from file | `fread(file, 1024)` |
| fwrite | Write to file | `fwrite(file, data);` |
| readline | Read a line | `readline(file)` |
| file_exists | Check if file exists | `file_exists("file.txt")` |
| read_file | Read entire file | `read_file("file.txt")` |
| write_file | Write entire file | `write_file("file.txt", data);` |
| system | Execute system command | `system("ls");` |
| read_stdin | Read line from stdin | `read_stdin()` |

**Example — File I/O (using Advanced blocks):**
```
import "std/fs.zc"

fn main() {
    let res = File::open("output.txt", "w");
    if res.is_err() {
        println "Failed to open file";
        return;
    }
    let f = res.unwrap();
    f.write_string("Hello from Zen-C!\n");
    f.close();
    println "File written successfully";
}
```

---

### Collections (Toolbox: Collections)

| Block | Description | Generated Code |
|---|---|---|
| Vec::new() | Create a new empty Vec | `Vec<int>::new()` |
| vec.push() | Push a value | `v.push(42);` |
| vec.pop() | Pop last value | `v.pop()` |
| vec.len | Get Vec length | `v.len` |
| vec.data[] | Access element by index | `v.data[0]` |

**Example — Vec & Collections:**
```
import "std.zc"

fn main() {
    let v = Vec<int>::new();
    defer v.free();
    v.push(10);
    v.push(20);
    v.push(30);
    println "length: {v.len}";
    for i in 0..(int)v.len {
        print "{v.data[i]} ";
    }
    println "";
    let popped = v.pop();
    println "popped: {popped}, new length: {v.len}";
}
```

---

### Advanced (Toolbox: Advanced)

| Block | Description | Generated Code |
|---|---|---|
| method call (value) | Call method, get return value | `obj.method(args)` |
| method call (statement) | Call method as statement | `obj.method(args);` |
| static call (value) | Static/associated method (value) | `Type::method(args)` |
| static call (statement) | Static/associated method (statement) | `Type::method(args);` |
| guard | Guard clause | `guard cond else { ... }` |
| def | Compile-time constant | `def MAX = 100;` |
| hex number | Hexadecimal literal | `0xFF` |
| @packed struct | Packed struct (no padding) | `@packed struct Name { ... }` |

**Example — Advanced Blocks:**
```
import "std.zc"

fn main() {
    def MAX = 5;
    let mask = 0xFF;
    guard MAX > 0 else {
        println "Invalid MAX!";
        return;
    }
    let v = Vec<int>::new();
    v.push(mask);
    v.push(42);
    println "MAX={MAX}, mask={mask}, vec len={v.len}";
    v.free();
}
```

---

### Build (Toolbox: Build)

| Block | Description | Generated Code |
|---|---|---|
| import | Import a module | `import "std/io.zc"` |
| cflags | Compiler flags directive | `//> cflags -lm` |
| target | Set compilation target | `//> target x86_64-linux` |
| freestanding | Freestanding mode (no OS) | `//> freestanding` |
| asm | Inline assembly | `asm("nop");` |
| linker | Linker script | `//> linker link.ld` |

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `Ctrl+S` | Save workspace |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Delete` | Delete selected block |

## How the Electron App Works

1. **Compile** writes the generated code to a temp file (`/tmp/zenckly_XXXX.zc`), then runs `zc build <file> -o <output>`.
2. **Run** compiles first, then executes the resulting binary. Output is streamed live to the output panel.
3. The `zc` compiler is searched for in: `PATH`, `../zc` (sibling directory), `/usr/local/bin/zc`, `~/.local/bin/zc`.

## License

Part of the Zen-C project.
