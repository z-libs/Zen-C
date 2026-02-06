<div align="center">
  <p>
    <a href="../README.md">English</a> •
    <a href="README_RU.md">Русский</a> •
    <a href="README_ZH_CN.md">简体中文</a> •
    <a href="README_ZH_TW.md">繁體中文</a> •
    <a href="README_ES.md">Español</a> •
    <a href="README_IT.md">Italiano</a> •
    <a href="README_PT_BR.md">Português Brasileiro</a>
  </p>
</div>

<div align="center">
  <h1>Zen C</h1>
  <h3>现代开发体验。零开销。纯净 C。</h3>
  <br>
  <p>
    <a href="#"><img src="https://img.shields.io/badge/build-passing-brightgreen" alt="构建状态"></a>
    <a href="#"><img src="https://img.shields.io/badge/license-MIT-blue" alt="许可证"></a>
    <a href="#"><img src="https://img.shields.io/github/v/release/z-libs/Zen-C?label=version&color=orange" alt="版本"></a>
    <a href="#"><img src="https://img.shields.io/badge/platform-linux-lightgrey" alt="平台"></a>
  </p>
  <p><em>像高级语言一样编写，像 C 一样运行。</em></p>
</div>

<hr>

<div align="center">
  <p>
    <b><a href="#概述">概述</a></b> •
    <b><a href="#社区">社区</a></b> •
    <b><a href="#快速入门">快速入门</a></b> •
    <b><a href="#语言参考">语言参考</a></b> •
    <b><a href="#标准库">标准库</a></b> •
    <b><a href="#工具链">工具链</a></b>
  </p>
</div>

---

## 概述

**Zen C** 是一种现代系统编程语言，可编译为人类可读的 `GNU C`/`C11`。它提供了一套丰富的特性，包括类型推断、模式匹配、泛型、Trait、async/await 以及具有 RAII 能力的手动内存管理，同时保持 100% 的 C ABI 兼容性。

## 社区

加入官方 Zen C Discord 服务器，参与讨论、展示 Demo、提问或报告 Bug！

- Discord: [点击加入](https://discord.com/invite/q6wEsCmkJP)

---

## 目录

<table>
  <tr>
    <td width="50%" valign="top">
      <h3>通用</h3>
      <ul>
        <li><a href="#概述">概述</a></li>
        <li><a href="#社区">社区</a></li>
        <li><a href="#快速入门">快速入门</a></li>
        <li><a href="#标准库">标准库</a></li>
        <li><a href="#工具链">工具链</a></li>
        <li><a href="#编译器支持与兼容性">编译器支持与兼容性</a></li>
        <li><a href="#贡献">贡献</a></li>
        <li><a href="#致谢与归属">致谢与归属</a></li>
      </ul>
    </td>
    <td width="50%" valign="top">
      <h3>语言参考</h3>
      <ul>
        <li><a href="#1-变量与常量">1. 变量与常量</a></li>
        <li><a href="#2-原始类型">2. 原始类型</a></li>
        <li><a href="#3-复合类型">3. 复合类型</a></li>
        <li><a href="#4-函数与-lambda">4. 函数与 Lambda</a></li>
        <li><a href="#5-控制流">5. 控制流</a></li>
        <li><a href="#6-运算符">6. 运算符</a></li>
        <li><a href="#7-打印与字符串插值">7. 打印与字符串插值</a></li>
        <li><a href="#8-内存管理">8. 内存管理</a></li>
        <li><a href="#9-面向对象编程">9. 面向对象编程</a></li>
        <li><a href="#10-泛型">10. 泛型</a></li>
        <li><a href="#11-并发-asyncawait">11. 并发</a></li>
        <li><a href="#12-元编程">12. 元编程</a></li>
        <li><a href="#13-属性">13. 属性</a></li>
        <li><a href="#14-内联汇编">14. 内联汇编</a></li>
        <li><a href="#15-构建指令">15. 构建指令</a></li>
        <li><a href="#16-关键字">16. 关键字</a></li>
        <li><a href="#17-c-互操作性">17. C 互操作性</a></li>
      </ul>
    </td>
  </tr>
</table>

---

## 快速入门

### 安装

```bash
git clone https://github.com/z-libs/Zen-C.git
cd Zen-C
make
sudo make install
```

### 便携式构建 (APE)

Zen C 可以通过 [Cosmopolitan Libc](https://github.com/jart/cosmopolitan) 编译为 **Actually Portable Executable (APE)**。这将生成一个单个的可执行文件 (`.com`)，能够原生运行在 Linux, macOS, Windows, FreeBSD, OpenBSD, 和 NetBSD 上的 x86_64 和 aarch64 架构上。

**前提条件：**
- `cosmocc` 工具链（必须在 PATH 中）

**构建与安装：**
```bash
make ape
sudo env "PATH=$PATH" make install-ape
```

**产物：**
- `out/bin/zc.com`: 便携式 Zen-C 编译器。已将标准库嵌入到可执行文件中。
- `out/bin/zc-boot.com`: 一个自包含的引导安装程序，用于设置新的 Zen-C 项目。

**用法：**
```bash
# 在任何支持的操作系统上运行
./out/bin/zc.com build hello.zc -o hello
```

### 用法

```bash
# 编译并运行
zc run hello.zc

# 构建可执行文件
zc build hello.zc -o hello

# 交互式 Shell
zc repl
```

### 环境变量

你可以设置 `ZC_ROOT` 来指定标准库的位置（标准导入如 `import "std/vector.zc"`）。这允许你从任何目录运行 `zc`。

```bash
export ZC_ROOT=/path/to/Zen-C
```

---

## 语言参考

### 1. 变量与常量

Zen C 区分编译时常量和运行时变量。

#### 清单常量 (`def`)
仅在编译时存在的值（折叠到代码中）。用于数组大小、固定配置和魔术数字。

```zc
def MAX_SIZE = 1024;
let buffer: char[MAX_SIZE]; // 有效的数组大小
```

#### 变量 (`let`)
内存中的存储位置。可以是可变的或只读的 (`const`)。

```zc
let x = 10;             // 可变
x = 20;                 // 允许

let y: const int = 10;  // 只读 (类型修饰)
// y = 20;              // 错误：无法赋值给 const 变量
```

> [!TIP]
> **类型推导**：Zen C 自动推导初始化变量的类型。在支持的编译器上编译为 C23 的 `auto`，否则使用 GCC 的 `__auto_type` 扩展。

### 2. 原始类型

| 类型 | C 等效类型 | 描述 |
|:---|:---|:---|
| `int`, `uint` | `int32_t`, `uint32_t` | 32位有符号/无符号整数 |
| `c_char`, `c_uchar` | `char`, `unsigned char` | C char (互操作) |
| `c_short`, `c_ushort` | `short`, `unsigned short` | C short (互操作) |
| `c_int`, `c_uint` | `int`, `unsigned int` | C int (互操作) |
| `c_long`, `c_ulong` | `long`, `unsigned long` | C long (互操作) |
| `I8` .. `I128` 或 `i8` .. `i128` | `int8_t` .. `__int128_t` | 有符号固定宽度整数 |
| `U8` .. `U128` 或 `u8` .. `u128` | `uint8_t` .. `__uint128_t` | 无符号固定宽度整数 |
| `isize`, `usize` | `ptrdiff_t`, `size_t` | 指针大小的整数 |
| `byte` | `uint8_t` | U8 的别名 |
| `F32`, `F64` 或 `f32`, `f64`  | `float`, `double` | 浮点数 |
| `bool` | `bool` | `true` 或 `false` |
| `char` | `char` | 单个字符 |
| `string` | `char*` | C-string (以 null 结尾) |
| `U0`, `u0`, `void` | `void` | 空类型 |
| `iN` (例 `i256`) | `_BitInt(N)` | 任意位宽有符号整数 (C23) |
| `uN` (例 `u42`) | `unsigned _BitInt(N)` | 任意位宽无符号整数 (C23) |

> [!IMPORTANT]
> **可移植代码最佳实践**
>
> - 对于所有纯 Zen C 逻辑，请使用 **可移植类型** (`int`、`uint`、`i64`、`u8` 等)。`int` 保证在所有架构上都是 32 位有符号整数。
> - 仅在与 C 库 (FFI) 交互时使用 **C 互操作类型** (`c_int`、`c_char`、`c_long`)。它们的大小因平台和 C 编译器而异。
> - 使用 `isize` 和 `usize` 进行数组索引和内存指针运算。

### 3. 复合类型

#### 数组
具有值语义的固定大小数组。
```zc
def SIZE = 5;
let ints: int[SIZE] = [1, 2, 3, 4, 5];
let zeros: [int; SIZE]; // 零初始化的
```

#### 元组
将多个值组合在一起，通过索引访问元素。
```zc
let pair = (1, "Hello");
let x = pair.0;  // 1
let s = pair.1;  // "Hello"
```

**多个返回值**

函数可以返回元组以提供多个结果：
```zc
fn add_and_subtract(a: int, b: int) -> (int, int) {
    return (a + b, a - b);
}

let result = add_and_subtract(3, 2);
let sum = result.0;   // 5
let diff = result.1;  // 1
```

**解构**

元组可以直接解构为多个变量：
```zc
let (sum, diff) = add_and_subtract(3, 2);
// sum = 5, diff = 1
```

带类型的解构允许显式类型注解：
```zc
let (a: string, b: u8) = ("hello", 42);
let (x, y: i32) = (1, 2);  // 混合：x 推断，y 显式
```

#### 结构体
带有可选位域的数据结构。
```zc
struct Point {
    x: int;
    y: int;
}

// 结构体初始化
let p = Point { x: 10, y: 20 };

// 位域
struct Flags {
    valid: U8 : 1;
    mode:  U8 : 3;
}
```

> [!NOTE]
> 结构体默认使用 [移动语义](#资源语义-默认移动)。即使是指针，也可以通过 `.` 访问字段（自动解引用）。

#### 不透明结构体
你可以将结构体定义为 `opaque`，以将对其字段的访问限制在定义该结构体的模块内部，同时仍允许在栈上分配该结构体（大小已知）。

```zc
// 在 user.zc 中
opaque struct User {
    id: int;
    name: string;
}

fn new_user(name: string) -> User {
    return User{id: 1, name: name}; // 允许：在模块内部
}

// 在 main.zc 中
import "user.zc";

fn main() {
    let u = new_user("Alice");
    // let id = u.id; // 错误：无法访问私有字段 'id'
}
```

#### 枚举
能够持有数据的标签联合 (Sum types)。
```zc
enum Shape {
    Circle(float),      // 持有半径
    Rect(float, float), // 持有宽、高
    Point               // 不带数据
}
```

#### 联合体
标准 C 联合体（不安全访问）。
```zc
union Data {
    i: int;
    f: float;
}
```

#### 类型别名
为现有类型创建新名称。
```zc
alias ID = int;
alias PointMap = Map<string, Point>;
```

#### 不透明类型别名
你可以将类型别名定义为 `opaque`，从而在定义模块之外创建一个与基础类型不同的新类型。这提供了强大的封装和类型安全性，而没有包装结构体的运行时开销。

```zc
// 在 library.zc 中
opaque alias Handle = int;

fn make_handle(v: int) -> Handle {
    return v; // 允许在模块内部进行隐式转换
}

// 在 main.zc 中
import "library.zc";

fn main() {
    let h: Handle = make_handle(42);
    // let i: int = h; // 错误：类型验证失败
    // let h2: Handle = 10; // 错误：类型验证失败
}
```

### 4. 函数与 Lambda

#### 函数
```zc
fn add(a: int, b: int) -> int {
    return a + b;
}

// 调用时支持命名参数
add(a: 10, b: 20);
```

> [!NOTE]
> 命名参数必须严格遵循定义的参数顺序。`add(b: 20, a: 10)` 是无效的。

#### 常量参数
函数参数可以标记为 `const` 以强制执行只读语义。这是一个类型修饰符，而不是清单常量。

```zc
fn print_val(v: const int) {
    // v = 10; // 错误：无法赋值给 const 变量
    println "{v}";
}
```

#### 默认参数
函数可以为尾部参数定义默认值。这些值可以是字面量、表达式或有效的 Zen C 代码（如结构体构造函数）。
```zc
// 简单默认值
fn increment(val: int, amount: int = 1) -> int {
    return val + amount;
}

// 表达式默认值（在调用处计算）
fn offset(val: int, pad: int = 10 * 2) -> int {
    return val + pad;
}

// 结构体默认值
struct Config { debug: bool; }
fn init(cfg: Config = Config { debug: true }) {
    if cfg.debug { println "调试模式"; }
}

fn main() {
    increment(10);      // 11
    offset(5);          // 25
    init();             // 打印 "调试模式"
}
```

#### Lambda (闭包)
可以捕获环境的匿名函数。
```zc
let factor = 2;
let double = x -> x * factor;  // 箭头语法
let full = fn(x: int) -> int { return x * factor; }; // 块语法
```

#### 原始函数指针
Zen C 使用 `fn*` 语法支持原始 C 函数指针。这允许与期望函数指针且没有闭包开销的 C 库进行无缝互操作。

```zc
// 接受原始函数指针的函数
fn set_callback(cb: fn*(int)) {
    cb(42);
}

// 返回原始函数指针的函数
fn get_callback() -> fn*(int) {
    return my_handler;
}

// 支持指向函数指针的指针 (fn**)
let pptr: fn**(int) = &ptr;
```

#### 变参函数
函数可以使用 `...` 和 `va_list` 类型接受可变数量的参数。
```zc
fn log(lvl: int, fmt: char*, ...) {
    let ap: va_list;
    va_start(ap, fmt);
    vprintf(fmt, ap); // 使用 C stdio
    va_end(ap);
}
```

### 5. 控制流

#### 条件语句
```zc
if x > 10 {
    print("Large");
} else if x > 5 {
    print("Medium");
} else {
    print("Small");
}

// 三元运算符
let y = x > 10 ? 1 : 0;

// If-表达式 (用于复杂条件)
let 类别 = if (x > 100) { "巨大" } else if (x > 10) { "大" } else { "小" };
```

#### 模式匹配
`switch` 的强大替代方案。
```zc
match val {
    1         => { print "One" },
    2 || 3    => { print "Two or Three" },    // 使用 || 进行 或 操作
    4 or 5    => { print "Four or Five" },    // 使用 'or' 进行 或 操作
    6, 7, 8   => { print "Six to Eight" },    // 使用逗号进行 或 操作
    10 .. 15  => { print "10 to 14" },        // 左闭右开区间 (旧语法)
    10 ..< 15 => { print "10 to 14" },        // 左闭右开区间 (显式)
    20 ..= 25 => { print "20 to 25" },        // 全闭区间
    _         => { print "Other" },
}

// 解构枚举
match shape {
    Shape::Circle(r)   => { println "半径: {r}" },
    Shape::Rect(w, h)  => { println "面积: {w*h}" },
    Shape::Point       => { println "点" },
}
```

#### 引用绑定
为了在不获取所有权（移动）的情况下检查一个值，在模式中使用 `ref` 关键字。这对于实现了移动语义的类型（如 `Option`, `Result`, 非 Copy 结构体）至关重要。

```zc
let opt = Some(NonCopyVal{...});
match opt {
    Some(ref x) => {
        // 'x' 是指向 'opt' 内部值的指针
        // 'opt' 在此处不会被移动/消耗
        println "{x.field}"; 
    },
    None => {}
}
```

#### 循環
```zc
// 區間迭代
for i in 0..10 { ... }      // 左閉右開 (0 到 9)
for i in 0..<10 { ... }     // 左閉右開 (顯式)
for i in 0..=10 { ... }     // 全閉 (0 到 10)
for i in 0..10 step 2 { ... }

// 迭代器 (Vec 或自定義 Iterable)
for item in vec { ... }

// 直接迭代固定大小数组
let arr: int[5] = [1, 2, 3, 4, 5];
for val in arr {
    // val 是 int
    println "{val}";
}

// While 循環
while x < 10 { ... }

// 帶標籤的無限循環
outer: loop {
    if done { break outer; }
}

// 重複 N 次
for _ in 0..5 { ... }
```

#### 高级控制
```zc
// Guard: 如果条件为假，则执行 else 块并返回
guard ptr != NULL else { return; }

// Unless: 除非为真（即如果为假）
unless is_valid { return; }
```

### 6. 运算符

Zen C 通过实现特定的方法名来支持用户定义结构体的运算符重载。

#### 可重载运算符

| 类别 | 运算符 | 方法名 |
|:---|:---|:---|
| **算术** | `+`, `-`, `*`, `/`, `%` | `add`, `sub`, `mul`, `div`, `rem` |
| **比较** | `==`, `!=` | `eq`, `neq` |
| | `<`, `>`, `<=`, `>=` | `lt`, `gt`, `le`, `ge` |
| **位运算** | `&`, `|`, `^` | `bitand`, `bitor`, `bitxor` |
| | `<<`, `>>` | `shl`, `shr` |
| **一元** | `-` | `neg` |
| | `!` | `not` |
| | `~` | `bitnot` |
| **索引** | `a[i]` | `get(a, i)` |
| | `a[i] = v` | `set(a, i, v)` |

> [!NOTE]
> **关于字符串相等性的说明**：
> - `string == string` 进行 **值比较**（等同于 `strcmp`）。
> - `char* == char*` 进行 **指针比较**（检查内存地址）。
> - 混合比较（例如 `string == char*`）默认为 **指针比较**。

**示例：**
```zc
impl Point {
    fn add(self, other: Point) -> Point {
        return Point{x: self.x + other.x, y: self.y + other.y};
    }
}

let p3 = p1 + p2; // 调用 p1.add(p2)
```

#### 语法糖

这些运算符是内置语言特性，不能直接重载。

| 运算符 | 名称 | 描述 |
|:---|:---|:---|
| `|>` | 管道 | `x |> f(y)` 脱糖为 `f(x, y)` |
| `??` | 空合并 | 如果 `val` 为 NULL，`val ?? default` 返回 `default` (用于指针) |
| `??=` | 空赋值 | 如果 `val` 为 NULL 则赋值 |
| `?.` | 安全导航 | 仅当 `ptr` 不为 NULL 时访问字段 |
| `?` | Try 运算符 | 如果存在错误则返回 (用于 Result/Option 类型) |

**自动解引用**：
指针字段访问 (`ptr.field`) 和方法调用 (`ptr.method()`) 会自动解引用指针，等同于 `(*ptr).field`。

### 7. 打印与字符串插值

Zen C 提供了多种控制台打印选项，包括关键字和简洁的简写形式。

#### 关键字

| 关键字 | 描述 |
|:---|:---|
| `print "text"` | 打印到 `stdout`，不带尾随换行符。 |
| `println "text"` | 打印到 `stdout`，带尾随换行符。 |
| `eprint "text"` | 打印到 `stderr`，不带尾随换行符。 |
| `eprintln "text"` | 打印到 `stderr`，带尾随换行符。 |

#### 简写形式

Zen C 允许直接将字符串字面量用作语句来进行快速打印：

| 语法 | 等效项 | 描述 |
|:---|:---|:---|
| `"Hello World"` | `println "Hello World"` | 隐式添加换行符。 |
| `"Hello World"..` | `print "Hello World"` | 不带尾随换行符。 |
| `!"Error"` | `eprintln "Error"` | 输出到 stderr。 |
| `!"Error"..` | `eprint "Error"` | 输出到 stderr，不带换行符。 |

#### 字符串插值 (F-strings)

你可以使用 `{}` 语法将表达式直接嵌入到字符串字面量中。这适用于所有打印方法和字符串简写。

```zc
let x = 42;
let name = "Zen";
println "值: {x}, 名称: {name}";
"值: {x}, 名称: {name}"; // 简写形式的 println
```

**转义花括号**：使用 `{{` 输出字面量 `{`，使用 `}}` 输出字面量 `}`：

```zc
let json = "JSON: {{\"键\": \"值\"}}";
// 输出: JSON: {"键": "值"}
```

#### 输入提示 (`?`)

Zen C 支持使用 `?` 前缀进行用户输入提示的简写。

- `? "提示文本"`: 打印提示信息（不换行）并等待输入（读取一行）。
- `? "输入年龄: " (age)`: 打印提示并扫描输入到变量 `age` 中。
    - 格式说明符会根据变量类型自动推断。

```zc
let age: int;
? "你多大了？ " (age);
println "你 {age} 岁了。";
```

### 8. 内存管理

Zen C 允许带有符合人体工程学辅助的手动内存管理。

#### Defer
在当前作用域退出时执行代码。Defer 语句按照后进先出 (LIFO) 的顺序执行。
```zc
let f = fopen("file.txt", "r");
defer fclose(f);
```

> 为了防止未定义行为，`defer` 块内不允许使用控制流语句（`return`, `break`, `continue`, `goto`）。

#### Autofree
在作用域退出时自动释放变量。
```zc
autofree let types = malloc(1024);
```

#### 资源语义 (默认移动)
Zen C 将带有析构函数（如 `File`, `Vec`, 或 malloc 的指针）的类型视为 **资源**。为了防止双重释放错误，资源不能被隐式复制。

- **默认移动**：分配资源变量会转移所有权。原始变量变得无效（已移动）。
- **复制类型**：没有析构函数的类型可以申请参与 `Copy` 行为，使赋值变成复制。

**诊断与哲学**：
如果你看到错误 "Use of moved value"，编译器是在告诉你：*"此类型拥有一个资源（如内存或句柄），盲目复制它是不安全的。"*

> **对比：** 与 C/C++ 不同，Zen C 不会隐式复制拥有资源的值。

**函数参数**：
将值传递给函数遵循与赋值相同的规则：资源会被移动，除非通过引用传递。

```zc
fn process(r: Resource) { ... } // 'r' 被移动进函数
fn peek(r: Resource*) { ... }   // 'r' 被借用 (引用)
```

**显式克隆**：
如果你 *确实* 想要一个资源的两个副本，请显式执行：

```zc
let b = a.clone(); // 调用 Clone trait 中的 'clone' 方法
```

**选择性复制 (值类型)**：
对于没有析构函数的小型类型：

```zc
struct Point { x: int; y: int; }
impl Copy for Point {} // 选择参与隐式复制

fn main() {
    let p1 = Point { x: 1, y: 2 };
    let p2 = p1; // 已复制。p1 保持有效。
}
```

#### RAII / Drop Trait
实现 `Drop` 以自动运行清理逻辑。
```zc
impl Drop for MyStruct {
    fn drop(self) {
        self.free();
    }
}
```

### 9. 面向对象编程

#### 方法
使用 `impl` 为类型定义方法。
```zc
impl Point {
    // 静态方法 (构造函数惯例)
    fn new(x: int, y: int) -> Self {
        return Point{x: x, y: y};
    }

    // 实例方法
    fn dist(self) -> float {
        return sqrt(self.x * self.x + self.y * self.y);
    }
}
```

**Self 简写**: 在具有 `self` 参数的方法中，可以使用 `.字段` 作为 `self.字段` 的简写：
```zc
impl Point {
    fn dist(self) -> float {
        return sqrt(.x * .x + .y * .y);  // 等同于 self.x, self.y
    }
}
```

#### 原始类型方法
Zen C 允许你使用相同的 `impl` 语法定义原始类型（如 `int`, `bool` 等）的方法。

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

#### Trait
定义共享行为。
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

#### 标准 Trait
Zen C 包含与语言语法集成的标准 Trait。

**Iterable**

实现 `Iterable<T>` 以便为你的自定义类型启用 `for-in` 循环。

```zc
import "std/iter.zc"

// 定义一个迭代器
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

// 实现 Iterable
impl MyRange {
    fn iterator(self) -> MyIter {
        return MyIter{curr: self.start, stop: self.end};
    }
}

// 在循环中使用
for i in my_range {
    println "{i}";
}
```

**Drop**

实现 `Drop` 来定义一个在对象超出范围时运行的析构函数 (RAII)。

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

> **注意：** 如果一个变量被移动，则原始变量不会调用 `drop`。它遵循 [资源语义](#资源语义-默认移动)。

**Copy**

标记 Trait，用于选择支持 `Copy` 行为（隐式复制）而不是移动语义。通过 `@derive(Copy)` 使用。

> **规则：** 实现了 `Copy` 的类型不得定义析构函数 (`Drop`)。

```zc
@derive(Copy)
struct Point { x: int; y: int; }

fn main() {
    let p1 = Point{x: 1, y: 2};
    let p2 = p1; // 已复制！p1 保持有效。
}
```

**Clone**

实现 `Clone` 以允许显式复制拥有资源的类型。

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
    let b2 = b1.clone(); // 显式复制
}
```

#### 组合
使用 `use` 嵌入其他结构体。你可以将它们混合进来（展平字段）或者为它们命名（嵌套字段）。

```zc
struct Entity { id: int; }

struct Player {
    // 混入 (未命名): 展平字段
    use Entity;  // 直接将 'id' 添加到 Player
    name: string;
}

struct Match {
    // 组合 (命名): 嵌套字段
    use p1: Player; // 通过 match.p1 访问
    use p2: Player; // 通过 match.p2 访问
}
```

### 10. 泛型

结构体和函数的类型安全模板。

```zc
// 泛型结构体
struct Box<T> {
    item: T;
}

// 泛型函数
fn identity<T>(val: T) -> T {
    return val;
}

// 多参数泛型
struct Pair<K, V> {
    key: K;
    value: V;
}
```

### 11. 并发 (Async/Await)

基于 pthreads 构建。

```zc
async fn fetch_data() -> string {
    // 在后台运行
    return "Data";
}

fn main() {
    let future = fetch_data();
    let result = await future;
}
```

### 12. 元编程

#### Comptime
在编译时运行代码以生成源代码或打印消息。
```zc
comptime {
    // 在编译时生成代码(写入 stdout)
    println "let build_date = \"2024-01-01\";";
}

println "Build Date: {build_date}";
```

**辅助函数**

`comptime` 块内可用的特殊函数:
- **`yield(str)`** - 显式输出生成的代码(printf 的替代方案)
- **`compile_error(msg)`** - 以致命错误消息停止编译
- **`compile_warn(msg)`** - 发出编译时警告(允许继续编译)

```zc
comptime {
    compile_warn("正在生成优化代码...");
    
    let ENABLE_FEATURE = 1;
    if (ENABLE_FEATURE == 0) {
        compile_error("必须启用功能!");
    }
    
    println "let FEATURE_ENABLED = 1;";
}
```

**构建元数据**

在编译时访问编译器构建信息:
- **`__COMPTIME_TARGET__`** - 平台字符串: `"linux"`, `"windows"` 或 `"macos"`
- **`__COMPTIME_FILE__`** - 当前正在编译的源文件名

```zc
comptime {
    // 平台特定的代码生成
    println "let PLATFORM = \"{__COMPTIME_TARGET__}\";";
}

println "运行于: {PLATFORM}";
```

> **注意**: 在 comptime 字符串内使用 `{{` 和 `}}` 来转义花括号。

#### Embed
将文件嵌入为指定类型。
```zc
// 默认 (Slice_char)
let data = embed "assets/logo.png";

// 类型化嵌入
let text = embed "shader.glsl" as string;    // 嵌入为 C-string
let rom  = embed "bios.bin" as u8[1024];     // 嵌入为固定数组
let wav  = embed "sound.wav" as u8[];        // 嵌入为 Slice_u8
```

#### 插件
导入编译器插件以扩展语法。
```zc
import plugin "regex"
let re = regex! { ^[a-z]+$ };
```

#### 泛型 C 宏
将预处理器宏传递给 C。

> **提示**：对于简单的常量，请使用 `def`。当你需要 C 预处理器宏或条件编译标志时，请使用 `#define`。

```zc
#define MAX_BUFFER 1024
```

### 13. 属性

修饰函数和结构体以修改编译器行为。

| 属性 | 作用域 | 描述 |
|:---|:---|:---|
| `@must_use` | 函数 | 如果忽略返回值则发出警告。 |
| `@deprecated("msg")` | 函数/结构体 | 使用时发出带有消息的警告。 |
| `@inline` | 函数 | 提示编译器进行内联。 |
| `@noinline` | 函数 | 防止内联。 |
| `@packed` | 结构体 | 移除字段间的填充。 |
| `@align(N)` | 结构体 | 强制按 N 字节对齐。 |
| `@constructor` | 函数 | 在 main 之前运行。 |
| `@destructor` | 函数 | 在 main 退出后运行。 |
| `@unused` | 函数/变量 | 抑制未使用变量警告。 |
| `@weak` | 函数 | 弱符号链接。 |
| `@section("name")` | 函数 | 将代码放置在特定段中。 |
| `@noreturn` | 函数 | 函数不会返回 (例如 exit)。 |
| `@pure` | 函数 | 函数无副作用 (优化提示)。 |
| `@cold` | 函数 | 函数不太可能被执行 (分支预测提示)。 |
| `@hot` | 函数 | 函数频繁执行 (优化提示)。 |
| `@export` | 函数/结构体 | 导出符号 (默认可见性)。 |
| `@global` | 函数 | CUDA: 内核入口点 (`__global__`)。 |
| `@device` | 函数 | CUDA: 设备函数 (`__device__`)。 |
| `@host` | 函数 | CUDA: 主机函数 (`__host__`)。 |
| `@comptime` | 函数 | 用于编译时执行的辅助函数。 |
| `@derive(...)` | 结构体 | 自动实现 Trait。支持 `Debug`, `Eq` (智能派生), `Copy`, `Clone`。 |
| `@ctype("type")` | 函数参数 | 覆盖参数生成的 C 类型。 |
| `@<custom>` | 任意 | 将泛型属性传递给 C (例如 `@flatten`, `@alias("name")`)。 |

#### 自定义属性

Zen C 支持强大的 **自定义属性** 系统，允许你在代码中直接使用任何 GCC/Clang 的 `__attribute__`。任何不被 Zen C 编译器显式识别的属性都会被视为泛型属性并传递给生成的 C 代码。

这提供了对高级编译器特性、优化和链接器指令的访问，而无需在语言核心中提供显式支持。

#### 语法映射
Zen C 属性直接映射到 C 属性：
- `@name` → `__attribute__((name))`
- `@name(args)` → `__attribute__((name(args)))`
- `@name("string")` → `__attribute__((name("string")))`

#### 智能派生

Zen C 提供了尊重移动语义的 "智能派生"：

- **`@derive(Eq)`**：生成一个通过引用获取参数的相等性方法 (`fn eq(self, other: T*)`)。
    - 当比较两个非 Copy 结构体 (`a == b`) 时，编译器会自动通过引用传递 `b` (`&b`) 以避免移动它。
    - 字段上的递归相等性检查也会优先使用指针访问，以防止所有权转移。

### 14. 内联汇编

Zen C 为内联汇编提供了一流支持，直接转译为 GCC 风格的扩展 `asm`。

#### 基本用法
在 `asm` 块内编写原始汇编。字符串会自动拼接。
```zc
asm {
    "nop"
    "mfence"
}
```

#### Volatile
防止编译器优化掉具有副作用的汇编代码。
```zc
asm volatile {
    "rdtsc"
}
```

#### 命名约束
Zen C 通过命名绑定简化了复杂的 GCC 约束语法。

```zc
// 语法: : out(变量) : in(变量) : clobber(寄存器)
// 使用 {变量} 占位符语法以提高可读性

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

| 类型 | 语法 | GCC 等效项 |
|:---|:---|:---|
| **输出** | `: out(variable)` | `"=r"(variable)` |
| **输入** | `: in(variable)` | `"r"(variable)` |
| **破坏** | `: clobber("rax")` | `"rax"` |
| **内存** | `: clobber("memory")` | `"memory"` |

> **注意：** 使用 Intel 语法时（通过 `-masm=intel`），必须确保你的构建配置正确（例如，`//> cflags: -masm=intel`）。TCC 不支持 Intel 语法的汇编。

### 15. 构建指令

Zen C 支持在源文件顶部使用特殊注释来配置构建过程，无需复杂的构建系统或 Makefile。

| 指令 | 参数 | 描述 |
|:---|:---|:---|
| `//> link:` | `-lfoo` 或 `path/to/lib.a` | 链接库或对象文件。 |
| `//> lib:` | `path/to/libs` | 添加库搜索路径 (`-L`)。 |
| `//> include:` | `path/to/headers` | 添加包含头文件搜索路径 (`-I`)。 |
| `//> framework:` | `Cocoa` | 链接 macOS Framework。 |
| `//> cflags:` | `-Wall -O3` | 向 C 编译器传递任意标志。 |
| `//> define:` | `MACRO` 或 `KEY=VAL` | 定义预处理器宏 (`-D`)。 |
| `//> pkg-config:` | `gtk+-3.0` | 运行 `pkg-config` 并追加 `--cflags` 和 `--libs`。 |
| `//> shell:` | `command` | 在构建期间执行 shell 命令。 |
| `//> get:` | `http://url/file` | 如果特定文件不存在，则下载该文件。 |

#### 特性

**1. 操作系统守护 (OS Guarding)**
在指令前加上操作系统名称，以使其仅在特定平台上应用。
受支持的前缀：`linux:`, `windows:`, `macos:` (或 `darwin:`)。

```zc
//> linux: link: -lm
//> windows: link: -lws2_32
//> macos: framework: Cocoa
```

**2. 环境变量展开**
使用 `${VAR}` 语法在指令中展开环境变量。

```zc
//> include: ${HOME}/mylib/include
//> lib: ${ZC_ROOT}/std
```

#### 示例

```zc
//> include: ./include
//> lib: ./libs
//> link: -lraylib -lm
//> cflags: -Ofast
//> pkg-config: gtk+-3.0

import "raylib.h"

fn main() { ... }
```

### 16. 关键字

以下关键字在 Zen C 中是保留的。

#### 声明
`alias`, `def`, `enum`, `fn`, `impl`, `import`, `let`, `module`, `opaque`, `struct`, `trait`, `union`, `use`

#### 控制流
`async`, `await`, `break`, `catch`, `continue`, `defer`, `else`, `for`, `goto`, `guard`, `if`, `loop`, `match`, `return`, `try`, `unless`, `while`

#### 特殊
`asm`, `assert`, `autofree`, `comptime`, `const`, `embed`, `launch`, `ref`, `sizeof`, `static`, `test`, `volatile`

#### 常量
`true`, `false`, `null`

#### C 保留字
以下标识符是保留的，因为它们是 C11 中的关键字：
`auto`, `case`, `char`, `default`, `do`, `double`, `extern`, `float`, `inline`, `int`, `long`, `register`, `restrict`, `short`, `signed`, `switch`, `typedef`, `unsigned`, `void`, `_Atomic`, `_Bool`, `_Complex`, `_Generic`, `_Imaginary`, `_Noreturn`, `_Static_assert`, `_Thread_local`

#### 运算符
`and`, `or`

### 17. C 互操作性

Zen C 提供了两种与 C 代码交互的方式：**信任导入 (Trusted Imports)** (方便) 和 **显式 FFI** (安全/精确)。

#### 方法 1: 信任导入 (方便)

你可以使用 `import` 关键字直接导入 `.h` 扩展名的 C 头文件。这会将头文件视为一个模块，并假设通过它访问的所有符号都存在。

```zc
//> link: -lm
import "math.h" as c_math;

fn main() {
    // 编译器信任不仅正确；直接生成 'cos(...)'
    let x = c_math::cos(3.14159);
}
```

> **优点**: 零样板代码。立即访问头文件中的所有内容。
> **缺点**: Zen C 不提供类型安全 (错误将在稍后由 C 编译器捕获)。

#### 方法 2: 显式 FFI (安全)

对于严格的类型检查，或当你不想包含头文件文本时，请使用 `extern fn`.

```zc
include <stdio.h> // 在生成的 C 代码中发出 #include <stdio.h>

// 定义严格签名
extern fn printf(fmt: char*, ...) -> c_int;

fn main() {
    printf("Hello FFI: %d\n", 42); // 由 Zen C 进行类型检查
}
```

> **优点**: Zen C 确保类型匹配。
> **缺点**: 需要手动声明函数。

#### `import` vs `include`

- **`import "file.h"`**: 将头文件注册为命名模块。启用对符号的隐式访问 (例如 `file::function()`)。
- **`include <file.h>`**: 纯粹在生成的 C 代码中发出 `#include <file.h>`。不向 Zen C 编译器引入任何符号；必须使用 `extern fn` 才能访问它们。


---

## 标准库

Zen C 包含一个涵盖基本功能的标准库 (`std`)。

[浏览标准库文档](../docs/std/README.md)

### 核心模块

<details>
<summary>点击查看所有标准库模块</summary>

| 模块 | 描述 | 文档 |
| :--- | :--- | :--- |
| **`std/vec.zc`** | 可增长动态数组 `Vec<T>`。 | [文档](../docs/std/vec.md) |
| **`std/string.zc`** | 堆分配的 `String` 类型，支持 UTF-8。 | [文档](../docs/std/string.md) |
| **`std/queue.zc`** | 先进先出队列 (环形缓冲区)。 | [文档](../docs/std/queue.md) |
| **`std/map.zc`** | 泛型哈希表 `Map<V>`。 | [文档](../docs/std/map.md) |
| **`std/fs.zc`** | 文件系统操作。 | [文档](../docs/std/fs.md) |
| **`std/io.zc`** | 标准输入/输出 (`print`/`println`)。 | [文档](../docs/std/io.md) |
| **`std/option.zc`** | 可选值 (`Some`/`None`)。 | [文档](../docs/std/option.md) |
| **`std/result.zc`** | 错误处理 (`Ok`/`Err`)。 | [文档](../docs/std/result.md) |
| **`std/path.zc`** | 跨平台路径操作。 | [文档](../docs/std/path.md) |
| **`std/env.zc`** | 进程环境变量。 | [文档](../docs/std/env.md) |
| **`std/net/`** | TCP, UDP, HTTP, DNS, URL. | [文档](../docs/std/net.md) |
| **`std/thread.zc`** | 线程与同步。 | [文档](../docs/std/thread.md) |
| **`std/time.zc`** | 时间测量与睡眠。 | [文档](../docs/std/time.md) |
| **`std/json.zc`** | JSON 解析与序列化。 | [文档](../docs/std/json.md) |
| **`std/stack.zc`** | 后进先出栈 `Stack<T>`。 | [文档](../docs/std/stack.md) |
| **`std/set.zc`** | 泛型哈希集合 `Set<T>`。 | [文档](../docs/std/set.md) |
| **`std/process.zc`** | 进程执行与管理。 | [文档](../docs/std/process.md) |

</details>

---

## 工具链

Zen C 提供内置的语言服务器 (LSP) 和 REPL 以增强开发体验。

### 语言服务器 (LSP)

Zen C 语言服务器 (LSP) 支持标准的 LSP 特性，用于编辑器集成：

*   **转到定义**
*   **查找引用**
*   **悬停信息**
*   **补全** (函数/结构体名，方法/字段的点补全)
*   **文档符号** (大纲)
*   **签名帮助**
*   **诊断** (语法/语义错误)

启动语言服务器（通常在编辑器的 LSP 设置中配置）：

```bash
zc lsp
```

它通过标准 I/O (JSON-RPC 2.0) 进行通信。

### REPL

Read-Eval-Print Loop 允许你交互式地尝试 Zen C 代码。

```bash
zc repl
```

#### 特性

*   **交互式编码**：输入表达式或语句以立即求值。
*   **持久历史**：命令保存在 `~/.zprep_history` 中。
*   **启动脚本**：自动加载 `~/.zprep_init.zc` 中的命令。

#### 命令

| 命令 | 描述 |
|:---|:---|
| `:help` | 显示可用命令。 |
| `:reset` | 清除当前会话历史 (变量/函数)。 |
| `:vars` | 显示活跃变量。 |
| `:funcs` | 显示用户定义的函数。 |
| `:structs` | 显示用户定义的结构体。 |
| `:imports` | 显示活跃导入。 |
| `:history` | 显示会话输入历史。 |
| `:type <expr>` | 显示表达式的类型。 |
| `:c <stmt>` | 显示语句生成的 C 代码。 |
| `:time <expr>` | 基准测试表达式 (运行 1000 次迭代)。 |
| `:edit [n]` | 在 `$EDITOR` 中编辑命令 `n` (默认：最后一条)。 |
| `:save <file>` | 将当前会话保存到 `.zc` 文件。 |
| `:load <file>` | 将 `.zc` 文件加载并执行到会话中。 |
| `:watch <expr>` | 监视表达式 (每次输入后重新求值)。 |
| `:unwatch <n>` | 移除监视。 |
| `:undo` | 从会话中移除最后一条命令。 |
| `:delete <n>` | 移除索引为 `n` 的命令。 |
| `:clear` | 清屏。 |
| `:quit` | 退出 REPL。 |
| `! <cmd>` | 运行 shell 命令 (如 `!ls`)。 |

---


## 编译器支持与兼容性

Zen C 旨在与大多数 C11 编译器配合使用。某些特性依赖于 GNU C 扩展，但这些扩展通常在其他编译器中也能工作。使用 `--cc` 标志切换后端。

```bash
zc run app.zc --cc clang
zc run app.zc --cc zig
```

### 测试套件状态

<details>
<summary>点击查看编译器支持详情</summary>

| 编译器 | 通过率 | 受支持特性 | 已知局限性 |
|:---|:---:|:---|:---|
| **GCC** | **100% (完全)** | 所有特性 | 无。 |
| **Clang** | **100% (完全)** | 所有特性 | 无。 |
| **Zig** | **100% (完全)** | 所有特性 | 无。使用 `zig cc` 作为替代 C 编译器。 |
| **TCC** | **~70% (基础)** | 基本语法, 泛型, Trait | 不支持 `__auto_type`, 不支持 Intel ASM, 不支持嵌套函数。 |

</details>

> [!TIP]
> **建议：** 生产环境构建请使用 **GCC**, **Clang**, 或 **Zig**。TCC 非常适合快速原型开发，因为它编译速度极快，但缺少 Zen C 全面支持所需的一些高级 C 扩展。

### 使用 Zig 构建

Zig 的 `zig cc` 命令提供了 GCC/Clang 的替代方案，具有出色的跨平台编译支持。使用 Zig：

```bash
# 使用 Zig 编译并运行 Zen C 程序
zc run app.zc --cc zig

# 使用 Zig 构建 Zen C 编译器本身
make zig
```

### C++ 互操作

Zen C 可以通过 `--cpp` 标志生成 C++ 兼容的代码，从而实现与 C++ 库的无缝集成。

```bash
# 直接使用 g++ 编译
zc app.zc --cpp

# 或者转译用于手动构建
zc transpile app.zc --cpp
g++ out.c my_cpp_lib.o -o app
```

#### 在 Zen C 中使用 C++

包含 C++ 头文件并在 `raw` 块中使用 C++ 代码：

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

> **注意：** `--cpp` 标志会将后端切换为 `g++` 并发出 C++ 兼容的代码（使用 `auto` 代替 `__auto_type`，使用函数重载代替 `_Generic`，以及对 `void*` 进行显式转换）。

#### CUDA 互操作

Zen C 通过转译为 **CUDA C++** 来支持 GPU 编程。这使你在维持 Zen C 人体工程学语法的同时，能够利用内核中的强大 C++ 特性（模板、constexpr）。

```bash
# 直接使用 nvcc 编译
zc run app.zc --cuda

# 或者转译用于手动构建
zc transpile app.zc --cuda -o app.cu
nvcc app.cu -o app
```

#### CUDA 特定属性

| 属性 | CUDA 等效项 | 描述 |
|:---|:---|:---|
| `@global` | `__global__` | 内核函数 (运行在 GPU，从主机调用) |
| `@device` | `__device__` | 设备函数 (运行在 GPU，从 GPU 调用) |
| `@host` | `__host__` | 主机函数 (明确仅 CPU 运行) |

#### 内核启动语法

Zen C 提供了一个简洁的 `launch` 语句用于调用 CUDA 内核：

```zc
launch kernel_name(args) with {
    grid: num_blocks,
    block: threads_per_block,
    shared_mem: 1024,  // 可选
    stream: my_stream   // 可选
};
```

这转译为：`kernel_name<<<grid, block, shared, stream>>>(args);`

#### 编写 CUDA 内核

使用带有 `@global` 的 Zen C 函数语法和 `launch` 语句：

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

    // ... 初始化数据 ...
    
    launch add_kernel(d_a, d_b, d_c, N) with {
        grid: (N + 255) / 256,
        block: 256
    };
    
    cuda_sync();
}
```

#### 标准库 (`std/cuda.zc`)
Zen C 为常见的 CUDA 操作提供了一个标准库，以减少 `raw` 块的使用：

```zc
import "std/cuda.zc"

// 内存管理
let d_ptr = cuda_alloc<float>(1024);
cuda_copy_to_device(d_ptr, h_ptr, 1024 * sizeof(float));
defer cuda_free(d_ptr);

// 同步
cuda_sync();

// 线程索引 (在内核内部使用)
let i = thread_id(); // 全局索引
let bid = block_id();
let tid = local_id();
```


> [!NOTE]
> **注意：** `--cuda` 标志设置 `nvcc` 为编译器并隐含 `--cpp` 模式。需要安装 NVIDIA CUDA Toolkit。

### C23 支持

当使用兼容的后端编译器（GCC 14+, Clang 14+）时，Zen C 支持现代 C23特性。

- **`auto`**: 如果 `__STDC_VERSION__ >= 202300L`，Zen C 会自动将类型推导映射到标准 C23 `auto`。
- **`_BitInt(N)`**: 使用 `iN` 和 `uN` 类型（例如 `i256`, `u12`, `i24`）访问 C23 任意位宽整数。

### Objective-C 互操作

Zen C 可以通过 `--objc` 标志编译为 Objective-C (`.m`)，允许你使用 Objective-C 框架（如 Cocoa/Foundation）和语法。

```bash
# 使用 clang (或 gcc/gnustep) 编译
zc app.zc --objc --cc clang
```

#### 在 Zen C 中使用 Objective-C

使用 `include` 包含头文件，并在 `raw` 块中使用 Objective-C 语法 (`@interface`, `[...]`, `@""`)。

```zc
//> macos: framework: Foundation
//> linux: cflags: -fconstant-string-class=NSConstantString -D_NATIVE_OBJC_EXCEPTIONS
//> linux: link: -lgnustep-base -lobjc

include <Foundation/Foundation.h>

fn main() {
    raw {
        NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
        NSLog(@"来自 Objective-C 的问候！");
        [pool drain];
    }
    println "Zen C 也能正常工作！";
}
```

> [!NOTE]
> **注意：** Zen C 字符串插值通过调用 `debugDescription` 或 `description` 同样适用于 Objective-C 对象 (`id`)。

---

## 贡献

我们欢迎各类贡献！无论是修复 Bug、完善文档，还是提出新功能建议。

请参阅 [CONTRIBUTING_ZH_CN.md](CONTRIBUTING_ZH_CN.md) 了解有关如何贡献、运行测试和提交拉取请求的详细指南。

---

## 安全

关于安全漏洞报告的说明，请参阅 [SECURITY_ZH_CN.md](SECURITY_ZH_CN.md)。

---

## 致谢与归属

本项目使用了第三方库。完整许可证文本可在 `LICENSES/` 目录中找到。

*   **[cJSON](https://github.com/DaveGamble/cJSON)** (MIT 许可证)：用于语言服务器中的 JSON 解析和生成。
*   **[zc-ape](https://github.com/OEvgeny/zc-ape)** (MIT 许可证)：由 [Eugene Olonov](https://github.com/OEvgeny) 开发的原版 Zen-C 实际上便携的可执行文件 (APE) 端口。
*   **[Cosmopolitan Libc](https://github.com/jart/cosmopolitan)** (ISC 许可证)：使 APE 成为可能的基础库。

---

<div align="center">
  <p>
    Copyright © 2026 Zen C 编程语言。<br>
    今天就开始你的旅程。
  </p>
  <p>
    <a href="https://discord.com/invite/q6wEsCmkJP">Discord</a> •
    <a href="https://github.com/z-libs/Zen-C">GitHub</a> •
    <a href="CONTRIBUTING_ZH_CN.md">贡献</a>
  </p>
</div>
