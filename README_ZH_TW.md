
<div align="center">

[English](README.md) • [简体中文](README_ZH_CN.md) • [繁體中文](README_ZH_TW.md) • [Español](README_ES.md) • [Italiano](README_IT.md)

</div>

<div align="center">

# Zen C

**現代開發體驗。零開銷。純淨 C。**

[![構建狀態](https://img.shields.io/badge/build-passing-brightgreen)]()
[![許可證](https://img.shields.io/badge/license-MIT-blue)]()
[![版本](https://img.shields.io/github/v/release/z-libs/Zen-C?label=version&color=orange)]()
[![平台](https://img.shields.io/badge/platform-linux-lightgrey)]()

*像高級語言一樣編寫，像 C 一樣運行。*

</div>

---

## 概述

**Zen C** 是一種現代系統編程語言，可編譯為人類可讀的 `GNU C`/`C11`。它提供了一套豐富的特性，包括類型推斷、模式匹配、泛型、Trait、async/await 以及具有 RAII 能力的手動內存管理，同時保持 100% 的 C ABI 兼容性。

## 社區

加入官方 Zen C Discord 服務器，參與討論、展示 Demo、提問或報告 Bug！

- Discord: [點擊加入](https://discord.com/invite/q6wEsCmkJP)

---

## 目錄

- [概述](#概述)
- [社區](#社區)
- [快速入門](#快速入門)
    - [安裝](#安裝)
    - [用法](#用法)
    - [環境變量](#環境變量)
- [語言參考](#語言參考)
    - [1. 變量與常量](#1-變量與常量)
    - [2. 原始類型](#2-原始類型)
    - [3. 複合類型](#3-複合類型)
        - [數組](#數組)
        - [元組](#元組)
        - [結構體](#結構體)
        - [不透明結構體](#不透明結構體)
        - [枚舉](#枚舉)
        - [聯合體](#聯合體)
        - [類型別名](#類型別名)
        - [不透明類型別名](#不透明類型別名)
    - [4. 函數與 Lambda](#4-函數與-lambda)
        - [函數](#函數)
        - [常量參數](#常量參數)
        - [默認參數](#默認參數)
        - [Lambda (閉包)](#lambda-閉包)
        - [原始函數指針](#原始函數指針)
        - [變參函數](#變參函數)
    - [5. 控制流](#5-控制流)
        - [條件語句](#條件語句)
        - [模式匹配](#模式匹配)
        - [循環](#循環)
        - [高級控制](#高級控制)
    - [6. 運算符](#6-運算符)
        - [可重載運算符](#可重載運算符)
        - [語法糖](#語法糖)
    - [7. 打印與字符串插值](#7-打印與字符串插值)
        - [關鍵字](#關鍵字)
        - [簡寫形式](#簡寫形式)
        - [字符串插值 (F-strings)](#字符串插值-f-strings)
        - [輸入提示 (`?`)](#輸入提示-)
    - [8. 內存管理](#8-內存管理)
        - [Defer](#defer)
        - [Autofree](#autofree)
        - [資源語義 (默認移動)](#資源語義-默認移動)
        - [RAII / Drop Trait](#raii--drop-trait)
    - [9. 面向對象編程](#9-面向對象編程)
        - [方法](#方法)
        - [Trait](#trait)
        - [標準 Trait](#標準-trait)
        - [組合](#組合)
    - [10. 泛型](#10-泛型)
    - [11. 並發 (Async/Await)](#11-並發-asyncawait)
    - [12. 元編程](#12-元編程)
        - [Comptime](#comptime)
        - [Embed](#embed)
        - [插件](#插件)
        - [泛型 C 宏](#泛型-c-宏)
    - [13. 屬性](#13-屬性)
    - [自定義屬性](#自定義屬性)
    - [智能派生](#智能派生)
    - [14. 內聯匯編](#14-內聯匯編)
        - [基本用法](#基本用法)
        - [Volatile](#volatile)
        - [命名約束](#命名約束)
    - [15. 構建指令](#15-構建指令)
    - [16. 關鍵字](#16-關鍵字)
- [標準庫](#標準庫)
- [工具鏈](#工具鏈)
    - [語言服務器 (LSP)](#語言服務器-lsp)
    - [REPL](#repl)
- [編譯器支持與兼容性](#編譯器支持與兼容性)
    - [測試套件狀態](#測試套件狀態)
    - [使用 Zig 構建](#使用-zig-構建)
    - [C++ 互操作](#c-互操作)
    - [CUDA 互操作](#cuda-互操作)
    - [Objective-C 互操作](#objective-c-互操作)
- [貢獻](#貢獻)
- [致謝與歸屬](#致謝與歸屬)

---

## 快速入門

### 安裝

```bash
git clone https://github.com/z-libs/Zen-C.git
cd Zen-C
make
sudo make install
```

### 便攜式構建 (APE)

Zen C 可以通過 [Cosmopolitan Libc](https://github.com/jart/cosmopolitan) 編譯為 **Actually Portable Executable (APE)**。這將生成一個單個的可執行文件 (`.com`)，能夠原生運行在 Linux, macOS, Windows, FreeBSD, OpenBSD, 和 NetBSD 上的 x86_64 和 aarch64 架構上。

**前提條件：**
- `cosmocc` 工具鏈（必須在 PATH 中）

**構建與安裝：**
```bash
make ape
sudo env "PATH=$PATH" make install-ape
```

**產物：**
- `out/bin/zc.com`: 便攜式 Zen-C 編譯器。已將標準庫嵌入到可執行文件中。
- `out/bin/zc-boot.com`: 一個自包含的引導安裝程序，用於設置新的 Zen-C 項目。

**用法：**
```bash
# 在任何支持的操作系統上運行
./out/bin/zc.com build hello.zc -o hello
```

### 用法

```bash
# 編譯並運行
zc run hello.zc

# 構建可執行文件
zc build hello.zc -o hello

# 交互式 Shell
zc repl
```

### 環境變量

你可以設置 `ZC_ROOT` 來指定標準庫的位置（標準導入如 `import "std/vector.zc"`）。這允許你從任何目錄運行 `zc`。

```bash
export ZC_ROOT=/path/to/Zen-C
```

---

## 語言參考

### 1. 變量與常量

Zen C 區分編譯時常量和運行時變量。

#### 清單常量 (`def`)
僅在編譯時存在的值（折疊到代碼中）。用於數組大小、固定配置和魔術數字。

```zc
def MAX_SIZE = 1024;
let buffer: char[MAX_SIZE]; // 有效的數組大小
```

#### 變量 (`let`)
內存中的存儲位置。可以是可變的或只讀的 (`const`)。

```zc
let x = 10;             // 可變
x = 20;                 // 允許

let y: const int = 10;  // 只讀 (類型修飾)
// y = 20;              // 錯誤：無法賦值給 const 變量
```

> **型別推導**：Zen C 自動推導初始化變數的型別。在支援的編譯器上編譯為 C23 的 `auto`，否則使用 GCC 的 `__auto_type` 擴充功能。

### 2. 原始類型

| 類型 | C 等效類型 | 描述 |
|:---|:---|:---|
| `int`, `uint` | `int`, `unsigned int` | 平台標準整數 |
| `I8` .. `I128` 或 `i8` .. `i128` | `int8_t` .. `__int128_t` | 有符號固定寬度整數 |
| `U8` .. `U128` 或 `u8` .. `u128` | `uint8_t` .. `__uint128_t` | 無符號固定寬度整數 |
| `isize`, `usize` | `ptrdiff_t`, `size_t` | 指針大小的整數 |
| `byte` | `uint8_t` | U8 的別名 |
| `F32`, `F64` 或 `f32`, `f64`  | `float`, `double` | 浮點數 |
| `bool` | `bool` | `true` 或 `false` |
| `char` | `char` | 單個字符 |
| `string` | `char*` | C-string (以 null 結尾) |
| `U0`, `u0`, `void` | `void` | 空類型 |
| `iN` (例 `i256`) | `_BitInt(N)` | 任意位元寬度有號整數 (C23) |
| `uN` (例 `u42`) | `unsigned _BitInt(N)` | 任意位元寬度無號整數 (C23) |

### 3. 複合類型

#### 數組
具有值語義的固定大小數組。
```zc
def SIZE = 5;
let ints: int[SIZE] = [1, 2, 3, 4, 5];
let zeros: [int; SIZE]; // 零初始化的
```

#### 元組
將多個值組合在一起，通過索引訪問元素。
```zc
let pair = (1, "Hello");
let x = pair.0;  // 1
let s = pair.1;  // "Hello"
```

**多個返回值**

函數可以返回元組以提供多個結果：
```zc
fn add_and_subtract(a: int, b: int) -> (int, int) {
    return (a + b, a - b);
}

let result = add_and_subtract(3, 2);
let sum = result.0;   // 5
let diff = result.1;  // 1
```

**解構**

元組可以直接解構為多個變量：
```zc
let (sum, diff) = add_and_subtract(3, 2);
// sum = 5, diff = 1
```

#### 結構體
帶有可選位域的數據結構。
```zc
struct Point {
    x: int;
    y: int;
}

// 結構體初始化
let p = Point { x: 10, y: 20 };

// 位域
struct Flags {
    valid: U8 : 1;
    mode:  U8 : 3;
}
```

> **注意**：結構體默認使用 [移動語義](#資源語義-默認移動)。即使是指針，也可以通過 `.` 訪問字段（自動解引用）。

#### 不透明結構體
你可以將結構體定義為 `opaque`，以將對其字段的訪問限制在定義該結構體的模塊內部，同時仍允許在棧上分配該結構體（大小已知）。

```zc
// 在 user.zc 中
opaque struct User {
    id: int;
    name: string;
}

fn new_user(name: string) -> User {
    return User{id: 1, name: name}; // 允許：在模塊內部
}

// 在 main.zc 中
import "user.zc";

fn main() {
    let u = new_user("Alice");
    // let id = u.id; // 錯誤：無法訪問私有字段 'id'
}
```

#### 枚舉
能夠持有數據的標籤聯合 (Sum types)。
```zc
enum Shape {
    Circle(float),      // 持有半徑
    Rect(float, float), // 持有寬、高
    Point               // 不帶數據
}
```

#### 聯合體
標準 C 聯合體（不安全訪問）。
```zc
union Data {
    i: int;
    f: float;
}
```

#### 類型別名
為現有類型創建新名稱。
```zc
alias ID = int;
alias PointMap = Map<string, Point>;
```

#### 不透明類型別名
你可以將類型別名定義為 `opaque`，從而在定義模塊之外創建一個與基礎類型不同的新類型。這提供了強大的封裝和類型安全性，而沒有包裝結構體的運行時開銷。

```zc
// 在 library.zc 中
opaque alias Handle = int;

fn make_handle(v: int) -> Handle {
    return v; // 允許在模塊內部進行隱式轉換
}

// 在 main.zc 中
import "library.zc";

fn main() {
    let h: Handle = make_handle(42);
    // let i: int = h; // 錯誤：類型驗證失敗
    // let h2: Handle = 10; // 錯誤：類型驗證失敗
}
```

### 4. 函數與 Lambda

#### 函數
```zc
fn add(a: int, b: int) -> int {
    return a + b;
}

// 調用時支持命名參數
add(a: 10, b: 20);
```

> **注意**：命名參數必須嚴格遵循定義的參數順序。`add(b: 20, a: 10)` 是無效的。

#### 常量參數
函數參數可以標記為 `const` 以強制執行只讀語義。這是一個類型修飾符，而不是清單常量。

```zc
fn print_val(v: const int) {
    // v = 10; // 錯誤：無法賦值給 const 變量
    println "{v}";
}
```

#### 默認參數
函數可以為尾部參數定義默認值。這些值可以是字面量、表達式或有效的 Zen C 代碼（如結構體構造函數）。
```zc
// 簡單默認值
fn increment(val: int, amount: int = 1) -> int {
    return val + amount;
}

// 表達式默認值（在調用處計算）
fn offset(val: int, pad: int = 10 * 2) -> int {
    return val + pad;
}

// 結構體默認值
struct Config { debug: bool; }
fn init(cfg: Config = Config { debug: true }) {
    if cfg.debug { println "調試模式"; }
}

fn main() {
    increment(10);      // 11
    offset(5);          // 25
    init();             // 打印 "調試模式"
}
```

#### Lambda (閉包)
可以捕獲環境的匿名函數。
```zc
let factor = 2;
let double = x -> x * factor;  // 箭頭語法
let full = fn(x: int) -> int { return x * factor; }; // 塊語法
```

#### 原始函數指針
Zen C 使用 `fn*` 語法支持原始 C 函數指針。這允許與期望函數指針且沒有閉包開銷的 C 庫進行無縫互操作。

```zc
// 接受原始函數指針的函數
fn set_callback(cb: fn*(int)) {
    cb(42);
}

// 返回原始函數指針的函數
fn get_callback() -> fn*(int) {
    return my_handler;
}

// 支持指向函數指針的指針 (fn**)
let pptr: fn**(int) = &ptr;
```

#### 變參函數
函數可以使用 `...` 和 `va_list` 類型接受可變數量的參數。
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

// 三元運算符
let y = x > 10 ? 1 : 0;
```

#### 模式匹配
`switch` 的強大替代方案。
```zc
match val {
    1         => { print "One" },
    2 || 3    => { print "Two or Three" },    // 使用 || 進行 或 操作
    4 or 5    => { print "Four or Five" },    // 使用 'or' 進行 或 操作
    6, 7, 8   => { print "Six to Eight" },    // 使用逗號進行 或 操作
    10 .. 15  => { print "10 to 14" },        // 左閉右開區間 (舊語法)
    10 ..< 15 => { print "10 to 14" },        // 左閉右開區間 (顯式)
    20 ..= 25 => { print "20 to 25" },        // 全閉區間
    _         => { print "Other" },
}

// 解構枚舉
match shape {
    Shape::Circle(r)   => println "半徑: {r}",
    Shape::Rect(w, h)  => println "面積: {w*h}",
    Shape::Point       => println "點"
}
```

#### 引用綁定
為了在不獲取所有權（移動）的情況下檢查一個值，在模式中使用 `ref` 關鍵字。這對於實現了移動語義的類型（如 `Option`, `Result`, 非 Copy 結構體）至關重要。

```zc
let opt = Some(NonCopyVal{...});
match opt {
    Some(ref x) => {
        // 'x' 是指向 'opt' 內部值的指針
        // 'opt' 在此處不會被移動/消耗
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

// 直接迭代固定大小數組
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

#### 高級控制
```zc
// Guard: 如果條件為假，則執行 else 塊並返回
guard ptr != NULL else { return; }

// Unless: 除非為真（即如果為假）
unless is_valid { return; }
```

### 6. 運算符

Zen C 通過實現特定的方法名來支持用戶定義結構體的運算符重載。

#### 可重載運算符

| 類別 | 運算符 | 方法名 |
|:---|:---|:---|
| **算術** | `+`, `-`, `*`, `/`, `%` | `add`, `sub`, `mul`, `div`, `rem` |
| **比較** | `==`, `!=` | `eq`, `neq` |
| | `<`, `>`, `<=`, `>=` | `lt`, `gt`, `le`, `ge` |
| **位運算** | `&`, `|`, `^` | `bitand`, `bitor`, `bitxor` |
| | `<<`, `>>` | `shl`, `shr` |
| **一元** | `-` | `neg` |
| | `!` | `not` |
| | `~` | `bitnot` |
| **索引** | `a[i]` | `get(a, i)` |
| | `a[i] = v` | `set(a, i, v)` |

> **關於字符串相等性的說明**：
> - `string == string` 進行 **值比較**（等同於 `strcmp`）。
> - `char* == char*` 進行 **指針比較**（檢查內存地址）。
> - 混合比較（例如 `string == char*`）默認為 **指針比較**。

**示例：**
```zc
impl Point {
    fn add(self, other: Point) -> Point {
        return Point{x: self.x + other.x, y: self.y + other.y};
    }
}

let p3 = p1 + p2; // 調用 p1.add(p2)
```

#### 語法糖

這些運算符是內置語言特性，不能直接重載。

| 運算符 | 名稱 | 描述 |
|:---|:---|:---|
| `|>` | 管道 | `x |> f(y)` 脫糖為 `f(x, y)` |
| `??` | 空合併 | 如果 `val` 為 NULL，`val ?? default` 返回 `default` (用於指針) |
| `??=` | 空賦值 | 如果 `val` 為 NULL 則賦值 |
| `?.` | 安全導航 | 僅當 `ptr` 不為 NULL 時訪問字段 |
| `?` | Try 運算符 | 如果存在錯誤則返回 (用於 Result/Option 類型) |

**自動解引用**：
指針字段訪問 (`ptr.field`) 和方法調用 (`ptr.method()`) 會自動解引用指針，等同於 `(*ptr).field`。

### 7. 打印與字符串插值

Zen C 提供了多種控制台打印選項，包括關鍵字和簡潔的簡寫形式。

#### 關鍵字

- `print "text"`: 打印到 `stdout`，不帶尾隨換行符。
- `println "text"`: 打印到 `stdout`，帶尾隨換行符。
- `eprint "text"`: 打印到 `stderr`，不帶尾隨換行符。
- `eprintln "text"`: 打印到 `stderr`，帶尾隨換行符。

#### 簡寫形式

Zen C 允許直接將字符串字面量用作語句來進行快速打印：

- `"Hello World"`: 等同於 `println "Hello World"`。(隱式添加換行符)
- `"Hello World"..`: 等同於 `print "Hello World"`。(不帶尾隨換行符)
- `!"Error"`: 等同於 `eprintln "Error"`。(輸出到 stderr)
- `!"Error"..`: 等同於 `eprint "Error"`。(輸出到 stderr，不帶換行符)

#### 字符串插值 (F-strings)

你可以使用 `{}` 語法將表達式直接嵌入到字符串字面量中。這適用於所有打印方法和字符串簡寫。

```zc
let x = 42;
let name = "Zen";
println "值: {x}, 名稱: {name}";
"值: {x}, 名稱: {name}"; // 簡寫形式的 println
```

#### 輸入提示 (`?`)

Zen C 支持使用 `?` 前綴進行用戶輸入提示的簡寫。

- `? "提示文本"`: 打印提示信息（不換行）並等待輸入（讀取一行）。
- `? "輸入年齡: " (age)`: 打印提示並掃描輸入到變量 `age` 中。
    - 格式說明符會根據變量類型自動推斷。

```zc
let age: int;
? "你多大了？ " (age);
println "你 {age} 歲了。";
```

### 8. 內存管理

Zen C 允許帶有符合人體工程學輔助的手動內存管理。

#### Defer
在當前作用域退出時執行代碼。Defer 語句按照後進先出 (LIFO) 的順序執行。
```zc
let f = fopen("file.txt", "r");
defer fclose(f);
```

> 為了防止未定義行為，`defer` 塊內不允許使用控制流語句（`return`, `break`, `continue`, `goto`）。

#### Autofree
在作用域退出時自動釋放變量。
```zc
autofree let types = malloc(1024);
```

#### 資源語義 (默認移動)
Zen C 將帶有析構函數（如 `File`, `Vec`, 或 malloc 的指針）的類型視為 **資源**。為了防止雙重釋放錯誤，資源不能被隱式複製。

- **默認移動**：分配資源變量會轉移所有權。原始變量變得無效（已移動）。
- **複製類型**：沒有析構函數的類型可以申請參與 `Copy` 行為，使賦值變成複製。

**診斷與哲學**：
如果你看到錯誤 "Use of moved value"，編譯器是在告訴你：*"此類型擁有一個資源（如內存或句柄），盲目複製它是不安全的。"*

> **對比：** 與 C/C++ 不同，Zen C 不會隱式複製擁有資源的值。

**函數參數**：
將值傳遞給函數遵循與賦值相同的規則：資源會被移動，除非通過引用傳遞。

```zc
fn process(r: Resource) { ... } // 'r' 被移動進函數
fn peek(r: Resource*) { ... }   // 'r' 被借用 (引用)
```

**顯式克隆**：
如果你 *確實* 想要一個資源的兩個副本，請顯式執行：

```zc
let b = a.clone(); // 調用 Clone trait 中的 'clone' 方法
```

**選擇性複製 (值類型)**：
對於沒有析構函數的小型類型：

```zc
struct Point { x: int; y: int; }
impl Copy for Point {} // 選擇參與隱式複製

fn main() {
    let p1 = Point { x: 1, y: 2 };
    let p2 = p1; // 已複製。p1 保持有效。
}
```

#### RAII / Drop Trait
實現 `Drop` 以自動運行清理邏輯。
```zc
impl Drop for MyStruct {
    fn drop(self) {
        self.free();
    }
}
```

### 9. 面向對象編程

#### 方法
使用 `impl` 為類型定義方法。
```zc
impl Point {
    // 靜態方法 (構造函數慣例)
    fn new(x: int, y: int) -> Self {
        return Point{x: x, y: y};
    }

    // 實例方法
    fn dist(self) -> float {
        return sqrt(self.x * self.x + self.y * self.y);
    }
}
```

#### Trait
定義共享行為。
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

#### 標準 Trait
Zen C 包含與語言語法集成的標準 Trait。

**Iterable**

實現 `Iterable<T>` 以便為你的自定義類型啟用 `for-in` 循環。

```zc
import "std/iter.zc"

// 定義一個迭代器
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

// 實現 Iterable
impl MyRange {
    fn iterator(self) -> MyIter {
        return MyIter{curr: self.start, stop: self.end};
    }
}

// 在循環中使用
for i in my_range {
    println "{i}";
}
```

**Drop**

實現 `Drop` 來定義一個在對象超出範圍時運行的析構函數 (RAII)。

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

> **注意：** 如果一個變量被移動，則原始變量不會調用 `drop`。它遵循 [資源語義](#資源語義-默認移動)。

**Copy**

標記 Trait，用於選擇支持 `Copy` 行為（隱式複製）而不是移動語義。通過 `@derive(Copy)` 使用。

> **規則：** 實現了 `Copy` 的類型不得定義析構函數 (`Drop`)。

```zc
@derive(Copy)
struct Point { x: int; y: int; }

fn main() {
    let p1 = Point{x: 1, y: 2};
    let p2 = p1; // 已複製！p1 保持有效。
}
```

**Clone**

實現 `Clone` 以允許顯式複製擁有資源的類型。

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
    let b2 = b1.clone(); // 顯式複製
}
```

#### 組合
使用 `use` 嵌入其他結構體。你可以將它們混合進來（展平字段）或者為它們命名（嵌套字段）。

```zc
struct Entity { id: int; }

struct Player {
    // 混入 (未命名): 展平字段
    use Entity;  // 直接將 'id' 添加到 Player
    name: string;
}

struct Match {
    // 組合 (命名): 嵌套字段
    use p1: Player; // 通過 match.p1 訪問
    use p2: Player; // 通過 match.p2 訪問
}
```

### 10. 泛型

結構體和函數的類型安全模板。

```zc
// 泛型結構體
struct Box<T> {
    item: T;
}

// 泛型函數
fn identity<T>(val: T) -> T {
    return val;
}

// 多參數泛型
struct Pair<K, V> {
    key: K;
    value: V;
}
```

### 11. 並發 (Async/Await)

基於 pthreads 構建。

```zc
async fn fetch_data() -> string {
    // 在後台運行
    return "Data";
}

fn main() {
    let future = fetch_data();
    let result = await future;
}
```

### 12. 元編程

#### Comptime
在編譯時運行代碼以生成源碼或打印消息。
```zc
comptime {
    // 在編譯時生成代碼 (寫入 stdout)
    println "let build_date = \"2024-01-01\";";
}

println "構建日期: {build_date}";
```

#### Embed
將文件嵌入為指定類型。
```zc
// 默認 (Slice_char)
let data = embed "assets/logo.png";

// 類型化嵌入
let text = embed "shader.glsl" as string;    // 嵌入為 C-string
let rom  = embed "bios.bin" as u8[1024];     // 嵌入為固定數組
let wav  = embed "sound.wav" as u8[];        // 嵌入為 Slice_u8
```

#### 插件
導入編譯器插件以擴展語法。
```zc
import plugin "regex"
let re = regex! { ^[a-z]+$ };
```

#### 泛型 C 宏
將預處理器宏傳遞給 C。

> **提示**：對於簡單的常量，請使用 `def`。當你需要 C 預處理器宏或條件編譯標誌時，請使用 `#define`。

```zc
#define MAX_BUFFER 1024
```

### 13. 屬性

修飾函數和結構體以修改編譯器行為。

| 屬性 | 作用域 | 描述 |
|:---|:---|:---|
| `@must_use` | 函數 | 如果忽略返回值則發出警告。 |
| `@deprecated("msg")` | 函數/結構體 | 使用時發出帶有消息的警告。 |
| `@inline` | 函數 | 提示編譯器進行內聯。 |
| `@noinline` | 函數 | 防止內聯。 |
| `@packed` | 結構體 | 移除字段間的填充。 |
| `@align(N)` | 結構體 | 強制按 N 字节對齊。 |
| `@constructor` | 函數 | 在 main 之前運行。 |
| `@destructor` | 函數 | 在 main 退出後運行。 |
| `@unused` | 函數/變量 | 抑制未使用變量警告。 |
| `@weak` | 函數 | 弱符號鏈接。 |
| `@section("name")` | 函數 | 將代碼放置在特定段中。 |
| `@noreturn` | 函數 | 函數不會返回 (例如 exit)。 |
| `@pure` | 函數 | 函數無副作用 (優化提示)。 |
| `@cold` | 函數 | 函數不太可能被執行 (分支預測提示)。 |
| `@hot` | 函數 | 函數頻繁執行 (優化提示)。 |
| `@export` | 函數/結構體 | 導出符號 (默認可見性)。 |
| `@global` | 函數 | CUDA: 內核入口點 (`__global__`)。 |
| `@device` | 函數 | CUDA: 設備函數 (`__device__`)。 |
| `@host` | 函數 | CUDA: 主機函數 (`__host__`)。 |
| `@comptime` | 函數 | 用於編譯時執行的輔助函數。 |
| `@derive(...)` | 結構體 | 自動實現 Trait。支持 `Debug`, `Eq` (智能派生), `Copy`, `Clone`。 |
| `@ctype("type")` | 函數參數 | 覆蓋參數生成的 C 類型。 |
| `@<custom>` | 任意 | 將泛型屬性傳遞給 C (例如 `@flatten`, `@alias("name")`)。 |

#### 自定義屬性

Zen C 支持強大的 **自定義屬性** 系統，允許你在代碼中直接使用任何 GCC/Clang 的 `__attribute__`。任何不被 Zen C 編譯器顯式識別的屬性都會被視為泛型屬性並傳遞給生成的 C 代碼。

這提供了對高級編譯器特性、優化和鏈接器指令的訪問，而無需在語言核心中提供顯式支持。

#### 語法映射
Zen C 屬性直接映射到 C 屬性：
- `@name` → `__attribute__((name))`
- `@name(args)` → `__attribute__((name(args)))`
- `@name("string")` → `__attribute__((name("string")))`

#### 智能派生

Zen C 提供了尊重移動語義的 "智能派生"：

- **`@derive(Eq)`**：生成一個通過引用獲取參數的相等性方法 (`fn eq(self, other: T*)`)。
    - 當比較兩個非 Copy 結構體 (`a == b`) 時，編譯器會自動通過引用傳遞 `b` (`&b`) 以避免移動它。
    - 字段上的遞歸相等性檢查也會優先使用指針訪問，以防止所有權轉移。

### 14. 內聯匯編

Zen C 為內聯匯編提供了一流支持，直接轉譯為 GCC 風格的擴展 `asm`。

#### 基本用法
在 `asm` 塊內編寫原始匯編。字符串會自動拼接。
```zc
asm {
    "nop"
    "mfence"
}
```

#### Volatile
防止編譯器優化掉具有副作用的匯編代碼。
```zc
asm volatile {
    "rdtsc"
}
```

#### 命名約束
Zen C 通過命名綁定簡化了複雜的 GCC 約束語法。

```zc
// 語法: : out(變量) : in(變量) : clobber(寄存器)
// 使用 {變量} 佔位符語法以提高可讀性

fn add(a: int, b: int) -> int {
    let result: int;
    asm {
        "add {result}, {a}, {b}"
        : out(result)
        : in(a), in(b)
        : clobber("cc")
    }
    return result;
}
```

| 類型 | 語法 | GCC 等效項 |
|:---|:---|:---|
| **輸出** | `: out(variable)` | `"=r"(variable)` |
| **輸入** | `: in(variable)` | `"r"(variable)` |
| **破壞** | `: clobber("rax")` | `"rax"` |
| **內存** | `: clobber("memory")` | `"memory"` |

> **注意：** 使用 Intel 語法時（通過 `-masm=intel`），必須確保你的構建配置正確（例如，`//> cflags: -masm=intel`）。TCC 不支持 Intel 語法的匯編。

### 15. 構建指令

Zen C 支持在源文件頂部使用特殊註釋來配置構建過程，無需複雜的構建系統或 Makefile。

| 指令 | 參數 | 描述 |
|:---|:---|:---|
| `//> link:` | `-lfoo` 或 `path/to/lib.a` | 鏈接庫或對象文件。 |
| `//> lib:` | `path/to/libs` | 添加庫搜索路徑 (`-L`)。 |
| `//> include:` | `path/to/headers` | 添加包含頭文件搜索路徑 (`-I`)。 |
| `//> framework:` | `Cocoa` | 鏈接 macOS Framework。 |
| `//> cflags:` | `-Wall -O3` | 向 C 編譯器傳遞任意標誌。 |
| `//> define:` | `MACRO` 或 `KEY=VAL` | 定義預處理器宏 (`-D`)。 |
| `//> pkg-config:` | `gtk+-3.0` | 運行 `pkg-config` 並追加 `--cflags` 和 `--libs`。 |
| `//> shell:` | `command` | 在構建期間執行 shell 命令。 |
| `//> get:` | `http://url/file` | 如果特定文件不存在，則下載該文件。 |

#### 特性

**1. 操作系統守護 (OS Guarding)**
在指令前加上操作系統名稱，以使其僅在特定平台上應用。
受支持的前綴：`linux:`, `windows:`, `macos:` (或 `darwin:`)。

```zc
//> linux: link: -lm
//> windows: link: -lws2_32
//> macos: framework: Cocoa
```

**2. 環境變量展開**
使用 `${VAR}` 語法在指令中展開環境變量。

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

### 16. 關鍵字

以下關鍵字在 Zen C 中是保留的。

#### 聲明
`alias`, `def`, `enum`, `fn`, `impl`, `import`, `let`, `module`, `opaque`, `struct`, `trait`, `union`, `use`

#### 控制流
`async`, `await`, `break`, `catch`, `continue`, `defer`, `else`, `for`, `goto`, `guard`, `if`, `loop`, `match`, `return`, `try`, `unless`, `while`

#### 特殊
`asm`, `assert`, `autofree`, `comptime`, `const`, `embed`, `launch`, `ref`, `sizeof`, `static`, `test`, `volatile`

#### 常量
`true`, `false`, `null`

#### C 保留字
以下標識符是保留的，因為它們是 C11 中的關鍵字：
`auto`, `case`, `char`, `default`, `do`, `double`, `extern`, `float`, `inline`, `int`, `long`, `register`, `restrict`, `short`, `signed`, `switch`, `typedef`, `unsigned`, `void`, `_Atomic`, `_Bool`, `_Complex`, `_Generic`, `_Imaginary`, `_Noreturn`, `_Static_assert`, `_Thread_local`

#### 運算符
`and`, `or`

---

## 標準庫

Zen C 包含一個涵蓋基本功能的標準庫 (`std`)。

[瀏覽標準庫文檔](docs/std/README.md)

### 核心模塊

| 模塊 | 描述 | 文檔 |
| :--- | :--- | :--- |
| **`std/vec.zc`** | 可增長動態數組 `Vec<T>`。 | [文檔](docs/std/vec.md) |
| **`std/string.zc`** | 堆分配的 `String` 類型，支持 UTF-8。 | [文檔](docs/std/string.md) |
| **`std/queue.zc`** | 先進先出隊列 (環形緩衝區)。 | [文檔](docs/std/queue.md) |
| **`std/map.zc`** | 泛型哈希表 `Map<V>`。 | [文檔](docs/std/map.md) |
| **`std/fs.zc`** | 文件系統操作。 | [文檔](docs/std/fs.md) |
| **`std/io.zc`** | 標準輸入/輸出 (`print`/`println`)。 | [文檔](docs/std/io.md) |
| **`std/option.zc`** | 可選值 (`Some`/`None`)。 | [文檔](docs/std/option.md) |
| **`std/result.zc`** | 錯誤處理 (`Ok`/`Err`)。 | [文檔](docs/std/result.md) |
| **`std/path.zc`** | 跨平台路徑操作。 | [文檔](docs/std/path.md) |
| **`std/env.zc`** | 進程環境變量。 | [文檔](docs/std/env.md) |
| **`std/net.zc`** | TCP 網絡 (套接字)。 | [文檔](docs/std/net.md) |
| **`std/thread.zc`** | 線程與同步。 | [文檔](docs/std/thread.md) |
| **`std/time.zc`** | 時間測量與睡眠。 | [文檔](docs/std/time.md) |
| **`std/json.zc`** | JSON 解析與序列化。 | [文檔](docs/std/json.md) |
| **`std/stack.zc`** | 後進先出棧 `Stack<T>`。 | [文檔](docs/std/stack.md) |
| **`std/set.zc`** | 泛型哈希集合 `Set<T>`。 | [文檔](docs/std/set.md) |
| **`std/process.zc`** | 進程執行與管理。 | [文檔](docs/std/process.md) |

---

## 工具鏈

Zen C 提供內置的語言服務器 (LSP) 和 REPL 以增強開發體驗。

### 語言服務器 (LSP)

Zen C 語言服務器 (LSP) 支持標準的 LSP 特性，用於編輯器集成：

*   **轉到定義**
*   **查找引用**
*   **懸停信息**
*   **補全** (函數/結構體名，方法/字段的點補全)
*   **文檔符號** (大綱)
*   **簽名幫助**
*   **診斷** (語法/語義錯誤)

啟動語言服務器（通常在編輯器的 LSP 設置中配置）：

```bash
zc lsp
```

它通過標準 I/O (JSON-RPC 2.0) 進行通信。

### REPL

Read-Eval-Print Loop 允許你交互式地嘗試 Zen C 代碼。

```bash
zc repl
```

#### 特性

*   **交互式編碼**：輸入表達式或語句以立即求值。
*   **持久歷史**：命令保存在 `~/.zprep_history` 中。
*   **啟動腳本**：自動加載 `~/.zprep_init.zc` 中的命令。

#### 命令

| 命令 | 描述 |
|:---|:---|
| `:help` | 顯示可用命令。 |
| `:reset` | 清除當前會話歷史 (變量/函數)。 |
| `:vars` | 顯示活躍變量。 |
| `:funcs` | 顯示用戶定義的函數。 |
| `:structs` | 顯示用戶定義的結構體。 |
| `:imports` | 顯示活躍導入。 |
| `:history` | 顯示會話輸入歷史。 |
| `:type <expr>` | 顯示表達式的類型。 |
| `:c <stmt>` | 顯示語句生成的 C 代碼。 |
| `:time <expr>` | 基准測試表達式 (運行 1000 次迭代)。 |
| `:edit [n]` | 在 `$EDITOR` 中編輯命令 `n` (默認：最後一條)。 |
| `:save <file>` | 將當前會話保存到 `.zc` 文件。 |
| `:load <file>` | 將 `.zc` 文件加載並執行到會話中。 |
| `:watch <expr>` | 監視表達式 (每次輸入後重新求值)。 |
| `:unwatch <n>` | 移除監視。 |
| `:undo` | 從會話中移除最後一條命令。 |
| `:delete <n>` | 移除索引為 `n` 的命令。 |
| `:clear` | 清屏。 |
| `:quit` | 退出 REPL。 |
| `! <cmd>` | 運行 shell 命令 (如 `!ls`)。 |

---


## 編譯器支持與兼容性

Zen C 旨在與大多數 C11 編譯器配合使用。某些特性依賴於 GNU C 擴展，但這些擴展通常在其他編譯器中也能工作。使用 `--cc` 標誌切換後端。

```bash
zc run app.zc --cc clang
zc run app.zc --cc zig
```

### 測試套件狀態

| 編譯器 | 通過率 | 受支持特性 | 已知局限性 |
|:---|:---:|:---|:---|
| **GCC** | **100%** | 所有特性 | 無。 |
| **Clang** | **100%** | 所有特性 | 無。 |
| **Zig** | **100%** | 所有特性 | 無。使用 `zig cc` 作為替代 C 編譯器。 |
| **TCC** | **~70%** | 基本語法, 泛型, Trait | 不支持 `__auto_type`, 不支持 Intel ASM, 不支持嵌套函數。 |

> **建議：** 生產環境構建請使用 **GCC**, **Clang**, 或 **Zig**。TCC 非常適合快速原型開發，因為它編譯速度極快，但缺少 Zen C 全面支持所需的一些高級 C 擴展。

### 使用 Zig 構建

Zig 的 `zig cc` 命令提供了 GCC/Clang 的替代方案，具有出色的跨平台編譯支持。使用 Zig：

```bash
# 使用 Zig 編譯並運行 Zen C 程序
zc run app.zc --cc zig

# 使用 Zig 構建 Zen C 編譯器本身
make zig
```

### C++ 互操作

Zen C 可以通過 `--cpp` 標誌生成 C++ 兼容的代碼，從而實現與 C++ 庫的無縫集成。

```bash
# 直接使用 g++ 編譯
zc app.zc --cpp

# 或者轉譯用於手動構建
zc transpile app.zc --cpp
g++ out.c my_cpp_lib.o -o app
```

#### 在 Zen C 中使用 C++

包含 C++ 頭文件並在 `raw` 塊中使用 C++ 代碼：

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

> **注意：** `--cpp` 標誌會將後端切換為 `g++` 並發出 C++ 兼容的代碼（使用 `auto` 代替 `__auto_type`，使用函數重載代替 `_Generic`，以及對 `void*` 進行顯式轉換）。

#### CUDA 互操作

Zen C 通過轉譯為 **CUDA C++** 來支持 GPU 編程。這使你在維持 Zen C 人體工程學語法的同時，能夠利用內核中的強大 C++ 特性（模板、constexpr）。

```bash
# 直接使用 nvcc 編譯
zc run app.zc --cuda

# 或者轉譯用於手動構建
zc transpile app.zc --cuda -o app.cu
nvcc app.cu -o app
```

#### CUDA 特定屬性

| 屬性 | CUDA 等效項 | 描述 |
|:---|:---|:---|
| `@global` | `__global__` | 內核函數 (運行在 GPU，從主機調用) |
| `@device` | `__device__` | 設備函數 (運行在 GPU，從 GPU 調用) |
| `@host` | `__host__` | 主機函數 (明確僅 CPU 運行) |

#### 內核啟動語法

Zen C 提供了一個簡潔的 `launch` 語句用於調用 CUDA 內核：

```zc
launch kernel_name(args) with {
    grid: num_blocks,
    block: threads_per_block,
    shared_mem: 1024,  // 可選
    stream: my_stream   // 可選
};
```

這轉譯為：`kernel_name<<<grid, block, shared, stream>>>(args);`

#### 編寫 CUDA 內核

使用帶有 `@global` 的 Zen C 函數語法和 `launch` 語句：

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

    // ... 初始化數據 ...
    
    launch add_kernel(d_a, d_b, d_c, N) with {
        grid: (N + 255) / 256,
        block: 256
    };
    
    cuda_sync();
}
```

#### 標準庫 (`std/cuda.zc`)
Zen C 為常見的 CUDA 操作提供了一個標準庫，以減少 `raw` 塊的使用：

```zc
import "std/cuda.zc"

// 內存管理
let d_ptr = cuda_alloc<float>(1024);
cuda_copy_to_device(d_ptr, h_ptr, 1024 * sizeof(float));
defer cuda_free(d_ptr);

// 同步
cuda_sync();

// 線程索引 (在內核內部使用)
let i = thread_id(); // 全局索引
let bid = block_id();
let tid = local_id();
```


> **注意：** `--cuda` 標誌設置 `nvcc` 為編譯器並隱含 `--cpp` 模式。需要安裝 NVIDIA CUDA Toolkit。

### C23 支援

當使用相容的後端編譯器（GCC 14+, Clang 14+）時，Zen C 支援現代 C23 特性。

-   **`auto`**: 如果 `__STDC_VERSION__ >= 202300L`，Zen C 會自動將型別推導映射到標準 C23 `auto`。
-   **`_BitInt(N)`**: 使用 `iN` 和 `uN` 型別（例如 `i256`, `u12`, `i24`）存取 C23 任意位元寬度整數。

### Objective-C 互操作

Zen C 可以通過 `--objc` 標誌編譯為 Objective-C (`.m`)，允許你使用 Objective-C 框架（如 Cocoa/Foundation）和語法。

```bash
# 使用 clang (或 gcc/gnustep) 編譯
zc app.zc --objc --cc clang
```

#### 在 Zen C 中使用 Objective-C

使用 `include` 包含頭文件，並在 `raw` 塊中使用 Objective-C 語法 (`@interface`, `[...]`, `@""`)。

```zc
//> macos: framework: Foundation
//> linux: cflags: -fconstant-string-class=NSConstantString -D_NATIVE_OBJC_EXCEPTIONS
//> linux: link: -lgnustep-base -lobjc

include <Foundation/Foundation.h>

fn main() {
    raw {
        NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
        NSLog(@"來自 Objective-C 的問候！");
        [pool drain];
    }
    println "Zen C 也能正常工作！";
}
```

> **注意：** Zen C 字符串插值通過調用 `debugDescription` 或 `description` 同樣適用於 Objective-C 對象 (`id`)。

---

## 貢獻

我們歡迎各類貢獻！無論是修復 Bug、完善文檔，還是提出新功能建議。

### 如何貢獻
1.  **Fork 倉庫**：標準的 GitHub 工作流程。
2.  **創建功能分支**：`git checkout -b feature/NewThing`。
3.  **代碼規範**：
    *   遵循現有的 C 風格。
    *   確保所有測試通過：`make test`。
    *   在 `tests/` 中為你的功能添加新測試。
4.  **提交拉取請求**：清晰地描述你的更改。

### 運行測試
測試套件是你最好的朋友。

```bash
# 運行所有測試 (GCC)
make test

# 運行特定的測試
./zc run tests/test_match.zc

# 使用不同的編譯器運行
./tests/run_tests.sh --cc clang
./tests/run_tests.sh --cc zig
./tests/run_tests.sh --cc tcc
```

### 擴展編譯器
*   **解析器 (Parser)**：`src/parser/` - 遞歸下降解析器。
*   **代碼生成 (Codegen)**：`src/codegen/` - 轉譯邏輯 (Zen C -> GNU C/C11)。
*   **標準庫 (Standard Library)**：`std/` - 使用 Zen C 本身編寫。

---

## 致謝與歸属

本項目使用了第三方庫。完整許可證文本可在 `LICENSES/` 目錄中找到。

*   **[cJSON](https://github.com/DaveGamble/cJSON)** (MIT 許可證)：用於語言服務器中的 JSON 解析和生成。
*   **[zc-ape](https://github.com/OEvgeny/zc-ape)** (MIT 許可證)：由 [Eugene Olonov](https://github.com/OEvgeny) 開發的原版 Zen-C 實際上便攜的可執行文件 (APE) 端口。
*   **[Cosmopolitan Libc](https://github.com/jart/cosmopolitan)** (ISC 許可證)：使 APE 成為可能納基礎庫。
