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
  <h3>Ergonomia moderna. Zero overhead. C puro.</h3>
  <br>
  <p>
    <a href="#"><img src="https://img.shields.io/badge/build-passing-brightgreen" alt="Status do Build"></a>
    <a href="#"><img src="https://img.shields.io/badge/license-MIT-blue" alt="Licença"></a>
    <a href="#"><img src="https://img.shields.io/github/v/release/z-libs/Zen-C?label=versão&color=orange" alt="Versão"></a>
    <a href="#"><img src="https://img.shields.io/badge/platform-linux-lightgrey" alt="Plataforma"></a>
  </p>
  <p><em>Programe como linguagem de alto nível, execute como C.</em></p>
</div>

<hr>

<div align="center">
  <p>
    <b><a href="#visão-geral">Visão Geral</a></b> •
    <b><a href="#comunidade">Comunidade</a></b> •
    <b><a href="#início-rápido">Início Rápido</a></b> •
    <b><a href="#referência-da-linguagem">Referência da Linguagem</a></b> •
    <b><a href="#biblioteca-padrão">Biblioteca Padrão</a></b> •
    <b><a href="#ferramentas">Ferramentas</a></b>
  </p>
</div>

---

## Visão Geral

**Zen C** é uma moderna linguagem de programação de sistemas que compila para `GNU C`/`C11` legível para humanos. Oferece um rico conjunto de funcionalidades, incluindo inferência de tipos, correspondência de padrões, genéricos, traits, async/await, e gerenciamento manual de memória com capacidades RAII - tudo enquanto mantém 100% de compatibilidade ABI com C.

## Comunidade

Participe das discussões, compartilhe demos, pergunte, ou reporte bugs no servidor oficial de Discord da Zen C!

- Discord: [Clique aqui](https://discord.com/invite/q6wEsCmkJP)

---

## Índice

<table>
  <tr>
    <td width="50%" valign="top">
      <h3>Geral</h3>
      <ul>
        <li><a href="#visão-geral">Visão Geral</a></li>
        <li><a href="#comunidade">Comunidade</a></li>
        <li><a href="#início-rápido">Início Rápido</a></li>
        <li><a href="#biblioteca-padrão">Biblioteca Padrão</a></li>
        <li><a href="#ferramentas">Ferramentas</a></li>
        <li><a href="#suporte-de-compiladores--compatibilidade">Suporte de Compiladores</a></li>
        <li><a href="#contribuindo">Contribuindo</a></li>
        <li><a href="#atribuições">Atribuições</a></li>
      </ul>
    </td>
    <td width="50%" valign="top">
      <h3>Referência da Linguagem</h3>
      <ul>
        <li><a href="#1-variáveis-e-constantes">1. Variáveis e Constantes</a></li>
        <li><a href="#2-tipos-primitivos">2. Tipos Primitivos</a></li>
        <li><a href="#3-tipos-agregados">3. Tipos Agregados</a></li>
        <li><a href="#4-funções--lambdas">4. Funções & Lambdas</a></li>
        <li><a href="#5-fluxo-de-controle">5. Fluxo de Controle</a></li>
        <li><a href="#6-operadores">6. Operadores</a></li>
        <li><a href="#7-print-e-interpolação-de-strings">7. Print e Interpolação</a></li>
        <li><a href="#8-gerenciamento-de-memória">8. Gerenciamento de Memória</a></li>
        <li><a href="#9-programação-orientada-a-objetos">9. POO</a></li>
        <li><a href="#10-genéricos">10. Genéricos</a></li>
        <li><a href="#11-concorrência-asyncawait">11. Concorrência</a></li>
        <li><a href="#12-metaprogramação">12. Metaprogramação</a></li>
        <li><a href="#13-atributos">13. Atributos</a></li>
        <li><a href="#14-assembly-inline">14. Assembly Inline</a></li>
        <li><a href="#15-diretivas-de-build">15. Diretivas de Build</a></li>
        <li><a href="#16-palavras-chave">16. Palavras-chave</a></li>
        <li><a href="#17-interoperabilidade-c">17. Interoperabilidade C</a></li>
      </ul>
    </td>
  </tr>
</table>

---

## Início Rápido

### Instalação

```bash
git clone https://github.com/z-libs/Zen-C.git
cd Zen-C
make
sudo make install
```

### Build Portátil (APE)

Zen C pode ser compilado como um **Actually Portable Executable (APE)** usando [Cosmopolitan Libc](https://github.com/jart/cosmopolitan). Isso gera um único binário (`.com`) que executa nativamente em Linux, macOS, Windows, FreeBSD, OpenBSD, e NetBSD, tanto em arquiteturas x86_64 quanto aarch64.

**Pré-requisitos:**
- Toolchain `cosmocc` (precisa estar no seu PATH)

**Build & Instalação:**
```bash
make ape
sudo env "PATH=$PATH" make install-ape
```

**Artifacts:**
- `out/bin/zc.com`: O compilador portátil de Zen-C. Inclui a biblioteca padrão embutida no executável.
- `out/bin/zc-boot.com`: Um instalador bootstrap independente para configurar novos projetos Zen-C.

**Uso:**
```bash
# Executa em qualquer SO suportado
./out/bin/zc.com build hello.zc -o hello
```

### Uso

```bash
# Compila e executa
zc run hello.zc

# Build do executável
zc build hello.zc -o hello

# Shell interativo
zc repl
```

### Variáveis de Ambiente

Você pode configurar `ZC_ROOT` para especificar a localização da Biblioteca Padrão (imports padrões como `import "std/vec.zc"`). Isso permite que você execute `zc` de qualquer diretório.

```bash
export ZC_ROOT=/path/to/Zen-C
```

---

## Referência da Linguagem

### 1. Variáveis e Constantes

Zen C distingue constantes em tempo de compilação e variáveis em tempo de execução.

#### Constantes Manifest (`def`)

Valores que existem apenas em tempo de compilação (encapsuladas no código). Utilize para tamanhos de arrays, configurações fixas e números mágicos.

```zc
def MAX_SIZE = 1024;
let buffer: char[MAX_SIZE]; // Tamanho de array válido
```

#### Variáveis (`let`, `const`)

Armazenam localizações na memória. Podem ser mutáveis (`let`) ou apenas leitura (`const`).

```zc
let x = 10;             // Mutável
x = 20;                 // OK

let y: const int = 10;  // Apenas leitura (tipo qualificado)
// y = 20;              // Erro: atribuição a const não permitida
```

> [!TIP]
> **Inferência de Tipo**: Zen C automaticamente infere tipos de variáveis inicializadas. Compila para C23 `auto` em compiladores suportados, ou para `__auto_type` da extensão GCC alternativamente.

### 2. Tipos Primitivos

| Tipo | Equivalente em C | Descrição |
|:---|:---|:---|
| `int`, `uint` | `int`, `unsigned int` | Inteiro padrão da plataforma |
| `I8` .. `I128` ou `i8` .. `i128` | `int8_t` .. `__int128_t` | Inteiros sinalizados de tamanho fixo |
| `U8` .. `U128` ou `u8` .. `u128` | `uint8_t` .. `__uint128_t` | Inteiros não-sinalizados de tamanho fixo |
| `isize`, `usize` | `ptrdiff_t`, `size_t` | Inteiros do tamanho de ponteiro |
| `byte` | `uint8_t` | Alias para U8 |
| `F32`, `F64` ou `f32`, `f64`  | `float`, `double` | Números de ponto flutuante |
| `bool` | `bool` | `true` (verdadeiro) ou `false` (falso) |
| `char` | `char` | Caractere único |
| `string` | `char*` | C-string (terminada em NULL) |
| `U0`, `u0`, `void` | `void` | Tipo vazio |
| `iN` (por exemplo, `i256`) | `_BitInt(N)` | Inteiro sinalizado de largura de bit arbitrária (C23) |
| `uN` (por exemplo, `u42`) | `unsigned _BitInt(N)` | Inteiro não-sinalizado de largura de bit arbitrária (C23) |

### 3. Tipos Agregados

#### Arrays
Arrays de tamanho fixo com semântica de valores.
```zc
def SIZE = 5;
let ints: int[SIZE] = [1, 2, 3, 4, 5];
let zeros: [int; SIZE]; // Inicialização-Zero
```

#### Tuplas
Agrupa múltiplos valores juntos, acessa elementos por índice.
```zc
let pair = (1, "Hello");
let x = pair.0;  // 1
let s = pair.1;  // "Hello"
```

**Retorno Múltiplo de Valores**

Funções podem retornar tuplas para fornecer múltiplos resultados:
```zc
fn add_and_subtract(a: int, b: int) -> (int, int) {
    return (a + b, a - b);
}

let result = add_and_subtract(3, 2);
let sum = result.0;   // 5
let diff = result.1;  // 1
```

**Desestruturação**

Tuplas podem ser desestruturadas diretamente em variáveis:
```zc
let (sum, diff) = add_and_subtract(3, 2);
// sum = 5, diff = 1
```

Desestruturação tipada permite anotações de tipo explícitas:
```zc
let (a: string, b: u8) = ("hello", 42);
let (x, y: i32) = (1, 2);  // Misto: x inferido, y explícito
```

#### Structs

Estruturas de dados com campos de bit opcionais.
```zc
struct Point {
    x: int;
    y: int;
}

// Inicialização de Struct
let p = Point { x: 10, y: 20 };

// Bitfields
struct Flags {
    valid: U8 : 1;
    mode:  U8 : 3;
}
```

> [!NOTE]
> Structs usam [Semântica de Move](#semântica-de-recursos-move-por-padrão) por padrão. Campos podem ser acessados via `.` mesmo em ponteiros (Auto-Dereferência).

#### Structs Opacos

Você pode definir um struct como `opaque` para restringir acesso aos seus campos exclusivamente ao módulo que o define, enquanto ainda permite que o struct seja alocado na pilha com tamanho conhecido.
```zc
// Em user.zc
opaque struct User {
    id: int;
    name: string;
}

fn new_user(name: string) -> User {
    return User{id: 1, name: name}; // OK: Dentro do módulo
}

// Em main.zc
import "user.zc";

fn main() {
    let u = new_user("Alice");
    // let id = u.id; // Erro: Acesso ao campo privado 'id' não permitido
}
```

#### Enums

Unions etiquetadas (tipos Sum) capazes de armazenar dados.
```zc
enum Shape {
    Circle(float),      // Armazena o raio
    Rect(float, float), // Armazena largura e altura
    Point               // Sem dados
}
```

#### Unions

Unions C padrão (acesso inseguro).
```zc
union Data {
    i: int;
    f: float;
}
```

#### Aliases de Tipos
Cria um novo nome para um tipo existente.
```zc
alias ID = int;
alias PointMap = Map<string, Point>;
```

#### Aliases Opacos de Tipos

Você pode definir um alias como `opaque` para criar um novo tipo que é distinto de seu tipo subjacente fora do módulo que o define. Isso fornece forte encapsulamento e segurança de tipos sem o overhead de runtime de um struct wrapper.

```zc
// Em library.zc
opaque alias Handle = int;

fn make_handle(v: int) -> Handle {
    return v; // Conversão implícita permitida dentro do módulo
}

// Em main.zc
import "library.zc";

fn main() {
    let h: Handle = make_handle(42);
    // let i: int = h; // Erro: validação de tipo falhou
    // let h2: Handle = 10; // Erro: validação de tipo falhou
}
```

### 4. Funções & Lambdas

#### Funções
```zc
fn add(a: int, b: int) -> int {
    return a + b;
}

// Argumentos nomeados são suportados na chamada
add(a: 10, b: 20);
```

> [!NOTE]
> Argumentos nomeados devem seguir estritamente a ordem dos parâmetros. `add(b: 20, a: 10)` é inválido.

#### Argumentos Const

Funções podem ser marcadas como `const` para impor semântica de apenas leitura. Isso é um qualificador de tipo, não uma constante manifest.

```zc
fn print_val(v: const int) {
    // v = 10; // Erro: atribuição a const não permitida
    println "{v}";
}
```

#### Argumentos Padrão

Funções podem definir valores-padrão para argumentos trailing. Estes podem ser literais, expressões ou código Zen C válido (como construtores struct).
```zc
// Valor padrão simples
fn increment(val: int, amount: int = 1) -> int {
    return val + amount;
}

// Expressão de valor padrão (avaliada no local da chamada)
fn offset(val: int, pad: int = 10 * 2) -> int {
    return val + pad;
}

// Valor padrão de struct
struct Config { debug: bool; }
fn init(cfg: Config = Config { debug: true }) {
    if cfg.debug { println "Debug Mode"; }
}

fn main() {
    increment(10);      // 11
    offset(5);          // 25
    init();             // Print "Debug Mode"
}
```

#### Lambdas (Fechamentos)

Funções anônimas que podem capturar seu ambiente.
```zc
let factor = 2;
let double = x -> x * factor;  // Sintaxe de seta
let full = fn(x: int) -> int { return x * factor; }; // Sintaxe de bloco
```

#### Ponteiros de Função Brutos

Zen C suporta ponteiros de função C brutos usando sintaxe `fn*`. Isso permite interoperabilidade ininterrupta com bibliotecas C que esperam ponteiros de função, sem overhead de fechamento.

```zc
// Função recebendo um ponteiro de função bruto 
fn set_callback(cb: fn*(int)) {
    cb(42);
}

// Função retornando um ponteiro de função bruto
fn get_callback() -> fn*(int) {
    return my_handler;
}

// Ponteiros para ponteiros de função são suportados (fn**)
let pptr: fn**(int) = &ptr;
```

#### Funções Variádicas

Funções podem aceitar um número variável de argumentos utilizando `...` e o tipo `va_list`.
```zc
fn log(lvl: int, fmt: char*, ...) {
    let ap: va_list;
    va_start(ap, fmt);
    vprintf(fmt, ap); // Use C stdio
    va_end(ap);
}
```

### 5. Fluxo de Controle

#### Condicionais
```zc
if x > 10 {
    print("Large");
} else if x > 5 {
    print("Medium");
} else {
    print("Small");
}

// Condicional Ternária
let y = x > 10 ? 1 : 0;

// If-Expression (para condições complexas)
let categoria = if (x > 100) { "enorme" } else if (x > 10) { "grande" } else { "pequeno" };
```

#### Correspondência de Padrões

Poderosa alternativa ao `switch`.
```zc
match val {
    1         => { print "One" },
    2 || 3    => { print "Two or Three" },    // OR com ||
    4 or 5    => { print "Four or Five" },    // OR com 'or'
    6, 7, 8   => { print "Six to Eight" },    // OR com vírgula
    10 .. 15  => { print "10 to 14" },        // Range excludente (Legado)
    10 ..< 15 => { print "10 to 14" },        // Range excludente (Explícito)
    20 ..= 25 => { print "20 to 25" },        // Range inclusivo
    _         => { print "Other" },
}

// Desestruturando Enums
match shape {
    Shape::Circle(r)   => println "Radius: {r}",
    Shape::Rect(w, h)  => println "Area: {w*h}",
    Shape::Point       => println "Point"
}
```

#### Associação de Referência**

Para inspecionar um valor sem tomar posse (movendo-o), use a palavra-chave `ref` no padrão. Isso é essencial para tipos que implementam semântica de movimento, como `Option`, `Result` e structs non-Copy.

```zc
let opt = Some(NonCopyVal{...});
match opt {
    Some(ref x) => {
        // 'x' é um ponteiro para o valor dentro de 'opt'
        // 'opt' NÃO É movido/consumido aqui
        println "{x.field}"; 
    },
    None => {}
}
```

#### Loops
```zc
// Range
for i in 0..10 { ... }      // Excludente (0 a 9)
for i in 0..<10 { ... }     // Excludente (Explícito)
for i in 0..=10 { ... }     // Inclusivo (0 a 10)
for i in 0..10 step 2 { ... }

// Iterador (Vec ou Iterável Customizado)
for item in vec { ... }

// Itera sobre arrays de tamanho fixo diretamente
let arr: int[5] = [1, 2, 3, 4, 5];
for val in arr {
    // val é um int
    println "{val}";
}

// While
while x < 10 { ... }

// Infinito com label
outer: loop {
    if done { break outer; }
}

// Repita N vezes
for _ in 0..5 { ... }
```

#### Controle Avançado
```zc
// Guard: Executa else e return se a condição for falsa
guard ptr != NULL else { return; }

// Unless: Se não for verdadeiro
unless is_valid { return; }
```

### 6. Operadores

Zen C suporta sobrecarga de operadores para structs definidos pelo usuário através da implementação de nomes específicos de métodos.

#### Operadores Sobrecarregáveis

| Categoria | Operador | Nome do Método |
|:---|:---|:---|
| **Aritmético** | `+`, `-`, `*`, `/`, `%` | `add`, `sub`, `mul`, `div`, `rem` |
| **Comparação** | `==`, `!=` | `eq`, `neq` |
| | `<`, `>`, `<=`, `>=` | `lt`, `gt`, `le`, `ge` |
| **Bitwise** | `&`, `\|`, `^` | `bitand`, `bitor`, `bitxor` |
| | `<<`, `>>` | `shl`, `shr` |
| **Unário** | `-` | `neg` |
| | `!` | `not` |
| | `~` | `bitnot` |
| **Índice** | `a[i]` | `get(a, i)` |
| | `a[i] = v` | `set(a, i, v)` |

> **Nota sobre a Igualdade de Strings**:
> - `string == string` performa uma **comparação de valores** (equivalente a `strcmp`).
> - `char* == char*` performa **comparação de ponteiros** (checa os endereços da memória).
> - Comparações mistas (e.g. `string == char*`) são **comparação de ponteiros** por padrão.

**Exemplo:**
```zc
impl Point {
    fn add(self, other: Point) -> Point {
        return Point{x: self.x + other.x, y: self.y + other.y};
    }
}

let p3 = p1 + p2; // Chama p1.add(p2)
```

#### Açúcar Sintático

Estes operadores são funcionalidades embutidas na linguagem e não podem ser sobrecarregados diretamente.

| Operador | Nome | Descrição |
|:---|:---|:---|
| `\|>` | Pipeline | `x \|> f(y)` des-açucara para `f(x, y)` |
| `??` | Coalescência Null | `val ?? default` retorna `default` se `val` for NULL (ponteiros) |
| `??=` | Atribuição Null | `val ??= init` atribui se `val` for NULL |
| `?.` | Navegação Segura | `ptr?.field` acessa o campo apenas se `ptr` não for NULL |
| `?` | Operador Try | `res?` retorna erro se presente (tipos Result/Option) |

**Auto-Deferência**:
Acesso a campos de ponteiro (`ptr.field`) e chamadas de métodos (`ptr.method()`) automaticamente dereferencia o ponteiro, equivalente a `(*ptr).field`.

### 7. Print e Interpolação de Strings

Zen C fornece opções versáteis para print no console, incluindo palavras-chave e abreviações concisas.

#### Palavras-Chave

| Palavra-chave | Descrição |
|:---|:---|
| `print` | Imprime na saída padrão (sem newline) |
| `println` | Imprime na saída padrão (com newline) |
| `eprint` | Imprime no erro padrão (sem newline) |
| `eprintln` | Imprime no erro padrão (com newline) |

#### Abreviações

Zen C permite utilizar literais string diretamente como declarações para print rápido:

| Sintaxe | Equivalente | Descrição |
|:---|:---|:---|
| `"Hello"` | `println "Hello"` | Implicitamente adiciona newline. |
| `"Hello"..` | `print "Hello"` | Sem trailing newline. |
| `!"Error"` | `eprintln "Error"` | Output para stderr. |
| `!"Error"..` | `eprint "Error"` | Output para stderr, sem newline. |

#### Interpolação de Strings (F-strings)

Você pode incorporar expressões diretamente em literais de string usando sintaxe `{}`. Isso funciona com todos os métodos de print e abreviações de string.

```zc
let x = 42;
let name = "Zen";
println "Value: {x}, Name: {name}";
"Value: {x}, Name: {name}"; // abreviação de println
```

**Escapando Chaves**: Use `{{` para produzir uma chave literal `{` e `}}` para uma `}` literal:

```zc
let json = "JSON: {{\"chave\": \"valor\"}}";
// Saída: JSON: {"chave": "valor"}
```

#### Prompts de Input (`?`)

Zen C suporta uma abreviação para solicitar entrada do usuário usando o prefixo `?`.

- `? "Prompt text"`: Imprime o prompt (sem newline) e aguarda entrada (lê uma linha).
- `? "Enter age: " (age)`: Imprime prompt e escaneia entrada para a variável `age`.
    - Especificadores de formato são automaticamente inferidos com base no tipo da variável.

```zc
let age: int;
? "How old are you? " (age);
println "You are {age} years old.";
```

### 8. Gerenciamento de Memória

Zen C permite gerenciamento manual de memória com auxílios ergonômicos.

#### Defer
Executa código quando sair do escopo atual. Declarações defer são executadas em ordem LIFO (last-in, first-out - último a entrar, primeiro a sair).
```zc
let f = fopen("file.txt", "r");
defer fclose(f);
```

> Para prevenir comportamento indefinido, declarações de fluxo de controle (`return`, `break`, `continue`, `goto`) **não são permitidas** dentro de um bloco `defer`.

#### Autofree
Libera automaticamente a variável quando sair do escopo.
```zc
autofree let types = malloc(1024);
```

#### Semântica de Recursos (Move by Default)
Zen C trata tipos com destrutores (como `File`, `Vec`, ou ponteiros mallocados) como **Recursos**. Para prevenir erros de double-free, recursos não podem ser implicitamente duplicados.

- **Move by Default**: Atribuir uma variável de recurso transfere propriedade. A variável original se torna inválida (Movida).
- **Copy Types**: Tipos sem destrutores podem optar pelo comportamento `Copy`, fazendo da atribuição uma duplicação.

**Diagnósticos & Filosofia**:
Se você vir um erro "Use of moved value", o compilador está dizendo: *"Este tipo possui um recurso (como memória ou um handle) e copiá-lo cegamente é inseguro."*

> **Contraste:** Diferente de C/C++, Zen C não duplica implicitamente valores que possuem recursos.

**Argumentos de Função**:
Passar um valor para uma função segue as mesmas regras que atribuição: recursos são movidos a menos que passados por referência.

```zc
fn process(r: Resource) { ... } // 'r' é movido para a função
fn peek(r: Resource*) { ... }   // 'r' é emprestado (referência)
```

**Clonagem Explícita**:
Se você *realmente* quiser duas cópias de um recurso, torne isso explícito:

```zc
let b = a.clone(); // Chama o método 'clone' do trait Clone
```

**Opt-in Copy (Tipos de Valor)**:
Para tipos pequenos sem destrutores:

```zc
struct Point { x: int; y: int; }
impl Copy for Point {} // Opt-in para duplicação implícita

fn main() {
    let p1 = Point { x: 1, y: 2 };
    let p2 = p1; // Copiado. p1 permanece válido.
}
```

#### RAII / Drop Trait
Implementa `Drop` para executar lógica de limpeza automaticamente.
```zc
impl Drop for MyStruct {
    fn drop(self) {
        self.free();
    }
}
```

### 9. Programação Orientada a Objetos

#### Métodos
Defina métodos em tipos usando `impl`.
```zc
impl Point {
    // Método estático (convenção de construtor)
    fn new(x: int, y: int) -> Self {
        return Point{x: x, y: y};
    }

    // Método de instância
    fn dist(self) -> float {
        return sqrt(self.x * self.x + self.y * self.y);
    }
}
```

**Atalho de Self**: Em métodos com um parâmetro `self`, você pode usar `.campo` como atalho para `self.campo`:
```zc
impl Point {
    fn dist(self) -> float {
        return sqrt(.x * .x + .y * .y);  // Equivalente a self.x, self.y
    }
}
```

#### Métodos Primitivos
Zen C permite definir métodos em tipos primitivos (como `int`, `bool`, etc.) usando a mesma sintaxe `impl`.

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
Defina comportamento compartilhado.
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

#### Traits Padrões
Zen C inclui traits padrões que se integram com a sintaxe da linguagem.

**Iterable**

Implemente `Iterable<T>` para habilitar loops `for-in` para seus tipos customizados.

```zc
import "std/iter.zc"

// Defina um Iterator
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

// Implemente Iterable
impl MyRange {
    fn iterator(self) -> MyIter {
        return MyIter{curr: self.start, stop: self.end};
    }
}

// Use no Loop
for i in my_range {
    println "{i}";
}
```

**Drop**

Implemente `Drop` para definir um destrutor que executa quando o objeto sai de escopo (RAII).

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

> **Nota:** Se uma variável é movida, `drop` NÃO É chamado na variável original. Ela se adere à [Semântica de Recursos](#semântica-de-recursos-move-por-padrão).

**Copy**

Trait marcador para optar pelo comportamento `Copy` (duplicação implícita) em vez de semântica Move. Usado via `@derive(Copy)`.

> **Regra:** Tipos que implementam `Copy` não devem definir um destrutor (`Drop`).

```zc
@derive(Copy)
struct Point { x: int; y: int; }

fn main() {
    let p1 = Point{x: 1, y: 2};
    let p2 = p1; // Copiado! p1 permanece válido.
}
```

**Clone**

Implemente `Clone` para permitir duplicação explícita de tipos que possuem recursos.

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
    let b2 = b1.clone(); // Cópia explícita
}
```

#### Composição
Use `use` para incorporar outros structs. Você pode misturá-los (achatar campos) ou nomeá-los (aninhar campos).

```zc
struct Entity { id: int; }

struct Player {
    // Mixin (Sem nome): Achata campos
    use Entity;  // Adiciona 'id' ao Player diretamente
    name: string;
}

struct Match {
    // Composição (Nomeado): Aninha campos
    use p1: Player; // Acessado via match.p1
    use p2: Player; // Acessado via match.p2
}
```

### 10. Genéricos

Templates type-safe para Structs e Funções.

```zc
// Struct Genérico
struct Box<T> {
    item: T;
}

// Função Genérica
fn identity<T>(val: T) -> T {
    return val;
}

// Genéricos Multi-parâmetro
struct Pair<K, V> {
    key: K;
    value: V;
}
```

### 11. Concorrência (Async/Await)

Construído sobre pthreads.

```zc
async fn fetch_data() -> string {
    // Executa em background
    return "Data";
}

fn main() {
    let future = fetch_data();
    let result = await future;
}
```

### 12. Metaprogramação

#### Comptime
Execute código em tempo de compilação para gerar código ou imprimir mensagens.
```zc
comptime {
    // Gera código em tempo de compilação (escrito em stdout)
    println "let data_compilacao = \"2024-01-01\";";
}

println "Data de compilação: {data_compilacao}";
```

**Funções Auxiliares**

Funções especiais disponíveis dentro de blocos `comptime`:
- **`yield(str)`** - Emite explicitamente código gerado (alternativa a `printf`)
- **`compile_error(msg)`** - Interrompe a compilação com uma mensagem de erro fatal
- **`compile_warn(msg)`** - Emite um aviso em tempo de compilação (permite continuar a compilação)

```zc
comptime {
    compile_warn("Gerando código otimizado...");
    
    let ENABLE_FEATURE = 1;
    if (ENABLE_FEATURE == 0) {
        compile_error("O recurso deve estar habilitado!");
    }
    
    println "let FEATURE_ENABLED = 1;";
}
```

**Metadados de Build**

Acesse informações de build do compilador em tempo de compilação:
- **`__COMPTIME_TARGET__`** - String da plataforma: `"linux"`, `"windows"` ou `"macos"`
- **`__COMPTIME_FILE__`** - Nome do arquivo fonte atual sendo compilado

```zc
comptime {
    // Geração de código específica da plataforma
    println "let PLATFORM = \"{__COMPTIME_TARGET__}\";";
}

println "Executando em: {PLATFORM}";
```

> **Nota**: Use `{{` e `}}` para escapar chaves dentro de strings comptime.

#### Embed
Incorpore arquivos como tipos especificados.
```zc
// Padrão (Slice_char)
let data = embed "assets/logo.png";

// Embed Tipado
let text = embed "shader.glsl" as string;    // Incorporado como C-string
let rom  = embed "bios.bin" as u8[1024];     // Incorporado como array fixa
let wav  = embed "sound.wav" as u8[];        // Incorporado como Slice_u8
```

#### Plugins
Importe plugins de compilação para sintaxe estendida
```zc
import plugin "regex"
let re = regex! { ^[a-z]+$ };
```

#### Macros Genéricos de C
Passe macros pré-processamento para o C.

> **Dica**: Para constantes simples, utilize `def` em vez disso. Utilize `#define` quando você precisar de macros pré-processamento em C ou de flags condicionais de compilação.

```zc
#define MAX_BUFFER 1024
```

### 13. Atributos
Decore funções e structs para modificar o comportamento do compilador.

| Atributo | Escopo | Descrição |
|:---|:---|:---|
| `@must_use` | Fn | Avisa se o valor de retorno é ignorado. |
| `@deprecated("msg")` | Fn/Struct | Aviso sobre o uso com mensagem. |
| `@inline` | Fn | Sugere compilador para inline. |
| `@noinline` | Fn | Previne inline. |
| `@packed` | Struct | Remove padding entre campos. |
| `@align(N)` | Struct | Força alinhamento para N bytes. |
| `@constructor` | Fn | Executa antes de main. |
| `@destructor` | Fn | Executa após sair de main. |
| `@unused` | Fn/Var | Suprime avisos de variáveis não utilizadas. |
| `@weak` | Fn | Linkagem fraca de símbolos. |
| `@section("name")` | Fn | Coloca código em seção específica. |
| `@noreturn` | Fn | Função não retorna (e.g. exit). |
| `@pure` | Fn | Função sem efeitos colaterais (sugestão de otimização). |
| `@cold` | Fn | Função provavelmente não é executada (sugestão de predição de branch). |
| `@hot` | Fn | Função é frequentemente executada (sugestão de otimização). |
| `@export` | Fn/Struct | Exporta símbolo (visibilidade padrão). |
| `@global` | Fn | CUDA: Entry point do Kernel (`__global__`). |
| `@device` | Fn | CUDA: Função de dispositivo (`__device__`). |
| `@host` | Fn | CUDA: Função Host (`__host__`). |
| `@comptime` | Fn | Função auxiliar disponível para execução em tempo de compilação. |
| `@derive(...)` | Struct | Auto-implementa traits. Supporta `Debug`, `Eq` (Smart Derive), `Copy`, `Clone`. |
| `@ctype("type")` | Fn Param | Sobreescreve tipo C gerado para um parâmetro. |
| `@<custom>` | Any | Passa atributos genéricos para o C (e.g. `@flatten`, `@alias("name")`). |

### Atributos Customizados

Zen C suporta um sistema poderoso de **Atributos Customizados** que te permite usar qualquer `__attribute__` de GCC/Clang diretamente no seu código. Qualquer atributo que não seja explicitamente reconhecido pelo compilador de Zen C é tratado como um atributo genérico e passado para o código C gerado.

Isso fornece acesso para funcionalidades avançadas do compilador, otimizações, e diretivas de linker sem precisar de suporte explícito para a linguagem núcleo.

#### Mapeamento de Sintaxe
Os atributos Zen C são mapeados diretamente para atributos C:
- `@name` → `__attribute__((name))`
- `@name(args)` → `__attribute__((name(args)))`
- `@name("string")` → `__attribute__((name("string")))`

### Smart Derives

Zen C fornece "Smart Derives" que respeitam a Semântica de Move (Move Semantics):

- **`@derive(Eq)`**: Gera um método de igualdade que recebe argumentos por referência (`fn eq(self, other: T*)`).
    - Ao comparar dois structs non-Copy (`a == b`), o compilador automaticamente passa `b` por referência (`&b`) para evitar que ele seja movido.
    - Checagens recursivas de igualdade nos campos também preferem acesso ao ponteiro para prevenir transferência de propriedade.
    
### 14. Assembly Inline

Zen C fornece suporte de primeira classe para assembly inline, transpilando diretamente para `asm`de estilo GCC estendido.

#### Uso Básico
Escreva assembly bruto dentro de blocos `asm`. Strings são concatenadas automaticamente.
```zc
asm {
    "nop"
    "mfence"
}
```

#### Volatile
Use `volatile` para prevenir otimizações do compilador em assembly com efeitos colaterais.
```zc
asm volatile {
    "rdtsc"
}
```


#### Restrições Nomeadas
Zen C simplifica a complexa restrição sintática do GCC com associações nomeadas.
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

| Tipo | Sintaxe | Equivalente GCC |
|:---|:---|:---|
| **Output** | `: out(variable)` | `"=r"(variable)` |
| **Input** | `: in(variable)` | `"r"(variable)` |
| **Clobber** | `: clobber("rax")` | `"rax"` |
| **Memory** | `: clobber("memory")` | `"memory"` |

> **Nota:** Ao utilizar sintaxe Intel (via `-masm=intel`), você deve garantir que seu build esteja configurado corretamente (por exemplo, `//> cflags: -masm=intel`). O TCC não suporta sintaxe assembly da Intel.

### 15. Diretivas de Build

Zen C suporta comentários especiais no topo de seu arquivo-fonte para configurar o processo de build sem precisar de um sistema de build complexo ou um Makefile.

| Diretiva | Argumentos | Descrição |
|:---|:---|:---|
| `//> link:` | `-lfoo` ou `path/to/lib.a` | Link uma biblioteca ou arquivo objeto. |
| `//> lib:` | `path/to/libs` | Adiciona um caminho de busca por bibliotecas (`-L`). |
| `//> include:` | `path/to/headers` | Adiciona um caminho de inclusão de buscas (`-I`). |
| `//> framework:` | `Cocoa` | Link para um framework macOS. |
| `//> cflags:` | `-Wall -O3` | Passa flags arbitrárias para o compilador C. |
| `//> define:` | `MACRO` ou `KEY=VAL` | Define um macro preprocessador (`-D`). |
| `//> pkg-config:` | `gtk+-3.0` | Executa `pkg-config` e acrescenta `--cflags` e `--libs`. |
| `//> shell:` | `command` | Executa um comando shell durante o build. |
| `//> get:` | `http://url/file` | Baixa um arquivo se o arquivo específico não existir. |

#### Features

**1. OS Guarding**
Diretivas de prefixo com o nome de um SO para aplicá-las apenas em plataformas específicas.
Prefixos suportados: `linux:`, `windows:`, `macos:` (or `darwin:`).

```zc
//> linux: link: -lm
//> windows: link: -lws2_32
//> macos: framework: Cocoa
```

**2. Expansão de Variáveis de Ambiente**
Use a sintaxe `${VAR}` para expandir as variáveis de ambiente em suas diretivas.

```zc
//> include: ${HOME}/mylib/include
//> lib: ${ZC_ROOT}/std
```

#### Exemplos

```zc
//> include: ./include
//> lib: ./libs
//> link: -lraylib -lm
//> cflags: -Ofast
//> pkg-config: gtk+-3.0

import "raylib.h"

fn main() { ... }
```

### 16. Palavras-chave

Zen C reserva as seguintes palavras-chave:

#### Declarações
`alias`, `def`, `enum`, `fn`, `impl`, `import`, `let`, `module`, `opaque`, `struct`, `trait`, `union`, `use`

#### Fluxo de Control
`async`, `await`, `break`, `catch`, `continue`, `defer`, `else`, `for`, `goto`, `guard`, `if`, `loop`, `match`, `return`, `try`, `unless`, `while`

#### Especiais
`asm`, `assert`, `autofree`, `comptime`, `const`, `embed`, `launch`, `ref`, `sizeof`, `static`, `test`, `volatile`

#### Constantes
`true`, `false`, `null`

#### Reservado pelo C
Os seguintes identificadores são reservados porque são palavras-chave em C11:
`auto`, `case`, `char`, `default`, `do`, `double`, `extern`, `float`, `inline`, `int`, `long`, `register`, `restrict`, `short`, `signed`, `switch`, `typedef`, `unsigned`, `void`, `_Atomic`, `_Bool`, `_Complex`, `_Generic`, `_Imaginary`, `_Noreturn`, `_Static_assert`, `_Thread_local`

#### Operadores
`and`, `or`

### 17. Interoperabilidade com C

Zen C oferece duas formas  de interagir com código C: **Trusted Imports** (Forma conveniente) e **Explicit FFI** (Forma segura/precisa).

#### Método 1: Trusted Imports (Conveniente)

Você pode importar um header C diretamente utilizando a palavra-chave `import` com a extensão `.h`. Isso trata o header como um módulo e assume que todos os símbolos acessados através dele existem.

```zc
//> link: -lm
import "math.h" as c_math;

fn main() {
    // Compilador confia na precisão; emite 'cos(...)' diretamente
    let x = c_math::cos(3.14159);
}
```

> **Prós**: Zero boilerplate. Acessa tudo no header imediatamente.
> **Contras**: Sem segurança de tipos por parte do Zen C (erros capturados apenas posteriormente, pelo compilador C).

#### Método 2: Explicit FFI (Segura)

Para checagem estrita de tipos ou quando você não quer incluir o texto de um header, utilize `extern fn`.

```zc
include <stdio.h> // Emite  #include <stdio.h> no C gerado

// Define assinatura estrita
extern fn printf(fmt: char*, ...) -> c_int;

fn main() {
    printf("Hello FFI: %d\n", 42); // Type checked by Zen C
}
```

> **Prós**: Zen C garante correspondência de tipos.
> **Contras**: Requer declaração manual de funções.

#### `import` vs `include`

- **`import "file.h"`**: Registra o header como um módulo nomeado. Habilita acesso implícito aos símbolos (por exemplo, `file::function()`).
- **`include <file.h>`**: Emite puramente `#include <file.h>` no código C gerado. Não introduz nenhum símbolo ao compilador Zen C; você deve usar `extern fn` para acessá-los.


---

## Biblioteca Padrão

O Zen C inclui a biblioteca padrão (`std`), que cobre as funcionalidades essenciais.

[Navegue pela Documentação da Biblioteca Padrão](../docs/std/README.md)

### Módulos Principais

<details>
<summary>Clique para ver todos os módulos da Biblioteca Padrão</summary>

| Módulo | Descrição | Docs |
| :--- | :--- | :--- |
| **`std/vec.zc`** | Growable dynamic array `Vec<T>`. | [Docs](../docs/std/vec.md) |
| **`std/string.zc`** | Heap-allocated `String` type with UTF-8 support. | [Docs](../docs/std/string.md) |
| **`std/queue.zc`** | FIFO queue (Ring Buffer). | [Docs](../docs/std/queue.md) |
| **`std/map.zc`** | Generic Hash Map `Map<V>`. | [Docs](../docs/std/map.md) |
| **`std/fs.zc`** | File system operations. | [Docs](../docs/std/fs.md) |
| **`std/io.zc`** | Standard Input/Output (`print`/`println`). | [Docs](../docs/std/io.md) |
| **`std/option.zc`** | Optional values (`Some`/`None`). | [Docs](../docs/std/option.md) |
| **`std/result.zc`** | Error handling (`Ok`/`Err`). | [Docs](../docs/std/result.md) |
| **`std/path.zc`** | Cross-platform path manipulation. | [Docs](../docs/std/path.md) |
| **`std/env.zc`** | Process environment variables. | [Docs](../docs/std/env.md) |
| **`std/net/`** | TCP, UDP, HTTP, DNS, URL. | [Docs](../docs/std/net.md) |
| **`std/thread.zc`** | Threads and Synchronization. | [Docs](../docs/std/thread.md) |
| **`std/time.zc`** | Time measurement and sleep. | [Docs](../docs/std/time.md) |
| **`std/json.zc`** | JSON parsing and serialization. | [Docs](../docs/std/json.md) |
| **`std/stack.zc`** | LIFO Stack `Stack<T>`. | [Docs](../docs/std/stack.md) |
| **`std/set.zc`** | Generic Hash Set `Set<T>`. | [Docs](../docs/std/set.md) |
| **`std/process.zc`** | Process execution and management. | [Docs](../docs/std/process.md) |

</details>

---

## Ferramentas

Zen C inclui um Language Server embutido (`zc lsp`) e um REPL para aprimorar a experiência do desenvolvimento.

### Language Server (LSP)

O Zen C Language Server (LSP) suporta funcionalidades padrão de LSP para integração com editores, fornecendo:

* **Go to Definition** - Vá para definição
* **Find References** - Encontrar referências
* **Hover Information** - Informação com sobreposição do ponteiro do mouse
* **Completion** - Auto-completar (Nomes de Função/Struct, compleção de ponto para métodos/campos)
* **Document Symbols** - Símbolos de documento (Outline)
* **Signature Help** - Ajuda de assinatura
* **Diagnostics** - Diagnóstico (Sintaxe/Erros semânticos)

Para inicializar o servidor da linguagem (tipicamente configurado nas configurações LSP do seu editor):

```bash
zc lsp
```

Ele se comunica via I/O padrão (JSON-RPC 2.0).

### REPL

O Read-Eval-Print Loop permite que você experimente seu código Zen C interativamente.

#### Funcionalidades:

* **Código Interativo**: Escreva expressões ou declarações para avaliação imediata.
* **Histórico Persistente**: Comandos são salvos em `~/.zprep_history`.
* **Script de Inicialização**: Automaticamente carrega comandos de `~/.zprep_init.zc`.

#### Comandos

| Comando | Descrição |
|:---|:---|
| `:help` | Mostra comandos disponíveis. |
| `:reset` | Reinicia a sessão atual (limpa todas as definições). |
| `:vars` | Mostra variáveis ativas. |
| `:funcs` | Mostra funções definidas pelo usuário. |
| `:structs` | Mostra structs definidos pelo usuário. |
| `:imports` | Mostra imports ativos. |
| `:history` | Mostra histórico de entrada da sessão. |
| `:type <expr>` | Mostra o tipo de uma expressão. |
| `:c <stmt>` | Mostra o código C gerado para uma declaração. |
| `:time <expr>` | Faz benchmark de uma expressão (executa 1000 iterações). |
| `:edit [n]` | Edita comando `n` (padrão: último) em `$EDITOR`. |
| `:save <file>` | Salva a sessão atual em um arquivo `.zc`. |
| `:load <file>` | Carrega e executa um arquivo `.zc` na sessão. |
| `:watch <expr>` | Observa uma expressão (reavaliada após cada entrada). |
| `:unwatch <n>` | Remove um watch. |
| `:undo` | Remove o último comando da sessão. |
| `:delete <n>` | Remove comando no índice `n`. |
| `:clear` | Limpa a tela. |
| `:quit` | Sai do REPL. |
| `! <cmd>` | Executa um comando shell (e.g. `!ls`). |

---

## Suporte de Compiladores & Compatibilidade

Zen C foi projetado para funcionar com a maioria dos compiladores C11. Algumas funcionalidades dependem de extensões GNU C, mas estas frequentemente funcionam em outros compiladores. Use a flag `--cc` para trocar backends.

```bash
zc run app.zc --cc clang
zc run app.zc --cc zig
```

### Status da Suíte de Testes

<details>
<summary>Clique para ver detalhes do Suporte de Compilador</summary>

| Compilador | Taxa de Aprovação | Funcionalidades Suportadas | Limitações Conhecidas |
|:---|:---:|:---|:---|
| **GCC** | **100% (Completo)** | Todas as Funcionalidades | Nenhuma. |
| **Clang** | **100% (Completo)** | Todas as Funcionalidades | Nenhuma. |
| **Zig** | **100% (Completo)** | Todas as Funcionalidades | Nenhuma. Usa `zig cc` como compilador C drop-in. |
| **TCC** | **~70% (Básico)** | Sintaxe Básica, Genéricos, Traits | Sem `__auto_type`, Sem ASM Intel, Sem Funções Aninhadas. |

</details>

> [!TIP]
> **Recomendação:** Use **GCC**, **Clang**, ou **Zig** para builds de produção. TCC é excelente para prototipagem rápida devido à sua velocidade de compilação, mas perde algumas extensões C avançadas nas quais Zen C se baseia para suporte completo de funcionalidades.

### Build com Zig

O comando `zig cc` do Zig fornece um substituto drop-in para GCC/Clang com excelente suporte de compilação cruzada. Para usar Zig:

```bash
# Compila e executa um programa Zen C com Zig
zc run app.zc --cc zig

# Faz build do próprio compilador Zen C com Zig
make zig
```

### Interoperabilidade C++

Zen C pode gerar código compatível com C++ com a flag `--cpp`, permitindo integração sem emendas com bibliotecas C++.

```bash
# Compilação direta com g++
zc app.zc --cpp

# Ou transpile para build manual
zc transpile app.zc --cpp
g++ out.c my_cpp_lib.o -o app
```

#### Usando C++ em Zen C

Inclua headers C++ e use blocos raw para código C++:

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

> **Nota:** A flag `--cpp` troca o backend para `g++` e emite código compatível com C++ (usa `auto` em vez de `__auto_type`, sobrecarga de função em vez de `_Generic`, e casts explícitos para `void*`).

### Interoperabilidade CUDA

Zen C suporta programação GPU transpilando para **CUDA C++**. Isso permite que você aproveite poderosas funcionalidades C++ (templates, constexpr) dentro de seus kernels enquanto mantém a sintaxe ergonômica do Zen C.

```bash
# Compilação direta com nvcc
zc run app.zc --cuda

# Ou transpile para build manual
zc transpile app.zc --cuda -o app.cu
nvcc app.cu -o app
```

#### Atributos Específicos do CUDA

| Atributo | Equivalente CUDA | Descrição |
|:---|:---|:---|
| `@global` | `__global__` | Função kernel (executa na GPU, chamada do host) |
| `@device` | `__device__` | Função device (executa na GPU, chamada da GPU) |
| `@host` | `__host__` | Função host (explicitamente apenas CPU) |

#### Sintaxe de Launch de Kernel

Zen C fornece uma instrução `launch` limpa para invocar kernels CUDA:

```zc
launch kernel_name(args) with {
    grid: num_blocks,
    block: threads_per_block,
    shared_mem: 1024,  // Opcional
    stream: my_stream   // Opcional
};
```

Isso transpila para: `kernel_name<<<grid, block, shared, stream>>>(args);`

#### Escrevendo Kernels CUDA

Use sintaxe de função Zen C com `@global` e a instrução `launch`:

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

#### Biblioteca Padrão (`std/cuda.zc`)
Zen C fornece uma biblioteca padrão para operações CUDA comuns para reduzir blocos `raw`:

```zc
import "std/cuda.zc"

// Gerenciamento de memória
let d_ptr = cuda_alloc<float>(1024);
cuda_copy_to_device(d_ptr, h_ptr, 1024 * sizeof(float));
defer cuda_free(d_ptr);

// Sincronização
cuda_sync();

// Indexação de Thread (use dentro de kernels)
let i = thread_id(); // Índice global
let bid = block_id();
let tid = local_id();
```

> [!NOTE]
> **Nota:** A flag `--cuda` define `nvcc` como o compilador e implica modo `--cpp`. Requer o NVIDIA CUDA Toolkit.

### Suporte C23

Zen C suporta funcionalidades modernas de C23 quando usa um compilador backend compatível (GCC 14+, Clang 14+, TCC (parcial)).

- **`auto`**: Zen C automaticamente mapeia inferência de tipo para o `auto` padrão C23 se `__STDC_VERSION__ >= 202300L`.
- **`_BitInt(N)`**: Use tipos `iN` e `uN` (e.g., `i256`, `u12`, `i24`) para acessar inteiros de largura arbitrária do C23.

### Interoperabilidade Objective-C

Zen C pode compilar para Objective-C (`.m`) usando a flag `--objc`, permitindo que você use frameworks Objective-C (como Cocoa/Foundation) e sintaxe.

```bash
# Compila com clang (ou gcc/gnustep)
zc app.zc --objc --cc clang
```

#### Usando Objective-C em Zen C

Use `include` para headers e blocos `raw` para sintaxe Objective-C (`@interface`, `[...]`, `@""`).

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
> **Nota:** Interpolação de strings do Zen C funciona com objetos Objective-C (`id`) chamando `debugDescription` ou `description`.


---

## Contribuindo

Nós damos boas-vindas a contribuições! Seja consertando bugs, adicionando documentação ou propondo novas funcionalidades.

Por favor, veja [CONTRIBUTING_PT_BR.md](CONTRIBUTING_PT_BR.md) para diretrizes detalhadas sobre como contribuir, executar testes e submeter pull requests.

---

## Segurança

Para instruções sobre relatórios de segurança, por favor veja [SECURITY_PT_BR.md](SECURITY_PT_BR.md).

---

## Atribuições

Este projeto usa bibliotecas de terceiros. Textos completos de licença podem ser encontrados no diretório `LICENSES/`.

*   **[cJSON](https://github.com/DaveGamble/cJSON)** (Licença MIT): Usado para parsing e geração JSON no Language Server.
*   **[zc-ape](https://github.com/OEvgeny/zc-ape)** (Licença MIT): O port original Actually Portable Executable do Zen-C por [Eugene Olonov](https://github.com/OEvgeny).
*   **[Cosmopolitan Libc](https://github.com/jart/cosmopolitan)** (Licença ISC): A biblioteca fundadora que torna APE possível.

---

<div align="center">
  <p>
    Copyright © 2026 Linguagem de Programação Zen C.<br>
    Comece sua jornada hoje.
  </p>
  <p>
    <a href="https://discord.com/invite/q6wEsCmkJP">Discord</a> •
    <a href="https://github.com/z-libs/Zen-C">GitHub</a> •
    <a href="CONTRIBUTING_PT_BR.md">Contribuir</a>
  </p>
</div>
