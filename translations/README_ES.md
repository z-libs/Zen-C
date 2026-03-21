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
  <h3>Ergonomía Moderna. Cero Overhead. C Puro.</h3>
  <br>
  <p>
    <a href="#"><img src="https://img.shields.io/badge/build-passing-brightgreen" alt="Estado de la Construcción"></a>
    <a href="#"><img src="https://img.shields.io/badge/license-MIT-blue" alt="Licencia"></a>
    <a href="#"><img src="https://img.shields.io/github/v/release/zenc-lang/zenc?label=version&color=orange" alt="Versión"></a>
    <a href="#"><img src="https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macos-lightgrey" alt="Plataforma"></a>
  </p>
  <p><em>Escribe como un lenguaje de alto nivel, ejecuta como C.</em></p>
</div>

<hr>

<div align="center">
  <p>
    <b><a href="#descripción-general">Descripción General</a></b> •
    <b><a href="#comunidad">Comunidad</a></b> •
    <b><a href="#inicio-rápido">Inicio Rápido</a></b> •
    <b><a href="#ecosistema">Ecosistema</a></b> •
    <b><a href="#referencia-del-lenguaje">Referencia del Lenguaje</a></b> •
    <b><a href="#biblioteca-estándar">Biblioteca Estándar</a></b> •
    <b><a href="#herramientas">Herramientas</a></b>
  </p>
</div>

---

## Descripción General

**Zen C** es un lenguaje de programación de sistemas moderno que se compila a `GNU C`/`C11` legible por humanos. Proporciona un conjunto rico de características que incluyen inferencia de tipos, coincidencia de patrones (pattern matching), genéricos, traits, async/await y gestión manual de memoria con capacidades RAII, todo manteniendo una compatibilidad total con el ABI de C.

## Comunidad

¡Únete a la discusión, comparte demos, haz preguntas o reporta errores en el servidor oficial de Discord de Zen C!

- Discord: [Únete aquí](https://discord.com/invite/q6wEsCmkJP)
- RFCs: [Proponer características](https://github.com/zenc-lang/rfcs)

## Ecosistema

El proyecto Zen C consta de varios repositorios. A continuación se presentan los principales:

| Repositorio | Descripción | Estado |
| :--- | :--- | :--- |
| **[zenc](https://github.com/zenc-lang/zenc)** | El compilador central de Zen C (`zc`), la CLI y la biblioteca estándar. | Desarrollo Activo |
| **[docs](https://github.com/zenc-lang/docs)** | La documentación técnica oficial y la especificación del lenguaje. | Activo |
| **[rfcs](https://github.com/zenc-lang/rfcs)** | El repositorio de Solicitud de Comentarios (RFC). Dale forma al futuro del lenguaje. | Activo |
| **[vscode-zenc](https://github.com/zenc-lang/vscode-zenc)** | Extensión oficial de VS Code (Resaltado de sintaxis, Snippets). | Alpha |
| **[www](https://github.com/zenc-lang/www)** | Código fuente de `zenc-lang.org`. | Activo |
| **[awesome-zenc](https://github.com/zenc-lang/awesome-zenc)** | Una lista curada de ejemplos asombrosos de Zen C. | Creciendo |
| **[zenc.vim](https://github.com/zenc-lang/zenc.vim)** | Plugin oficial para Vim/Neovim (Sintaxis, Sangría). | Activo |

## Proyectos Destacados

Echa un vistazo a estos proyectos construidos con Zen C:

- **[ZC-pong-3ds](https://github.com/5quirre1/ZC-pong-3ds)**: Un clon de Pong para Nintendo 3DS.
- **[zen-c-parin](https://github.com/Kapendev/zen-c-parin)**: Un ejemplo básico usando Zen C con Parin.
- **[almond](https://git.sr.ht/~leanghok/almond)**: Un navegador web minimalista escrito en Zen C.

---

## Índice

<table align="center">
  <tr>
    <th width="50%">General</th>
    <th width="50%">Referencia del Lenguaje</th>
  </tr>
  <tr>
    <td valign="top">
      <ul>
        <li><a href="#descripción-general">Descripción General</a></li>
        <li><a href="#comunidad">Comunidad</a></li>
        <li><a href="#ecosistema">Ecosistema</a></li>
        <li><a href="https://github.com/zenc-lang/rfcs">RFCs</a></li>
        <li><a href="#inicio-rápido">Inicio Rápido</a></li>
        <li><a href="https://github.com/zenc-lang/docs">Documentación</a></li>
        <li><a href="#biblioteca-estándar">Biblioteca Estándar</a></li>
        <li><a href="#herramientas">Herramientas</a>
          <ul>
            <li><a href="#protocolo-de-servidor-de-lenguaje-lsp">LSP</a></li>
            <li><a href="#depuración-de-zen-c">Depuración</a></li>
          </ul>
        </li>
        <li><a href="#soporte-del-compilador-y-compatibilidad">Soporte del Compilador</a></li>
        <li><a href="#contribuyendo">Contribuyendo</a></li>
        <li><a href="#atribuciones">Atribuciones</a></li>
      </ul>
    </td>
    <td valign="top">
      <ul>
        <li><a href="#1-variables-y-constantes">1. Variables y Constantes</a></li>
        <li><a href="#2-tipos-primitivos">2. Tipos Primitivos</a>
          <ul>
            <li><a href="#unicode-y-runas">Unicode y Runas</a></li>
          </ul>
        </li>
        <li><a href="#3-tipos-agregados">3. Tipos Agregados</a></li>
        <li><a href="#4-funciones-y-lambdas">4. Funciones y Lambdas</a></li>
        <li><a href="#5-flujo-de-control">5. Flujo de Control</a></li>
        <li><a href="#6-operadores">6. Operadores</a></li>
        <li><a href="#7-impresión-e-interpolación-de-cadenas">7. Impresión e Interpolación</a></li>
        <li><a href="#8-gestión-de-memoria">8. Gestión de Memoria</a></li>
        <li><a href="#9-programación-orientada-a-objetos">9. POO</a></li>
        <li><a href="#10-genéricos">10. Genéricos</a></li>
        <li><a href="#11-concurrencia-asyncawait">11. Concurrencia</a></li>
        <li><a href="#12-metaprogramación">12. Metaprogramación</a></li>
        <li><a href="#13-atributos">13. Atributos</a></li>
        <li><a href="#14-ensamblador-inline">14. Ensamblador Inline</a></li>
        <li><a href="#15-directivas-de-construcción">15. Directivas de Construcción</a></li>
        <li><a href="#16-palabras-clave">16. Palabras Clave</a></li>
        <li><a href="#17-interoperabilidad-c">17. Interoperabilidad C</a></li>
        <li><a href="#18-marco-de-pruebas-unitarias">18. Marco de Pruebas Unitarias</a></li>
      </ul>
    </td>
  </tr>
</table>

---

## Inicio Rápido

### Instalación

```bash
git clone https://github.com/zenc-lang/zenc.git
cd Zen-C
make clean # eliminar archivos de construcción antiguos
make
sudo make install
```

### Windows

Zen C tiene soporte nativo completo para Windows (x86_64). Puedes construirlo usando el script de procesamiento por lotes (batch) proporcionado con GCC (MinGW):

```cmd
build.bat
```

Esto construirá el compilador (`zc.exe`). Las operaciones de Red, Sistema de Archivos y Procesos están totalmente soportadas a través de la Capa de Abstracción de Plataforma (PAL).

Alternativamente, puedes usar `make` si tienes un entorno tipo Unix (MSYS2, Cygwin, git-bash).

### Construcción Portable (APE)

Zen C puede compilarse como un **Ejecutable Realmente Portable (APE)** usando [Cosmopolitan Libc](https://github.com/jart/cosmopolitan). Esto produce un único binario (`.com`) que se ejecuta de forma nativa en Linux, macOS, Windows, FreeBSD, OpenBSD y NetBSD en arquitecturas x86_64 y aarch64.

**Prerrequisitos:**
- Toolchain `cosmocc` (debe estar en tu PATH)

**Construcción e Instalación:**
```bash
make ape
sudo env "PATH=$PATH" make install-ape
```

**Artefactos:**
- `out/bin/zc.com`: El compilador Zen-C portable. Incluye la biblioteca estándar embebida dentro del ejecutable.
- `out/bin/zc-boot.com`: Un instalador bootstrap autónomo para configurar nuevos proyectos Zen-C.

**Uso:**
```bash
# Ejecutar en cualquier SO compatible
./out/bin/zc.com build hello.zc -o hello
```

### Uso

```bash
# Compilar y ejecutar
zc run hello.zc

# Construir ejecutable
zc build hello.zc -o hello

# Shell Interactiva
zc repl

# Mostrar Zen Facts
zc build hello.zc --zen
```

### Variables de Entorno

Puedes configurar `ZC_ROOT` para especificar la ubicación de la Biblioteca Estándar (importaciones estándar como `import "std/vector.zc"`). Esto te permite ejecutar `zc` desde cualquier directorio.

```bash
export ZC_ROOT=/ruta/a/Zen-C
```

---

## Referencia del Lenguaje

### 1. Variables y Constantes

Zen C distingue entre constantes en tiempo de compilación y variables en tiempo de ejecución.

#### Constantes Manifiestas (`def`)
Valores que existen solo en tiempo de compilación (se pliegan en el código). Úsalos para tamaños de arrays, configuración fija y números mágicos.

```zc
def MAX_SIZE = 1024;
let buffer: char[MAX_SIZE]; // Tamaño de array válido
```

#### Variables (`let`)
Ubicaciones de almacenamiento en memoria. Pueden ser mutables o de solo lectura (`const`).

```zc
let x = 10;             // Mutable
x = 20;                 // OK

let y: const int = 10;  // Solo lectura (Calificado por tipo)
// y = 20;              // Error: no se puede asignar a una constante
```

> [!TIP]
> **Inferencia de tipos**: Zen C infiere automáticamente los tipos para variables inicializadas. Se compila a `auto` de C23 en compiladores compatibles, o a la extensión `__auto_type` de GCC en otros casos.

### 2. Tipos Primitivos

| Tipo | Equivalente en C | Descripción |
|:---|:---|:---|
| `int`, `uint` | `int32_t`, `uint32_t` | Entero de 32 bits con signo/sin signo |
| `c_char`, `c_uchar` | `char`, `unsigned char` | C char (Interoperabilidad) |
| `c_short`, `c_ushort` | `short`, `unsigned short` | C short (Interoperabilidad) |
| `c_int`, `c_uint` | `int`, `unsigned int` | C int (Interoperabilidad) |
| `c_long`, `c_ulong` | `long`, `unsigned long` | C long (Interoperabilidad) |
| `c_long_long`, `c_ulong_long` | `long long`, `unsigned long long` | C long long / unsigned long long (Interoperabilidad) |
| `I8` .. `I128` o `i8` .. `i128` | `int8_t` .. `__int128_t` | Enteros con signo de ancho fijo |
| `U8` .. `U128` o `u8` .. `u128` | `uint8_t` .. `__uint128_t` | Enteros sin signo de ancho fijo |
| `isize`, `usize` | `ptrdiff_t`, `size_t` | Enteros del tamaño de un puntero |
| `byte` | `uint8_t` | Alias para U8 |
| `F32`, `F64` o `f32`, `f64`  | `float`, `double` | Números de coma flotante |
| `bool` | `bool` | `true` o `false` |
| `char` | `char` | Carácter único |
| `string` | `char*` | Cadena de C (terminada en null) |
| `U0`, `u0`, `void` | `void` | Tipo vacío |
| `iN` (ej. `i256`) | `_BitInt(N)` | Entero con signo de ancho arbitrario (C23) |
| `uN` (ej. `u42`) | `unsigned _BitInt(N)` | Entero sin signo de ancho arbitrario (C23) |
| `rune` | `uint32_t` | Valor escalar Unicode (punto de código UTF-32) |

#### Literales
- **Enteros**: Decimal (`123`), Hex (`0xFF`), Octal (`0o755`), Binario (`0b1011`).
  - *Nota*: Los números con ceros a la izquierda se tratan como decimales (`0123` es `123`), a diferencia de C.
  - *Nota*: Los números pueden contener guiones bajos para mejorar la legibilidad (`1_000_000`, `0b_1111_0000`).
- **Flotantes**: Estándar (`3.14`), Científico (`1e-5`, `1.2E3`). Los números de punto flotante también soportan guiones bajos (`3_14.15_92`).

#### Unicode y Runas

Zen C proporciona soporte de primera clase para valores escalares Unicode a través del tipo `rune`. Una `rune` representa un único punto de código Unicode (codificado como un entero sin signo de 32 bits).

| Literal | Descripción |
|:---|:---|
| `'a'` | Carácter ASCII estándar |
| `'🚀'` | Carácter Unicode multi-byte |
| `'\u{2764}'` | Secuencia de escape Unicode (Hex) |

```zc
import "std.zc"

fn main() {
    let c = 'a';
    println "El carácter '{c}' tiene un código de {(int)c} en ASCII/Unicode";

    let codigo = 97;
    println "El código {codigo} corresponde al carácter {(char)codigo}";

    let r: rune = '🚀';
    println "La runa '{r}' tiene un código de {(uint)r} en Unicode";
    
    let r_codigo: uint = 128640;
    println "El código {r_codigo} corresponde a la runa '{(rune)r_codigo}'";

    let r_esc: rune = '\u{2764}';
    println "La runa '{r_esc}' tiene el código {(uint)r_esc} (0x{(uint)r_esc:X})";
}
```

> [!IMPORTANT]
> **Mejores Prácticas para Código Portable**
>
> - Usa **Tipos Portables** (`int`, `uint`, `i64`, `u8`, etc.) para toda la lógica pura de Zen C. `int` garantiza ser 32-bits con signo en todas las arquitecturas.
> - Usa **Tipos de Interoperabilidad C** (`c_int`, `c_char`, `c_long`, ``c_ulong``, ``c_long_long``, ``c_ulong_long``) **sólo** al interactuar con bibliotecas C (FFI). Su tamaño varía según la plataforma y el compilador C.
> - Usa `isize` y `usize` para indexado de arrays y aritmética de punteros.

### 3. Tipos Agregados

#### Arrays
Arrays de tamaño fijo con semántica de valor.
```zc
def SIZE = 5;
let ints: int[SIZE] = [1, 2, 3, 4, 5];
let zeros: [int; SIZE]; // Inicializado a cero
```

#### Tuplas
Agrupa múltiples valores, accede a los elementos por índice.
```zc
let pair = (1, "Hola");
let x = pair.0;  // 1
let s = pair.1;  // "Hola"
```

**Múltiples Valores de Retorno**

Las funciones pueden retornar tuplas para proporcionar múltiples resultados:
```zc
fn sumar_y_restar(a: int, b: int) -> (int, int) {
    return (a + b, a - b);
}

let resultado = sumar_y_restar(3, 2);
let suma = resultado.0;   // 5
let resta = resultado.1;  // 1
```

**Desestructuración**

Las tuplas pueden desestructurarse directamente en variables:
```zc
let (suma, resta) = sumar_y_restar(3, 2);
// suma = 5, resta = 1
```

La desestructuración tipada permite anotaciones de tipo explícitas:
```zc
let (a: string, b: u8) = ("hello", 42);
let (x, y: i32) = (1, 2);  // Mixto: x inferido, y explícito
```

#### Structs
Estructuras de datos con campos de bits opcionales.
```zc
struct Point {
    x: int;
    y: int;
}

// Inicialización de struct
let p = Point { x: 10, y: 20 };

// Campos de bits
struct Flags {
    valid: U8 : 1;
    mode:  U8 : 3;
}
```

> [!NOTE]
> Los structs usan [Semántica de Movimiento](#semántica-de-recursos-movimiento-por-defecto) por defecto. Los campos se pueden acceder mediante `.` incluso en punteros (Auto-Dereferencia).

#### Structs Opacos
Puedes definir un struct como `opaque` para restringir el acceso a sus campos solo al módulo que lo define, permitiendo aún que el struct sea asignado en el stack (el tamaño es conocido).

```zc
// En user.zc
opaque struct User {
    id: int;
    name: string;
}

fn new_user(name: string) -> User {
    return User{id: 1, name: name}; // OK: Dentro del módulo
}

// En main.zc
import "user.zc";

fn main() {
    let u = new_user("Alice");
    // let id = u.id; // Error: No se puede acceder al campo privado 'id'
}
```

#### Enums
Uniones etiquetadas (Tipos suma) capaces de contener datos.
```zc
enum Shape {
    Circle(float),      // Contiene el radio
    Rect(float, float), // Contiene ancho y alto
    Point               // Sin datos
}
```

#### Uniones
Uniones estándar de C (acceso inseguro).
```zc
union Data {
    i: int;
    f: float;
}
```

#### Vectores SIMD
Tipos de vectores SIMD nativos utilizando extensiones de vectores de GCC/Clang. Anota un struct con `@vector(N)` para definir un vector de N elementos.
```zc
import "std/simd.zc";

fn main() {
    let a = f32x4{v: 1.0};              // Difusión: {1.0, 1.0, 1.0, 1.0}
    let b = f32x4{1.0, 2.0, 3.0, 4.0};  // Inicialización por elemento
    let c = a + b;                       // Suma por elementos
    let x = c[0];                        // Acceso a elementos (float)
}
```
Los operadores aritméticos (`+`, `-`, `*`, `/`) y bit a bit (`&`, `|`, `^`) funcionan por elementos. Consulta [`std/simd.zc`](../std/simd.zc) para los tipos predefinidos.

#### Alias de Tipos
Crea un nuevo nombre para un tipo existente.
```zc
alias ID = int;
alias PointMap = Map<string, Point>
alias OpFunc = fn(int, int) -> int
```
> **Nota:** El punto y coma final es opcional para los alias de tipo.

#### Alias de Tipos Opacos
Puedes definir un alias de tipo como `opaque` para crear un nuevo tipo que sea distinto de su tipo subyacente fuera del módulo que lo define. Esto proporciona una fuerte encapsulación y seguridad de tipos sin la sobrecarga en tiempo de ejecución de un struct envoltorio.

```zc
// En library.zc
opaque alias Handle = int;

fn make_handle(v: int) -> Handle {
    return v; // Conversión implícita permitida dentro del módulo
}

// En main.zc
import "library.zc";

fn main() {
    let h: Handle = make_handle(42);
    // let i: int = h; // Error: Falló la validación de tipos
    // let h2: Handle = 10; // Error: Falló la validación de tipos
}
```

### 4. Funciones y Lambdas

#### Funciones
```zc
fn suma(a: int, b: int) -> int {
    return a + b;
}

// Argumentos con nombre soportados en las llamadas
suma(a: 10, b: 20);
```

> [!NOTE]
> Los argumentos con nombre deben seguir estrictamente el orden de los parámetros definidos. `suma(b: 20, a: 10)` es inválido.

#### Argumentos Const
Los argumentos de las funciones pueden marcarse como `const` para imponer una semántica de solo lectura. Este es un calificador de tipo, no una constante manifiesta.

```zc
fn print_val(v: const int) {
    // v = 10; // Error: No se puede asignar a una variable const
    println "{v}";
}
```

#### Argumentos por Defecto
Las funciones pueden definir valores por defecto para los argumentos finales. Estos pueden ser literales, expresiones o código válido de Zen C (como constructores de structs).
```zc
// Valor por defecto simple
fn incrementar(val: int, cantidad: int = 1) -> int {
    return val + cantidad;
}

// Valor por defecto por expresión (evaluado en el sitio de la llamada)
fn offset(val: int, pad: int = 10 * 2) -> int {
    return val + pad;
}

// Valor por defecto de tipo struct
struct Config { debug: bool; }
fn init(cfg: Config = Config { debug: true }) {
    if cfg.debug { println "Modo Debug"; }
}

fn main() {
    incrementar(10);    // 11
    offset(5);          // 25
    init();             // Imprime "Modo Debug"
}
```

#### Lambdas (Clausuras)
Funciones anónimas que pueden capturar su entorno.
```zc
let factor = 2;
let doble = x -> x * factor;  // Sintaxis de flecha
let completo = fn(x: int) -> int { return x * factor; }; // Sintaxis de bloque

// Captura por Referencia (Sintaxis de Bloque)
let val = 10;
let modificar = fn[&]() { val += 1; }; 
modificar(); // val ahora es 11

// Captura por Referencia (Sintaxis de Flecha)
let modificar_flecha = [&] x -> val += x;
modificar_flecha(5); // val ahora es 16

// Captura por Referencia (Sintaxis de Flecha con Múltiples Argumentos)
let sumar_en = [&] (a, b) -> val += (a + b);
sumar_en(2, 2); // val ahora es 20

// Captura por Valor (Por Defecto)
let original = 100;
let implicita = x -> original + x;      // Captura implícita por valor (sin corchetes)
let explicita = [=] x -> original + x;  // Captura explícita por valor
// let fallar = x -> original += x;     // Error: no se puede asignar a valor capturado

```

#### Punteros a Funciones Crudos
Zen C soporta punteros a funciones de C crudos usando la sintaxis `fn*`. Esto permite una interoperabilidad perfecta con bibliotecas de C que esperan punteros a funciones sin la sobrecarga de las clausuras.

```zc
// Función que recibe un puntero a función crudo
fn set_callback(cb: fn*(int)) {
    cb(42);
}

// Función que retorna un puntero a función crudo
fn get_callback() -> fn*(int) {
    return mi_manejador;
}

// Se soportan punteros a punteros de funciones (fn**)
let pptr: fn**(int) = &ptr;
```

#### Funciones Variádicas
Las funciones pueden aceptar un número variable de argumentos usando `...` y el tipo `va_list`.
```zc
fn log(lvl: int, fmt: char*, ...) {
    let ap: va_list;
    va_start(ap, fmt);
    vprintf(fmt, ap); // Usa stdio de C
    va_end(ap);
}
```

### 5. Flujo de Control

#### Condicionales
```zc
if x > 10 {
    print("Grande");
} else if x > 5 {
    print("Mediano");
} else {
    print("Pequeño");
}

// Ternario
let y = x > 10 ? 1 : 0;

// If-Expression (para condiciones complejas)
let categoria = if (x > 100) { "enorme" } else if (x > 10) { "grande" } else { "pequeño" };
```

#### Coincidencia de Patrones (Pattern Matching)
Una alternativa potente al `switch`.
```zc
match val {
    1         => { print "Uno" },
    2 || 3    => { print "Dos o Tres" },      // OR con ||
    4 or 5    => { print "Cuatro o Cinco" },  // OR con 'or'
    6, 7, 8   => { print "Seis a Ocho" },     // OR con coma
    10 .. 15  => { print "10 a 14" },         // Rango exclusivo (Legado)
    10 ..< 15 => { print "10 a 14" },         // Rango exclusivo (Explícito)
    20 ..= 25 => { print "20 a 25" },         // Rango inclusivo
    _         => { print "Otro" },
}

// Desestructuración de Enums
match shape {
    Shape::Circle(r)   => { println "Radio: {r}" },
    Shape::Rect(w, h)  => { println "Área: {w*h}" },
    Shape::Point       => { println "Punto" },
}
```

#### Vinculación por Referencia (Reference Binding)
Para inspeccionar un valor sin tomar posesión de él (sin moverlo), usa la palabra clave `ref` en el patrón. Esto es esencial para tipos que implementan la Semántica de Movimiento (como `Option`, `Result`, structs que no son Copy).

```zc
let opt = Some(ValorNoCopy{...});
match opt {
    Some(ref x) => {
        // 'x' es un puntero al valor dentro de 'opt'
        // 'opt' NO se mueve ni se consume aquí
        println "{x.field}"; 
    },
    None => {}
}
```

#### Bucles
```zc
// Rango
for i in 0..10 { ... }      // Exclusivo (0 al 9)
for i in 0..<10 { ... }     // Exclusivo (Explícito)
for i in 0..=10 { ... }     // Inclusivo (0 al 10)
for i in 0..10 step 2 { ... }
for i in 10..0 step -1 { ... }  // Descending loop

// Iterador (Vec o Iterable personalizado)
for item in vec { ... }

// Enumerado: obtener índice y valor
for i, val in arr { ... }       // i = 0, 1, 2, ...
for i, val in 0..10 step 2 { ... } // i = 0, 1, 2, ...; val = 0, 2, 4, ...

// Iterar sobre arrays de tamaño fijo directamente
let arr: int[5] = [1, 2, 3, 4, 5];
for val in arr {
    // val es int
    println "{val}";
}

// While
while x < 10 { ... }

// Do-While
do { ... } while x < 10;

// Infinito con etiqueta
externo: loop {
    if terminado { break externo; }
}

// Repetir N veces
for _ in 0..5 { ... }
```


#### Control Avanzado
```zc
// Guard: Ejecuta else y retorna si la condición es falsa
guard ptr != NULL else { return; }

// Unless: Si no es verdadero
unless es_valido { return; }
```

### 6. Operadores

Zen C soporta la sobrecarga de operadores para structs definidos por el usuario implementando nombres de métodos específicos.

#### Operadores Sobrecargables

| Categoría | Operador | Nombre del Método |
|:---|:---|:---|
| **Aritméticos** | `+`, `-`, `*`, `/`, `%`, `**` | `add`, `sub`, `mul`, `div`, `rem`, `pow` |
| **Comparación** | `==`, `!=` | `eq`, `neq` |
| | `<`, `>`, `<=`, `>=` | `lt`, `gt`, `le`, `ge` |
| **Bitwise** | `&`, `|`, `^` | `bitand`, `bitor`, `bitxor` |
| | `<<`, `>>` | `shl`, `shr` |
| **Unarios** | `-` | `neg` |
| | `!` | `not` |
| | `~` | `bitnot` |
| **Índice** | `a[i]` | `get(a, i)` |
| | `a[i, j]` | `get(a, i, j)` |
| | `a[i] = v` | `set(a, i, v)` |

> **Nota sobre la igualdad de cadenas**:
> - `string == string` realiza una **comparación de valores** (equivalente a `strcmp`).
> - `char* == char*` realiza una **comparación de punteros** (comprueba direcciones de memoria).
> - Comparaciones mixtas (ej. `string == char*`) por defecto realizan una **comparación de punteros**.

**Ejemplo:**
```zc
impl Point {
    fn add(self, other: Point) -> Point {
        return Point{x: self.x + other.x, y: self.y + other.y};
    }
}

let p3 = p1 + p2; // Llama a p1.add(p2)
```

**Ejemplo Multi-Índice:**
```zc
struct Matriz {
    data: int[9];
}

impl Matriz {
    fn get(self, fila: int, col: int) -> int {
        return self.data[fila * 3 + col];
    }
}

let m = Matriz{data: [1,0,0, 0,1,0, 0,0,1]};
let val = m[1, 2]; // Llama a Matriz.get(m, 1, 2)
```

#### Azúcar Sintáctico

Estos operadores son características integradas del lenguaje y no pueden sobrecargarse directamente.

| Operador | Nombre | Descripción |
|:---|:---|:---|
| `|>` | Pipeline | `x |> f(y)` se desazucara a `f(x, y)` |
| `??` | Null Coalescing | `val ?? default` retorna `default` si `val` es NULL (punteros) |
| `??=` | Null Assignment | `val ??= init` asigna si `val` es NULL |
| `?.` | Navegación Segura | `ptr?.campo` accede al campo solo si `ptr` no es NULL |
| `?` | Operador Try | `res?` retorna el error si está presente (tipos Result/Option) |

**Auto-Dereferencia**:
El acceso a campos por puntero (`ptr.campo`) y las llamadas a métodos (`ptr.metodo()`) dereferencian automáticamente el puntero, equivalente a `(*ptr).campo`.

### 7. Impresión e Interpolación de Cadenas

Zen C proporciona opciones versátiles para imprimir en la consola, incluyendo palabras clave y abreviaturas concisas.

#### Palabras Clave

| Palabra Clave | Descripción |
|:---|:---|
| `print "texto"` | Imprime en `stdout` sin un salto de línea al final. |
| `println "texto"` | Imprime en `stdout` **con** un salto de línea al final. |
| `eprint "texto"` | Imprime en `stderr` sin un salto de línea al final. |
| `eprintln "texto"` | Imprime en `stderr` **con** un salto de línea al final. |

#### Abreviaturas

Zen C permite usar literales de cadena directamente como sentencias para una impresión rápida:

| Sintaxis | Equivalente | Descripción |
|:---|:---|:---|
| `"Hz"` | `println "Hz"` | Imprime en `stdout` con salto de línea. |
| `"Hz"..` | `print "Hz"` | Imprime en `stdout` sin salto de línea. |
| `!"Err"` | `eprintln "Err"` | Imprime en `stderr` con salto de línea. |
| `!"Err"..` | `eprint "Err"` | Imprime en `stderr` sin salto de línea. |

#### Interpolación de Cadenas

Puedes embeber expresiones directamente dentro de literales de cadena usando la sintaxis `{}`. Esto funciona con todos los métodos de impresión y abreviaturas de cadena.

La interpolación de cadenas en Zen C es **implícita**: si tu cadena contiene `{...}`, se analizará automáticamente como una cadena interpolada. También puedes usar explícitamente el prefijo `f` (ej. `f"..."`) para forzar la semántica de interpolación.

```zc
let x = 42;
let nombre = "Zen";
println "Valor: {x}, Nombre: {nombre}";
"Valor: {x}, Nombre: {nombre}"; // abreviatura println
```

**Escapando Llaves**: Usa `{{` para producir una llave literal `{` y `}}` para una `}` literal:

```zc
let json = "JSON: {{\"clave\": \"valor\"}}";
// Salida: JSON: {"clave": "valor"}
```

**Cadenas Crudas (Raw Strings)**: Para definir una cadena donde la interpolación y las secuencias de escape se ignoran por completo, usa el prefijo `r` (ej. `r"..."`):

```zc
let regex = r"\w+"; // Contiene exactamente \ w +
let raw_json = r'{"clave": "valor"}'; // No es necesario escapar llaves
```

#### Cadenas Multilínea

Zen C soporta bloques de cadenas multilínea crudas usando el delimitador `"""`. Esto es extremadamente útil para escribir lenguajes embebidos (GLSL, HTML) o para generar código C en bloques `comptime` sin tener que escapar manualmente los saltos de línea y las comillas internas.

Al igual que las cadenas estándar, las cadenas multilínea soportan **interpolación implícita**. También puedes prefijarlas explícitamente:
- `f"""..."""`: Marca explícitamente como un bloque de cadena interpolada.
- `r"""..."""`: Marca explícitamente como un bloque de cadena cruda (sin interpolación, sin secuencias de escape).

```zc
let prompt = """
  Por favor, introduzca su nombre:
  Escribe "exit" para cancelar.
""";

let mundo = "mundo";
let script = """
  fn hola() {
      println "hola, {mundo}!";
  }
""";

let pure_raw = r"""
  Aquí las {llaves} son solo texto, y \n es literalmente una barra y una n.
""";
```

#### Prompts de Entrada (`?`)

Zen C soporta una abreviatura para solicitar entrada al usuario usando el prefijo `?`.

- `? "Texto del prompt"`: Imprime el prompt (sin salto de línea) y espera la entrada (lee una línea).
- `? "Ingresa la edad: " (edad)`: Imprime el prompt y escanea la entrada en la variable `edad`.
    - Los especificadores de formato se infieren automáticamente según el tipo de variable.

```zc
let edad: int;
? "¿Cuántos años tienes? " (edad);
println "Tienes {edad} años.";
```

### 8. Gestión de Memoria

Zen C permite la gestión manual de memoria con ayudas ergonómicas.

#### Defer
Ejecuta código cuando el ámbito actual finaliza. Las sentencias defer se ejecutan en orden LIFO (último en entrar, primero en salir).
```zc
let f = fopen("archivo.txt", "r");
defer fclose(f);
```

> Para prevenir comportamientos indefinidos, las sentencias de flujo de control (`return`, `break`, `continue`, `goto`) **no están permitidas** dentro de un bloque `defer`.

#### Autofree
Libera automáticamente la variable cuando finaliza el ámbito.
```zc
autofree let tipos = malloc(1024);
```

#### Semántica de Recursos (Movimiento por Defecto)
Zen C trata los tipos con destructores (como `File`, `Vec` o punteros de malloc) como **Recursos**. Para prevenir errores de doble liberación (double-free), los recursos no pueden duplicarse implícitamente.

- **Movimiento por Defecto**: La asignación de una variable de recurso transfiere la posesión. La variable original se vuelve inválida (Movida).
- **Tipos Copy**: Los tipos sin destructores pueden optar por el comportamiento `Copy`, haciendo que la asignación sea una duplicación.

**Diagnóstico y Filosofía**:
Si ves un error "Use of moved value", el compilador te está diciendo: *"Este tipo posee un recurso (como memoria o un manejador) y copiarlo a ciegas es inseguro."*

> **Contraste:** A diferencia de C/C++, Zen C no duplica implícitamente los valores que poseen recursos.

**Argumentos de Función**:
Pasar un valor a una función sigue las mismas reglas que la asignación: los recursos se mueven a menos que se pasen por referencia.

```zc
fn procesar(r: Recurso) { ... } // 'r' se mueve dentro de la función
fn mirar(r: Recurso*) { ... }   // 'r' es prestado (referencia)
```

**Clonación Explícita**:
Si *realmente* quieres dos copias de un recurso, hazlo explícito:

```zc
let b = a.clone(); // Llama al método 'clone' del trait Clone
```

**Optar por Copy (Tipos de Valor)**:
Para tipos pequeños sin destructores:

```zc
struct Point { x: int; y: int; }
impl Copy for Point {} // Optar por la duplicación implícita

fn main() {
    let p1 = Point { x: 1, y: 2 };
    let p2 = p1; // Copiado. p1 sigue siendo válido.
}
```

#### RAII / Drop Trait
Implementa `Drop` para ejecutar lógica de limpieza automáticamente.
```zc
impl Drop for MiEstructura {
    fn drop(self) {
        self.free();
    }
}
```

### 9. Programación Orientada a Objetos

#### Métodos
Define métodos en los tipos usando `impl`.
```zc
impl Point {
    // Método estático (convención de constructor)
    fn new(x: int, y: int) -> Self {
        return Point{x: x, y: y};
    }

    // Método de instancia
    fn dist(self) -> float {
        return sqrt(self.x * self.x + self.y * self.y);
    }
}
```

**Atajo de Self**: En métodos con un parámetro `self`, puedes usar `.campo` como abreviatura de `self.campo`:
```zc
impl Point {
    fn dist(self) -> float {
        return sqrt(.x * .x + .y * .y);  // Equivalente a self.x, self.y
    }
}
```

#### Métodos primitivos
Zen C permite definir métodos en tipos primitivos (como `int`, `bool`, etc.) usando la misma sintaxis `impl`.

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
Define un comportamiento compartido.
```zc
struct Circle { radio: f32; }

trait Dibujable {
    fn dibujar(self);
}

impl Dibujable for Circle {
    fn dibujar(self) { ... }
}

let circulo = Circle{};
let dibujable: Dibujable = &circulo;
```

#### Traits Estándar
Zen C incluye traits estándar que se integran con la sintaxis del lenguaje.

**Iterable**

Implementa `Iterable<T>` para habilitar bucles `for-in` para tus tipos personalizados.

```zc
import "std/iter.zc"

// Define un Iterador
struct MiIter {
    actual: int;
    final: int;
}

impl MiIter {
    fn next(self) -> Option<int> {
        if self.actual < self.final {
            self.actual += 1;
            return Option<int>::Some(self.actual - 1);
        }
        return Option<int>::None();
    }
}

// Implementa Iterable
impl MiRango {
    fn iterator(self) -> MiIter {
        return MiIter{actual: self.inicio, final: self.fin};
    }
}

// Uso en un Bucle
for i in mi_rango {
    println "{i}";
}
```

**Drop**

Implementa `Drop` para definir un destructor que se ejecuta cuando el objeto sale de ámbito (RAII).

```zc
import "std/mem.zc"

struct Recurso {
    ptr: void*;
}

impl Drop for Recurso {
    fn drop(self) {
        if self.ptr != NULL {
            free(self.ptr);
        }
    }
}
```

> [!NOTE]
> Si una variable es movida, no se llama a `drop` en la variable original. Se adhiere a la [Semántica de Recursos](#semántica-de-recursos-movimiento-por-defecto).

**Copy**

Trait marcador para optar por el comportamiento `Copy` (duplicación implícita) en lugar de la semántica de movimiento. Se usa mediante `@derive(Copy)`.

> [!CAUTION]
> **Regla:** Los tipos que implementan `Copy` no deben definir un destructor (`Drop`).

```zc
@derive(Copy)
struct Point { x: int; y: int; }

fn main() {
    let p1 = Point{x: 1, y: 2};
    let p2 = p1; // ¡Copiado! p1 sigue siendo válido.
}
```

**Clone**

Implementa `Clone` para permitir la duplicación explícita de tipos que poseen recursos.

```zc
import "std/mem.zc"

struct MiBox { val: int; }

impl Clone for MiBox {
    fn clone(self) -> MiBox {
        return MiBox{val: self.val};
    }
}

fn main() {
    let b1 = MiBox{val: 42};
    let b2 = b1.clone(); // Copia explícita
}
```

#### Composición
Usa `use` para embeber otros structs. Puedes mezclarlos (aplanar campos) o nombrarlos (anidar campos).

```zc
struct Entity { id: int; }

struct Player {
    // Mezcla (Mixin - Sin nombre): Aplana los campos
    use Entity;  // Añade 'id' a Player directamente
    nombre: string;
}

struct Match {
    // Composición (Con nombre): Anida los campos
    use p1: Player; // Accedido mediante match.p1
    use p2: Player; // Accedido mediante match.p2
}
```

### 10. Genéricos

Plantillas seguras para tipos para Structs y Funciones.

```zc
// Struct Genérico
struct Box<T> {
    item: T;
}

// Función Genérica
fn identidad<T>(val: T) -> T {
    return val;
}

// Genéricos con múltiples parámetros
struct Par<K, V> {
    llave: K;
    valor: V;
}
```

### 11. Concurrencia (Async/Await)

Construido sobre pthreads.

```zc
async fn obtener_datos() -> string {
    // Se ejecuta en segundo plano
    return "Datos";
}

fn main() {
    let futuro = obtener_datos();
    let resultado = await futuro;
}
```

### 12. Metaprogramación

#### Comptime
Ejecuta código en tiempo de compilación para generar código fuente o imprimir mensajes.
```zc
comptime {
    // Genera código en tiempo de compilación (escrito en stdout)
    println "let fecha_compilacion = \"2024-01-01\";";
}

println "Fecha de compilación: {fecha_compilacion}";
```

<details>
<summary><b>Funciones Auxiliares</b></summary>

Funciones especiales disponibles dentro de bloques `comptime`:

<table>
<tr>
<th>Función</th>
<th>Descripción</th>
</tr>
<tr>
<td><code>yield(str)</code></td>
<td>Emite código generado explícitamente (alternativa a <code>printf</code>)</td>
</tr>
<tr>
<td><code>code(str)</code></td>
<td>Alias de <code>yield()</code> - intención más clara para generación de código</td>
</tr>
<tr>
<td><code>compile_error(msg)</code></td>
<td>Detiene la compilación con un mensaje de error fatal</td>
</tr>
<tr>
<td><code>compile_warn(msg)</code></td>
<td>Emite una advertencia en tiempo de compilación (permite continuar)</td>
</tr>
</table>

**Ejemplo:**
```zc
comptime {
    compile_warn("Generando código optimizado...");
    
    let ENABLE_FEATURE = 1;
    if (ENABLE_FEATURE == 0) {
        compile_error("¡La función debe estar habilitada!");
    }
    
    // Usa code() con raw strings para generación limpia
    code(r"let FEATURE_ENABLED = 1;");
}
```
</details>

<details>
<summary><b>Metadatos de Construcción</b></summary>

Accede a información de construcción del compilador en tiempo de compilación:

<table>
<tr>
<th>Constante</th>
<th>Tipo</th>
<th>Descripción</th>
</tr>
<tr>
<td><code>__COMPTIME_TARGET__</code></td>
<td>string</td>
<td>Plataforma: <code>"linux"</code>, <code>"windows"</code> o <code>"macos"</code></td>
</tr>
<tr>
<td><code>__COMPTIME_FILE__</code></td>
<td>string</td>
<td>Nombre del archivo fuente actual siendo compilado</td>
</tr>
</table>

**Ejemplo:**
```zc
comptime {
    // Generación de código específica de plataforma
    println "let PLATFORM = \"{__COMPTIME_TARGET__}\";";
}

println "Ejecutando en: {PLATFORM}";
```
</details>

> [!TIP]
> Usa raw strings (`r"..."`) en comptime para evitar escapar llaves: `code(r"fn test() { return 42; }")`. De lo contrario, usa `{{` y `}}` para escapar llaves en strings regulares.


#### Embed
Embebe archivos como los tipos especificados.
```zc
// Por defecto (Slice_char)
let datos = embed "assets/logo.png";

// Embed tipado
let texto = embed "shader.glsl" as string;    // Embebe como C-string
let rom   = embed "bios.bin" as u8[1024];     // Embebe como array fijo
let wav   = embed "sound.wav" as u8[];        // Embebe como Slice_u8
```

#### Plugins
Importa plugins del compilador para extender la sintaxis.
```zc
import plugin "regex"
let re = regex! { ^[a-z]+$ };
```

#### Macros de C Genéricas
Pasa macros del preprocesador directamente a C.

> [!TIP]
> Para constantes simples, usa `def` en su lugar. Usa `#define` cuando necesites macros del preprocesador de C o flags de compilación condicional.

```zc
#define MAX_BUFFER 1024
```

#### Compilación Condicional
Usa `@cfg()` para incluir o excluir condicionalmente cualquier declaración de nivel superior basándote en flags `-D`.

```zc
// Compilar con: zc build app.zc -DUSE_OPENGL

@cfg(USE_OPENGL)
import "opengl_backend.zc";

@cfg(USE_VULKAN)
import "vulkan_backend.zc";

@cfg(not(USE_OPENGL))
@cfg(not(USE_VULKAN))
fn fallback_init() { println "No se seleccionó backend"; }
```

| Forma | Significado |
|:---|:---|
| `@cfg(NAME)` | Incluir si `-DNAME` está definido |
| `@cfg(not(NAME))` | Incluir si `-DNAME` NO está definido |
| `@cfg(any(A, B, ...))` | Incluir si ALGUNA condición es verdadera (OR) |
| `@cfg(all(A, B, ...))` | Incluir si TODAS las condiciones son verdaderas (AND) |

Múltiples `@cfg` en una declaración se combinan con AND. `not()` se puede usar dentro de `any()` y `all()`. Funciona con cualquier declaración de nivel superior: `fn`, `struct`, `import`, `impl`, `raw`, `def`, `test`, etc.

### 13. Atributos

Decora funciones y structs para modificar el comportamiento del compilador.

| Atributo | Ámbito | Descripción |
|:---|:---|:---|
| `@required` | Fn | Advierte si el valor de retorno es ignorado. |
| `@deprecated("msg")` | Fn/Struct | Advierte sobre el uso con un mensaje. |
| `@inline` | Fn | Sugiere al compilador hacer inlininig. |
| `@noinline` | Fn | Previene el inlining. |
| `@packed` | Struct | Elimina el padding entre campos. |
| `@align(N)` | Struct | Fuerza el alineamiento a N bytes. |
| `@constructor` | Fn | Se ejecuta antes de main. |
| `@destructor` | Fn | Se ejecuta después de que main termine. |
| `@unused` | Fn/Var | Suprime advertencias de variables no usadas. |
| `@weak` | Fn | Enlace de símbolo débil (weak symbol linkage). |
| `@section("nombre")` | Fn | Coloca el código en una sección específica. |
| `@noreturn` | Fn | La función no retorna (ej. exit). |
| `@pure` | Fn | La función no tiene efectos secundarios (sugestión de optimización). |
| `@cold` | Fn | Es poco probable que la función se ejecute (sugestión de predicción de saltos). |
| `@hot` | Fn | La función se ejecuta frecuentemente (sugestión de optimización). |
| `@export` | Fn/Struct | Exporta el símbolo (visibilidad por defecto). |
| `@global` | Fn | CUDA: Punto de entrada del kernel (`__global__`). |
| `@device` | Fn | CUDA: Función de dispositivo (`__device__`). |
| `@host` | Fn | CUDA: Función de host (`__host__`). |
| `@comptime` | Fn | Función auxiliar disponible para ejecución en tiempo de compilación. |
| `@cfg(NAME)` | Cualquiera | Compilación condicional: incluye solo si se pasa `-DNAME`. Soporta `not()`, `any()`, `all()`. |
| `@derive(...)` | Struct | Implementa traits automáticamente. Soporta `Debug`, `Eq` (Derivación Inteligente), `Copy`, `Clone`. |
| `@ctype("tipo")` | Parámetro Fn | Sobrescribe el tipo C generado para un parámetro. |
| `@<custom>` | Cualquier | Pasa atributos genéricos a C (ej. `@flatten`, `@alias("nombre")`). |

#### Atributos Personalizados

Zen C soporta un potente sistema de **Atributos Personalizados** que te permite usar cualquier `__attribute__` de GCC/Clang directamente en tu código. Cualquier atributo que no sea reconocido explícitamente por el compilador de Zen C es tratado como un atributo genérico y se pasa al código C generado.

Esto proporciona acceso a características avanzadas del compilador, optimizaciones y directivas del enlazador sin necesidad de soporte explícito en el núcleo del lenguaje.

#### Mapeo de Sintaxis
Los atributos de Zen C se mapean directamente a atributos de C:
- `@nombre` → `__attribute__((nombre))`
- `@nombre(args)` → `__attribute__((nombre(args)))`
- `@nombre("string")` → `__attribute__((nombre("string")))`

#### Derivaciones Inteligentes

Zen C proporciona "Derivaciones Inteligentes" que respetan la Semántica de Movimiento:

- **`@derive(Eq)`**: Genera un método de igualdad que recibe los argumentos por referencia (`fn eq(self, other: T*)`).
    - Al comparar dos structs que no son Copy (`a == b`), el compilador pasa automáticamente `b` por referencia (`&b`) para evitar moverlo.
    - Las comprobaciones de igualdad recursivas en los campos también prefieren el acceso por puntero para prevenir la transferencia de posesión.

### 14. Ensamblador Inline

Zen C proporciona soporte de primera clase para ensamblador inline, transpilando directamente a `asm` extendido de estilo GCC.

#### Uso Básico
Escribe ensamblador crudo dentro de bloques `asm`. Las cadenas se concatenan automáticamente.
```zc
asm {
    "nop"
    "mfence"
}
```

#### Volatile
Previene que el compilador optimice y elimine el ensamblador que tiene efectos secundarios.
```zc
asm volatile {
    "rdtsc"
}
```

#### Restricciones con Nombre
Zen C simplifica la compleja sintaxis de restricciones de GCC con vinculaciones con nombre.

```zc
// Sintaxis: : out(variable) : in(variable) : clobber(reg)
// Usa la sintaxis de marcador de posición {variable} para legibilidad

fn sumar(x: int) -> int {
    let resultado: int;
    asm {
        "mov {x}, {resultado}"
        "add $5, {resultado}"
        : out(resultado)
        : in(x)
        : clobber("cc")
    }
    return resultado;
}
```

| Tipo | Sintaxis | Equivalente GCC |
|:---|:---|:---|
| **Salida** | `: out(variable)` | `"=r"(variable)` |
| **Entrada** | `: in(variable)` | `"r"(variable)` |
| **Clobber** | `: clobber("rax")` | `"rax"` |
| **Memoria** | `: clobber("memory")` | `"memory"` |

> **Nota:** Cuando uses la sintaxis de Intel (mediante `-masm=intel`), debes asegurarte de que tu construcción esté configurada correctamente (por ejemplo, `//> cflags: -masm=intel`). TCC no soporta el ensamblador con sintaxis Intel.

### 15. Directivas de Construcción

Zen C soporta comentarios especiales en la parte superior de tu archivo fuente para configurar el proceso de construcción sin necesidad de un complejo sistema de construcción o Makefile.

| Directiva | Argumentos | Descripción |
|:---|:---|:---|
| `//> link:` | `-lfoo` o `ruta/a/lib.a` | Enlaza contra una biblioteca o archivo objeto. |
| `//> lib:` | `ruta/a/libs` | Añade una ruta de búsqueda de biblioteca (`-L`). |
| `//> include:` | `ruta/a/headers` | Añade una ruta de búsqueda de cabeceras (`-I`). |
| `//> framework:` | `Cocoa` | Enlaza contra un framework de macOS. |
| `//> cflags:` | `-Wall -O3` | Pasa flags arbitrarios al compilador de C. |
| `//> define:` | `MACRO` o `LLAVE=VAL` | Define una macro del preprocesador (`-D`). |
| `//> pkg-config:` | `gtk+-3.0` | Ejecuta `pkg-config` y añade `--cflags` y `--libs`. |
| `//> shell:` | `comando` | Ejecuta un comando de shell durante la construcción. |
| `//> get:` | `http://url/archivo` | Descarga un archivo si el archivo específico no existe. |

#### Características

**1. Protección de SO (OS Guarding)**
Prefija las directivas con el nombre de un SO para aplicarlas solo en plataformas específicas.
Prefijos soportados: `linux:`, `windows:`, `macos:` (o `darwin:`).

```zc
//> linux: link: -lm
//> windows: link: -lws2_32
//> macos: framework: Cocoa
```

**2. Expansión de Variables de Entorno**
Usa la sintaxis `${VAR}` para expandir variables de entorno en tus directivas.

```zc
//> include: ${HOME}/mylib/include
//> lib: ${ZC_ROOT}/std
```

#### Ejemplos

```zc
//> include: ./include
//> lib: ./libs
//> link: -lraylib -lm
//> cflags: -Ofast
//> pkg-config: gtk+-3.0

import "raylib.h"

fn main() { ... }
```

### 16. Palabras Clave

Las siguientes palabras clave están reservadas en Zen C.

#### Declaraciones
`alias`, `def`, `enum`, `fn`, `impl`, `import`, `let`, `module`, `opaque`, `struct`, `trait`, `union`, `use`

#### Flujo de Control
`async`, `await`, `break`, `catch`, `continue`, `defer`, `do`, `else`, `for`, `goto`, `guard`, `if`, `loop`, `match`, `return`, `try`, `unless`, `while`

#### Especiales
`asm`, `assert`, `autofree`, `comptime`, `const`, `embed`, `launch`, `ref`, `sizeof`, `static`, `test`, `volatile`

#### Constantes
`true`, `false`, `null`

#### Reservadas de C
Los siguientes identificadores están reservados porque son palabras clave en C11:
`auto`, `case`, `char`, `default`, `double`, `extern`, `float`, `inline`, `int`, `long`, `register`, `restrict`, `short`, `signed`, `switch`, `typedef`, `unsigned`, `void`, `_Atomic`, `_Bool`, `_Complex`, `_Generic`, `_Imaginary`, `_Noreturn`, `_Static_assert`, `_Thread_local`

#### Operadores
`and`, `or`

### 17. Interoperabilidad C
Zen C ofrece dos formas de interactuar con código C: **Importaciones de Confianza** (Conveniente) y **FFI Explícita** (Seguro/Preciso).

#### Método 1: Importaciones de Confianza (Conveniente)

Puedes importar una cabecera C directamente usando la palabra clave `import` con la extensión `.h`. Esto trata la cabecera como un módulo y asume que todos los símbolos accedidos existen.

```zc
//> link: -lm
import "math.h" as c_math;

fn main() {
    // El compilador confía en la corrección; emite 'cos(...)' directamente
    let x = c_math::cos(3.14159);
}
```

> **Pros**: Cero código repetitivo. Acceso a todo el contenido de la cabecera inmediato.
> **Cons**: Sin seguridad de tipos desde Zen C (errores capturados por el compilador C después).

#### Método 2: FFI Explícita (Seguro)

Para una comprobación estricta de tipos o cuando no quieres incluir el texto de una cabecera, usa `extern fn`.

```zc
include <stdio.h> // Emite #include <stdio.h> en el C generado

// Define firma estricta
extern fn printf(fmt: char*, ...) -> c_int;

fn main() {
    printf("Hola FFI: %d\n", 42); // Comprobado por tipos por Zen C
}
```

> **Pros**: Zen C asegura que los tipos coincidan.
> **Cons**: Requiere declaración manual de funciones.

#### `import` vs `include`

- **`import "file.h"`**: Registra la cabecera como un módulo con nombre. Habilita el acceso implícito a símbolos (ej. `file::function()`).
- **`include <file.h>`**: Puramente emite `#include <file.h>` en el código C generado. No introduce ningún símbolo al compilador de Zen C; debes usar `extern fn` para acceder a ellos.


---

## Biblioteca Estándar

Zen C incluye una biblioteca estándar (`std`) que cubre las funcionalidades esenciales.

[Explorar la Documentación de la Biblioteca Estándar](../docs/std/README.md)

### Módulos Clave

<details>
<summary>Click para ver todos los módulos de la Biblioteca Estándar</summary>

| Módulo | Descripción | Docs |
| :--- | :--- | :--- |
| **`std/bigfloat.zc`** | Aritmética de punto flotante de precisión arbitraria. | [Docs](../docs/std/bigfloat.md) |
| **`std/bigint.zc`** | Entero de precisión arbitraria `BigInt`. | [Docs](../docs/std/bigint.md) |
| **`std/bits.zc`** | Operaciones bit a bit de bajo nivel (`rotl`, `rotr`, etc). | [Docs](../docs/std/bits.md) |
| **`std/complex.zc`** | Aritmética de números complejos `Complex`. | [Docs](../docs/std/complex.md) |
| **`std/vec.zc`** | Array dinámico creíble `Vec<T>`. | [Docs](../docs/std/vec.md) |
| **`std/string.zc`** | Tipo `String` asignado en el heap con soporte UTF-8. | [Docs](../docs/std/string.md) |
| **`std/queue.zc`** | Cola FIFO (Ring Buffer). | [Docs](../docs/std/queue.md) |
| **`std/map.zc`** | Mapa Hash Genérico `Map<V>`. | [Docs](../docs/std/map.md) |
| **`std/fs.zc`** | Operaciones del sistema de archivos. | [Docs](../docs/std/fs.md) |
| **`std/io.zc`** | Entrada/Salida estándar (`print`/`println`). | [Docs](../docs/std/io.md) |
| **`std/option.zc`** | Valores opcionales (`Some`/`None`). | [Docs](../docs/std/option.md) |
| **`std/result.zc`** | Gestión de errores (`Ok`/`Err`). | [Docs](../docs/std/result.md) |
| **`std/path.zc`** | Manipulación de rutas multiplataforma. | [Docs](../docs/std/path.md) |
| **`std/env.zc`** | Variables de entorno del proceso. | [Docs](../docs/std/env.md) |
| **`std/net/`** | TCP, UDP, HTTP, DNS, URL. | [Docs](../docs/std/net.md) |
| **`std/thread.zc`** | Hilos y Sincronización. | [Docs](../docs/std/thread.md) |
| **`std/time.zc`** | Medición de tiempo y espera (sleep). | [Docs](../docs/std/time.md) |
| **`std/json.zc`** | Parseo y serialización de JSON. | [Docs](../docs/std/json.md) |
| **`std/stack.zc`** | Pila LIFO `Stack<T>`. | [Docs](../docs/std/stack.md) |
| **`std/set.zc`** | Conjunto Hash Genérico `Set<T>`. | [Docs](../docs/std/set.md) |
| **`std/process.zc`** | Ejecución y gestión de procesos. | [Docs](../docs/std/process.md) |
| **`std/regex.zc`** | Expresiones Regulares (basado en TRE). | [Docs](../docs/std/regex.md) |
| **`std/simd.zc`** | Tipos de vectores SIMD nativos. | [Docs](../docs/std/simd.md) |

</details>

---

## Herramientas

Zen C proporciona un Servidor de Lenguaje y un REPL integrados para mejorar la experiencia de desarrollo.

### Servidor de Lenguaje (LSP)

El Servidor de Lenguaje de Zen C (LSP) soporta las características estándar de LSP para integración con editores, proporcionando:

*   **Ir a la Definición**
*   **Encontrar Referencias**
*   **Información al pasar el ratón (Hover)**
*   **Autocompletado** (Nombres de funciones/structs, autocompletado tras punto para métodos/campos)
*   **Símbolos del Documento** (Esquema)
*   **Ayuda de Firma**
*   **Diagnósticos** (Errores sintácticos/semánticos)

Para iniciar el servidor de lenguaje (normalmente configurado en los ajustes de LSP de tu editor):

```bash
zc lsp
```

Se comunica mediante I/O estándar (JSON-RPC 2.0).

### REPL

El bucle Read-Eval-Print te permite experimentar con el código de Zen C de forma interactiva.

```bash
zc repl
```

#### Características

*   **Codificación Interactiva**: Escribe expresiones o sentencias para su evaluación inmediata.
*   **Historial Persistente**: Los comandos se guardan en `~/.zprep_history`.
*   **Script de Inicio**: Carga automáticamente comandos desde `~/.zprep_init.zc`.

#### Comandos

| Comando | Descripción |
|:---|:---|
| `:help` | Muestra los comandos disponibles. |
| `:reset` | Limpia el historial de la sesión actual (variables/funciones). |
| `:vars` | Muestra las variables activas. |
| `:funcs` | Muestra las funciones definidas por el usuario. |
| `:structs` | Muestra los structs definidos por el usuario. |
| `:imports` | Muestra las importaciones activas. |
| `:history` | Muestra el historial de entrada de la sesión. |
| `:type <expr>` | Muestra el tipo de una expresión. |
| `:c <stmt>` | Muestra el código C generado para una sentencia. |
| `:time <expr>` | Benchmark de una expresión (ejecuta 1000 iteraciones). |
| `:edit [n]` | Edita el comando `n` (por defecto: el último) en `$EDITOR`. |
| `:save <file>` | Guarda la sesión actual en un archivo `.zc`. |
| `:load <file>` | Carga y ejecuta un archivo `.zc` en la sesión. |
| `:watch <expr>` | Observa una expresión (se revalúa tras cada entrada). |
| `:unwatch <n>` | Elimina una observación. |
| `:undo` | Elimina el último comando de la sesión. |
| `:delete <n>` | Elimina el comando en el índice `n`. |
| `:clear` | Limpia la pantalla. |
| `:quit` | Sale del REPL. |
| `! <cmd>` | Ejecuta un comando de shell (ej. `!ls`). |

---


### Protocolo de Servidor de Lenguaje (LSP)

Zen C incluye un Servidor de Lenguaje integrado para la integración con editores.

- **[Guía de Instalación y Configuración](translations/LSP_ES.md)**
- **Editores Soportados**: VS Code, Neovim, Vim, Zed, y cualquier editor capaz de LSP.

Usa `zc lsp` para iniciar el servidor.

### Depuración de Zen C

Los programas de Zen C se pueden depurar utilizando depuradores de C estándar como **LLDB** o **GDB**.

#### Visual Studio Code

Para obtener la mejor experiencia en VS Code, instale la [extensión oficial de Zen C](https://marketplace.visualstudio.com/items?itemName=Z-libs.zenc). Para la depuración, puede utilizar la extensión **C/C++** (de Microsoft) o **CodeLLDB**.

Agregue estas configuraciones a su directorio `.vscode` para habilitar la depuración con un solo clic:

**`tasks.json`** (Tarea de compilación):
```json
{
    "label": "Zen C: Build Debug",
    "type": "shell",
    "command": "zc",
    "args": [ "${file}", "-g", "-o", "${fileDirname}/app", "-O0" ],
    "group": { "kind": "build", "isDefault": true }
}
```

**`launch.json`** (Depurador):
```json
{
    "name": "Zen C: Debug (LLDB)",
    "type": "lldb",
    "request": "launch",
    "program": "${fileDirname}/app",
    "preLaunchTask": "Zen C: Build Debug"
}
```

## Soporte del Compilador y Compatibilidad

Zen C está diseñado para funcionar con la mayoría de los compiladores C11. Algunas características dependen de extensiones de GNU C, pero estas suelen funcionar en otros compiladores. Usa la flag `--cc` para cambiar de backend.

```bash
zc run app.zc --cc clang
zc run app.zc --cc zig
```

### Estado de la Suite de Pruebas

<details>
<summary>Click para ver detalles de Soporte del Compilador</summary>

| Compilador | Tasa de Acierto | Características Soportadas | Limitaciones Conocidas |
|:---|:---:|:---|:---|
| **GCC** | **100% (Completo)** | Todas las características | Ninguna. |
| **Clang** | **100% (Completo)** | Todas las características | Ninguna. |
| **Zig** | **100% (Completo)** | Todas las características | Ninguna. Usa `zig cc` como compilador C. |
| **TCC** | **98% (Alto)** | Estructuras, Genéricos, Traits, Coincidencia de Patrones | Sin ASM Intel, Sin `__attribute__((constructor))`. |

</details>

> [!WARNING]
> **ADVERTENCIA DE COMPILACIÓN:** Aunque **Zig CC** funciona excelentemente como backend para tus programas Zen C, compilar el *propio compilador Zen C* con el puede verificar pero producir un binario inestable que falla en las pruebas. Recomendamos compilar el compilador con **GCC** o **Clang** y usar Zig solo como backend para tu código operativo.

### Construyendo con Zig

El comando `zig cc` de Zig proporciona un reemplazo directo para GCC/Clang con un excelente soporte de compilación cruzada (cross-compilation). Para usar Zig:

```bash
# Compilar y ejecutar un programa Zen C con Zig
zc run app.zc --cc zig

# Construir el propio compilador Zen C con Zig
make zig
```

### Interop con C++

Zen C puede generar código compatible con C++ con la flag `--cpp`, permitiendo una integración perfecta con bibliotecas de C++.

```bash
# Compilación directa con g++
zc app.zc --cpp

# O transpilar para construcción manual
zc transpile app.zc --cpp
g++ out.c mi_lib_cpp.o -o app
```

#### Usando C++ en Zen C

Incluye cabeceras de C++ y usa bloques `raw` para el código C++:

```zc
include <vector>
include <iostream>

raw {
    std::vector<int> hacer_vec(int a, int b) {
        return {a, b};
    }
}

fn main() {
    let v = hacer_vec(1, 2);
    raw { std::cout << "Tamaño: " << v.size() << std::endl; }
}
```

> [!NOTE]
> La flag `--cpp` cambia el backend a `g++` y emite código compatible con C++ (usa `auto` en lugar de `__auto_type`, sobrecarga de funciones en lugar de `_Generic`, y casts explícitos para `void*`).

#### Interop con CUDA

Zen C soporta la programación de GPU transpilando a **CUDA C++**. Esto te permite aprovechar las potentes características de C++ (plantillas, constexpr) dentro de tus kernels mientras mantienes la sintaxis ergonómica de Zen C.

```bash
# Compilación directa con nvcc
zc run app.zc --cuda

# O transpilar para construcción manual
zc transpile app.zc --cuda -o app.cu
nvcc app.cu -o app
```

#### Atributos Específicos de CUDA

| Atributo | Equivalente CUDA | Descripción |
|:---|:---|:---|
| `@global` | `__global__` | Función de kernel (se ejecuta en GPU, se llama desde el host) |
| `@device` | `__device__` | Función de dispositivo (se ejecuta en GPU, se llama desde GPU) |
| `@host` | `__host__` | Función de host (explícitamente solo CPU) |

#### Sintaxis de Lanzamiento de Kernel

Zen C proporciona una sentencia `launch` limpia para invocar kernels de CUDA:

```zc
launch nombre_del_kernel(args) with {
    grid: num_bloques,
    block: hilos_por_bloque,
    shared_mem: 1024,  // Opcional
    stream: mi_stream   // Opcional
};
```

Esto se transpila a: `nombre_del_kernel<<<grid, bloque, compartido, stream>>>(args);`

#### Escribiendo Kernels de CUDA

Usa la sintaxis de funciones de Zen C con `@global` y la sentencia `launch`:

```zc
import "std/cuda.zc"

@global
fn kernel_suma(a: float*, b: float*, c: float*, n: int) {
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

    // ... inicialización de datos ...
    
    launch kernel_suma(d_a, d_b, d_c, N) with {
        grid: (N + 255) / 256,
        block: 256
    };
    
    cuda_sync();
}
```

#### Biblioteca Estándar (`std/cuda.zc`)
Zen C proporciona una biblioteca estándar para operaciones comunes de CUDA para reducir los bloques `raw`:

```zc
import "std/cuda.zc"

// Gestión de memoria
let d_ptr = cuda_alloc<float>(1024);
cuda_copy_to_device(d_ptr, h_ptr, 1024 * sizeof(float));
defer cuda_free(d_ptr);

// Sincronización
cuda_sync();

// Indexación de hilos (usar dentro de kernels)
let i = thread_id(); // Índice global
let bid = block_id();
let tid = local_id();
```


> [!NOTE]
> **Nota:** La flag `--cuda` establece `nvcc` como el compilador e implica el modo `--cpp`. Requiere el NVIDIA CUDA Toolkit.

### Soporte C23

Zen C soporta características modernas de C23 cuando se utiliza un compilador backend compatible (GCC 14+, Clang 14+).

- **`auto`**: Zen C mapea automáticamente la inferencia de tipos a `auto` estándar de C23 si `__STDC_VERSION__ >= 202300L`.
- **`_BitInt(N)`**: Use tipos `iN` y `uN` (ej. `i256`, `u12`, `i24`) para acceder a enteros de ancho arbitrario de C23.

### Interop con Objective-C

Zen C puede compilarse a Objective-C (`.m`) usando la flag `--objc`, permitiéndote usar frameworks de Objective-C (como Cocoa/Foundation) y su sintaxis.

```bash
# Compilar con clang (o gcc/gnustep)
zc app.zc --objc --cc clang
```

#### Usando Objective-C en Zen C

Usa `include` para las cabeceras y bloques `raw` para la sintaxis de Objective-C (`@interface`, `[...]`, `@""`).

```zc
//> macos: framework: Foundation
//> linux: cflags: -fconstant-string-class=NSConstantString -D_NATIVE_OBJC_EXCEPTIONS
//> linux: link: -lgnustep-base -lobjc

include <Foundation/Foundation.h>

fn main() {
    raw {
        NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
        NSLog(@"¡Hola desde Objective-C!");
        [pool drain];
    }
    println "¡Zen C también funciona!";
}
```

> [!NOTE]
> **Nota:** La interpolación de cadenas de Zen C funciona con objetos de Objective-C (`id`) llamando a `debugDescription` o `description`.

### 18. Marco de Pruebas Unitarias

Zen C incluye un marco de pruebas integrado que permite escribir pruebas unitarias directamente en los archivos fuente utilizando la palabra clave `test`.

#### Sintaxis
Un bloque `test` contiene un nombre descriptivo y un cuerpo de código para ejecutar. Las pruebas no requieren una función `main` para ejecutarse.

```zc
test "unittest1" {
    "Esta es una prueba unitaria";

    let a = 3;
    assert(a > 0, "a debería ser un entero positivo");

    "unittest1 pasado.";
}
```

#### Ejecución de Pruebas
Para ejecutar todas las pruebas en un archivo, usa el comando `run`. El compilador detectará y ejecutará automáticamente todos los bloques `test` de nivel superior.

```bash
zc run mi_archivo.zc
```

#### Aserciones
Usa la función integrada `assert(condición, mensaje)` para verificar las expectativas. Si la condición es falsa, la prueba fallará y se imprimirá el mensaje proporcionado.

---

## Contribuyendo

¡Damos la bienvenida a las contribuciones! Ya sea corrigiendo errores, añadiendo documentación o proponiendo nuevas características.

Por favor, consulta [CONTRIBUTING_ES.md](CONTRIBUTING_ES.md) para ver las guías detalladas sobre cómo contribuir, ejecutar pruebas y enviar pull requests.

---

## Seguridad

Para instrucciones sobre reportes de seguridad, por favor vea [SECURITY_ES.md](SECURITY_ES.md).

---

## Atribuciones

Este proyecto utiliza bibliotecas de terceros. Los textos completos de las licencias pueden encontrarse en el directorio `LICENSES/`.

*   **[cJSON](https://github.com/DaveGamble/cJSON)** (Licencia MIT): Usado para el parseo y generación de JSON en el Servidor de Lenguaje.
*   **[zc-ape](https://github.com/OEvgeny/zc-ape)** (Licencia MIT): El port original de Ejecutable Realmente Portable de Zen-C por [Eugene Olonov](https://github.com/OEvgeny).
*   **[Cosmopolitan Libc](https://github.com/jart/cosmopolitan)** (Licencia ISC): La biblioteca fundamental que hace posible APE.
*   **[TRE](https://github.com/laurikari/tre)** (Licencia BSD): Usado para el motor de expresiones regulares en la biblioteca estándar.
*   **[zenc.vim](https://github.com/zenc-lang/zenc.vim)** (Licencia MIT): El plugin oficial para Vim/Neovim, escrito principalmente por **[davidscholberg](https://github.com/davidscholberg)**.

---

<div align="center">
  <p>
    Copyright © 2026 Lenguaje de Programación Zen C.<br>
    Comienza tu viaje hoy.
  </p>
  <p>
    <a href="https://discord.com/invite/q6wEsCmkJP">Discord</a> •
    <a href="https://github.com/zenc-lang/zenc">GitHub</a> •
    <a href="https://github.com/zenc-lang/docs">Documentación</a> •
    <a href="https://github.com/zenc-lang/awesome-zenc">Ejemplos</a> •
    <a href="https://github.com/zenc-lang/rfcs">RFCs</a> •
    <a href="CONTRIBUTING_ES.md">Contribuir</a>
  </p>
</div>
