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
  <h3>Ergonomia Moderna. Zero Overhead. C Puro.</h3>
  <br>
  <p>
    <a href="#"><img src="https://img.shields.io/badge/build-passing-brightgreen" alt="Stato Build"></a>
    <a href="#"><img src="https://img.shields.io/badge/license-MIT-blue" alt="Licenza"></a>
    <a href="#"><img src="https://img.shields.io/github/v/release/z-libs/Zen-C?label=versione&color=orange" alt="Versione"></a>
    <a href="#"><img src="https://img.shields.io/badge/platform-linux-lightgrey" alt="Piattaforma"></a>
  </p>
  <p><em>Comodità di un linguaggio ad alto livello, veloce come il C</em></p>
</div>

<hr>

<div align="center">
  <p>
    <b><a href="#panoramica">Panoramica</a></b> •
    <b><a href="#comunità">Comunità</a></b> •
    <b><a href="#guida-rapida">Guida Rapida</a></b> •
    <b><a href="#riferimento-del-linguaggio">Riferimento del Linguaggio</a></b> •
    <b><a href="#libreria-standard">Libreria Standard</a></b> •
    <b><a href="#tooling">Toolchain</a></b>
  </p>
</div>

---

## Panoramica

**Zen C** è un linguaggio di programmazione di sistemi moderno che genera codice `GNU C`/`C11`. Fornisce allo sviluppatore un ricco set di funzionalità, tra cui inferenza di tipo, pattern matching, generici, tratti, async/await, e gestione manuale della memoria con funzionalità RAII, mantenendo al contempo una compatibilità al 100% con l'ABI C

## Community

Unisciti alla conversazione, condividi delle demo, fai domande o segnala dei bug nel server ufficiale Discord Zen C

- Discord: [Unisciti qui](https://discord.com/invite/q6wEsCmkJP)

---

## Indice

<table>
  <tr>
    <td width="50%" valign="top">
      <h3>Generale</h3>
      <ul>
        <li><a href="#panoramica">Panoramica</a></li>
        <li><a href="#community">Community</a></li>
        <li><a href="#guida-rapida">Guida Rapida</a></li>
        <li><a href="#libreria-standard">Libreria Standard</a></li>
        <li><a href="#tooling">Tooling</a></li>
        <li><a href="#supporto-del-compilatore-e-compatibilità">Supporto del Compilatore</a></li>
        <li><a href="#contribuisci">Contribuisci</a></li>
        <li><a href="#attribuzioni">Attribuzioni</a></li>
      </ul>
    </td>
    <td width="50%" valign="top">
      <h3>Riferimenti del Linguaggio</h3>
      <ul>
        <li><a href="#1-variabili-e-costanti">1. Variabili e Costanti</a></li>
        <li><a href="#2-tipi-primitivi">2. Tipi Primitivi</a></li>
        <li><a href="#3-tipi-aggregati">3. Tipi Aggregati</a></li>
        <li><a href="#4-funzioni-e-lambda">4. Funzioni e Lambda</a></li>
        <li><a href="#5-controllo-di-flusso">5. Controllo di Flusso</a></li>
        <li><a href="#6-operatori">6. Operatori</a></li>
        <li><a href="#7-stampaggio-e-interpolazione-delle-stringhe">7. Stampaggio e Interpolazione</a></li>
        <li><a href="#8-gestione-della-memoria">8. Gestione della memoria</a></li>
        <li><a href="#9-programmazione-orientata-a-oggetti">9. Programmazione Orientata a Oggetti</a></li>
        <li><a href="#10-generici">10. Generici</a></li>
        <li><a href="#11-concorrenza-asincrona-asyncawait">11. Concorrenza</a></li>
        <li><a href="#12-metaprogramming">12. Metaprogramming</a></li>
        <li><a href="#13-attributi">13. Attributi</a></li>
        <li><a href="#14-assembly-inline">14. Assembly Inline</a></li>
        <li><a href="#15-direttive-della-build">15. Direttive della Build</a></li>
        <li><a href="#16-keyword">16. Keyword</a></li>
        <li><a href="#17-interoperabilità-c">17. Interoperabilità C</a></li>
      </ul>
    </td>
  </tr>
</table>

---

## Guida Rapida

### Installazione

```bash
git clone https://github.com/z-libs/Zen-C.git
cd Zen-C
make
sudo make install
```

### Build Portatile (APE)

Il codice Zen C può come un **Actually Portable Executable (APE)** (lett. _Eseguibile Effetivamente Portatile_) utilizzando la [Cosmopolitan Libc](https://github.com/jart/cosmopolitan). Ciò produrrà un singolo eseguibile (`.com`) che potrà essere eseguito nativamente su Linux, macOS, Windows, FreeBSD, OpenBSD e NetBSD sia sulle architetture x86_64 e aarch64.

**Prerequisiti:**
- Strumenti `cosmocc` (deve trovarsi nella tua PATH)

**Builda e Installa:**
```bash
make ape
sudo env "PATH=$PATH" make install-ape
```

**Artefatti:**
- `out/bin/zc.com`: Il compilatore Zen-C portatile. Inlude la libreria standard, incorporata nell'eseguibile.
- `out/bin/zc-boot.com`: Un installer bootstrap auto-contenuto per configurare nuovi progetti Zen-C rapidamente.

**Utilizzo:**
```bash
# Eseguibile su qualunque OS supportato
./out/bin/zc.com build hello.zc -o hello
```

### Utilizzo

```bash
# Compila e avvia
zc run hello.zc

# Builda eseguibile
zc build hello.zc -o hello

# Shell interattiva
zc repl
```

### Variabili d'ambiente

Puoi impostare `ZC_ROOT` per specificare la posizione della Libreria Standard (per inclusioni standard come `import "std/vector.zc"`). Ciò ti permetterà di eseguire il comando `zc` da qualsiasi directory.

```bash
export ZC_ROOT=/path/to/Zen-C
```

---

## Riferimenti Del Linguaggio

### 1. Variabili e Costanti

Zen C differenzia le costanti al tempo di compilazione e le variabili di esecuzione.

#### Costanti Manifesto (`def`)
Valori che esistono solo durante la compilazione (integrate nel codice). Utilizzale per le grandezze degli array, configurazioni fisse, e numeri magici.

```zc
def MAX_SIZE = 1024;
let buffer: char[MAX_SIZE]; // Grandezza valida per l'array
```

#### Variabili (`let`)
Locazioni di memoria. Possono essere mutabili o di sola lettura (`const`).

```zc
let x = 10;             // Mutabile
x = 20;                 // OK

let y: const int = 10;  // Sola lettura (Tipo qualificato)
// y = 20;              // Errore: impossibile assegnare un valore ad una variabile costante
```

> [!TIP]
> **Inferenza di tipo**: Zen C inferisce automaticamente il tipo per le variabili inizializzate. Compilando ciò alla keyword `auto` dello standard C23 nei compilatori supportati, oppure alla estensione GCC `__auto_type`.

### 2. Tipi Primitivi

| Tipo | C Equivalent | Descrizione |
|:---|:---|:---|
| `int`, `uint` | `int32_t`, `uint32_t` | Intero a 32 bit con segno/senza segno |
| `c_char`, `c_uchar` | `char`, `unsigned char` | C char (Interop) |
| `c_short`, `c_ushort` | `short`, `unsigned short` | C short (Interop) |
| `c_int`, `c_uint` | `int`, `unsigned int` | C int (Interop) |
| `c_long`, `c_ulong` | `long`, `unsigned long` | C long (Interop) |
| `I8` .. `I128` or `i8` .. `i128` | `int8_t` .. `__int128_t` | Interi a grandezza fissa con segno |
| `U8` .. `U128` or `u8` .. `u128` | `uint8_t` .. `__uint128_t` | Interi a grandezza fissa senza segno |
| `isize`, `usize` | `ptrdiff_t`, `size_t` | Interi con grandezza di un puntatore |
| `byte` | `uint8_t` | Alias per U8 |
| `F32`, `F64` or `f32`, `f64`  | `float`, `double` | Numeri con parte decimale |
| `bool` | `bool` | `true` (lett. _vero_) o `false` (lett. _falso_) |
| `char` | `char` | Carattere singolo |
| `string` | `char*` | Stringhe C terminate da NULL |
| `U0`, `u0`, `void` | `void` | Tipo vuoto |
| `iN` (Per esempio, `i256`) | `_BitInt(N)` | Intero con segno a larghezza arbitraria di bit (C23) |
| `uN` (Per esempio, `u42`) | `unsigned _BitInt(N)` | Intero senza segno a larghezza arbitraria di bit (C23) |

> [!IMPORTANT]
> **Best Practice per Codice Portabile**
>
> - Usa **Tipi Portabili** (`int`, `uint`, `i64`, `u8`, ecc.) per tutta la logica Zen C pura. `int` è garantito essere a 32-bit con segno su tutte le architetture.
> - Usa **Tipi di Interop C** (`c_int`, `c_char`, `c_long`) **solo** quando interagisci con librerie C (FFI). La loro dimensione varia in base alla piattaforma e al compilatore C.
> - Usa `isize` e `usize` per indicizzazione di array e aritmetica dei puntatori.

### 3. Tipi Aggregati

#### Array
Array a lunghezza fissa con valori arbitrari.
```zc
def GRANDEZZA = 5;
let interi: int[GRANDEZZA] = [1, 2, 3, 4, 5];
let zeri: [int; GRANDEZZA]; // Inizializzato a zero
```

#### Tuple
Valori molteplici raggruppati assieme, accesso agli elementi indicizzato.
```zc
let paio = (1, "Ciao!");
let x = paio.0;  // 1
let s = paio.1;  // "Ciao!"
```

**Molteplici Valori di Ritorno**

Le funzioni posso restituire delle tuple per fornire diversi risultati:
```zc
fn somma_e_differenza(a: int, b: int) -> (int, int) {
    return (a + b, a - b);
}

let risultato = somma_e_differenza(3, 2);
let somma = risultato.0;   // 5
let differenza = risultato.1;  // 1
```

**Separazione**

Le tuple possono essere separate direttamente in variabili singole.
```zc
let (somma, differenza) = somma_e_differenza(3, 2);
// somma = 5, differenza = 1
```

La separazione delle tuple tipizzata permette annotazioni di tipo esplicite:
```zc
let (a: string, b: u8) = ("hello", 42);
let (x, y: i32) = (1, 2);  // Misto: x inferito, y esplicito
```

#### Structs
Strutture dati con campi di bit opzionali.
```zc
struct Punto {
    x: int;
    y: int;
}

// Inizializzazione struct
let p = Punto { x: 10, y: 20 };

// Campi di bit
struct Flags {
    valido: U8 : 1;
    modalità:  U8 : 3;
}
```

> [!NOTE]
> Gli struct usano le [Semantiche di Spostamento](#semantiche-di-movimento--copia-sicura) di default. I campi di uno struct possono essere acceduti via `.` anche sui puntatori (Dereferenza-Automatica).

#### Struct Opachi
Puoi definire uno struct come `opaque` (lett. _opaco_) per restringere l'accesso ai suoi campi al modulo che lo ha definito, permettendo comunque l'allocazione sullo stack dello struct (la grandezza è data).

```zc
// In utente.zc
opaque struct Utente {
    id: int;
    nome: string;
}

fn nuovo_utente(nome: string) -> Utente {
    return Utente{id: 1, nome: nome}; // OK: Dentro il modulo
}

// In main.zc
import "utente.zc";

fn main() {
    let u = nuovo_utente("Alice");
    // let id = u.id; // Error: Impossibile accedere al campo privato 'id'
}
```

#### Enum
Unioni taggate (tipi somma) capaci di contenere dati.
```zc
enum Forma {
    Cerchio(float),           // Contiene il raggio
    Rettangolo(float, float), // Contiene la larghezza e l'altezza
    Punto                     // Non contiene dati
}
```

#### Unioni
Unioni standard C (accesso non sicuro).
```zc
union Dati {
    i: int;
    f: float;
}
```

#### Alias del tipo
Crea un alias per un tipo già esistente.
```zc
alias ID = int;
alias PuntoDellaMappa = Mappa<string, Punto>;
```

#### Alias del tipo opachi
Puoi definire un alias del tipo come `opaque` (lett. _opaco_) per creare un nuovo tipo che si distingue dal suo tipo sottostante al di fuori del modulo che l'ha definito. Questo permette una forte incapsulamento e sicurezza dei tipi senza overhead extra durante l'esecuzione di un wrapper struct.

```zc
// In libreria.zc
opaque alias Handle = int;

fn crea_handle(v: int) -> Handle {
    return v; // Conversione implicita consentita all'interno del modulo
}

// In main.zc
import "libreria.zc";

fn main() {
    let h: Handle = crea_handle(42);
    // let i: int = h; // Errore: Validazione del tipo fallita
    // let h2: Handle = 10; // Errore: Validazione del tipo fallita
}
```

### 4. Funzioni e Lambda

#### Funzioni
```zc
fn somma(a: int, b: int) -> int {
    return a + b;
}

// Supporto per argomenti nominati nelle chiamate
somma(a: 10, b: 20);
```

> **Nota**: Gli argomenti nominati devono seguire rigorosamente l'ordine predefinito dei parametri. `somma(b: 20, a: 10)` è errato.

#### Argomenti Costanti
Gli argomenti di una funzione possono essere marcati come `const` (lett. _costanti_) per reinforzare semantiche di sola lettura. Questo è un qualificatore del tipo, non una costante esplicita.

```zc
fn stampa_valore(v: const int) {
    // v = 10; // Errore: Impossibile assegnare un valore ad una variabile costante
    println "{v}";
}
```

#### Argomenti di default
Le funzioni posso definire dei valori default per gli argomenti in caso che questi non vengano specificati durante la chiamata. Questi valori possono essere letterali, espressioni, o codice Zen C valido (come il costruttore di uno struct).
```zc
// Valore default semplice
fn incrementa(val: int, quantità: int = 1) -> int {
    return val + quantità;
}

// Espressione come valore default (calcolato)
fn offset(val: int, pad: int = 10 * 2) -> int {
    return val + pad;
}

// Struct come valore default
struct Config { debug: bool; }
fn init(cfg: Config = Config { debug: true }) {
    if cfg.debug { println "Modalità Debug"; }
}

fn main() {
    incrementa(10);      // 11
    offset(5);          // 25
    init();             // Stampa "Modalità Debug"
}
```

#### Lambda (Closure)
Funzioni anonime che possono catturare il loro ambiente.
```zc
let fattore = 2;
let double = x -> x * fattore;  // Sintassi con freccia
let pieno = fn(x: int) -> int { return x * fattore; }; // Sintassi a blocco
```

#### Puntatori-Funzione grezzi
Zen C supporta i puntatori-funzione grezzi utilizzando la sintassi `fn*`. Questo permette un'interoperabilità fluida con le librerie C che si aspettano puntatori-funzione senza overhead di closure.

```zc
// Funzione che prende un puntatore-funzione grezzo
fn imposta_callback(cb: fn*(int)) {
    cb(42);
}

// Funzione che restituisce un puntatore-funzione grezzo
fn ottieni_callback() -> fn*(int) {
    return il_mio_handler;
}

// I puntatori a puntatori-funzione sono supportati (fn**)
let pptr: fn**(int) = &ptr;
```

#### Argomenti Variadici
Le funzioni possono accettare un numero variabile di argomenti utilizzando la sintassi `...` e il tipo `va_list`.
```zc
fn log(lvl: int, fmt: char*, ...) {
    let ap: va_list;
    va_start(ap, fmt);
    vprintf(fmt, ap); // Usa lo stdio C
    va_end(ap);
}
```

### 5. Controllo di Flusso

#### Condizionali
```zc
if x > 10 {
    print("Grande");
} else if x > 5 {
    print("Medio");
} else {
    print("Piccolo");
}

// Operatore ternario
let y = x > 10 ? 1 : 0; // Se x è maggiore di 10 y sarà uguale a 1, in ogni altro caso, y sarà uguale a 0

// If-Expression (per condizioni complesse)
let categoria = if (x > 100) { "enorme" } else if (x > 10) { "grande" } else { "piccolo" };
```

#### Pattern Matching
Alternativa potente agli `switch`.
```zc
match val {
    1         => { print "Uno" },
    2 || 3    => { print "Due o Tre" },           // OR logico con ||
    4 or 5    => { print "Quattro or Cinque" },   // OR logico con 'or'
    6, 7, 8   => { print "Da Sei a Otto" },       // OR logico con la virgola (,)
    10 .. 15  => { print "Da 10 a 14" },          // Range Esclusivo (Legacy)
    10 ..< 15 => { print "Da 10 a 14" },          // Range Esclusivo (Esplicito)
    20 ..= 25 => { print "Da 20 a 25" },          // Range Inclusivo
    _         => { print "Altro" },
}

// Destrutturazione degli Enums
match forma {
    Forma::Cerchio(r)        => { println "Raggio: {r}" },
    Forma::Rettangolo(w, h)  => { println "Area: {w*h}" },
    Forma::Punto             => { println "Punto" },
}
```

#### Associaione di riferiemnto
Per ispezionare un valore senza assumerne la proprietà (spostarlo) puoi usare la keyword `ref` nel pattern. Questo è essenziale per i tipi che implementano Semantiche di Movimento (come `Option`, `Result`, struct non-copiabile).

```zc
let opt = Qualche(ValoreNonCopiable{...});
match opt {
    Some(ref x) => {
        // 'x' è un puntatore che punta al valore contenuto in 'opt'
        // 'opt' NON viene né mosso né consumato qui
        println "{x.field}"; 
    },
    None => {}
}
```

#### Loops
```zc
// Range
for i in 0..10 { ... }      // Esclusivo (Da 0 a 9)
for i in 0..<10 { ... }     // Esclusivo (Esplicito)
for i in 0..=10 { ... }     // Inclusivo (Da 0 a 10)
for i in 0..10 step 2 { ... }

// Iteratore (Vec, Array, oppure un Iteratore personalizzato)
for item in collection { ... }

// While (lett. mentre)
while x < 10 { ... }

// Infinito con etichetta
esterno: loop {
    if done { break esterno; }
}

// Ripeti N volte
for _ in 0..5 { ... }
```

#### Controllo Avanzato
```zc
// Guard (lett. 'guardia'): Esegue il caso 'else' e ritorna se la condizione è falsa
guard ptr != NULL else { return; }

// Unless (lett. 'a meno che'): Se non vero
unless è_valido { return; }
```

### 6. Operatori

Zen C supporta l'overloading di operatori per gli struct definiti dall'utente per implementare nomi specifici di metodi.

#### Operatori Overload-abili

| Categoria | Operatore | Nome del Metodo |
|:---|:---|:---|
| **Aritmetico** | `+`, `-`, `*`, `/`, `%` | `add`, `sub`, `mul`, `div`, `rem` |
| **Paragone** | `==`, `!=` | `eq`, `neq` |
| | `<`, `>`, `<=`, `>=` | `lt`, `gt`, `le`, `ge` |
| **Bitwise** | `&`, `\|`, `^` | `bitand`, `bitor`, `bitxor` |
| | `<<`, `>>` | `shl`, `shr` |
| **Unari** | `-` | `neg` |
| | `!` | `not` |
| | `~` | `bitnot` |
| **Indice** | `a[i]` | `get(a, i)` |
| | `a[i] = v` | `set(a, i, v)` |

> **Nota sull'uguaglianza delle stringhe**:
> - `string == string` performa un controllo del **valore** (equivalente a `strcmp`).
> - `char* == char*` performa un controllo dei **puntatori** (controlla gli indirizzi di memoria).
> - Paragoni misti (e.g. `string == char*`) defaulta al controllo dei **pointer**.

**Esempio:**
```zc
impl Punto {
    fn add(self, altro: Punto) -> Punto {
        return Punto{x: self.x + altro.x, y: self.y + altro.y};
    }
}

let p3 = p1 + p2; // Chiama p1.somma(p2)
```

#### Zucchero Sintattico

Questi operatori sono funzionalità integrate del linguaggio e non è possibile overloadarli.

| Operatore | Nome | Descrizione |
|:---|:---|:---|
| `\|>` | Pipeline | `x \|> f(y)` viene dezuccherato a `f(x, y)` |
| `??` | Coalescenza nulla | `val ?? default` restituisce `default` se `val` è NULL (puntatori) |
| `??=` | Assegnazione nulla | `val ??= init` assegna se `val` è NULL |
| `?.` | Safe Navigation | `ptr?.campo` accede a 'campo' solo se `ptr` non è NULL |
| `?` | Try Operator | `res?` restituisce un errore se presente (tipi Result/Option) |

**Dereferenza Automatica**:
Pointer field access (`ptr.field`) and method calls (`ptr.method()`) automatically dereference the pointer, equivalent to `(*ptr).field`.
Accesso ai campi da un puntatore (`puntatore.campo`) e chiamate ai metodi (`puntatore.metodo()`) dereferenzano automaticamente il puntatore, ciò è equivalente a `(*puntatore).campo`

### 7. Stampaggio e Interpolazione delle Stringhe

Zen C fornisce opzioni versatili per stampare alla console, includendo keyword e scorciatoie coincise.

#### Keyword

| Keyword | Descrizione |
|:---|:---|
| `print "testo"` | Stampa a `stdout` senza aggiunzione di una newline automatica. |
| `println "testo"` | Stampa a `stdout` aggiungendo una newline automatica. |
| `eprint "testo"` | Stampa a `stderr` senza aggiunzione di una newline automatica. |
| `eprintln "testo"` | Stampa a `stderr` aggiungendo una newline automatica. |

#### Scorciatoie

Zen C ti permette di utilizzare stringhe letterali direttamente come istruzione di stampaggio veloce:

| Sintassi | Equivalente | Descrizione |
|:---|:---|:---|
| `"Ciao!"` | `println "Ciao!"` | Aggiunge una newline implicitamente. |
| `"Ciao!"..` | `print "Ciao!"` | Non aggiunge una newline. |
| `!"Errore"` | `eprintln "Errore"` | Output a stderr. |
| `!"Errore"..` | `eprint "Errore"` | Output a stderr, senza newline. |

#### Interpolazione delle Stringhe (F-strings)

Puoi incorporare espressioni direttamente all'interno di stringhe letterali utilizzando la sintassi `{}`. Questo funziona con tutti i metodi di stampaggio, incluse le scorciatoie.

```zc
let x = 42;
let nome = "Max";
println "Valore: {x}, Nome: {name}";
"Valore: {x}, Nome: {name}"; // scorciatoia per println
```

**Escape delle Parentesi Graffe**: Usa `{{` per produrre una parentesi graffa letterale `{` e `}}` per una `}` letterale:

```zc
let json = "JSON: {{\"chiave\": \"valore\"}}";
// Output: JSON: {"chiave": "valore"}
```

#### Prompt di Input (`?`)

Zen C supporta una scorciatoia per richiedere input dall'utente utilizzando il prefisso `?`.

- `? "Inserisci il tuo nome"`: Stampa il prompt (senza newline) e aspetta per dell'input (legge una linea).
- `? "Inserisci la tua età: " (età)`: Stampa il prompt e memorizza l'input nella variabile `età`.
    - Gli specificatori del formato vengono automaticamente inferiti in base al tipo della variabile.

```zc
let età: int;
? "Inserisci la tua età: " (età);
println "Hai {età} anni.";
```

### 8. Gestione della memoria

Zen C permette una gestione manuale della memoria con aiuti ergonomici.

#### Rimando
Esegui il codice quando l’ambito corrente termina. Le istruzioni defer vengono eseguite in ordine LIFO (last-in, first-out).
```zc
let f = fopen("file.txt", "r");
defer fclose(f);
```

> Per prevenire comportamenti indefiniti, le istruzioni del controllo di flusso (`return`, `break`, `continue`, `goto`) **non sono ammesse** dentro un blocco `defer`.

#### Liberazione automatica
Libera automaticamente la memoria occupata dalla variabile quando l'ambito corrente termina.
```zc
autofree let tipi = malloc(1024);
```

#### Semantiche delle risorse (Muovi di Default)
Zen C tratta i tipi con distruttori (come `File`, `Vec`, o puntatori allocati manualmente con `malloc`) come **Risorse**. Per prevenire errori di doppia-liberazione, le risorse non possono essere implicitamente duplicate.

- **Muovi di Default**: Assegnare una risorsa variabile ne trasferisce il proprietario. La variabile originale diventa invalida (Spostata).
- **Tipi di Copia**: Tipi senza distruttori possono opzionalmente avere un comportamento `Copy`, rendendo l'assegnazione una duplicazione.

**Diagnostica & Filosofia**:
Se vedi un errore "Utilizzo di una variabile spostata", il compilatore ti sta dicendo: *"Questo tipo è proprietario di una risorsa (come memoria o un handle) e copiarlo ciecamente non è sicuro."*

> **Contrasto:** Al contrario di come fanno C/C++, Zen C non duplica implicitamente i valori che posseggono risorse.

**Argomento di una funzione**:
Passare un valore ad una funzione segue le stesse regole dell'assegnazione: le risorse vengono spostate se non passate per referenza.

```zc
fn processo(r: Risorsa) { ... } // 'r' viene spostato nella funzione
fn peek(r: Risorsa*) { ... }    // 'r' viene preso in prestito (referenza)
```

**Clonazione Esplicita**:
Se *vuoi* avere più copie di una risorsa, rendilo esplicito:

```zc
let b = a.clona(); // Chiama il metodo `clona` dal tratto `Clone`
```

**Duplicazione opt-in (Tipi dei valori)**:
Per tipi piccoli senza distruttore:

```zc
struct Punto { x: int; y: int; }
impl Copy for Punto {} // Opt-in per la duplicazione implicita

fn main() {
    let p1 = Point { x: 1, y: 2 };
    let p2 = p1; // Copiato. p1 rimane valido.
}
```

#### RAII / Rilascio Tratti
Implementa `Drop` per una logica di pulizia automatica.
```zc
impl Drop for MioStruct {
    fn drop(self) {
        self.free();
    }
}
```

### 9. Programmazione Orientata a Oggetti

#### Metodi
Definisci metodi sui tipi utilizziando `impl`.
```zc
impl Punto {
    // Metodo statico (convenzione del costruttore)
    fn nuovo(x: int, y: int) -> Self {
        return Point{x: x, y: y};
    }

    // Metodo d'instanza
    fn dist(self) -> float {
        return sqrt(self.x * self.x + self.y * self.y);
    }
}
```

**Scorciatoia di Self**: Nei metodi con un parametro `self`, puoi usare `.campo` come abbreviazione per `self.campo`:
```zc
impl Point {
    fn dist(self) -> float {
        return sqrt(.x * .x + .y * .y);  // Equivalente a self.x, self.y
    }
}
```

#### Métodos primitivos
Zen C permette di definire metodi su tipi primitivi (come `int`, `bool`, etc.) usando la stessa sintassi `impl`.

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

#### Tratti
Definisci un comportamento condiviso.
```zc
struct Cerchio { raggio: f32; }

trait Disegnabile {
    fn disegna(self);
}

impl Disegna for Cerchio {
    fn disegna(self) { ... }
}

let cerchio = Cerchio{};
let disegnabile: Disegnabile = &cerchio;
```

#### Tratti Standard
Zen C include dei tratti standard che si integrano con la sintassi del linguaggio.

**Iterable** (lett. _Iterabile_)

Implementa `Iterable<T>` per abilitare loop `for-in` (lett. _per in_) nei tuoi tipi personalizzati.

```zc
import "std/iter.zc"

// Definisci un Iteratore
struct MioIteratore {
    curr: int;
    stop: int;
}

impl MioIteratore {
    fn next(self) -> Option<int> {
        if self.curr < self.stop {
            self.curr += 1;
            return Option<int>::Some(self.curr - 1);
        }
        return Option<int>::None();
    }
}

// Implementa Iterable
impl MioRange {
    fn iterator(self) -> MioIteratore {
        return MioIteratore{curr: self.start, stop: self.end};
    }
}

// Usalo in un loop
for i in mio_range {
    println "{i}";
}
```

**Drop** (lett. _rilascia_)

Implementa `Drop` per definire un distruttore che esegue quando l'oggetto va fuori ambito (RAII).

```zc
import "std/mem.zc"

struct Risorsa {
    ptr: void*;
}

impl Drop for Risorsa {
    fn drop(self) {
        if self.ptr != NULL {
            free(self.ptr);
        }
    }
}
```

> **Nota:** Se una variabile viene spostata, `drop` NON verrà chiamato sulla variabile originale. Aderisce alle [Semantiche delle Risorse](#semantiche-delle-risorse)

**Copy** (lett. _copia_)

Tratto marcatore opt-in per il comportamento `Copy` (duplicazione implicita) al posto delle semantiche Move. Utilizzato tramite `@derive(Copy)`

> **Regola:** I tipi che implementano `Copy` non dovrà definire un distruttore (`Drop`).

```zc
@derive(Copy)
struct Punto { x: int; y: int; }

fn main() {
    let p1 = Punto{x: 1, y: 2};
    let p2 = p1; // Copiato! p1 rimane valido.
}
```

**Clone** (lett. _clona_)

Implementa `Clone` per permettere la duplicazione esplicita di tipi che posseggono risorse.

```zc
import "std/mem.zc"

struct Scatola { val: int; }

impl Clone for Scatola {
    fn clone(self) -> Scatola {
        return Scatola{val: self.val};
    }
}

fn main() {
    let b1 = Scatola{val: 42};
    let b2 = b1.clone(); // Explicit copy
}
```

#### Composizione
Usa `use` per incorporare altri struct. Puoi mischiarli (campi piatti) o nominarli (campi nidificato).

```zc
struct Entità { id: int; }

struct Giocatore {
    // Mischiati (Non nominati): Campi piatti
    use Entità;  // Aggiunge 'id' a 'Giocatore' direttamente
    nome: string;
}

struct Partita {
    // Composizione (Nominati): Campi nidificati
    use p1: Giocatore; // Vi si accede tramite partita.p1
    use p2: Giocatore; // Vi si accede tramite partita.p2
}
```

### 11. Generici

Template type-safe per struct e funzioni.

```zc
// Struct Generico
struct Scatola<T> {
    oggetto: T;
}

// Funzione Generica
fn identità<T>(valore: T) -> T {
    return valore;
}

// Generici Multi-parametro
struct Paio<K, V> {
    chiavi: K;
    valore: V;
}
```

### 11. Concorrenza Asincrona (Async/Await)

Costruito sui pthreads.

```zc
async fn ottieni_dati() -> string {
    // Esegue in background
    return "Dati";
}

fn main() {
    let futuro    = ottieni_dati();
    let risultato = await futuro; // (lett. 'aspetta')
}
```

### 12. Metaprogramming

#### Comptime
Esegui codice al momento della compilazione per generare sorgente o stampare messaggi.
```zc
comptime {
    // Genera codice al momento della compilazione (scritto su stdout)
    println "let data_compilazione = \"2024-01-01\";";
}

println "Data compilazione: {data_compilazione}";
```

<details>
<summary><b>🔧 Funzioni Helper</b></summary>

Funzioni speciali disponibili all'interno dei blocchi `comptime`:

<table>
<tr>
<th>Funzione</th>
<th>Descrizione</th>
</tr>
<tr>
<td><code>yield(str)</code></td>
<td>Emette esplicitamente codice generato (alternativa a <code>printf</code>)</td>
</tr>
<tr>
<td><code>code(str)</code></td>
<td>Alias di <code>yield()</code> - intento più chiaro per generazione codice</td>
</tr>
<tr>
<td><code>compile_error(msg)</code></td>
<td>❌ Interrompe la compilazione con un messaggio di errore fatale</td>
</tr>
<tr>
<td><code>compile_warn(msg)</code></td>
<td>⚠️ Emette un avviso al momento della compilazione (consente di continuare)</td>
</tr>
</table>

**Esempio:**
```zc
comptime {
    compile_warn("Generazione codice ottimizzato...");
    
    let ENABLE_FEATURE = 1;
    if (ENABLE_FEATURE == 0) {
        compile_error("La funzionalità deve essere abilitata!");
    }
    
    // Usa code() con raw strings per generazione pulita
    code(r"let FEATURE_ENABLED = 1;");
}
```
</details>

<details>
<summary><b>📦 Metadati di Build</b></summary>

Accedi alle informazioni di build del compilatore al momento della compilazione:

<table>
<tr>
<th>Costante</th>
<th>Tipo</th>
<th>Descrizione</th>
</tr>
<tr>
<td><code>__COMPTIME_TARGET__</code></td>
<td>string</td>
<td>Piattaforma: <code>"linux"</code>, <code>"windows"</code> o <code>"macos"</code></td>
</tr>
<tr>
<td><code>__COMPTIME_FILE__</code></td>
<td>string</td>
<td>Nome del file sorgente corrente in compilazione</td>
</tr>
</table>

**Esempio:**
```zc
comptime {
    // Generazione codice specifico per piattaforma
    println "let PLATFORM = \"{__COMPTIME_TARGET__}\";";
}

println "In esecuzione su: {PLATFORM}";
```
</details>

> **💡 Suggerimento:** Usa raw strings (`r"..."`) in comptime per evitare di eseguire l'escape delle parentesi graffe: `code(r"fn test() { return 42; }")`. Altrimenti, usa `{{` e `}}` per l'escape nelle stringhe normali.


#### Incorporati
Incorpora file come tipi specificati.
```zc
// Default (Slice_char)
let data = embed "assets/logo.png";

// Incorporazioni tipizzate
let testo = embed "shader.glsl" as string;    // Incorpora come una stringa C
let rom   = embed "bios.bin" as u8[1024];     // Incorpora come un array a dimensione fissa
let wav   = embed "sound.wav" as u8[];        // Incorpora come Slice_u8
```

#### Plugin
Importa plugin del compilatore per estendere la sintassi.
```zc
import plugin "regex"
let re = regex! { ^[a-z]+$ };
```

#### Macro C Generiche
Passa delle macro del preprocessore C.

> **Consiglio**: Per delle semplici costanti, utilizza `def`. Usa `#define` solo quanto ti servono macro del preprocessore C o flag di compilazione condizionale.

```zc
#define BUFFER_MASSIMO 1024
```

### 13. Attributi

Decora le funzioni e gli struct per modificare il comportamento del compilatore.

| Attributo | Ambito | Descrizione |
|:---|:---|:---|
| `@must_use` | Fn | Avvereti se il valore di ritorno viene ignorato. |
| `@deprecated("msg")` | Fn/Struct | Avverti all'uso con 'msg' |
| `@inline` | Fn | Suggerisci al compilatore di rendere il codice inline |
| `@noinline` | Fn | Previeni l'inline automatico |
| `@packed` | Struct | Rimuovi il padding (lett. _imbottitura_) automatico in mezzo ai campi. |
| `@align(N)` | Struct | Forza l'allineamento a N byte. |
| `@constructor` | Fn | Esegui prima di `main`. |
| `@destructor` | Fn | Esegue dopo la terminazione di `main`. |
| `@unused` | Fn/Var | Sopprimi gli errori di 'variabile inutilizzata' |
| `@weak` | Fn | Linking dei simboli _weak_ (lett. _debole_). |
| `@section("name")` | Fn | Inserisci il codice in una specifica sezione. |
| `@noreturn` | Fn | La funzione non restituisce valori. (e.g. `exit`). |
| `@pure` | Fn | La funzione non ha effetti collaterali (indizio per l'ottimizzazione). |
| `@cold` | Fn | La funzione è usata poco spesso (indizio per la branch prediction). |
| `@hot` | Fn | La funzione è usata molto spesso (indizio per l'ottimizzazione). |
| `@export` | Fn/Struct | Esporta simbolo (visibilità default). |
| `@global` | Fn | CUDA: Entry point del Kernel (`__global__`). |
| `@device` | Fn | CUDA: Funzione del Device (`__device__`). |
| `@host` | Fn | CUDA: Funzione dell'Host (`__host__`). |
| `@comptime` | Fn | Funzione di supporto disponibile per l'esecuzione al tempo di compilazione. |
| `@derive(...)` | Struct | Implementa automaticamente i tratti. Supporta `Debug`, `Eq` (Derivazione Intelligente), `Copy`, `Clone`. |
| `@<custom>` | Any | Passa gli attributi generici direttamente al C (e.g. `@flatten`, `@alias("nome")`) |

#### Attributi Personalizzati

Zen C supporta un potente sistema di **Attributi Personalizzati** che ti permettono di utilizzare ogni `__attributo__` GCC/Clang direttamente nel tuo codice Zen C. Qualsiasi attributo non riconosciuto dal compilatore Zen C viene trattato come un attributo generico e passato direttamente nel codice C generato.

Ciò fornisce accesso a delle avanzate funzionalità, ottimizzazioni e direttive del linker senza necessitare di un supporto esplicito nel cuore del linguaggio.

#### Mappatura della Sintassi
Zen C attributes are mapped directly to C attributes:
- `@name` → `__attribute__((name))`
- `@name(args)` → `__attribute__((name(args)))`
- `@name("string")` → `__attribute__((name("string")))`

#### Derivazioni Intelligenti

Zen C fornisce delle "derivazioni intelligenti" che rispettano le Semantiche di Movimento:

- **`@derive(Eq)`**: Genera un metodo di uguaglianza che prende argomenti per referenza (`fn eq(self, other: T*)`).
    - Quando si confrontano due struct non-Copy (`a == b`), il compilatore passa automaticamente `b` per referenza (`&b`) per non doverlo spostare.
    - I controlli di uguaglianza ricorsivi preferiscono l'accesso da puntatore per prevenire il trasferimento del proprietario.

### 14. Assembly Inline

Zen C fornisce supporto di prima-classe per l'assembly _inline_, traspilando direttamente ad `asm` con estensioni in stile GCC.

#### Utilizzo Base
Scrivi assembly grezzo all'interno di blocchi `asm`. Le stringhe vengono concatenate automaticamente.
```zc
asm {
    "nop"
    "mfence"
}
```

#### Volatile
Impedisci al compilatore di eliminare automaticamente istruzioni assembly (e.g. ottimizzazione) se ciò potrebbe avere ripercussioni.
```zc
asm volatile {
    "rdtsc"
}
```

#### Vincoli Nominati
Zen C semplifica la sintassi complessa dei vincoli di GCC con dei binding nominati.

**Nota per i lettori italiani**: Con 'clobber' si intende la *sovrascrizione*.

```zc
// Sintassi: : out(variable) : in(variable) : clobber(reg)
// Usa una sintassi placeholder (`{variabile}`) per la leggibilità

fn aggiungi_cinque(x: int) -> int {
    let risultato: int;
    asm {
        "mov {x}, {risultato}"
        "add $5, {risultato}"
        : out(risultato)
        : in(x)
        : clobber("cc")
    }
    return risultato;
}
```

| Tipo | Sintassi | Equivalente GCC |
|:---|:---|:---|
| **Output** | `: out(variabile)` | `"=r"(variabile)` |
| **Input** | `: in(variabile)` | `"r"(variabile)` |
| **Clobber** | `: clobber("rax")` | `"rax"` |
| **Memory** | `: clobber("memoria")` | `"memoria"` |

> **Nota:** Quando si usa la sintassi Intel (via `-masm=intel`), dovrai assicurarti che la tua build sia configurata correttamente (per esempio, `//> cflags: -masm=intel`). TCC non supporta la sintassi assembly Intel.


### 15. Direttive della Build

Zen C supporta dei commenti speciali all'inizio del tuo file sorgente che ti permettono di configurare il processo di build senza necessitare di un sistema di build complesso o di un *Makefile*.

| Direttiva | Argomenti | Descrizione |
|:---|:---|:---|
| `//> link:` | `-lfoo` oppure `path/to/lib.a` | Linka con una libreria o un file object. |
| `//> lib:` | `path/to/libs` | Aggiunge una directory dove cercare le librerie (`-L`). |
| `//> include:` | `path/to/headers` | Aggiunge una directory dove cercare i file include (`-I`). |
| `//> framework:` | `Cocoa` | Linka con un framework macOS. |
| `//> cflags:` | `-Wall -O3` | Passa flag arbitrare al compilatore C. |
| `//> define:` | `MACRO` or `KEY=VAL` | Definisci una macro del preprocessore (`-D`). |
| `//> pkg-config:` | `gtk+-3.0` | Esegui `pkg-config` e aggiungi `--cflags` e `--libs`. |
| `//> shell:` | `command` | Esegui un comando sulla shell durante il processo di build. |
| `//> get:` | `http://url/file` | Scarica un file se un file specifico non esiste. |

#### Feature

**1. OS Guarding** (lett. _Protezione OS_)
Prefissa delle direttive con il nome di un OS per applicarle solo su piattaforme specifiche.
Prefissi supportati: `linux:`, `windows:`, `macos:` (or `darwin:`).

```zc
//> linux: link: -lm
//> windows: link: -lws2_32
//> macos: framework: Cocoa
```

**2. Environment Variable Expansion**
Utilizza la sintassi `${VAR}` per espandare variabili d'ambiente nelle tue direttive.

```zc
//> include: ${HOME}/MiaLibreria/include
//> lib: ${ZC_ROOT}/std
```

#### Esempi

```zc
//> include: ./include
//> lib: ./librerie
//> link: -lraylib -lm
//> cflags: -Ofast
//> pkg-config: gtk+-3.0

import "raylib.h"

fn main() { ... }
```

### 16. Keyword

Le keyword che seguono sono riservate in Zen C.

#### Dichiarazioni
`alias`, `def`, `enum`, `fn`, `impl`, `import`, `let`, `module`, `opaque`, `struct`, `trait`, `union`, `use`

#### Controllo del Flusso
`async`, `await`, `break`, `catch`, `continue`, `defer`, `else`, `for`, `goto`, `guard`, `if`, `loop`, `match`, `return`, `try`, `unless`, `while`

#### Speciali
`asm`, `assert`, `autofree`, `comptime`, `const`, `embed`, `launch`, `ref`, `sizeof`, `static`, `test`, `volatile`

#### Costanti
`true`, `false`, `null`

#### Riservate del C
Gli identifiers seguenti sono riservati poiché sono keyword nello standard C11:
`auto`, `case`, `char`, `default`, `do`, `double`, `extern`, `float`, `inline`, `int`, `long`, `register`, `restrict`, `short`, `signed`, `switch`, `typedef`, `unsigned`, `void`, `_Atomic`, `_Bool`, `_Complex`, `_Generic`, `_Imaginary`, `_Noreturn`, `_Static_assert`, `_Thread_local`

#### Operatori
`and`, `or`

### 17. Interoperabilità C
Zen C offre due modi per interagire con il codice C: **Import Trusted** (Conveniente) e **FFI Esplicita** (Sicuro/Preciso).

#### Metodo 1: Import Trusted (Conveniente)
Puoi importare un header C direttamente usando la parola chiave `import` con l'estensione `.h`. Questo tratta l'header come un modulo e assume che tutti i simboli acceduti esistano.

```zc
//> link: -lm
import "math.h" as c_math;

fn main() {
    // Il compilatore si fida della correttezza; emette 'cos(...)' direttamente
    let x = c_math::cos(3.14159);
}
```

> **Pro**: Zero boilerplate. Accesso immediato a tutto nell'header.
> **Contro**: Nessuna sicurezza dei tipi da Zen C (errori catturati dal compilatore C dopo).

#### Metodo 2: FFI Esplicita (Sicuro)
Per un controllo rigoroso dei tipi o quando non vuoi includere il testo di un header, usa `extern fn`.

```zc
include <stdio.h> // Emette #include <stdio.h> nel C generato

// Definisci firma rigorosa
extern fn printf(fmt: char*, ...) -> c_int;

fn main() {
    printf("Ciao FFI: %d\n", 42); // Controllato nei tipi da Zen C
}
```

> **Pro**: Zen C assicura che i tipi corrispondano.
> **Contro**: Richiede dichiarazione manuale delle funzioni.

#### `import` vs `include`

- **`import "file.h"`**: Registra l'header come un modulo con nome. Abilita l'accesso implicito ai simboli (es. `file::function()`).
- **`include <file.h>`**: Emette puramente `#include <file.h>` nel codice C generato. Non introduce alcun simbolo nel compilatore Zen C; devi usare `extern fn` per accedervi.


---

## Libreria Standard

Zen C include una libreria standard (`std`) che ricopre funzionalità essenziali.

[Scopri la documentazione della Libreria Standard](../docs/std/README.md)

### Moduli Chiave

<details>
<summary>Clicca per vedere tutti i moduli della Libreria Standard</summary>

| Modulo | Descrizione | Documentazione |
| :--- | :--- | :--- |
| **`std/vec.zc`** | Array dinamico espandibile `Vec<T>`. | [Docs](../docs/std/vec.md) |
| **`std/string.zc`** | Tipo `String` allocato sull'Heap con supporto UTF-8. | [Docs](../docs/std/string.md) |
| **`std/queue.zc`** | Coda FIFO (Buffer Circolare). | [Docs](../docs/std/queue.md) |
| **`std/map.zc`** | Hash Map Generica `Map<V>`. | [Docs](../docs/std/map.md) |
| **`std/fs.zc`** | Operazioni del File System. | [Docs](../docs/std/fs.md) |
| **`std/io.zc`** | Standard Input/Output (`print`/`println`). | [Docs](../docs/std/io.md) |
| **`std/option.zc`** | Valori opzionali (`Some`/`None`). | [Docs](../docs/std/option.md) |
| **`std/result.zc`** | Gestione degli errori (`Ok`/`Err`). | [Docs](../docs/std/result.md) |
| **`std/path.zc`** | Manipolazione dei percorsi Cross-platform. | [Docs](../docs/std/path.md) |
| **`std/env.zc`** | Variabili d'ambiente del processo. | [Docs](../docs/std/env.md) |
| **`std/net/`** | TCP, UDP, HTTP, DNS, URL. | [Docs](../docs/std/net.md) |
| **`std/thread.zc`** | Thread e Sincronizzazione. | [Docs](../docs/std/thread.md) |
| **`std/time.zc`** | Misuramenti di tempo e `sleep`. | [Docs](../docs/std/time.md) |
| **`std/json.zc`** | Parsing JSON e serializzazione. | [Docs](../docs/std/json.md) |
| **`std/stack.zc`** | Stack LIFO `Stack<T>`. | [Docs](../docs/std/stack.md) |
| **`std/set.zc`** | Hash Set Generico `Set<T>`. | [Docs](../docs/std/set.md) |
| **`std/process.zc`** | Esecuzione e gestione di processi. | [Docs](../docs/std/process.md) |

</details>

---

## Tooling

Zen C fornisce un Language Server (LSP) e un REPL per migliorare l'esperienza degli sviluppatori.

### Language Server (LSP)

Il server del linguaggio (LSP) di Zen C supporta le feature standard per l'integrazione con gli editor, esso fornisce:

*   **Vai alla definizione**
*   **Trova riferimenti**
*   **Informazioni sull'hover**
*   **Completamenti automatici** (Nomi di funzioni/struct, Completamento dal punto per i methods/campi)
*   **Simboli dei documenti** (Outline)
*   **Aiuto con le signature delle funzioni**
*   **Diagnostiche** (Errori sintattici/semantici)

Per avviare il server del linguaggio (tipicamente configurato nelle impostazioni LSP del tuo editor):

```bash
zc lsp
```

Il server comunica via lo Standard I/o (JSON-RPC 2.0).

### REPL

Il Read-Eval-Print-Loop (REPL, lett. _Leggi-Esegui-Stampa-Ripeti_) ti permette ti sperimentare con il codice Zen C in maniera interattiva.

```bash
zc repl
```

#### Funzionalità

*   **Coding interattivo**: Scrivi espressioni o istruzioni per una esecuzione immediata.
*   **Storia persistente**: I comandi vengono salvati in `~/.zprep_history`.
*   **Script di avvio**: I comandi di avvio (auto-load) sono salvati in `~/.zprep_init.zc`.

#### Comandi

| Comande | Descrizione |
|:---|:---|
| `:help` | Mostra i comandi disponibili. |
| `:reset` | Cancella la storia della sessione corrente (variabili/funzioni). |
| `:vars` | Mostra le variabili attive. |
| `:funcs` | Mostra le funzioni definite dall'utente. |
| `:structs` | Mostra gli struct definiti dall'utente. |
| `:imports` | Mostra gli 'import' attivi. |
| `:history` | Mostra la storia dell'input della sessione. |
| `:type <expr>` | Mostra il tipo di un espressione. |
| `:c <stmt>` | Mostra il codice C generato per un istruzione. |
| `:time <expr>` | Esegui un benchmark per l'espressione data. (Esegue 1000 iterazioni). |
| `:edit [n]` | Modifica il comando `n` (default: l'ultimo comando) in `$EDITOR`. |
| `:save <file>` | Salva la sessione corrente in un file `.zc`. |
| `:load <file>` | Carica ed esegui un file `.zc` nella sessione corrente. |
| `:watch <expr>` | Watch (lett. _guarda_) un espressione (rieseguita dopo ogni entry). |
| `:unwatch <n>` | Rimuovi un watch. |
| `:undo` | Rimuovi l'ultimo comando dalla sessione. |
| `:delete <n>` | Rimuovi il comando all'indice `n`. |
| `:clear` | Pulisce lo schermo. |
| `:quit` | Esce dal REPL. |
| `! <cmd>` | Esegue un comando sulla shell (e.g. `!ls`). |

---


## Supporto del Compilatore e Compatibilità

Zen C è stato creato in modo tale da poter funzionare con la maggior parte dei compilatori C11. Alcune funzionalità potrebbero affidarsi ad estensioni GNU C,  ma spesso queste funzionano anche su altri compilatori. Utilizza la flag `--cc` per modificare il backend.

```bash
zc run app.zc --cc clang
zc run app.zc --cc zig
```

### Stato della suite di test

<details>
<summary>Clicca per vedere i dettagli del supporto del compilatore</summary>

| Compilatore | Percentuale di Superamento | Funzionalità Supportate | Limitazioni Nota |
|:---|:---:|:---|:---|
| **GCC** | **100% (Completo)** | Tutte le funzioni. | Nessuna. |
| **Clang** | **100% (Completo)** | Tutte le funzioni. | Nessuna. |
| **Zig** | **100% (Completo)** | Tutte le funzioni. | Nessuna. Usa `zig cc` come compilatore C _drop-in_. |
| **TCC** | **~70% (Base)** | Sintaxis base, Generici, Tratti | Nessun `__auto_type`, Nessuna sintassi Intel per ASM, Nessuna funzione innestata. |

</details>

> [!TIP]
> **Consiglio:** Usag **GCC**, **Clang**, o **Zig** per le build di produzione. TCC è eccellente per prototipazione rapida per via dei suoi tempi di compilazione molto brevi ma è carente di alcune estensioni C avanzate che servono a Zen C per un set completo di funzionalità.

### Buildare con Zig

Il comando `zig cc` di Zig fornisce un rimpiazzamento drop-in per GCC/Clang con eccellente supporto per la cross-compilation. Per usare Zig:

```bash
# Compila ed esegui un programma Zen C con Zig
zc run app.zc --cc zig

# Puoi compilare persino il compilatore Zen C stesso con Zig
make zig
```

### Interop C++

Zen C può generare codice compatibile con C++ utilizzando l'opzione `--cpp`, permettendo una integrazione fluida con le librerie C++.

```bash
# Compilazione diretta con g++
zc app.zc --cpp

# O traspila per le build manuali
zc transpile app.zc --cpp
g++ out.c my_cpp_lib.o -o app
```

#### Usare C++ in Zen C

Includi header C++ e usa blocchi grezzi per codice C++:

```zc
include <vector>
include <iostream>

raw {
    std::vector<int> crea_vettore(int a, int b) {
        return {a, b};
    }
}

fn main() {
    let v = crea_vettore(1, 2);
    raw { std::cout << "Dimensione: " << v.size() << std::endl; }
}
```

> **Nota:** L'opzione `--cpp` rende il backend `g++` ed emette codice valido per C++ (utilizza `auto` al posto di `__auto_type`, overload delle funzioni al posto di `_Generic` e i cast espliciti per `void*`)

#### Interop CUDA

Zen C supporta la programmazione GPU traspilando a **CUDA C++**. Questo ti permette di utilizzare potenti funzionalità C++ (template, `constexpr`) all'interno dei tuoi kernel mantenendo la sintassi ergonomica di Zen C.

```bash
# Compilazione diretta con nvcc
zc run app.zc --cuda

# O traspila per le build manuali
zc transpile app.zc --cuda -o app.cu
nvcc app.cu -o app
```

#### Attributi specifici CUDA

| Attributo | Equivalente CUDA | Descrizione |
|:---|:---|:---|
| `@global` | `__global__` | Function Kernel (esegue sulla GPU, chiamato dall'host) |
| `@device` | `__device__` | Funzione Device (esegue sulla GPU, chiamato dalla GPU) |
| `@host` | `__host__` | Funzione Host (Solo CPU esplicita) |

#### Kernel Launch Syntax

Zen C fornisce un'istruzione chiara `launch` per richiamare kernel CUDA:

```zc
launch kernel_name(args) with {
    grid: num_blocks,
    block: threads_per_block,
    shared_mem: 1024,  // Opzionale
    stream: my_stream   // Opzionale
};
```

Questo traspila a: `kernel_name<<<grid, block, shared, stream>>>(args);` 

#### Scrivere kernel CUDA

Utilizza la sintassi delle funzioni Zen C con `@global` e l'istruzione `launch`:

```zc
import "std/cuda.zc"

@global
fn aggiungi_kernel(a: float*, b: float*, c: float*, n: int) {
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
    
    launch aggiungi_kernel(d_a, d_b, d_c, N) with {
        grid: (N + 255) / 256,
        block: 256
    };
    
    cuda_sync();
}
```

#### Libreria Standard (`std/cuda.zc`)
Zen C fornisce una libreria standard per delle operazioni comuni in CUDA per ridurre la mole di blocchi `raw` (grezzi):

```zc
import "std/cuda.zc"

// Gestione della memoria
let d_ptr = cuda_alloc<float>(1024);
cuda_copy_to_device(d_ptr, h_ptr, 1024 * sizeof(float));
defer cuda_free(d_ptr);

// Sincronizzazione
cuda_sync();

// Indicizzazione dei thread (usa all'interno del kernel)
let i = thread_id(); // Indice globale
let bid = block_id();
let tid = local_id();
```


> [!NOTE]
> **Nota:** La flag `--cuda` imposta `nvcc` come compilatore e implica la modalità `--cpp`. Richiede l'installazione dell'NVIDIA CUDA Toolkit.

### Supporto C23

Zen C supporta le funzionalità moderne dello standard C23 quando si usa un backend compatibile (GCC 14+, Clang 14+, _TCC_ (_parziale_)).

- **`auto`**: Zen C mappa automaticamente l'inferenza del tipo alla keyword `auto` di C23 (se `__STDC_VERSION__ >= 202300L`).
- **`_BitInt(N)`**: Usa i tipi `iN` e `uN` (e.g., `i256`, `u12`, `i24`) per accedere agli interi di lunghezza arbitraria di C23.

### Interop Objective-C

Zen C può compilare a Objective-C (`.m`) utilizzando la flag `--objc`, permettendoti di utilizzare i framework (come Cocoa/Foundation) e la sintassi Obj-C

```bash
# Compila con clang (o gcc/gnustep)
zc app.zc --objc --cc clang
```

#### Usando l'Objective-C in Zen C

Utilizza `include` per gli header e i blocchi `raw` per la sintassi Obj-C (`@interface`, `[...]`, `@""`).

```zc
//> macos: framework: Foundation
//> linux: cflags: -fconstant-string-class=NSConstantString -D_NATIVE_OBJC_EXCEPTIONS
//> linux: link: -lgnustep-base -lobjc

include <Foundation/Foundation.h>

fn main() {
    raw {
        NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
        NSLog(@"Ciao da Objective-C!");
        [pool drain];
    }
    println "Funziona anche Zen C!";
}
```

> [!NOTE]
> **Nota:** L'interpolazione delle stringhe di Zen C funziona con gli oggetti dell'Objective-C (`id`) chiamando `debugDescription` oppure `description`.

---

## Contribuisci

Qui accogliamo tutti i contributi! Che siano fix di bug, miglioramenti alla documentazione, o la proposta di nuove funzionalità.

Per favore, consulta [CONTRIBUTING_IT.md](CONTRIBUTING_IT.md) per le linee guida dettagliate su come contribuire, eseguire i test e inviare pull request.

---

## Sicurezza

Per istruzioni sulla segnalazione di vulnerabilità, vedi [SECURITY_IT.md](SECURITY_IT.md).

---

## Attribuzioni

Questo progetto utilizza librerie esterne. I testi di licenza completi possono essere trovati nella directory `LICENSES/`.

* **[cJSON](https://github.com/DaveGamble/cJSON)** (Licenza MIT): Usato per il parsing e la generazione di JSON nel Language Server.
* **[zc-ape](https://github.com/OEvgeny/zc-ape)** (Licenza MIT): La versione originale di Actually Portable Executable di Zen-C, realizzata da [Eugene Olonov](https://github.com/OEvgeny).
* **[Cosmopolitan Libc](https://github.com/jart/cosmopolitan)** (Licenza ISC): La libreria fondamentale che rende possibile APE.

---

<div align="center">
  <p>
    Copyright © 2026 Zen C Programming Language.<br>
    Inizia il tuo viaggio oggi.
  </p>
  <p>
    <a href="https://discord.com/invite/q6wEsCmkJP">Discord</a> •
    <a href="https://github.com/z-libs/Zen-C">GitHub</a> •
    <a href="CONTRIBUTING_IT.md">Contribuisci</a>
  </p>
</div>

