<div align="center">
  <p>
    <a href="../README.md">English</a> •
    <a href="README_DE.md">Deutsch</a> •
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
  <h3>Moderne Ergonomie. Null Overhead. Pures C.</h3>  
  <br>  
  <p>  
	<a href="#"><img src="https://img.shields.io/badge/build-passing-brightgreen" alt="Build Status"></a>  
	<a href="#"><img src="https://img.shields.io/badge/license-MIT-blue" alt="Lizenz"></a>  
	<a href="#"><img src="https://img.shields.io/github/v/release/zenc-lang/zenc?label=version&color=orange" alt="Version"></a>  
	<a href="#"><img src="https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macos-lightgrey" alt="Plattform"></a>  
  </p>  
  <p><em>Schreiben wie in einer Hochsprache, ausführen wie in C.</em></p>  
</div>  

<hr>  

<div align="center">  
  <p>  
	<b><a href="#übersicht">Übersicht</a></b> •  
	<b><a href="#community">Community</a></b> •  
	<b><a href="#schnellstart">Schnellstart</a></b> •  
	<b><a href="#ökosystem">Ökosystem</a></b> •  
	<b><a href="#sprachreferenz">Sprachreferenz</a></b> •  
	<b><a href="#standardbibliothek">Standardbibliothek</a></b> •  
	<b><a href="#tooling">Tooling</a></b>  
  </p>  
</div>  

---

## Übersicht

**Zen C** ist eine moderne Systemprogrammiersprache, die zu menschenlesbarem `GNU C`/`C11` kompiliert. Es bietet einen reichhaltigen Funktionsumfang, darunter Typinferenz, Pattern Matching, Generics, Traits, Async/Await und manuelles Speichermanagement mit RAII-Fähigkeiten – und das alles bei 100%iger C-ABI-Kompatibilität.

## Community

Diskutiere mit, teile Demos, stelle Fragen oder melde Fehler auf dem offiziellen Zen C Discord-Server!

- Discord: [Hier beitreten](https://discord.com/invite/q6wEsCmkJP)
- RFCs: [Features vorschlagen](https://github.com/zenc-lang/rfcs)

## Ökosystem

Das Zen C-Projekt besteht aus mehreren Repositories:

| Repository | Beschreibung | Status |
| :--- | :--- | :--- |
| **[zenc](https://github.com/zenc-lang/zenc)** | Der Kern-Compiler (zc), CLI und Standardbibliothek. | Aktive Entwicklung |
| **[docs](https://github.com/zenc-lang/docs)** | Offizielle Dokumentation und Spezifikation. | Aktiv |
| **[rfcs](https://github.com/zenc-lang/rfcs)** | Request for Comments (RFCs). Gestalte die Zukunft mit. | Aktiv |
| **[vscode-zenc](https://github.com/zenc-lang/vscode-zenc)** | Offizielle VS Code Erweiterung. | Alpha |
| **[www](https://github.com/zenc-lang/www)** | Quellcode für zenc-lang.org.| Aktiv |
| **[awesome-zenc](https://github.com/zenc-lang/awesome-zenc)** | Eine sorgfältig zusammengestellte Liste großartiger Zen C-Beispiele. | Wachsend |
| **[zenc.vim](https://github.com/zenc-lang/zenc.vim)** | Offizielles Vim/Neovim-Plugin (Syntax, Einrückung). | Aktiv |

## Showcase

Projekte, die mit Zen C erstellt wurden:

- **[ZC-pong-3ds](https://github.com/5quirre1/ZC-pong-3ds)**: Ein Pong-Klon für den Nintendo 3DS.
- **[zen-c-parin](https://github.com/Kapendev/zen-c-parin)**: Ein einfaches Beispiel mit Zen C und Parin.
- **[almond](https://git.sr.ht/~leanghok/almond)**: Ein minimaler Webbrowser in Zen C.

---

## Index

<table align="center">  
<tr>  
	<th width="50%">Allgemeines</th>  
	<th width="50%">Sprachreferenz</th>  
  </tr>  
  <tr>  
	<td valign="top">  
	  <ul>  
		<li><a href="#übersicht">Übersicht</a></li>  
		<li><a href="#community">Community</a></li>  
		<li><a href="#schnellstart">Schnellstart</a></li>  
		<li><a href="#ökosystem">Ökosystem</a></li>  
		<li><a href="https://github.com/zenc-lang/docs">Documentation</a></li>
		<li><a href="#standardbibliothek">Standardbibliothek</a></li>  
		<li><a href="#tooling">Tooling</a>
		  <ul>  
			 <li><a href="#language-server-protocol-lsp">LSP</a></li>
	        <li><a href="#debugging-zen-c">Debugging</a></li>
	      </ul>
	    </li>
	    <li><a href="#compilerunterstützung--kompatibilität">Compilerunterstützung & Kompatibilität</a></li>
	    <li><a href="#mitwirken">Mitwirken</a></li>
	    <li><a href="#quellenangaben">Quellenangaben</a></li>
	  </ul>
	</td>  
	<td valign="top">
      <p><a href="https://docs.zenc-lang.org/tour/"><b>Browse the Language Reference</b></a></p>  
	</td>  
</tr>  
</table>  

---

## Schnellstart

### Installation

```bash
git clone https://github.com/zenc-lang/zenc.git
cd Zen-C
make clean # Entferne alte Build-Dateien
make
sudo make install
```

### Windows

Zen C unterstützt Windows (x86_64) nativ. Nutze das Batch-Skript mit GCC (MinGW):

```cmd
build.bat
```

Dadurch wird der Compiler (`zc.exe`) erstellt. Netzwerk-, Dateisystem- und Prozessoperationen werden vollständig über die Plattformabstraktionsschicht (PAL) unterstützt.

Alternativ kannst du `make` verwenden, wenn du eine Unix-ähnliche Umgebung (MSYS2, Cygwin, git-bash) nutzt.

### Portable Binärdatei (APE)

Zen C kann als **Actually Portable Executable (APE)** mit [Cosmopolitan Libc](https://github.com/jart/cosmopolitan) kompiliert werden. Dies erzeugt eine einzige Datei (`.com`), die nativ auf Linux, macOS, Windows, FreeBSD, OpenBSD und NetBSD sowohl auf x86_64- als auch auf aarch64-Architekturen läuft.

**Voraussetzungen:**
- `cosmocc`-Toolchain (muss sich im PATH befinden)

**Build & Installation:**
```bash
make ape
sudo env "PATH=$PATH" make install-ape
```

**Artefakte:**
- `out/bin/zc.com`: Der portable Zen-C-Compiler. Enthält die Standardbibliothek in der ausführbaren Datei.
- `out/bin/zc-boot.com`: Ein eigenständiges Bootstrap-Installationsprogramm zum Einrichten neuer Zen-C-Projekte.

**Verwendung:**
```bash
# Läuft auf jedem unterstützten Betriebssystem
./out/bin/zc.com build hello.zc -o hello
```

### Verwendung

```bash
# Kompilieren und Ausführen
zc run hello.zc

# Executable erstellen
zc build hello.zc -o hello

# Interaktive Shell
zc repl

# Zen-Fakten zeigen
zc build hello.zc --zen
```

### Umgebungsvariablen

Du kannst `ZC_ROOT` setzen, um den Speicherort der Standardbibliothek anzugeben (Standardimporte wie `import "std/vec.zc"`). Dadurch kannst du `zc` aus jedem beliebigen Verzeichnis ausführen.

```bash
export ZC_ROOT=/path/to/Zen-C
```

---

## Sprachreferenz

Weitere Details finden Sie in der offiziellen [Sprachreferenz](https://docs.zenc-lang.org/tour/01-variables-constants/).

## Standardbibliothek

Zen C enthält eine Standardbibliothek (`std`), die grundlegende Funktionalität abdeckt.

[Zur Dokumentation der Standardbibliothek](docs/std/README.md)

### Wichtige Module

<details>
<summary>Klicke, um alle Standardbibliotheks-Module zu sehen</summary>

| Modul | Beschreibung | Docs |
| :--- | :--- | :--- |
| **`std/bigfloat.zc`** | Gleitkomma-Arithmetik mit beliebiger Genauigkeit. | [Docs](docs/std/bigfloat.md) |
| **`std/bigint.zc`** | Ganzzahlen mit beliebiger Genauigkeit `BigInt`. | [Docs](docs/std/bigint.md) |
| **`std/bits.zc`** | Niedrigstufige Bitoperationen (`rotl`, `rotr`). | [Docs](docs/std/bits.md) |
| **`std/complex.zc`** | Komplexe Zahlen `Complex`. | [Docs](docs/std/complex.md) |
| **`std/vec.zc`** | Dynamisches, wachsendes Array `Vec<T>`. | [Docs](docs/std/vec.md) |
| **`std/string.zc`** | Heap-allokierter `String` mit UTF-8 Unterstützung. | [Docs](docs/std/string.md) |
| **`std/queue.zc`** | FIFO-Warteschlange (Ringpuffer). | [Docs](docs/std/queue.md) |
| **`std/map.zc`** | Generische Hash-Map `Map<V>`. | [Docs](docs/std/map.md) |
| **`std/fs.zc`** | Dateisystemoperationen. | [Docs](docs/std/fs.md) |
| **`std/io.zc`** | Standard Ein-/Ausgabe (`print`/`println`). | [Docs](docs/std/io.md) |
| **`std/option.zc`** | Optionale Werte (`Some`/`None`). | [Docs](docs/std/option.md) |
| **`std/result.zc`** | Fehlerbehandlung (`Ok`/`Err`). | [Docs](docs/std/result.md) |
| **`std/path.zc`** | Plattformübergreifende Pfadmanipulation. | [Docs](docs/std/path.md) |
| **`std/env.zc`** | Prozess-Umgebungsvariablen. | [Docs](docs/std/env.md) |
| **`std/net/`** | TCP, UDP, HTTP, DNS, URL. | [Docs](docs/std/net.md) |
| **`std/thread.zc`** | Threads und Synchronisation. | [Docs](docs/std/thread.md) |
| **`std/time.zc`** | Zeitmessung und Sleep-Funktionen. | [Docs](docs/std/time.md) |
| **`std/json.zc`** | JSON Parsing und Serialisierung. | [Docs](docs/std/json.md) |
| **`std/stack.zc`** | LIFO-Stack `Stack<T>`. | [Docs](docs/std/stack.md) |
| **`std/set.zc`** | Generisches Hash-Set `Set<T>`. | [Docs](docs/std/set.md) |
| **`std/process.zc`** | Prozessausführung und Management. | [Docs](docs/std/process.md) |
| **`std/regex.zc`** | Reguläre Ausdrücke (TRE-basiert). | [Docs](docs/std/regex.md) |
| **`std/simd.zc`** | Native SIMD-Vektortypen. | [Docs](docs/std/simd.md) |

</details>

### 18. Unit-Testing-Framework

Zen C bietet ein eingebautes Test-Framework, um Unit-Tests direkt in den Quellcode-Dateien zu schreiben, mittels des `test`-Schlüsselworts.

#### Syntax
Ein `test`-Block enthält einen beschreibenden Namen und einen Codeblock, der ausgeführt wird. Es wird keine `main`-Funktion benötigt.

```zc
test "unittest1" {
    "Dies ist ein Unit-Test";

    let a = 3;
    assert(a > 0, "a sollte eine positive Zahl sein");

    "unittest1 erfolgreich.";
}
```

#### Tests ausführen
Um alle Tests einer Datei auszuführen, nutze den `run`-Befehl. Der Compiler erkennt automatisch alle top-level `test`-Blöcke.

```bash
zc run my_file.zc
```

#### Assertions
Verwende die eingebaute Funktion `assert(condition, message)` zur Überprüfung von Erwartungen. Wenn die Bedingung falsch ist, schlägt der Test fehl und die Nachricht wird ausgegeben.

---

## Tooling

Zen C bietet einen eingebauten **Language Server** und eine REPL, um die Entwicklungsarbeit zu erleichtern. Außerdem kann Zen C mit LLDB oder GDB debuggt werden.

### Language Server (LSP)

Der Zen C Language Server unterstützt das Language Server Protocol (LSP) und bietet die typischen Editor-Funktionen:

* **Gehe zu Definition** (`Go to Definition`)
* **Finde Referenzen** (`Find References`)
* **Hover-Informationen**
* **Autovervollständigung** (Funktions-/Struct-Namen, Methoden/Felder via Punkt)
* **Dokumentstruktur** (`Document Symbols` / Outline)
* **Signatur-Hilfe**
* **Diagnosen** (Syntax- und Semantikfehler)

Starten des Sprachserver (normalerweise in den LSP-Einstellungen deinem Editors konfiguriert):

```bash
zc lsp
```

Es kommuniziert über Standard I/O (JSON-RPC 2.0).

### REPL

Die Read-Eval-Print-Schleife ermöglicht es, interaktiv mit Zen C-Code zu experimentieren.

```bash
zc repl
```

#### Features

*   **Interaktives Coden**: Ausdrücke oder Statements sofort auswerten.
*   **Persistente Historie**: Befehle werden in `~/.zprep_history` gespeichert.
*   **Startup-Skript**: Lädt automatisch `~/.zprep_init.zc`.

#### Befehle

| Befehl | Beschreibung |
|:---|:---|
| `:help` | Zeigt alle verfügbaren Kommandos an |
| `:reset` | Löscht aktuelle Session-Historie (Variablen/Funktionen) |
| `:vars` | Zeigt aktive Variablen |
| `:funcs` | Zeigt benutzerdefinierte Funktionen |
| `:structs` | Zeigt benutzerdefinierte Structs |
| `:imports` | Zeigt aktive Importe |
| `:history` | Zeigt Session-Eingabeverlauf |
| `:type <expr>` | Zeigt den Typ eines Ausdrucks |
| `:c <stmt>` | Zeigt den generierten C-Code für ein Statement |
| `:time <expr>` | Benchmark eines Ausdrucks (1000 Iterationen) |
| `:edit [n]` | Bearbeite Befehl `n` im `$EDITOR` (Standard: letzter) |
| `:save <file>` | Speichert die aktuelle Session in einer `.zc` Datei |
| `:load <file>` | Lädt und führt eine `.zc` Datei in die Session aus |
| `:watch <expr>` | Beobachtet einen Ausdruck (automatisch nach jeder Eingabe aktualisiert) |
| `:unwatch <n>` | Entfernt einen Watch |
| `:undo` | Entfernt den letzten Befehl aus der Session |
| `:delete <n>` | Löscht Befehl an Index `n` |
| `:clear` | Bildschirm leeren |
| `:quit` | REPL beenden |
| `! <cmd>` | Führe Shell-Befehl aus (z.B. `!ls`) |

---

### Language Server Protocol (LSP)

Zen C enthält einen integrierten Sprachserver zur Editorintegration.

- **[Installations- und Einrichtungsanleitung](../docs/LSP.md)**
- **Unterstützte Editoren**: VS Code, Neovim, Vim ([zenc.vim](https://github.com/zenc-lang/zenc.vim)), Zed und alle LSP-fähigen Editoren.

Verwende `zc lsp`, um den Server zu starten.

### Debugging Zen C

Zen C Programme können mit Standard-C-Debuggern wie **LLDB** oder **GDB** debuggt werden.

#### Visual Studio Code

Für eine optimale Benutzererfahrung in VS Code installiere die offizielle [Zen C-Erweiterung](https://marketplace.visualstudio.com/items?itemName=Z-libs.zenc). Verwende zum Debuggen die **C/C++**-Erweiterung (von Microsoft) oder die **CodeLLDB**-Erweiterung.

Füge diese Konfigurationen in den `.vscode`-Verzeichnis hinzu, um das Debuggen mit einem Klick zu aktivieren:

**`tasks.json`** (Build Task):
```json
{
    "label": "Zen C: Build Debug",
    "type": "shell",
    "command": "zc",
    "args": [ "${file}", "-g", "-o", "${fileDirname}/app", "-O0" ],
    "group": { "kind": "build", "isDefault": true }
}
```

**`launch.json`** (Debugger):
```json
{
    "name": "Zen C: Debug (LLDB)",
    "type": "lldb",
    "request": "launch",
    "program": "${fileDirname}/app",
    "preLaunchTask": "Zen C: Build Debug"
}
```

## Compilerunterstützung & Kompatibilität

Zen C ist so konzipiert, dass es mit den meisten **C11-Compilern** funktioniert. Einige Features basieren auf **GNU-C-Erweiterungen**, funktionieren aber oft auch in anderen Compilern. Mit dem `--cc`-Flag kannst du das Backend wechseln.

```bash
zc run app.zc --cc clang
zc run app.zc --cc zig
```

### Status der Test-Suite

<details>
<summary>Klicke, um Compiler-Support-Details anzuzeigen</summary>

| Compiler | Erfolgsrate | Unterstützte Features | Bekannte Einschränkungen |
|:---|:---:|:---|:---|
| **GCC** | **100 % (Vollständig)** | Alle Features | Keine |
| **Clang** | **100 % (Vollständig)** | Alle Features | Keine |
| **Zig** | **100 % (Vollständig)** | Alle Features | Keine. Nutzt `zig cc` als Drop-in-C-Compiler |
| **TCC** | **98 % (Hoch)** | Structs, Generics, Traits, Pattern Matching | Kein Intel-ASM, kein `__attribute__((constructor))` |

</details>

> [!WARNING]
> **COMPILER BUILD WARNING:** Obwohl **Zig CC** hervorragend als Backend für Zen C Programme funktioniert, kann das **Bauen des Zen C Compilers selbst** damit zwar erfolgreich verifizieren, aber instabile Binaries erzeugen, die Tests nicht bestehen. Empfehlung: Den Compiler selbst mit **GCC** oder **Clang** bauen und Zig nur als Backend für Produktionscode verwenden.

### Build mit Zig

Zigs `zig cc` dient als Drop-in-Ersatz für GCC/Clang mit exzellenter Cross-Compilation-Unterstützung. Um Zig zu verwenden:

```bash
# Zen C Programm mit Zig kompilieren und ausführen
zc run app.zc --cc zig

# Den Zen C Compiler selbst mit Zig bauen
make zig
```

### C++-Interoperabilität

Zen C kann mit dem `--cpp`-Flag C++-kompatiblen Code generieren und dadurch nahtlos mit C++-Bibliotheken interagieren.

```bash
# Direkte Kompilierung mit g++
zc app.zc --cpp

# Oder transpilen und manuell bauen
zc transpile app.zc --cpp
g++ out.c my_cpp_lib.o -o app
```

#### Verwendung von C++ in Zen C

C++-Header einbinden und raw-Blöcke für nativen C++-Code verwenden:

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
> Das `--cpp`-Flag wechselt auf `g++` als Backend und erzeugt C++-kompatiblen Code (`auto` statt `__auto_type`, Overloads statt `_Generic`, explizite `void*`-Casts).

### CUDA-Interoperabilität

Zen C unterstützt GPU-Programmierung durch Transpilierung nach **CUDA C++**. Dadurch lassen sich moderne C++-Features (Templates, constexpr) in CUDA-Kernels nutzen, während die ergonomische Zen C Syntax erhalten bleibt.

```bash
# Direkt mit nvcc kompilieren
zc run app.zc --cuda

# Oder transpilen und manuell bauen
zc transpile app.zc --cuda -o app.cu
nvcc app.cu -o app
```

#### CUDA-spezifische Attribute

| Attribut | CUDA-Äquivalent | Beschreibung |
|:---|:---|:---|
| `@global` | `__global__` | Kernel-Funktion (läuft auf GPU, wird vom Host aufgerufen) |
| `@device` | `__device__` | Device-Funktion (läuft auf GPU, wird von GPU aufgerufen) |
| `@host` | `__host__` | Host-Funktion (explizit CPU-only) |

#### Kernel-Launch-Syntax

Zen C bietet ein sauberes `launch`-Statement zum Aufruf von CUDA-Kernels:

```zc
launch kernel_name(args) with {
    grid: num_blocks,
    block: threads_per_block,
    shared_mem: 1024,  // Optional
    stream: my_stream   // Optional
};
```

This transpiles to: `kernel_name<<<grid, block, shared, stream>>>(args);`

#### Schreiben von CUDA-Kernels

Verwende normale Zen C Funktionen mit `@global` und `launch`:

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

    // ... Daten initialisieren ...
    
    launch add_kernel(d_a, d_b, d_c, N) with {
        grid: (N + 255) / 256,
        block: 256
    };
    
    cuda_sync();
}
```

#### Standardbibliothek (`std/cuda.zc`)
Zen C stellt eine Standardbibliothek für gängige CUDA-Operationen zur Verfügung, um `raw`-Blöcke zu reduzieren:

```zc
import "std/cuda.zc"

// Speicherverwaltung
let d_ptr = cuda_alloc<float>(1024);
cuda_copy_to_device(d_ptr, h_ptr, 1024 * sizeof(float));
defer cuda_free(d_ptr);

// Synchronisation
cuda_sync();

// Thread-Indizes (innerhalb von Kernels)
let i = thread_id(); // Globaler Index
let bid = block_id();
let tid = local_id();
```

> [!NOTE]  
> **Hinweis:** Das `--cuda`-Flag setzt automatisch `nvcc` als Compiler und aktiviert implizit `--cpp`. Setzt NVIDIA CUDA Toolkit voraus.

### C23-Unterstützung

Zen C unterstützt moderne **C23-Features**, wenn ein kompatibler Backend-Compiler verwendet wird  
(GCC 14+, Clang 14+, TCC (teilweise)).

- **`auto`**: Zen C bildet Typinferenz automatisch auf das standardisierte C23-`auto` ab, wenn `__STDC_VERSION__ >= 202300L`.
- **`_BitInt(N)`**: Verwende `iN`- und `uN`-Typen (z. B. `i256`, `u12`, `i24`), um auf Ganzzahlen mit beliebiger Bitbreite aus C23 zuzugreifen.

### Objective-C-Interoperabilität

Zen C kann mit dem `--objc`-Flag nach **Objective-C (`.m`)** kompilieren, sodass Objective-C-Frameworks (wie Cocoa/Foundation) und deren Syntax direkt genutzt werden können.

```bash
# Mit clang kompilieren (oder gcc/gnustep)
zc app.zc --objc --cc clang
```

#### Verwendung von Objective-C in Zen C

Verwende `include` für Header und `raw`-Blöcke für Objective-C-Syntax (`@interface`, `[...]`, `@""`).

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
    println "Zen C funktioniert ebenfalls!";
}
```

> [!NOTE]  
> **Hinweis:** Zen C String-Interpolation funktioniert auch mit Objective-C-Objekten (`id`), indem automatisch `debugDescription` oder `description` aufgerufen wird.

---

## Mitwirken

Wir freuen uns über Beiträge!  
Egal ob Bugfixes, Dokumentation oder neue Sprachfeatures.

Siehe [CONTRIBUTING.md](4%20CONTRIBUTING_EN.md) für detaillierte Richtlinien zum Mitwirken, Testen und Einreichen von Pull Requests.

---

## Sicherheit

Hinweise zum Melden von Sicherheitslücken findest du in [SECURITY.md](5%20SECURITY_EN.md).

---

## Quellenangaben

Dieses Projekt verwendet Bibliotheken von Drittanbietern. Die vollständigen Lizenztexte befinden sich im Verzeichnis `LICENSES/`.

*   **[cJSON](https://github.com/DaveGamble/cJSON)** (MIT-Lizenz): Wird für JSON-Parsing und -Generierung im Language Server verwendet.
*   **[zc-ape](https://github.com/OEvgeny/zc-ape)** (MIT-Lizenz): Der ursprüngliche Actually Portable Executable Port von Zen C von **[Eugene Olonov](https://github.com/OEvgeny)**.
*   **[Cosmopolitan Libc](https://github.com/jart/cosmopolitan)** (ISC-Lizenz): Die zugrunde liegende Bibliothek, die APE ermöglicht.
*   **[TRE](https://github.com/laurikari/tre)** (BSD-Lizenz): Wird für die Regex-Engine der Standardbibliothek verwendet.
*   **[zenc.vim](https://github.com/zenc-lang/zenc.vim)** (MIT-Lizenz): Das offizielle Vim/Neovim-Plugin, hauptsächlich entwickelt von **[davidscholberg](https://github.com/davidscholberg)**.

---

<div align="center">
  <p>
    Copyright © 2026 Zen C Programmiersprache.<br>
    Starte deine Reise noch heute.
  </p>
  <p>
    <a href="https://discord.com/invite/q6wEsCmkJP">Discord</a> •
    <a href="https://github.com/zenc-lang/zenc">GitHub</a> •
    <a href="https://github.com/zenc-lang/docs">Dokumentation</a> •
    <a href="https://github.com/zenc-lang/awesome-zenc">Beispiele</a> •
    <a href="https://github.com/zenc-lang/rfcs">RFCs</a> •
    <a href="CONTRIBUTING_DE.md">Mitwirken</a>
  </p>
</div>