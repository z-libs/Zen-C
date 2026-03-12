# Standard Library: Process (`std/process.zc`)

The `std/process` module allows you to spawn and interact with child processes.

## Usage

```zc
import "std/process.zc"

fn main() {
    let output = Command::new("echo")
        .arg("hello")
        .output();
        
    if (output.exit_code == 0) {
        output.stdout.print();
        // Or access raw C string: output.stdout.c_str()
    }
}
```

## Struct Definitions

### `Command`

A builder for spawning a process.

```zc
struct Command {
    program: String;
    args: Vec<String>;
}
```

### `Output`

The output of a finished process.

```zc
struct Output {
    stdout: String;
    exit_code: int;
}
```

## Methods

### `Command` Methods

| Method | Signature | Description |
| :--- | :--- | :--- |
| **new** | `Command::new(program: char*) -> Command` | Creates a new Command for the given program. |
| **arg** | `arg(self, arg: char*) -> Command*` | Adds an argument to the command (chains). |
| **output** | `output(self) -> Output` | Executes and waits for the process, capturing stdout. |
| **status** | `status(self) -> int` | Executes and returns the exit status code. |

### `Output` Methods

| Method | Signature | Description |
| :--- | :--- | :--- |
| **Trait** | `impl Drop for Output` | Automatically frees the captured `stdout` string. |
