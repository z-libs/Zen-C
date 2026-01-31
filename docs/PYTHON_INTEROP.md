# Python Interop for Zen-C

Zen-C can embed CPython and call Python code through the `std/python.zc` module. Write inline Python directly with `raw python { }` blocks, or use the `Py::` and `PyObj` API for structured interop.

## Build Requirements

You need the Python 3 development headers installed:

```bash
# Ubuntu/Debian
sudo apt install python3-dev

# Fedora
sudo dnf install python3-devel

# macOS
brew install python3
```

## Quick Start

```zc
import "std/python.zc"

fn main() {
    Py::init();

    // Write Python directly
    raw python {
        import math
        print(f"sqrt(16) = {math.sqrt(16)}")
    }

    // Or eval into Zen-C
    let val = Py::eval("2 ** 10");
    println "2^10 = {val.as_int()}";

    Py::finalize();
}
```

Build and run:

```bash
./zc run examples/features/python_interop.zc
```

## `raw python { }` Blocks

Write Python code inline. The compiler wraps it in `PyRun_SimpleString()` automatically:

```zc
raw python {
    import json
    data = {"name": "Zen-C", "version": 1}
    print(json.dumps(data))
}
```

Set Python variables and pull them into Zen-C with `Py::eval()`:

```zc
raw python {
    import math
    result = math.factorial(10)
}
let fact = Py::eval("result");
println "10! = {fact.as_int()}";
```

## API Reference

### Py — Interpreter Operations

| Method | Returns | Description |
|---|---|---|
| `Py::init()` | — | Initialize the Python interpreter |
| `Py::finalize()` | — | Shut down the Python interpreter |
| `Py::import(name)` | `PyObj` | Import a Python module by name |
| `Py::run(code)` | `int` | Execute Python code string (0 on success) |
| `Py::eval(expr)` | `PyObj` | Evaluate expression, return result |
| `Py::tuple(args, count)` | `PyObj` | Build tuple from array |
| `Py::list(args, count)` | `PyObj` | Build list from array |
| `Py::dict()` | `PyObj` | Create empty dict |
| `Py::err_print()` | — | Print Python traceback to stderr |
| `Py::err_clear()` | — | Clear Python error state |

### PyObj — Creating Values

| Method | Description |
|---|---|
| `PyObj::from_int(v: i64)` | Create a Python int |
| `PyObj::from_f64(v: f64)` | Create a Python float |
| `PyObj::from_str(s: char*)` | Create a Python str |
| `PyObj::from_bool(v: bool)` | Create a Python bool |
| `PyObj::none()` | Get Python `None` |

### PyObj — Extracting Values

| Method | Returns | Description |
|---|---|---|
| `obj.as_int()` | `i64` | Extract as integer |
| `obj.as_f64()` | `f64` | Extract as float |
| `obj.as_str()` | `char*` | Extract as C string |
| `obj.as_bool()` | `bool` | Extract as boolean |

### PyObj — Calling and Attributes

| Method | Description |
|---|---|
| `obj.attr(name)` | Get attribute (like Python's `getattr`) |
| `obj.call0()` | Call with no arguments |
| `obj.call1(a)` | Call with 1 argument |
| `obj.call2(a, b)` | Call with 2 arguments |
| `obj.call3(a, b, c)` | Call with 3 arguments |

### PyObj — Dict Operations

| Method | Description |
|---|---|
| `obj.set(key, val)` | Set entry by string key |
| `obj.get(key)` | Get entry by string key |
| `obj.set_obj(key, val)` | Set entry with PyObj key |
| `obj.get_obj(key)` | Get entry with PyObj key |

### PyObj — List Operations

| Method | Description |
|---|---|
| `obj.len()` | Get list length |
| `obj.item(i)` | Get element at index |
| `obj.set_item(i, v)` | Set element at index |

### PyObj — Inspection

| Method | Returns | Description |
|---|---|---|
| `obj.is_none()` | `bool` | Check if None or NULL |
| `obj.is_err()` | `bool` | Check if Python error occurred |
| `obj.repr()` | `char*` | String representation |

## Type Conversions

| Zen-C Type | Python Type | To Python | From Python |
|---|---|---|---|
| `i64` | `int` | `PyObj::from_int(v)` | `obj.as_int()` |
| `f64` | `float` | `PyObj::from_f64(v)` | `obj.as_f64()` |
| `char*` | `str` | `PyObj::from_str(s)` | `obj.as_str()` |
| `bool` | `bool` | `PyObj::from_bool(v)` | `obj.as_bool()` |
| — | `None` | `PyObj::none()` | `obj.is_none()` |

## Memory Management

`PyObj` implements `Drop`, automatically calling `Py_DECREF` when objects go out of scope.

## Examples

### Inline Python (simplest)

```zc
import "std/python.zc"

fn main() {
    Py::init();

    raw python {
        import json
        data = {"key": "value", "count": 42}
        print(json.dumps(data, indent=2))
    }

    Py::finalize();
}
```

### Structured calls

```zc
import "std/python.zc"

fn main() {
    Py::init();

    let math = Py::import("math");
    let sqrt = math.attr("sqrt");
    let result = sqrt.call1(PyObj::from_f64(256.0));
    println "sqrt(256) = {result.as_f64()}";

    Py::finalize();
}
```

### Mixed: Python computes, Zen-C reads

```zc
import "std/python.zc"

fn main() {
    Py::init();

    raw python {
        import math
        factorial_result = math.factorial(20)
    }

    let val = Py::eval("factorial_result");
    println "20! = {val.as_int()}";

    Py::finalize();
}
```

### NumPy

```zc
import "std/python.zc"

fn main() {
    Py::init();

    raw python {
        import numpy as np
        arr = np.array([1, 2, 3, 4, 5])
        print(f"mean = {arr.mean()}")
    }

    let mean = Py::eval("float(arr.mean())");
    println "Zen-C got mean = {mean.as_f64()}";

    Py::finalize();
}
```

See `examples/features/python_numpy.zc` for a full NumPy example.

### HuggingFace AI/ML (CPU)

Use Python for ML inference and Zen-C for application logic, data processing, and presentation.

Install dependencies:

```bash
pip3 install transformers torch --break-system-packages
```

```zc
import "std/python.zc"
import "std/io.zc"

fn main() {
    Py::init();

    // Python: load model and run inference
    raw python {
        from transformers import pipeline
        classifier = pipeline("sentiment-analysis", device=-1)

        texts = [
            "Zen-C is an amazing programming language!",
            "I hate waiting for slow builds.",
            "The weather is okay today.",
        ]
        results = classifier(texts)
        _count = len(results)
    }

    // Zen-C: process results, format output, compute stats
    let count = Py::eval("_count").as_int();
    let positive = 0;

    for (let i = 0; i < (int)count; i = i + 1) {
        let idx_cmd = format("_i = %d", i);
        Py::run(idx_cmd);

        let text  = Py::eval("texts[_i]");
        let label = Py::eval("results[_i]['label']");
        let score = Py::eval("results[_i]['score']");

        println "\"{text.as_str()}\"";
        println "  -> {label.as_str()} ({score.as_f64()})";

        if (strcmp(label.as_str(), "POSITIVE") == 0) {
            positive = positive + 1;
        }
    }

    println "Positive: {positive}/{count}";
    Py::finalize();
}
```

This pattern works for any HuggingFace pipeline — text generation, translation, summarization, question answering, etc. Python handles the model, Zen-C handles everything else.

| Layer | Language | Why |
|---|---|---|
| ML inference (LLM, classifiers, embeddings) | Python | HuggingFace, PyTorch ecosystem |
| Data processing, loops, formatting | Zen-C | Fast, no GIL, native control |
| HTTP/networking | Zen-C | `std/net.zc` |
| GPU compute (custom kernels) | Zen-C | `std/cuda.zc` |

See `examples/features/python_huggingface.zc` for the full example with progress bars and summary stats.

## Limitations

- Python interpreter is global state; only one can be active at a time.
- `Py::init()` / `Py::finalize()` should be called once per program.
- String pointers from `as_str()` are owned by the Python object and become invalid after the object is freed.
- `callN` supports up to 3 arguments. For more, build a tuple with `Py::tuple()`.
- `raw python { }` blocks require balanced braces (which valid Python always has).
