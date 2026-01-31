# LLaMA.cpp Integration for Zen-C

This directory contains examples demonstrating how to use LLaMA.cpp with Zen-C for LLM inference.

## Prerequisites

- CMake 3.14+
- C/C++ compiler (GCC or Clang)
- Git
- wget or curl (for downloading models)
- (Optional) CUDA toolkit for GPU acceleration

## Quick Start

### Step 1: Build llama.cpp

Run the build script to clone and compile llama.cpp:

```bash
cd examples/llama
chmod +x deps.sh
./deps.sh
```

This will:
- Clone llama.cpp from GitHub
- Detect CUDA if available and enable GPU support
- Build the shared libraries
- Install to `./llama_lib/`

### Step 2: Download a Model

Download a GGUF model for testing. TinyLlama is recommended for quick tests:

```bash
mkdir -p models
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O models/tinyllama.gguf
```

### Step 3: Set Library Path

Before running any example, set the library path:

```bash
export LD_LIBRARY_PATH="./llama_lib/lib:$LD_LIBRARY_PATH"
```

To make this permanent, add to your `~/.bashrc`:
```bash
echo 'export LD_LIBRARY_PATH="/full/path/to/examples/llama/llama_lib/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Step 4: Run Examples

```bash
# Make sure you're in the examples/llama directory
cd examples/llama

# Set library path (if not already done)
export LD_LIBRARY_PATH="./llama_lib/lib:$LD_LIBRARY_PATH"

# Basic CPU inference
zc run basic_inference.zc

# GPU inference (requires CUDA)
zc run gpu_inference.zc --cuda

# Interactive chat
zc run chat.zc

# Sampling strategies demo
zc run sampling_demo.zc
```

## Configuration

Each example has hardcoded configuration at the top of the `main()` function. Edit these values to customize:

```zc
fn main() -> int {
    // Hardcoded configuration - modify these as needed
    var model_path = "models/tinyllama.gguf";  // Path to your model
    var prompt = "Hello, world!";               // Your prompt
    var gpu_layers = -1;                        // -1 = all on GPU, 0 = CPU only
    ...
}
```

## Examples

### basic_inference.zc

Simple CPU-only inference example. Loads a model, prints model info, and generates text completion.

### gpu_inference.zc

GPU-accelerated inference with streaming output and performance metrics (tokens/sec).

### chat.zc

Interactive chat mode with conversation history.

Commands:
- `quit` or `exit` - Exit the chat
- `clear` - Clear conversation history
- `help` - Show help

### sampling_demo.zc

Demonstrates different sampling strategies:
- **Greedy**: Temperature = 0, always picks most likely token
- **Balanced**: Default settings with top-k/top-p filtering
- **Creative**: High temperature for more varied output

## Troubleshooting

### "libllama.so: cannot open shared object file"

Set the library path before running:
```bash
export LD_LIBRARY_PATH="./llama_lib/lib:$LD_LIBRARY_PATH"
```

### "No such file or directory: models/tinyllama.gguf"

Download a model first:
```bash
mkdir -p models
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O models/tinyllama.gguf
```

### CUDA not detected during build

1. Ensure CUDA toolkit is installed
2. Check that `nvcc` is in your PATH: `which nvcc`
3. Rebuild: `rm -rf llama_build llama_lib && ./deps.sh`

### Out of memory (GPU)

Edit the example file and reduce GPU layers:
```zc
var gpu_layers = 10;  // Use fewer layers on GPU
```

### Slow generation

- Use GPU acceleration if available
- Use a smaller/quantized model (Q4_K_M recommended)
- Reduce context size in the code

## Recommended Models

| Model | Size | RAM/VRAM | Quality | Download |
|-------|------|----------|---------|----------|
| TinyLlama-1.1B-Q4 | ~700MB | ~1GB | Basic | [Link](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) |
| Phi-2-Q4 | ~1.5GB | ~2GB | Good | [Link](https://huggingface.co/TheBloke/phi-2-GGUF) |
| Mistral-7B-Q4 | ~4GB | ~5GB | Excellent | [Link](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) |

## Complete Setup Example

```bash
# 1. Navigate to the llama examples directory
cd examples/llama

# 2. Build llama.cpp
./deps.sh

# 3. Download a test model
mkdir -p models
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O models/tinyllama.gguf

# 4. Set library path
export LD_LIBRARY_PATH="./llama_lib/lib:$LD_LIBRARY_PATH"

# 5. Run the chat example
zc run chat.zc
```

## API Reference

### Loading Models

```zc
import "../../std/llama.zc"

// CPU only
var model = LlamaModel::load_cpu("model.gguf").unwrap();

// GPU with all layers
var model = LlamaModel::load_gpu_all("model.gguf").unwrap();

// GPU with specific layer count
var model = LlamaModel::load_gpu("model.gguf", 20).unwrap();
```

### Creating Context

```zc
// Default context
var ctx = LlamaContext::from_model(&model).unwrap();

// Custom context parameters
var params = LlamaContextParams::defaults();
params.n_ctx = 4096;
params.n_batch = 512;
var ctx = LlamaContext::new(&model, params).unwrap();
```

### Sampling

```zc
// Greedy (deterministic)
var sampler = LlamaSampler::greedy(&model);

// Default balanced
var sampler = LlamaSampler::defaults(&model, seed);

// Creative (high temperature)
var sampler = LlamaSampler::creative(&model, seed);
```

### Text Generation

```zc
// Batch generation
var result = llama_generate(&ctx, &sampler, "Hello", 128).unwrap();
println "{result.text.c_str()}"
result.free();
```

### Cleanup

```zc
sampler.free();
ctx.free();
model.free();
llama_cleanup();
```

## License

The llama.cpp library is licensed under MIT. See the [llama.cpp repository](https://github.com/ggerganov/llama.cpp) for details.
