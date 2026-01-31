#!/bin/bash
# One-command setup script for llama.cpp examples
# Usage: ./setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Zen-C LLaMA Setup ==="
echo ""

# Step 1: Build llama.cpp
echo "[1/3] Building llama.cpp..."
if [ ! -d "llama_lib" ]; then
    chmod +x deps.sh
    ./deps.sh
else
    echo "  llama_lib already exists, skipping build."
    echo "  (Delete llama_lib/ and llama_build/ to rebuild)"
fi
echo ""

# Step 2: Download model
echo "[2/3] Checking for model..."
if [ ! -f "models/tinyllama.gguf" ]; then
    echo "  Downloading TinyLlama model (~700MB)..."
    mkdir -p models
    wget -q --show-progress https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O models/tinyllama.gguf
else
    echo "  Model already exists at models/tinyllama.gguf"
fi
echo ""

# Step 3: Instructions
echo "[3/3] Setup complete!"
echo ""
echo "=== How to Run ==="
echo ""
echo "Before running examples, set the library path:"
echo ""
echo "  export LD_LIBRARY_PATH=\"$SCRIPT_DIR/llama_lib/lib:\$LD_LIBRARY_PATH\""
echo ""
echo "Then run any example:"
echo ""
echo "  zc run basic_inference.zc    # Basic text generation"
echo "  zc run chat.zc               # Interactive chat"
echo "  zc run sampling_demo.zc      # Sampling comparison"
echo "  zc run gpu_inference.zc --cuda  # GPU inference"
echo ""
echo "Or run this one-liner:"
echo ""
echo "  LD_LIBRARY_PATH=\"./llama_lib/lib:\$LD_LIBRARY_PATH\" zc run chat.zc"
echo ""
