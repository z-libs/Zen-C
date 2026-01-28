#!/bin/bash
# Build script for llama.cpp dependency
# This script clones, builds, and installs llama.cpp for Zen-C integration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/llama_build"
INSTALL_DIR="$SCRIPT_DIR/llama_lib"
LLAMA_REPO="https://github.com/ggerganov/llama.cpp.git"
LLAMA_BRANCH="master"

echo "=== llama.cpp Build Script for Zen-C ==="
echo ""

# Check for required tools
check_requirements() {
    echo "Checking requirements..."

    if ! command -v git &> /dev/null; then
        echo "Error: git is required but not installed."
        exit 1
    fi

    if ! command -v cmake &> /dev/null; then
        echo "Error: cmake is required but not installed."
        exit 1
    fi

    if ! command -v make &> /dev/null; then
        echo "Error: make is required but not installed."
        exit 1
    fi

    echo "All requirements satisfied."
}

# Detect CUDA availability
detect_cuda() {
    if command -v nvcc &> /dev/null; then
        echo "CUDA detected: $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')"
        return 0
    elif [ -d "/usr/local/cuda" ]; then
        echo "CUDA directory found at /usr/local/cuda"
        export PATH="/usr/local/cuda/bin:$PATH"
        if command -v nvcc &> /dev/null; then
            echo "CUDA detected: $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')"
            return 0
        fi
    fi
    echo "CUDA not detected - building CPU-only version"
    return 1
}

# Clone llama.cpp
clone_repo() {
    if [ -d "$BUILD_DIR/llama.cpp" ]; then
        echo "llama.cpp already cloned. Updating..."
        cd "$BUILD_DIR/llama.cpp"
        git fetch origin
        git checkout $LLAMA_BRANCH
        git pull origin $LLAMA_BRANCH
    else
        echo "Cloning llama.cpp..."
        mkdir -p "$BUILD_DIR"
        cd "$BUILD_DIR"
        git clone --depth 1 --branch $LLAMA_BRANCH "$LLAMA_REPO"
    fi
}

# Build llama.cpp
build_llama() {
    echo "Building llama.cpp..."
    cd "$BUILD_DIR/llama.cpp"

    # Clean previous build
    rm -rf build
    mkdir build
    cd build

    # Configure cmake
    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"
    CMAKE_ARGS="$CMAKE_ARGS -DLLAMA_BUILD_EXAMPLES=OFF"
    CMAKE_ARGS="$CMAKE_ARGS -DLLAMA_BUILD_TESTS=OFF"
    CMAKE_ARGS="$CMAKE_ARGS -DLLAMA_BUILD_SERVER=OFF"
    CMAKE_ARGS="$CMAKE_ARGS -DBUILD_SHARED_LIBS=ON"


    echo "CMake arguments: $CMAKE_ARGS"
    cmake .. $CMAKE_ARGS

    # Build
    cmake --build . --config Release -j$(nproc)

    echo "Build completed."
}

# Install to local directory
install_lib() {
    echo "Installing to $INSTALL_DIR..."

    mkdir -p "$INSTALL_DIR/lib"
    mkdir -p "$INSTALL_DIR/include"

    # Copy libraries
    cd "$BUILD_DIR/llama.cpp/build"

    # Find and copy shared libraries
    find . -name "*.so*" -exec cp {} "$INSTALL_DIR/lib/" \;
    find . -name "*.dylib" -exec cp {} "$INSTALL_DIR/lib/" \; 2>/dev/null || true

    # Copy static libraries if no shared libs found
    if [ -z "$(ls -A $INSTALL_DIR/lib/*.so* 2>/dev/null)" ]; then
        find . -name "*.a" -exec cp {} "$INSTALL_DIR/lib/" \;
    fi

    # Copy headers
    cd "$BUILD_DIR/llama.cpp"
    cp include/llama.h "$INSTALL_DIR/include/"
    # Copy all ggml headers
    cp ggml/include/*.h "$INSTALL_DIR/include/" 2>/dev/null || true
    # Also check for headers in src directories
    find . -name "ggml*.h" -exec cp {} "$INSTALL_DIR/include/" \; 2>/dev/null || true

    echo "Installation completed."
    echo ""
    echo "Libraries installed to: $INSTALL_DIR/lib"
    echo "Headers installed to: $INSTALL_DIR/include"
}

# Print usage instructions
print_usage() {
    echo ""
    echo "=== Usage Instructions ==="
    echo ""
    echo "To use llama.cpp with Zen-C, add these build directives to your .zc file:"
    echo ""
    echo "  //> include: $INSTALL_DIR/include"
    echo "  //> lib: $INSTALL_DIR/lib"
    echo "  //> link: -lllama -lm -lpthread"
    echo ""
    echo "Or set LD_LIBRARY_PATH before running:"
    echo ""
    echo "  export LD_LIBRARY_PATH=\"$INSTALL_DIR/lib:\$LD_LIBRARY_PATH\""
    echo ""
    echo "Download a GGUF model for testing:"
    echo ""
    echo "  mkdir -p models"
    echo "  wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O models/tinyllama.gguf"
    echo ""
}

# Main
main() {
    check_requirements
    clone_repo
    build_llama
    install_lib
    print_usage

    echo "=== Done! ==="
}

# Run
main "$@"
