#!/bin/bash

# Zen-C Test Suite Runner
# Usage: ./tests/run_tests.sh [zc options]
#
# Examples:
#   ./tests/run_tests.sh                    # Test with default compiler (gcc)
#   ./tests/run_tests.sh --cc clang         # Test with clang
#   ./tests/run_tests.sh --cc zig           # Test with zig cc
#   ./tests/run_tests.sh --cc tcc           # Test with tcc

# Configuration
ZC="./zc"
if [ ! -f "$ZC" ]; then
    if [ -f "./zc.exe" ]; then
        ZC="./zc.exe"
    elif [ -f "./build/zc" ]; then
        ZC="./build/zc"
    elif [ -f "./build/zc.exe" ]; then
        ZC="./build/zc.exe"
    fi
fi
TEST_DIR="tests"
PASSED=0
FAILED=0
FAILED_TESTS=""

# Display which compiler is being used
CC_NAME="gcc (default)"
USE_TYPECHECK=0
filtered_args=()
sys_type=$(uname -s)
sys_arch=$(uname -m)

for arg in "$@"; do
    if [ "$prev_arg" = "--cc" ]; then
        CC_NAME="$arg"
    fi
    if [ "$arg" = "--typecheck" ]; then
        USE_TYPECHECK=1
    fi
    
    filtered_args+=("$arg")
    prev_arg="$arg"
done

# Replace $@ with filtered_args
set -- "${filtered_args[@]}"

echo "** Running Zen C test suite (compiler: $CC_NAME) **"

if [ ! -f "$ZC" ]; then
    echo "Error: zc binary not found. Please build it first."
    exit 1
fi

while read -r test_file; do
    [ -e "$test_file" ] || continue

    # Skip tests known to fail with TCC
    if [[ "$CC_NAME" == *"tcc"* ]]; then
        if [[ "$test_file" == *"test_intel.zc"* ]]; then
            echo "Skipping $test_file (Intel assembly not supported by TCC)"
            continue
        fi
        if [[ "$test_file" == *"test_attributes.zc"* ]]; then
            echo "Skipping $test_file (Constructor attribute not supported by TCC)"
            continue
        fi
        if [[ "$test_file" == *"test_simd_native.zc"* ]]; then
            echo "Skipping $test_file (SIMD vector extensions not supported by TCC)"
            continue
        fi
    fi

    # Skip architecture-specific tests
    if [[ "$sys_arch" != *"86"* && "$sys_arch" != "amd64" ]]; then
        if [[ "$test_file" == *"test_asm.zc"* ]] || \
           [[ "$test_file" == *"test_asm_clobber.zc"* ]] || \
           [[ "$test_file" == *"test_intel.zc"* ]]; then
            echo "Skipping $test_file (x86 assembly not supported on $sys_arch)"
            continue
        fi
    fi

    if [[ "$sys_arch" != *"arm64"* && "$sys_arch" != "aarch64" ]]; then
        if [[ "$test_file" == *"_arm64.zc"* ]]; then
            echo "Skipping $test_file (ARM64 assembly not supported on $sys_arch)"
            continue
        fi
    fi

    # Skip tests that require typechecking if not enabled
    if grep -q "// REQUIRE: TYPECHECK" "$test_file"; then
        if [ $USE_TYPECHECK -eq 0 ]; then
             echo "Skipping $test_file (requires --typecheck)"
             continue
        fi
    fi

    echo -n "Testing $test_file... "
    
    # Add -w to suppress warnings as requested
    tmp_out="test_out_$$.out"
    output=$($ZC run "$test_file" -o "$tmp_out" -w "$@" 2>&1)
    exit_code=$?
    rm -f "$tmp_out"
    
    # Check for expected failure annotation
    if grep -q "// EXPECT: FAIL" "$test_file"; then
        if [ $exit_code -ne 0 ]; then
            echo "PASS (Expected Failure)"
            ((PASSED++))
        else
            echo "FAIL (Unexpected Success)"
            ((FAILED++))
            FAILED_TESTS="$FAILED_TESTS\n- $test_file (Unexpected Success)"
        fi
    else
        if [ $exit_code -eq 0 ]; then
            echo "PASS"
            ((PASSED++))
        else
            echo "FAIL"
            echo "$output"
            ((FAILED++))
            FAILED_TESTS="$FAILED_TESTS\n- $test_file"
        fi
    fi
done < <(find "$TEST_DIR" -name "*.zc" -not -name "_*.zc" | sort)

echo "----------------------------------------"
echo "Summary:"
echo "-> Passed: $PASSED"
echo "-> Failed: $FAILED"
echo "----------------------------------------"

if [ $FAILED -ne 0 ]; then
    echo -e "Failed tests:$FAILED_TESTS"
    rm -f test_out_*.out out.c
    exit 1
else
    echo "All tests passed!"
    rm -f test_out_*.out out.c
    exit 0
fi
