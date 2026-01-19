#!/bin/bash

# Codegen Verification Test Runner
ZC="./zc"
TEST_DIR="tests/codegen"
PASSED=0
FAILED=0

if [ ! -f "$ZC" ]; then
    echo "Error: zc binary not found."
    exit 1
fi

echo "** Running Codegen Verification Tests **"

# Test 1: Duplicate Typedefs
TEST_NAME="dedup_typedefs.zc"
echo -n "Testing $TEST_DIR/$TEST_NAME (Duplicate Typedefs)... "

$ZC "$TEST_DIR/$TEST_NAME" --emit-c > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "FAIL (Compilation error)"
    ((FAILED++))
else
    # Check out.c for duplicates
    # We expect "typedef struct Vec2f Vec2f;" to appear exactly once
    COUNT=$(grep -c "typedef struct Vec2f Vec2f;" out.c)
    
    if [ "$COUNT" -eq 1 ]; then
        echo "PASS"
        ((PASSED++))
    else
        echo "FAIL (Found $COUNT typedefs for Vec2f, expected 1)"
        ((FAILED++))
    fi
fi

# Cleanup
rm -f out.c a.out

echo "----------------------------------------"
echo "Summary:"
echo "-> Passed: $PASSED"
echo "-> Failed: $FAILED"
echo "----------------------------------------"

if [ $FAILED -ne 0 ]; then
    exit 1
else
    exit 0
fi
