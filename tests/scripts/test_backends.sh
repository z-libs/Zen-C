#!/usr/bin/env bash
# Test every output backend — structural format validation + runtime execution.
set -o pipefail

ZC=${ZC:-./zc}
STRUCTURAL_INPUT=${STRUCTURAL_INPUT:-tests/backends/test_minimal.zc}
RUNTIME_INPUT=${RUNTIME_INPUT:-tests/backends/test_runtime.zc}
PASS=0; FAIL=0; SKIP=0
PASSED_LIST=(); FAILED_LIST=(); SKIPPED_LIST=()

###############################################################################
# Phase 1 — Structural validation: format well-formedness for ALL backends
###############################################################################
for be in json lisp dot ast-dump c cpp cuda objc; do
    out=$(mktemp /tmp/zc_backend_test_XXXXXX)

    if ! "$ZC" transpile "$STRUCTURAL_INPUT" --backend "$be" -o "$out" 2>/dev/null; then
        echo "FAIL: $be (transpile error)"
        FAIL=$((FAIL+1)); FAILED_LIST+=("$be"); rm -f "$out"; continue
    fi
    if [ ! -s "$out" ]; then
        echo "FAIL: $be (empty output)"
        FAIL=$((FAIL+1)); FAILED_LIST+=("$be"); rm -f "$out"; continue
    fi

    result=0
    case "$be" in
        json)
            python3 -c "
import json, sys
with open('$out') as f:
    d = json.load(f)
assert d.get('node') == 'ROOT', 'root should be ROOT'
c = d.get('children', [])
assert len(c) >= 1, 'root should have children'
fn = c[0]
assert fn.get('node') == 'FUNCTION', 'first child should be FUNCTION'
assert fn.get('name') == 'main', 'function should be named main'
bd = fn.get('body', [])
if isinstance(bd, list):
    assert len(bd) >= 1, 'function body not empty'
blk = bd[0] if isinstance(bd, list) else bd
assert blk.get('node') == 'BLOCK', 'body should be BLOCK'
" 2>/dev/null || result=1
            ;;
        lisp)
            python3 -c "
import sys
s = open('$out').read()
depth = 0
for i, line in enumerate(s.split(chr(10))):
    cl = line.split(';')[0] if ';' in line else line
    for c in cl:
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
    assert depth >= 0, f'unbalanced at line {i+1}'
assert depth == 0, f'unmatched parens at EOF ({depth})'
assert '(defun main' in s, 'missing (defun main)'
assert '(main)' in s, 'missing (main) call'
assert s.count('(princ ') + s.count('(format ') + s.count('(write') > 0, 'no output'
" 2>/dev/null || result=1
            ;;
        dot)
            python3 -c "
import sys, re
s = open('$out').read()
assert s.startswith('digraph AST {'), 'must start with digraph AST'
assert s.strip().endswith('}'), 'must end with }'
assert s.count('->') >= 2, 'expected >=2 edges'
assert len(re.findall(r'\\d+ \\[label=', s)) >= 3, 'expected >=3 nodes'
" 2>/dev/null || result=1
            ;;
        ast-dump)
            python3 -c "
import sys
s = open('$out').read()
lines = [l for l in s.split(chr(10)) if l.strip()]
assert 'ROOT' in s and 'FUNCTION' in s and 'BLOCK' in s, 'missing core nodes'
assert 'main' in s, 'should contain main'
assert any(c in s for c in ('├', '└', '│')), 'should have tree chars'
assert len(lines) >= 4, f'expected >=4 non-empty lines, got {len(lines)}'
" 2>/dev/null || result=1
            ;;
        c)
            python3 -c "
import sys
s = open('$out').read()
assert '#include <stdio.h>' in s, 'should include stdio.h'
assert 'int main(' in s or 'void main(' in s, 'should have main'
" 2>/dev/null || result=1
            if [ $result -eq 0 ]; then
                type gcc >/dev/null 2>&1 && { gcc -fsyntax-only -x c "$out" 2>/dev/null || result=1; } || result=2
            fi
            ;;
        cpp)
            python3 -c "
import sys
s = open('$out').read()
assert 'int main(' in s or 'void main(' in s or 'auto main(' in s, 'should have main'
assert 'fprintf' in s or 'printf' in s or 'std::cout' in s, 'should have print'
" 2>/dev/null || result=1
            if [ $result -eq 0 ]; then
                type g++ >/dev/null 2>&1 && { g++ -fsyntax-only -fpermissive -x c++ "$out" 2>/dev/null || result=1; } || result=2
            fi
            ;;
        cuda)
            python3 -c "
import sys
s = open('$out').read()
includes = [l for l in s.split(chr(10)) if l.strip().startswith('#include')]
assert len(includes) >= 3, f'expected >=3 includes, got {len(includes)}'
assert 'printf' in s or 'fprintf' in s, 'should contain output call'
" 2>/dev/null || result=1
            ;;
        objc)
            python3 -c "
import sys
s = open('$out').read()
assert 'int main(' in s or 'void main(' in s, 'should have main'
assert '#include <stdio.h>' in s or '#import <Foundation' in s, 'should include stdio/Foundation'
" 2>/dev/null || result=1
            if [ $result -eq 0 ]; then
                type clang >/dev/null 2>&1 && { clang -fsyntax-only -x objective-c "$out" 2>/dev/null || result=1; } || result=2
            fi
            ;;
    esac

    rm -f "$out"
    if [ $result -eq 0 ]; then
        echo "PASS: $be (structural)"
        PASS=$((PASS+1)); PASSED_LIST+=("$be")
    elif [ $result -eq 2 ]; then
        echo "SKIP: $be (compiler not found)"
        SKIP=$((SKIP+1)); SKIPPED_LIST+=("$be")
    else
        echo "FAIL: $be (structural)"
        FAIL=$((FAIL+1)); FAILED_LIST+=("$be")
    fi
done

###############################################################################
# Phase 2 — Runtime validation: actually run the compiled output
###############################################################################
bin=$(mktemp /tmp/zc_backend_bin_XXXXXX)

for be in c lisp; do
    ext=$([ "$be" = "lisp" ] && echo ".lisp" || echo ".c")
    src=$(mktemp "/tmp/zc_backend_src_XXXXXX$ext")
    actual=$(mktemp /tmp/zc_backend_actual_XXXXXX)

    if ! "$ZC" transpile "$RUNTIME_INPUT" --backend "$be" -o "$src" 2>/dev/null; then
        echo "FAIL: $be (runtime transpile error)"
        FAIL=$((FAIL+1)); FAILED_LIST+=("${be}_runtime"); rm -f "$src" "$actual"; continue
    fi

    case "$be" in
        c)
            if ! type gcc >/dev/null 2>&1; then
                echo "SKIP: $be (runtime, gcc not found)"
                SKIP=$((SKIP+1)); SKIPPED_LIST+=("${be}_runtime")
                rm -f "$src" "$actual"; continue
            fi
            gcc -x c "$src" -o "$bin" 2>/dev/null || {
                echo "FAIL: $be (runtime, compile failed)"
                FAIL=$((FAIL+1)); FAILED_LIST+=("${be}_runtime"); rm -f "$src" "$actual"; continue
            }
            "$bin" > "$actual" 2>/dev/null
            ;;
        lisp)
            if ! type sbcl >/dev/null 2>&1; then
                echo "SKIP: $be (runtime, sbcl not found)"
                SKIP=$((SKIP+1)); SKIPPED_LIST+=("${be}_runtime")
                rm -f "$src" "$actual"; continue
            fi
            sbcl --script "$src" > "$actual" 2>/dev/null
            ;;
    esac

    expected="PASS"
    if grep -q "^$expected" "$actual" 2>/dev/null; then
        echo "PASS: $be (runtime)"
        PASS=$((PASS+1)); PASSED_LIST+=("${be}_runtime")
    else
        echo "FAIL: $be (runtime, expected '$expected', got '$(cat "$actual" | tr -d '\n')')"
        FAIL=$((FAIL+1)); FAILED_LIST+=("${be}_runtime")
    fi
    rm -f "$src" "$actual"
done

rm -f "$bin"

###############################################################################
# Summary
###############################################################################
echo ""
echo "----------------------------------------"
echo "Backend test results:"
echo "-> Passed:  $PASS"
echo "-> Failed:  $FAIL"
echo "-> Skipped: $SKIP"
echo "----------------------------------------"
if [ ${#FAILED_LIST[@]} -gt 0 ]; then
    echo "Failed: ${FAILED_LIST[*]}"
fi
exit $FAIL
