#!/usr/bin/env bash
# LSP Protocol Compliance Test Suite
# Usage: ./run_lsp_tests.sh

ZC="./zc"
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

PASSED=0
FAILED=0
FAILED_TESTS=()

send_lsp_message() {
    local json="$1"
    local len
    len=$(printf '%s' "$json" | wc -c)

    response=$(printf "Content-Length: %d\r\n\r\n%s" "$len" "$json" | "$ZC" lsp 2>/dev/null)
    printf '%s' "$response"
}

check_response() {
    local response="$1"
    local name="$2"
    local expected_id="$3"
    local expect_error="${4:-0}"
    local is_notification="${5:-0}"

    local status="FAIL"
    local detail=""

    if [[ $is_notification -eq 1 ]]; then
        if [[ -z "$response" ]]; then
            status="PASS"
        else
            detail="→ got response but notification should be silent"
        fi
    elif [[ -z "$response" ]]; then
        if [[ $expect_error -eq 1 ]]; then
            status="PASS"
            detail="→ no response (correct for invalid request)"
        else
            detail="→ no response received"
        fi
    else
        id=$(echo "$response" | grep -oP '"id":\s*(\d+|"[a-zA-Z0-9]+?"|\[[^\]]+\]|true|false|null)' | head -1 | cut -d: -f2- | tr -d ' "')

        has_error=$(echo "$response" | grep -c '"error"' || true)
        has_result=$(echo "$response" | grep -c '"result"' || true)

        if [[ $expect_error -eq 1 ]]; then
            [[ $has_error -gt 0 ]] && status="PASS" || detail="→ expected error but none found"
        else
            if [[ "$id" = "$expected_id" ]]; then
                if [[ $has_result -gt 0 ]]; then
                    status="PASS"
                elif [[ $has_error -gt 0 ]]; then
                    detail="→ got error instead of result"
                else
                    detail="→ missing both result & error"
                fi
            else
                detail="→ id mismatch (got '$id', expected '$expected_id')"
            fi
        fi
    fi

    if [[ $status = "PASS" ]]; then
        echo -e "Testing $name... PASS"
        ((PASSED++))
    else
        echo -e "Testing $name... FAIL$"
        [[ -n "$detail" ]] && echo "    $detail"
        ((FAILED++))
        FAILED_TESTS+=("$name")
    fi
}

# ────────────────────────────────────────────────
echo "Running LSP Compliance Tests..."
echo

# ALL tests

response=$(send_lsp_message '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"rootUri":"file:///tmp"}}')
check_response "$response" "initialize (id=1)" "1" 0 0

response=$(send_lsp_message '{"jsonrpc":"2.0","id":"abc","method":"initialize","params":{"rootUri":"file:///tmp"}}')
check_response "$response" "initialize (id=\"abc\")" "abc" 0 0

response=$(send_lsp_message '{"jsonrpc":"2.0","id":true,"method":"initialize","params":{"rootUri":"file:///tmp"}}')
check_response "$response" "initialize (invalid id=true)" "true" 1 0

response=$(send_lsp_message '{"jsonrpc":"2.0","id":[1,2],"method":"initialize","params":{"rootUri":"file:///tmp"}}')
check_response "$response" "initialize (invalid id=array)" "[1,2]" 1 0

response=$(send_lsp_message '{"jsonrpc":"2.0","method":"initialize","params":{"rootUri":"file:///tmp"}}')
check_response "$response" "initialize as notification (no id)" "" 1 0

response=$(send_lsp_message '{"jsonrpc":"2.0","id":2,"method":"shutdown","params":null}')
check_response "$response" "shutdown (id=2)" "2" 0 0

response=$(send_lsp_message '{"jsonrpc":"2.0","method":"exit"}')
check_response "$response" "exit notification" "" 0 1

response=$(send_lsp_message '{"jsonrpc":"2.0","id":99,"method":"unknownMethod","params":{}}')
check_response "$response" "unknown method" "99" 1 0

echo
echo "----------------------------------------"
echo "Summary:"
echo "-> Passed: $PASSED"
echo "-> Failed: $FAILED"
echo "----------------------------------------"