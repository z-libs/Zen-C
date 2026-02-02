#!/usr/bin/env bash
set -e

LSP_BIN=./zc  # ton binaire LSP

# --- Couleurs ---
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # no color

# --- Helper pour envoyer un message et récupérer la réponse ---
send_lsp_message() {
    local json="$1"
    local len
    len=$(printf "%s" "$json" | wc -c)
    
    # envoyer le message et capturer stdout
    response=$( { 
        printf "Content-Length: %d\r\n\r\n%s" "$len" "$json"
    } | "$LSP_BIN" lsp )
    
    echo "$response"
}

# --- Helper pour tester la réponse ---
# $1 = response
# $2 = expected id
# $3 = expect_error (1 si id invalide)
# $4 = is_notification (1 si pas d'id)
check_response() {
    local response="$1"
    local expected_id="$2"
    local expect_error="$3"
    local is_notification="$4"

    if [[ "$is_notification" == "1" ]]; then
        if [[ -z "$response" ]]; then
            echo -e "${GREEN}✅ Notification: pas de réponse (correct)${NC}"
        else
            echo -e "${RED}❌ Notification: réponse reçue alors qu'aucune attendue${NC}"
        fi
        return
    fi

    if [[ -z "$response" ]]; then
        if [[ "$expect_error" == "1" ]]; then
            echo -e "${GREEN}✅ No response for invalid id (treated as error)${NC}"
        else
            echo -e "${RED}❌ No response${NC}"
        fi
        return
    fi

    # extraire id
    id=$(echo "$response" | grep -Po '"id":\s*([^,}]+)' | head -1 | awk -F: '{print $2}' | tr -d ' "')

    if [[ "$expect_error" == "1" ]]; then
        if echo "$response" | grep -q '"error"'; then
            echo -e "${GREEN}✅ Error present (invalid id)${NC}"
        else
            echo -e "${RED}❌ Expected error but none found${NC}"
        fi
    else
        if [[ "$id" == "$expected_id" ]]; then
            echo -e "${GREEN}✅ id matches: $id${NC}"
        else
            echo -e "${RED}❌ id mismatch: got $id, expected $expected_id${NC}"
        fi

        # vérifier presence de result
        if echo "$response" | grep -q '"result"'; then
            echo -e "${GREEN}✅ result present${NC}"
        elif echo "$response" | grep -q '"error"'; then
            echo -e "${RED}❌ got error instead of result${NC}"
        else
            echo -e "${RED}❌ neither result nor error present${NC}"
        fi
    fi
}

# -------------------------
# TESTS
# -------------------------

# 1️⃣ Initialize valide (id=1)
JSON='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"rootUri":"file:///tmp"}}'
echo
echo "=== Test initialize (id=1) ==="
response=$(send_lsp_message "$JSON")
check_response "$response" "1" 0 0

# 2️⃣ Initialize valide (id="abc")
JSON='{"jsonrpc":"2.0","id":"abc","method":"initialize","params":{"rootUri":"file:///tmp"}}'
echo
echo "=== Test initialize (id=\"abc\") ==="
response=$(send_lsp_message "$JSON")
check_response "$response" "abc" 0 0

# 3️⃣ Initialize invalide (id=true)
JSON='{"jsonrpc":"2.0","id":true,"method":"initialize","params":{"rootUri":"file:///tmp"}}'
echo
echo "=== Test initialize (id=true, invalide) ==="
response=$(send_lsp_message "$JSON")
check_response "$response" "true" 1 0

# 4️⃣ Initialize invalide (id=array)
JSON='{"jsonrpc":"2.0","id":[1,2],"method":"initialize","params":{"rootUri":"file:///tmp"}}'
echo
echo "=== Test initialize (id=[1,2], invalide) ==="
response=$(send_lsp_message "$JSON")
check_response "$response" "[1,2]" 1 0

# 5️⃣ Initialize notification (id absent)
JSON='{"jsonrpc":"2.0","method":"initialize","params":{"rootUri":"file:///tmp"}}'
echo
echo "=== Test initialize notification (pas d'id) ==="
response=$(send_lsp_message "$JSON")
check_response "$response" "" 1 0

# 6️⃣ Shutdown valide (id=2)
JSON='{"jsonrpc":"2.0","id":2,"method":"shutdown","params":null}'
echo
echo "=== Test shutdown ==="
response=$(send_lsp_message "$JSON")
check_response "$response" "2" 0 0

# 7️⃣ Exit notification
JSON='{"jsonrpc":"2.0","id":99,"method":"exit"}'
echo
echo "=== Test exit notification ==="
response=$(send_lsp_message "$JSON")
check_response "$response" "" 0 1

# 8️⃣ Méthode inconnue (id=99)
JSON='{"jsonrpc":"2.0","id":99,"method":"unknownMethod","params":{}}'
echo
echo "=== Test méthode inconnue ==="
response=$(send_lsp_message "$JSON")
check_response "$response" "99" 1 0
