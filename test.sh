#!/usr/bin/env bash

LSP_BIN=./zc

JSON='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"rootUri":"file:///home/judanana/code/Zen-C_fork/examples/algorithms/binsearch.zc"}}'

LEN=$(printf "%s" "$JSON" | wc -c)

{
  printf "Content-Length: %d\r\n" "$LEN"
  printf "\r\n"
  printf "%s" "$JSON"
} | "$LSP_BIN" lsp
