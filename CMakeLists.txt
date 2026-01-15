cmake_minimum_required(VERSION 3.17)

project (zc C)

file(GLOB SRCS CONFIGURE_DEPENDS src/*.c src/analysis/*.c src/ast/*.c src/codegen/*.c src/lexer/*.c src/lsp/*.c src/parser/*.c src/plugins/*.c src/repl/*.c src/utils/*.c src/zen/*.c)

add_executable(zc ${SRCS})

set(PLUGINS befunge brainfuck forth lisp regex sql)
foreach(plugin ${PLUGINS})
add_library(${plugin} MODULE plugins/${plugin}.c)    
endforeach()

target_include_directories(zc PRIVATE src plugins src/analysis src/ast src/codegen src/lexer src/lsp src/parser src/repl src/utils src/plugins src/zen)
