cmake_minimum_required(VERSION 3.10)
project(ZenC C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Compiler warnings
if(MSVC)
    add_compile_options(/W4)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra -g)
endif()

# Platform-specific compat layer
if(WIN32)
    set(COMPAT_SRC src/compat/compat_win32.c)
else()
    set(COMPAT_SRC src/compat/compat_posix.c)
endif()

# Source files
set(SRCS
    src/main.c
    src/parser/parser_core.c
    src/parser/parser_expr.c
    src/parser/parser_stmt.c
    src/parser/parser_type.c
    src/parser/parser_utils.c
    src/ast/ast.c
    src/codegen/codegen.c
    src/codegen/codegen_decl.c
    src/codegen/codegen_main.c
    src/codegen/codegen_utils.c
    src/utils/utils.c
    src/lexer/token.c
    src/analysis/typecheck.c
    src/lsp/json_rpc.c
    src/lsp/lsp_main.c
    src/lsp/lsp_analysis.c
    src/lsp/lsp_index.c
    src/zen/zen_facts.c
    src/repl/repl.c
    src/plugins/plugin_manager.c
    ${COMPAT_SRC}
    plugins/befunge.c
    plugins/brainfuck.c
    plugins/forth.c
    plugins/lisp.c
    plugins/regex.c
    plugins/sql.c
)

# Executable
add_executable(zc ${SRCS})

# Include directories
target_include_directories(zc PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/ast
    ${CMAKE_SOURCE_DIR}/src/parser
    ${CMAKE_SOURCE_DIR}/src/codegen
    ${CMAKE_SOURCE_DIR}/src/zen
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/src/lexer
    ${CMAKE_SOURCE_DIR}/src/analysis
    ${CMAKE_SOURCE_DIR}/src/lsp
    ${CMAKE_SOURCE_DIR}/src/compat
    ${CMAKE_SOURCE_DIR}/plugins
)

# Platform-specific libraries
if(WIN32)
    # Windows: no -lm, -ldl, pthread handled via Windows API
    # Nothing to link
else()
    # Unix/Linux/macOS
    target_link_libraries(zc PRIVATE m pthread dl)
endif()

# Installation
install(TARGETS zc RUNTIME DESTINATION bin)
install(FILES man/zc.1 DESTINATION share/man/man1 OPTIONAL)

# Output build info
message(STATUS "Building Zen-C compiler")
message(STATUS "Platform: ${CMAKE_SYSTEM_NAME}")
