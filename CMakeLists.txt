# --- Embed compat.h as embed.h before build ---
add_executable(header2string src/utils/header2string.c)
add_custom_command(
    OUTPUT src/compat/embed.h
    COMMAND header2string ${CMAKE_SOURCE_DIR}/src/compat/compat.h ${CMAKE_SOURCE_DIR}/src/compat/embed.h
    DEPENDS header2string ${CMAKE_SOURCE_DIR}/src/compat/compat.h
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Embedding compat.h as embed.h"
)
add_custom_target(embed_compat ALL DEPENDS src/compat/embed.h)
cmake_minimum_required(VERSION 3.10)
project(zc C)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Compiler flags (set per compiler)
if(MSVC)
    # MSVC: Enable warnings, debug info, disable security warnings
    add_compile_options(/W4 /Zi /D_CRT_SECURE_NO_WARNINGS)
elseif(CMAKE_C_COMPILER_ID MATCHES "Clang|AppleClang")
    add_compile_options(-Wall -Wextra -g -lws2_32)
elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wall -Wextra -g -lws2_32)
endif()

# --- Find Required Packages (Modern CMake Way) ---
find_package(Threads REQUIRED)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/ast
    ${PROJECT_SOURCE_DIR}/src/parser
    ${PROJECT_SOURCE_DIR}/src/codegen
    ${PROJECT_SOURCE_DIR}/plugins
    ${PROJECT_SOURCE_DIR}/src/zen
    ${PROJECT_SOURCE_DIR}/src/utils
    ${PROJECT_SOURCE_DIR}/src/lexer
    ${PROJECT_SOURCE_DIR}/src/analysis
    ${PROJECT_SOURCE_DIR}/src/lsp
)

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
)

# Platform-specific library logic
set(PLATFORM_LIBS Threads::Threads) # Standard thread linking

if(WIN32)
    # Winsock 2 for Networking, ws2_32 is required for our socket abstraction
    list(APPEND PLATFORM_LIBS ws2_32)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    set(PLUGIN_EXT ".dll")
    set(BINARY_EXT ".exe")
    set(SHARED_FLAGS "")
elseif(APPLE)
    # macOS requires CoreFoundation for _NSGetExecutablePath
    find_library(COREFOUNDATION_FRAMEWORK CoreFoundation)
    list(APPEND PLATFORM_LIBS ${CMAKE_DL_LIBS} ${COREFOUNDATION_FRAMEWORK})
    set(SHARED_FLAGS "-fPIC")
    set(BINARY_EXT "")
    set(PLUGIN_EXT ".dylib")
else()
    # Linux/BSD: Link with libdl for zc_dlopen and libm for math
    list(APPEND PLATFORM_LIBS ${CMAKE_DL_LIBS} m)
    set(SHARED_FLAGS "-fPIC")
    set(BINARY_EXT "")
    set(PLUGIN_EXT ".so")
endif()

# Add executable
add_executable(zc ${SRCS})
add_dependencies(zc embed_compat)
target_link_libraries(zc PRIVATE ${PLATFORM_LIBS})

# Build plugins as shared libraries
set(PLUGIN_NAMES befunge brainfuck forth lisp regex sql)
foreach(plugin ${PLUGIN_NAMES})
    add_library(${plugin}_plugin MODULE plugins/${plugin}.c)
    
    # Plugins need to link against the same platform libs (especially for zc_compat symbols)
    target_link_libraries(${plugin}_plugin PRIVATE ${PLATFORM_LIBS})
    
    set_target_properties(${plugin}_plugin PROPERTIES
        OUTPUT_NAME ${plugin}
        PREFIX ""
        SUFFIX ${PLUGIN_EXT}
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins
    )
    target_compile_options(${plugin}_plugin PRIVATE ${SHARED_FLAGS})
endforeach()

# --- Install Rules ---
install(TARGETS zc DESTINATION bin)
install(DIRECTORY std DESTINATION share/zenc)
install(FILES plugins/zprep_plugin.h DESTINATION include/zenc)
install(DIRECTORY ${CMAKE_BINARY_DIR}/plugins/ DESTINATION lib/zenc/plugins 
        FILES_MATCHING PATTERN "*.${PLUGIN_EXT}")

# Test target
add_custom_target(test
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_tests.sh
    DEPENDS zc
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Clean target
add_custom_target(distclean
    COMMAND ${CMAKE_COMMAND} --build . --target clean
    COMMAND ${CMAKE_COMMAND} -E rm -f out.c
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_BINARY_DIR}/plugins
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)