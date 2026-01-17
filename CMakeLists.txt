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
)

# Plugin sources
set(PLUGIN_SRCS
    plugins/befunge.c
    plugins/brainfuck.c
    plugins/forth.c
    plugins/lisp.c
    plugins/regex.c
    plugins/sql.c
)

# Executable
add_executable(zc ${SRCS})

# Build plugins as shared libraries
foreach(plugin_src ${PLUGIN_SRCS})
    get_filename_component(plugin_name ${plugin_src} NAME_WE)
    add_library(${plugin_name} SHARED ${plugin_src})
    target_include_directories(${plugin_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/plugins
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/compat
    )
    set_target_properties(${plugin_name} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins"
    )
    if(WIN32)
        set_target_properties(${plugin_name} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug/plugins"
            LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release/plugins"
            RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug/plugins"
            RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release/plugins"
        )
    endif()
endforeach()

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
include(GNUInstallDirs)

install(TARGETS zc RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Standard library location (Windows: alongside exe, Unix: share/zc)
if(WIN32)
    set(ZC_DATA_DIR ${CMAKE_INSTALL_BINDIR})
    set(ZC_STD_INSTALL_PATH "")  # Empty = search relative to exe
else()
    set(ZC_DATA_DIR ${CMAKE_INSTALL_DATADIR}/zc)
    set(ZC_STD_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}/${ZC_DATA_DIR}")
endif()

# Plugin install path
if(WIN32)
    set(ZC_PLUGIN_INSTALL_PATH "")  # Empty = search relative to exe
else()
    set(ZC_PLUGIN_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/zc/plugins")
endif()

# Pass install paths to compiler
target_compile_definitions(zc PRIVATE
    ZC_STD_INSTALL_PATH="${ZC_STD_INSTALL_PATH}"
    ZC_PLUGIN_INSTALL_PATH="${ZC_PLUGIN_INSTALL_PATH}"
)

install(DIRECTORY std/ DESTINATION ${ZC_DATA_DIR}/std)
install(FILES std.zc DESTINATION ${ZC_DATA_DIR})

# Install plugins
foreach(plugin_src ${PLUGIN_SRCS})
    get_filename_component(plugin_name ${plugin_src} NAME_WE)
    if(WIN32)
        # Windows: install DLLs alongside exe in bin/plugins/
        install(TARGETS ${plugin_name} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/plugins)
    else()
        # Unix: install .so to lib/zc/plugins/
        install(TARGETS ${plugin_name} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/zc/plugins)
    endif()
endforeach()

# Man page (Unix only)
if(NOT WIN32)
    install(FILES man/zc.1 DESTINATION ${CMAKE_INSTALL_MANDIR}/man1 OPTIONAL)
endif()

# Output build info
message(STATUS "Building Zen-C compiler")
message(STATUS "Platform: ${CMAKE_SYSTEM_NAME}")
