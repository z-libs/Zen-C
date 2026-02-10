cmake_minimum_required(VERSION 3.10)
project(zc C)
include(GNUInstallDirs)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Derive ZEN_VERSION from git (similar to Makefile)
set(ZEN_VERSION "dev")
execute_process(
    COMMAND git describe --tags --always --dirty
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RESULT_VARIABLE GIT_DESCRIBE_RESULT
    OUTPUT_VARIABLE GIT_DESCRIBE_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(GIT_DESCRIBE_RESULT EQUAL 0 AND NOT "${GIT_DESCRIBE_OUTPUT}" STREQUAL "")
    set(ZEN_VERSION "${GIT_DESCRIBE_OUTPUT}")
endif()

# Set ZEN_SHARE_DIR for runtime file lookups (similar intent as Makefile)
if(NOT DEFINED CMAKE_INSTALL_FULL_DATADIR)
    set(CMAKE_INSTALL_FULL_DATADIR "${CMAKE_INSTALL_PREFIX}/share")
endif()
set(ZEN_SHARE_DIR "${CMAKE_INSTALL_FULL_DATADIR}/zc")

# Provide these as compile definitions for all targets
add_compile_definitions(
    ZEN_VERSION="${ZEN_VERSION}"
    ZEN_SHARE_DIR="${ZEN_SHARE_DIR}"
)
# --- Compiler flags (set per compiler) ---
set(ZC_COMPILE_FLAGS "")
if(WIN32)
    if(MSVC)
        set(ZC_COMPILE_FLAGS "/W4 /Zi /D_CRT_SECURE_NO_WARNINGS")
    elseif(CMAKE_C_COMPILER_ID MATCHES "Clang|AppleClang")
        set(ZC_COMPILE_FLAGS -Wall -Wextra -g)
    elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        set(ZC_COMPILE_FLAGS -Wall -Wextra -g)
    elseif(CMAKE_C_COMPILER_ID STREQUAL "TinyCC")
        set(ZC_COMPILE_FLAGS -Wall -g)
    endif()
else()
    if(CMAKE_C_COMPILER_ID MATCHES "Clang|AppleClang")
        set(ZC_COMPILE_FLAGS -Wall -Wextra -g)
    elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        set(ZC_COMPILE_FLAGS -Wall -Wextra -g)
    elseif(CMAKE_C_COMPILER_ID STREQUAL "TinyCC")
        set(ZC_COMPILE_FLAGS -Wall -g)
    endif()
endif()

# --- Find Required Packages (Modern CMake Way) ---
set(THREADS_PREFER_PTHREAD_FLAG ON)
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
    ${PROJECT_SOURCE_DIR}/src/diagnostics
)

# Source files
set(SRCS
    src/main.c
    src/diagnostics/diagnostics.c
    src/parser/parser_core.c
    src/parser/parser_decl.c
    src/parser/parser_expr.c
    src/parser/parser_stmt.c
    src/parser/parser_struct.c
    src/parser/parser_type.c
    src/parser/parser_utils.c
    src/ast/ast.c
    src/codegen/codegen.c
    src/codegen/codegen_decl.c
    src/codegen/codegen_main.c
    src/codegen/codegen_utils.c
    src/codegen/codegen_stmt.c
    src/utils/utils.c
    src/lexer/token.c
    src/analysis/typecheck.c
    src/lsp/cJSON.c
    src/lsp/json_rpc.c
    src/lsp/lsp_main.c
    src/lsp/lsp_project.c
    src/lsp/lsp_analysis.c
    src/lsp/lsp_index.c
    src/lsp/lsp_semantic.c
    src/zen/zen_facts.c
    src/repl/repl.c
    src/repl/repl_os.c
    src/plugins/plugin_manager.c
)

# Platform-specific library logic
set(PLATFORM_LIBS Threads::Threads)
if(WIN32)
    # Winsock 2 for Networking, ws2_32 is required for our socket abstraction
    list(APPEND PLATFORM_LIBS ws2_32)
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
target_link_libraries(zc PRIVATE ${PLATFORM_LIBS})
if(NOT "${ZC_COMPILE_FLAGS}" STREQUAL "")
    target_compile_options(zc PRIVATE ${ZC_COMPILE_FLAGS})
endif()

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
    if(NOT "${SHARED_FLAGS}" STREQUAL "")
        target_compile_options(${plugin}_plugin PRIVATE ${SHARED_FLAGS})
    endif()
endforeach()

# --- Install Rules ---
install(TARGETS zc DESTINATION bin)
install(DIRECTORY std DESTINATION share/zenc)
install(FILES src/zen/facts.json DESTINATION share/zenc)
install(FILES src/repl/docs.json DESTINATION share/zenc)
install(FILES plugins/zprep_plugin.h DESTINATION include/zenc)
install(DIRECTORY ${CMAKE_BINARY_DIR}/plugins/ DESTINATION lib/zenc/plugins 
        FILES_MATCHING PATTERN "*.${PLUGIN_EXT}")

if(UNIX AND NOT WIN32)
    install(FILES man/zc.1 DESTINATION share/man/man1)
    install(FILES man/zc.5 DESTINATION share/man/man5)
    install(FILES man/zc.7 DESTINATION share/man/man7)
endif()
# Test target
add_custom_target(test
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/scripts/run_tests.sh
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