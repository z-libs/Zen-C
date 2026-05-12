@echo off
setlocal

rem Compiler configuration (default to gcc)
if "%CC%"=="" set CC=gcc

rem Version
set ZEN_VERSION=0.1.0
for /f "delims=" %%i in ('git describe --tags --always --dirty 2^>nul') do set ZEN_VERSION=%%i

rem Compilation flags
set CFLAGS=-std=gnu11 -Wall -Wextra -Wshadow -g ^
 -I./src -I./src/ast -I./src/parser -I./src/codegen -I./plugins -I./src/zen ^
 -I./src/utils -I./src/lexer -I./src/analysis -I./src/lsp -I./src/diagnostics ^
 -I./std/third-party/tre/include
set CFLAGS=%CFLAGS% -DZEN_VERSION=\"%ZEN_VERSION%\" -DZEN_SHARE_DIR=\".\" -DZC_ALLOW_INTERNAL

if "%ZC_HAS_JIT%"=="" set ZC_HAS_JIT=1
if "%ZC_HAS_JIT%"=="1" (
    set CFLAGS=%CFLAGS% -DZC_HAS_JIT
    set LIBS=-lws2_32 -ltcc
) else (
    set LIBS=-lws2_32
)

if "%NO_PLUGINS%"=="1" (
    set CFLAGS=%CFLAGS% -DZC_NO_PLUGINS
)

rem Source files
REM Source list duplicated from src-sources.txt (keep in sync!)
REM Primary build files: Makefile and CMakeLists.txt read src-sources.txt directly.
set SRCS=src\main.c ^
 src\parser\parser_core.c ^
 src\parser\parser_expr.c ^
 src\parser\parser_stmt.c ^
 src\parser\parser_type.c ^
 src\parser\parser_utils.c ^
 src\parser\parser_decl.c ^
 src\parser\parser_struct.c ^
 src\ast\ast.c ^
 src\ast\primitives.c ^
 src\ast\symbols.c ^
 src\codegen\codegen.c ^
 src\codegen\codegen_stmt.c ^
 src\codegen\codegen_decl.c ^
 src\codegen\codegen_main.c ^
 src\codegen\codegen_utils.c ^
 src\codegen\codegen_shared.c ^
 src\codegen\codegen_backend_c.c ^
 src\codegen\codegen_backend_astdump.c ^
 src\utils\emitter.c ^
 src\utils\format_expr.c ^
 src\utils\utils.c ^
 src\utils\colors.c ^
 src\utils\cmd.c ^
 src\platform\os.c ^
 src\platform\console.c ^
 src\platform\dylib.c ^
 src\platform\misra.c ^
 src\utils\config.c ^
 src\diagnostics\diagnostics.c ^
 src\driver\driver.c ^
 src\lexer\token.c ^
 src\analysis\typecheck.c ^
 src\analysis\typecheck_expr.c ^
 src\analysis\typecheck_stmt.c ^
 src\analysis\comptime_interpreter.c ^
 src\analysis\move_check.c ^
 src\analysis\const_fold.c ^
 src\lsp\json_rpc.c ^
 src\lsp\lsp_main.c ^
 src\lsp\lsp_formatter.c ^
 src\lsp\lsp_analysis.c ^
 src\lsp\lsp_semantic.c ^
 src\lsp\lsp_index.c ^
 src\lsp\lsp_project.c ^
 src\lsp\cJSON.c ^
 src\zen\zen_facts.c ^
 src\zen\zen_doc.c ^
 src\repl\repl.c ^
 src\repl\repl_highlight.c ^
 src\repl\repl_readline.c ^
 src\repl\repl_eval.c ^
 src\repl\repl_jit.c ^
 src\repl\repl_commands.c ^
 src\plugins\plugin_manager.c ^
 src\plugins\static_plugins.c ^
 std\third-party\tre\lib\regcomp.c ^
 std\third-party\tre\lib\regerror.c ^
 std\third-party\tre\lib\regexec.c ^
 std\third-party\tre\lib\tre-ast.c ^
 std\third-party\tre\lib\tre-compile.c ^
 std\third-party\tre\lib\tre-filter.c ^
 std\third-party\tre\lib\tre-match-approx.c ^
 std\third-party\tre\lib\tre-match-backtrack.c ^
 std\third-party\tre\lib\tre-match-parallel.c ^
 std\third-party\tre\lib\tre-mem.c ^
 std\third-party\tre\lib\tre-parse.c ^
 std\third-party\tre\lib\tre-stack.c ^
 std\third-party\tre\lib\xmalloc.c

rem Build
echo Building Zen C (%ZEN_VERSION%)...
%CC% %CFLAGS% %SRCS% -o zc.exe %LIBS%
if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    exit /b %ERRORLEVEL%
)

echo Build success! zc.exe created.

rem Build plugins
if "%NO_PLUGINS%"=="1" (
    echo Plugins disabled by NO_PLUGINS flag.
    goto end
)
echo Building plugins...
if not exist plugins mkdir plugins
for %%f in (plugins\*.zc) do (
    echo Compiling native plugin %%f...
    .\zc.exe build %%f -shared -o %%~dpnf.dll
    if errorlevel 1 (
        echo Plugin build failed for %%f
        exit /b 1
    )
)

:end
