@echo off
REM Zen-C Test Suite Runner
REM Usage: tests\run_tests.cmd [zc options]
REM
REM Examples:
REM   tests\run_tests.cmd                    Test with default compiler (gcc)
REM   tests\run_tests.cmd --cc clang         Test with clang
REM   tests\run_tests.cmd --cc zig           Test with zig cc
REM   tests\run_tests.cmd --cc tcc           Test with tcc

setlocal enabledelayedexpansion

REM Configuration
set ZC=zc.exe
set TEST_DIR=tests
set PASSED=0
set FAILED=0
set FAILED_TESTS=

REM Display which compiler is being used
set CC_NAME=gcc (default)
set prev_arg=
for %%A in (%*) do (
    if "!prev_arg!"=="--cc" (
        set CC_NAME=%%A
        goto :cc_found
    )
    set prev_arg=%%A
)

:cc_found
echo ** Running Zen C test suite (compiler: !CC_NAME!) **

if not exist "!ZC!" (
    echo Error: zc.exe binary not found. Please build it first.
    exit /b 1
)

REM Find and run all .zc test files
for /r "%TEST_DIR%" %%F in (*.zc) do (
    set test_file=%%F
    set test_name=%%~nxF
    
    setlocal enabledelayedexpansion
    set /p =Testing !test_name!... <nul
    
    REM Run the test and capture output
    for /f "delims=" %%O in ('!ZC! run "!test_file!" %* 2^>^&1') do (
        set output=%%O
    )
    
    if !errorlevel! equ 0 (
        echo PASS
        set /a PASSED=!PASSED!+1
    ) else (
        echo FAIL
        set /a FAILED=!FAILED!+1
        set FAILED_TESTS=!FAILED_TESTS! - !test_name!
    )
    endlocal
)

echo ----------------------------------------
echo Summary:
echo ^-> Passed: !PASSED!
echo ^-> Failed: !FAILED!
echo ----------------------------------------

if !FAILED! neq 0 (
    echo Failed tests:
    for %%T in (!FAILED_TESTS!) do echo %%T
    if exist a.exe del a.exe
    if exist out.c del out.c
    exit /b 1
) else (
    echo All tests passed!
    if exist a.exe del a.exe
    if exist out.c del out.c
    exit /b 0
)
