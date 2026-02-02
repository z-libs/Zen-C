# Zen-C Test Suite Runner (PowerShell)
# Usage: .\run_tests.ps1 [zc options]
#
# Examples:
#   .\run_tests.ps1                    # Test with default compiler (gcc)
#   .\run_tests.ps1 --cc clang         # Test with clang
#   .\run_tests.ps1 --cc zig           # Test with zig cc
#   .\run_tests.ps1 --cc tcc           # Test with tcc

if (Test-Path "./build/Debug/zc.exe") {
    $ZC = "./build/Debug/zc.exe"
} elseif (Test-Path "./build/zc.exe") {
    $ZC = "./build/zc.exe"
} else {
    Write-Host "Error: zc binary not found in ./build/Debug or ./build. Please build it first."
    exit 1
}
$TEST_DIR = "tests"
$PASSED = 0
$FAILED = 0
$FAILED_TESTS = @()

# Display which compiler is being used
$CC_NAME = "gcc (default)"
for ($i = 0; $i -lt $args.Count; $i++) {
    if ($args[$i - 1] -eq "--cc") {
        $CC_NAME = $args[$i]
        break
    }
}

Write-Host "** Running Zen C test suite (compiler: $CC_NAME) **"

if (!(Test-Path $ZC)) {
    Write-Host "Error: zc binary not found. Please build it first."
    exit 1
}

$testFiles = Get-ChildItem -Path $TEST_DIR -Filter *.zc -Recurse | Sort-Object FullName
foreach ($testFile in $testFiles) {
    $baseName = $testFile.Name
    $padLen = 60
    $msg = "Testing $baseName..."
    if ($msg.Length -lt $padLen) {
        $msg = $msg + (" " * ($padLen - $msg.Length))
    }
    Write-Host -NoNewline $msg
    $output = & $ZC run "$($testFile.FullName)" --emit-c @args 2>&1
    $exitCode = $LASTEXITCODE
    $outC = "$($testFile.FullName).c"
    # Wait for out.c to be created (up to 5 seconds)
    $tries = 0
    while (-not (Test-Path "out.c") -and $tries -lt 50) {
        Start-Sleep -Milliseconds 100
        $tries++
    }
    if (Test-Path "out.c") {
        Move-Item -Force "out.c" $outC
    } else {
        Write-Host "WARN: No output file generated." -ForegroundColor Yellow
    }
    if ($exitCode -eq 0) {
        Write-Host "PASS" -ForegroundColor Blue
        $PASSED++
    } else {
        Write-Host "FAIL" -ForegroundColor Red
        $FAILED++
        $FAILED_TESTS += "- $baseName"
        Write-Host "Output:"
        $output -split "`r?`n" | ForEach-Object { Write-Host $_ }
    }
}

Write-Host "----------------------------------------"
Write-Host "Summary:"
Write-Host "-> Passed: $PASSED" -ForegroundColor Blue
Write-Host "-> Failed: $FAILED" -ForegroundColor Red
Write-Host "----------------------------------------"

if ($FAILED -ne 0) {
    Write-Host "Failed tests:"
    $FAILED_TESTS | ForEach-Object { Write-Host $_ -ForegroundColor Red }
    Remove-Item -ErrorAction SilentlyContinue a.out, out.c
    exit 1
} else {
    Write-Host "All tests passed!" -ForegroundColor Blue
    Remove-Item -ErrorAction SilentlyContinue a.out, out.c
    exit 0
}
