# Zen-C Test Suite Runner (PowerShell)
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\tests\run_tests.ps1 [zc options]
#
# Examples:
#   powershell -ExecutionPolicy Bypass -File .\tests\run_tests.ps1
#   powershell -ExecutionPolicy Bypass -File .\tests\run_tests.ps1 --cc clang
#   powershell -ExecutionPolicy Bypass -File .\tests\run_tests.ps1 --cc zig
#   powershell -ExecutionPolicy Bypass -File .\tests\run_tests.ps1 --cc tcc

$ZC = ".\zig-out\bin\zc.exe"
$TEST_DIR = "tests"
$PASSED = 0
$FAILED = 0
$FAILED_TESTS = @()

# Determine compiler name (matches bash behavior)
$CC_NAME = "gcc (default)"
for ($i = 0; $i -lt $args.Length; $i++) {
  if ($args[$i] -eq "--cc" -and ($i + 1) -lt $args.Length) {
    $CC_NAME = $args[$i + 1]
    break
  }
}

Write-Host "** Running Zen C test suite (compiler: $CC_NAME) **"

if (-not (Test-Path $ZC)) {
  Write-Host "Error: zc binary not found at $ZC. Please build it first."
  exit 1
}

# Find *.zc excluding _*.zc, sort
$test_files =
  Get-ChildItem -Path $TEST_DIR -Recurse -File -Filter "*.zc" |
  Where-Object { $_.Name -notlike "_*.zc" } |
  Sort-Object FullName

foreach ($f in $test_files) {
  $test_file = $f.FullName
  Write-Host -NoNewline "Testing $test_file... "

  # Equivalent of: zc run "$test_file" "$@" 2>&1
  $out = & $ZC run $test_file @args 2>&1
  $exit_code = $LASTEXITCODE

  if ($exit_code -eq 0) {
    Write-Host "PASS"
    $PASSED++
  } else {
    Write-Host "FAIL"
    $FAILED++
    $FAILED_TESTS += $test_file

    # If you want to see output immediately when failing, uncomment:
    # $out | ForEach-Object { Write-Host $_ }
  }
}

Write-Host "----------------------------------------"
Write-Host "Summary:"
Write-Host "-> Passed: $PASSED"
Write-Host "-> Failed: $FAILED"
Write-Host "----------------------------------------"

# Cleanup like bash script
Remove-Item -Force -ErrorAction SilentlyContinue a.out, out.c

if ($FAILED -ne 0) {
  Write-Host "Failed tests:"
  foreach ($t in $FAILED_TESTS) { Write-Host "- $t" }
  exit 1
} else {
  Write-Host "All tests passed!"
  exit 0
}
