# Zen-C Single Test Runner (PowerShell)
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\tests\run_single.ps1 <test_file.zc> [zc options]
#
# Examples:
#   powershell -ExecutionPolicy Bypass -File .\tests\run_single.ps1 tests/collections/test_string_suite.zc
#   powershell -ExecutionPolicy Bypass -File .\tests\run_single.ps1 tests/collections/test_string_suite.zc --cc zig
#   powershell -ExecutionPolicy Bypass -File .\tests\run_single.ps1 .\tests\basic\test_basics.zc --cc clang

$ZC = ".\zig-out\bin\zc.exe"

if (-not (Test-Path $ZC)) {
  Write-Host "Error: zc binary not found at $ZC. Please build it first."
  exit 1
}

if ($args.Length -lt 1) {
  Write-Host "Usage: .\tests\run_single.ps1 <test_file.zc> [zc options]"
  exit 2
}

# First argument is the test file; the rest are passed through to zc
$test_file = $args[0]
$zc_args = @()
if ($args.Length -gt 1) { $zc_args = $args[1..($args.Length - 1)] }

# Normalize path separators and resolve relative path
$test_file_norm = $test_file -replace '/', '\'
$test_file_path = Resolve-Path -ErrorAction SilentlyContinue $test_file_norm

if (-not $test_file_path) {
  Write-Host "Error: test file not found: $test_file"
  exit 2
}

# Determine compiler name (matches the suite script behavior)
$CC_NAME = "gcc (default)"
for ($i = 0; $i -lt $zc_args.Length; $i++) {
  if ($zc_args[$i] -eq "--cc" -and ($i + 1) -lt $zc_args.Length) {
    $CC_NAME = $zc_args[$i + 1]
    break
  }
}

Write-Host "** Running Zen C single test (compiler: $CC_NAME) **"
Write-Host -NoNewline "Testing $($test_file_path.Path)... "

# Equivalent of: zc run "<file>" <options> 2>&1
$out = & $ZC run $test_file_path.Path @zc_args 2>&1
$exit_code = $LASTEXITCODE

if ($exit_code -eq 0) {
  Write-Host "PASS"
  # Cleanup (same as suite script)
  Remove-Item -Force -ErrorAction SilentlyContinue a.out, out.c
  exit 0
} else {
  Write-Host "FAIL"
  # Show output on failure (helpful for single-test runner)
  if ($out) {
    Write-Host "---- zc output ----"
    $out | ForEach-Object { Write-Host $_ }
    Write-Host "-------------------"
  }
  # Cleanup
  Remove-Item -Force -ErrorAction SilentlyContinue a.out, out.c
  exit 1
}
