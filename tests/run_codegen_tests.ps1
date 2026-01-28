# Codegen Verification Test Runner (PowerShell)
$ZC = ".\zig-out\bin\zc.exe"
$TEST_DIR = "tests\codegen"
$PASSED = 0
$FAILED = 0

if (-not (Test-Path $ZC)) {
  Write-Host "Error: zc binary not found at $ZC."
  exit 1
}

Write-Host "** Running Codegen Verification Tests **"

# Test 1: Duplicate Typedefs
$TEST_NAME = "dedup_typedefs.zc"
$TEST_PATH = Join-Path $TEST_DIR $TEST_NAME

Write-Host -NoNewline "Testing $TEST_PATH (Duplicate Typedefs)... "

& $ZC $TEST_PATH --emit-c *> $null
$exit_code = $LASTEXITCODE

if ($exit_code -ne 0) {
  Write-Host "FAIL (Compilation error)"
  $FAILED++
} else {
  if (-not (Test-Path "out.c")) {
    Write-Host "FAIL (out.c not generated)"
    $FAILED++
  } else {
    # Expect exactly one occurrence
    $count = (Select-String -Path "out.c" -SimpleMatch "typedef struct Vec2f Vec2f;" | Measure-Object).Count
    if ($count -eq 1) {
      Write-Host "PASS"
      $PASSED++
    } else {
      Write-Host "FAIL (Found $count typedefs for Vec2f, expected 1)"
      $FAILED++
    }
  }
}

# Cleanup
Remove-Item -Force -ErrorAction SilentlyContinue out.c, a.out

Write-Host "----------------------------------------"
Write-Host "Summary:"
Write-Host "-> Passed: $PASSED"
Write-Host "-> Failed: $FAILED"
Write-Host "----------------------------------------"

if ($FAILED -ne 0) { exit 1 } else { exit 0 }
