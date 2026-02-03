import os
import sys
import subprocess
import argparse
import tempfile
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import time

import re

def strip_ansi(text):
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

import shutil
import time
import tempfile

def safe_copy(src: Path, dst: Path, retries=3, delay=0.1):
    """Attempts to copy a file, retrying if it's locked or busy."""
    for i in range(retries):
        try:
            # If the file is a compiler artifact (like .c or .exe), 
            # it might be locked. We can often just skip these.
            if src.suffix in ['.c', '.exe', '.obj', '.o']:
                return 
            
            shutil.copy2(src, dst)
            return
        except (PermissionError, OSError):
            if i < retries - 1:
                time.sleep(delay)
            else:
                # If it's still failing, it's likely a file we don't 
                # actually need for the test (like a locked log or binary)
                pass 

def run_test(test_path: Path, zc_path: Path, extra_args: list):
    test_name = test_path.name
    orig_dir = test_path.parent

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        shutil.copytree("std", tmp_path / "std")
        shutil.copy("std.zc", tmp_path / "std.zc")

        # 1. Improved Copy Logic
        for item in orig_dir.iterdir():
            dest = tmp_path / item.name
            if item.is_file():
                # ONLY copy .zc files and non-build artifacts
                # This prevents Test B from trying to copy Test A's output
                safe_copy(item, dest)
            elif item.is_dir():
                # Avoid copying build folders if they exist inside tests
                if item.name not in ["build", "bin", "obj"]:
                    shutil.copytree(item, dest, dirs_exist_ok=True, 
                                    ignore=shutil.ignore_patterns('*.c', '*.exe'))

        # 2. RUN THE COMPILER
        # Crucial: Output to the TEMP directory, NOT the source directory
        tmp_output_c = tmp_path / (test_path.stem + ".zc.c")
        tmp_test_file = tmp_path / test_path.name

        cmd = [
            str(zc_path.absolute()), 
            "run", str(tmp_test_file), 
            "-o", str(tmp_output_c)
        ] + extra_args

        try:
            proc = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=60)
            output = proc.stdout + proc.stderr

            if proc.returncode != 0:
                return test_name, False, output

            # 3. SUCCESS - If you want the .c file for debugging, 
            # copy it back AFTER the test is done.
            final_dest = orig_dir / (test_path.name + ".c")
            shutil.copy2(tmp_output_c, final_dest)

            return test_name, True, output
        except Exception as e:
            return test_name, False, f"Exception: {str(e)}"

def find_zc_binary():
    build_dirs = [Path("."), Path("./build"), Path("./build/Debug"), Path("./build/Release")]
    name = "zc.exe" if os.name == "nt" else "zc"
    for d in build_dirs:
        if (d / name).is_file(): return d / name
    return None

def main():
    parser = argparse.ArgumentParser(description="Zen C Test Suite Runner")
    parser.add_argument("--cc", default="gcc", help="C compiler backend to use")
    parser.add_argument("--dir", default="tests", help="Directory containing .zc tests")
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count(), help="Number of parallel jobs")
    args, extra = parser.parse_known_args()

    zc_path = find_zc_binary()
    if not zc_path:
        print(f"{RED}Error: 'zc' binary not found.{RESET}")
        sys.exit(1)
    else:
        zc_path = zc_path.absolute()

    test_dir = Path(args.dir)
    # Sorting ensures the submission order is deterministic
    test_files = sorted(list(test_dir.rglob("*.zc")))
    test_files = [f.resolve() for f in test_files]
    
    if not test_files:
        print(f"No tests found in {test_dir}")
        sys.exit(0)

    # Setup Log File
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"test_results_{args.cc}_{timestamp}.log"
    log_path = test_dir / log_name
    log_entries = []

    header = f"** Zen C Test Suite: {args.cc} **\nStarted: {timestamp}\n"
    print(f"{BOLD}{header}{RESET}")
    log_entries.append(header)

    passed_count = 0
    failed_tests = []
    zc_extra_args = ["--cc", args.cc] + extra

    # Using a list to keep track of futures in order
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        # 1. Submit all tests and store the 'future' objects in a list
        # This preserves the sorted order from test_files
        future_list = [
            executor.submit(run_test, f, zc_path, zc_extra_args) 
            for f in test_files
        ]
        
        # 2. Iterate through the futures in the order they were submitted
        for i, future in enumerate(future_list, 1):
            # This blocks until THIS specific test is done
            name, success, log = future.result()
            
            status_text = "PASS" if success else "FAIL"
            color = GREEN if success else RED
            
            # Print to console in order
            print(f"[{i:3}/{len(test_files)}] Testing {name.ljust(40)} {color}{status_text}{RESET}")
            
            log_entries.append(f"[{status_text}] {name}")
            
            if success:
                passed_count += 1
            else:
                failed_tests.append((name, log))

    # Summary Generation
    summary = (
        "\n" + "-"*50 + 
        f"\nSummary {args.cc}:\n  Passed: {passed_count}\n  Failed: {len(failed_tests)}\n" + 
        "-"*50 + "\n"
    )
    
    if failed_tests:
        summary += "\nFailed Test Details:\n"
        for name, log in failed_tests:
            summary += f"\n--- {name} ---\n{log}\n"

    print(summary)
    log_entries.append(summary)
    
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(strip_ansi(entry) for entry in log_entries))

    print(f"{BOLD}Log saved to:{RESET} {log_path}")
    sys.exit(0 if not failed_tests else 1)

if __name__ == "__main__":
    main()