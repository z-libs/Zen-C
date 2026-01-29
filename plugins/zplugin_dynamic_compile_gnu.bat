zig cc -shared -O2 -o brainfuck.dll brainfuck.c -Iplugins -target x86_64-windows-gnu -Wl,--export-all-symbols
