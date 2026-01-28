zig cc -shared -O2 -o brainfuck.dll brainfuck.c -Iplugins -target x86_64-windows-msvc -Wl,/EXPORT:z_plugin_init
