# Installation paths (must be before CFLAGS)
PREFIX = /usr/local
BINDIR = $(PREFIX)/bin
LIBDIR = $(PREFIX)/lib
DATADIR = $(PREFIX)/share/zc
MANDIR = $(PREFIX)/share/man
PLUGINDIR = $(LIBDIR)/zc/plugins

# Compiler configuration
# Default: gcc
# To build with clang: make CC=clang
# To build with zig:   make CC="zig cc"
CC = gcc
CFLAGS = -Wall -Wextra -g -I./src -I./src/ast -I./src/parser -I./src/codegen -I./plugins -I./src/zen -I./src/utils -I./src/lexer -I./src/analysis -I./src/lsp -I./src/compat -DZC_STD_INSTALL_PATH='"$(DATADIR)"' -DZC_PLUGIN_INSTALL_PATH='"$(PLUGINDIR)"'
TARGET = zc
LIBS = -lm -lpthread -ldl

SRCS = src/main.c \
       src/parser/parser_core.c \
       src/parser/parser_expr.c \
       src/parser/parser_stmt.c \
       src/parser/parser_type.c \
       src/parser/parser_utils.c \
       src/ast/ast.c \
       src/codegen/codegen.c \
       src/codegen/codegen_decl.c \
       src/codegen/codegen_main.c \
       src/codegen/codegen_utils.c \
       src/utils/utils.c \
       src/lexer/token.c \
       src/analysis/typecheck.c \
       src/lsp/json_rpc.c \
       src/lsp/lsp_main.c \
       src/lsp/lsp_analysis.c \
       src/lsp/lsp_index.c \
       src/zen/zen_facts.c \
       src/repl/repl.c \
       src/plugins/plugin_manager.c \
       src/compat/compat_posix.c

OBJ_DIR = obj
OBJS = $(patsubst %.c, $(OBJ_DIR)/%.o, $(SRCS))

PLUGINS = plugins/befunge.so plugins/brainfuck.so plugins/forth.so plugins/lisp.so plugins/regex.so plugins/sql.so

# Default target
all: $(TARGET) $(PLUGINS)

# Build plugins
plugins/%.so: plugins/%.c
	$(CC) $(CFLAGS) -shared -fPIC -o $@ $<

# Link
$(TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)
	@echo "=> Build complete: $(TARGET)"

# Compile
$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# Install
install: $(TARGET)
	install -d $(BINDIR)
	install -m 755 $(TARGET) $(BINDIR)/$(TARGET)
	
	# Install man pages
	install -d $(MANDIR)/man1 $(MANDIR)/man5 $(MANDIR)/man7
	test -f man/zc.1 && install -m 644 man/zc.1 $(MANDIR)/man1/zc.1 || true
	test -f man/zc.5 && install -m 644 man/zc.5 $(MANDIR)/man5/zc.5 || true
	test -f man/zc.7 && install -m 644 man/zc.7 $(MANDIR)/man7/zc.7 || true
	
	# Install standard library
	install -d $(DATADIR)/std
	install -m 644 std.zc $(DATADIR)/
	install -m 644 std/*.zc $(DATADIR)/std/ 2>/dev/null || true
	install -m 644 std/*.h $(DATADIR)/std/ 2>/dev/null || true
	
	# Install plugin headers
	install -d $(DATADIR)/include
	install -m 644 plugins/zprep_plugin.h $(DATADIR)/include/zprep_plugin.h
	
	# Install plugins
	install -d $(PLUGINDIR)
	install -m 755 plugins/*.so $(PLUGINDIR)/ 2>/dev/null || true
	@echo "=> Installed to $(BINDIR)/$(TARGET)"
	@echo "=> Man pages installed to $(MANDIR)"
	@echo "=> Standard library installed to $(DATADIR)/"
	@echo "=> Plugins installed to $(PLUGINDIR)/"

# Uninstall
uninstall:
	rm -f $(BINDIR)/$(TARGET)
	rm -f $(MANDIR)/man1/zc.1
	rm -f $(MANDIR)/man5/zc.5
	rm -f $(MANDIR)/man7/zc.7
	rm -rf $(DATADIR)
	rm -rf $(PLUGINDIR)
	@echo "=> Uninstalled $(TARGET)"

# Clean
clean:
	rm -rf $(OBJ_DIR) $(TARGET) out.c plugins/*.so
	@echo "=> Clean complete!"

# Test
test: $(TARGET)
	./tests/run_tests.sh
	./tests/run_codegen_tests.sh

# Build with alternative compilers
zig:
	$(MAKE) CC="zig cc"

clang:
	$(MAKE) CC=clang

.PHONY: all clean install uninstall test zig clang
