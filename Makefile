
CC = gcc
CFLAGS = -Wall -Wextra -g -I./src -I./src/ast -I./src/parser -I./src/codegen -I./plugins -I./src/zen -I./src/utils -I./src/lexer -I./src/analysis -I./src/lsp -I./src/compat
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
       src/compat/compat_posix.c \
       plugins/befunge.c \
       plugins/brainfuck.c \
       plugins/forth.c \
       plugins/lisp.c \
       plugins/regex.c \
       plugins/sql.c

OBJ_DIR = obj
OBJS = $(patsubst %.c, $(OBJ_DIR)/%.o, $(SRCS))

# Installation paths
PREFIX = /usr/local
BINDIR = $(PREFIX)/bin
MANDIR = $(PREFIX)/share/man/man1

# Default target
all: $(TARGET)

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
	install -d $(MANDIR)
	# Install man page if it exists
	test -f man/zc.1 && install -m 644 man/zc.1 $(MANDIR)/zc.1 || true
	@echo "=> Installed to $(BINDIR)/$(TARGET)"

# Uninstall
uninstall:
	rm -f $(BINDIR)/$(TARGET)
	rm -f $(MANDIR)/zc.1
	@echo "=> Uninstalled from $(BINDIR)/$(TARGET)"

# Clean
clean:
	rm -rf $(OBJ_DIR) $(TARGET) out.c
	@echo "=> Clean complete!"

# Test
test: $(TARGET)
	./tests/run_tests.sh

.PHONY: all clean install uninstall test
