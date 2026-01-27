export PATH := $(realpath usr/bin):$(PATH)
CC=cosmocc
ZC=zc.com

all: out/hello.com

out/hello.com: hello.zc
	@mkdir -p out
	$(ZC) build --cc $(CC) -o $@ $<

clean:
	rm -rf out

.PHONY: all clean