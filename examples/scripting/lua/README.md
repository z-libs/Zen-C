# Lua Scripting with Zen-C

This examples demonstrates how to build your Zen-C application with Lua C bindings,<br>
and how you can call Lua code from Zen-C (.zc) context, or exposed Zen-C functions from Lua (.lua) context.

> Lua is a powerful, efficient, lightweight, embeddable scripting language, often used in games.

---

Example output:

![alt text](https://i.imgur.com/5vfGuVg.png)

## Setup

Get Lua 5.4 headers and lib:

```sh
./deps.sh
```

This will leave the headers (.h) and lib (.a) inside a folder called ``lua``.

## Build

```sh
zc build lua.zc -o lua
```

## Run

Either

```sh
zc run lua.zc
```

or (after compilation)

```sh
./lua
```

## Documentation

- https://www.lua.org/manual/5.4/manual.html
