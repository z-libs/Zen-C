const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const c_flags = [_][]const u8{
        "-Wall",
        "-Wextra",
        "-g",
    };

    const include_dirs = [_][]const u8{
        "src",
        "src/ast",
        "src/parser",
        "src/codegen",
        "plugins",
        "src/zen",
        "src/utils",
        "src/lexer",
        "src/analysis",
        "src/lsp",
    };

    const zc_sources = [_][]const u8{
        "src/main.c",
        "src/parser/parser_core.c",
        "src/parser/parser_expr.c",
        "src/parser/parser_stmt.c",
        "src/parser/parser_type.c",
        "src/parser/parser_utils.c",
        "src/parser/parser_decl.c",
        "src/parser/parser_struct.c",
        "src/ast/ast.c",
        "src/codegen/codegen.c",
        "src/codegen/codegen_stmt.c",
        "src/codegen/codegen_decl.c",
        "src/codegen/codegen_main.c",
        "src/codegen/codegen_utils.c",
        "src/utils/utils.c",
        "src/utils/path_utils.c",
        "src/utils/zc_path_resolve.c",
        "src/lexer/token.c",
        "src/analysis/typecheck.c",
        "src/lsp/json_rpc.c",
        "src/lsp/lsp_main.c",
        "src/lsp/lsp_analysis.c",
        "src/lsp/lsp_index.c",
        "src/lsp/lsp_project.c",
        "src/lsp/cJSON.c",
        "src/zen/zen_facts.c",
        "src/repl/repl.c",
        "src/plugins/plugin_manager.c",
    };

    // --- zc executable (C project) ---
    const exe = b.addExecutable(.{
        .name = "zc",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,

            // replacement for the removed exe.linkLibC()
            .link_libc = true,
        }),
    });

    for (include_dirs) |p| {
        exe.root_module.addIncludePath(b.path(p));
    }

    exe.root_module.addCSourceFiles(.{
        .files = &zc_sources,
        .flags = &c_flags,
    });

    // Match Makefile libs where relevant
    const os_tag = target.result.os.tag;
    switch (os_tag) {
        .linux => {
            exe.root_module.linkSystemLibrary("m", .{});
            exe.root_module.linkSystemLibrary("pthread", .{});
            exe.root_module.linkSystemLibrary("dl", .{});
        },
        .freebsd, .netbsd, .openbsd, .dragonfly => {
            exe.root_module.linkSystemLibrary("m", .{});
            exe.root_module.linkSystemLibrary("pthread", .{});
        },
        .macos => {
            // Typically no explicit -lm/-lpthread needed
        },
        .windows => {
            // -lm/-lpthread/-ldl don't apply
        },
        else => {},
    }

    b.installArtifact(exe);

    // --- plugins as dynamic libraries ---
    const plugin_names = [_][]const u8{
        "befunge",
        "brainfuck",
        "forth",
        "lisp",
        "regex",
        "sql",
    };

    const plugin_ext = switch (os_tag) {
        .windows => "dll",
        .macos => "dylib",
        else => "so",
    };

    const plugins_step = b.step("plugins", "Build all plugins");
    const install_step = b.getInstallStep();

    // For plugins, we install into: zig-out/bin/plugins/<name>.<ext>
    // (so zc.exe and plugins are together under bin/)
    for (plugin_names) |name| {
        const lib = b.addLibrary(.{
            .name = name,
            .linkage = .dynamic,
            .root_module = b.createModule(.{
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            }),
        });

        for (include_dirs) |p| {
            lib.root_module.addIncludePath(b.path(p));
        }

        lib.root_module.addCSourceFile(.{
            .file = b.path(b.fmt("plugins/{s}.c", .{name})),
            .flags = &c_flags,
        });

        // Force install into bin/ (not lib/) and rename to match Makefile-ish names
        const inst = b.addInstallArtifact(lib, .{
            .dest_dir = .{ .override = .{ .bin = {} } },
            .dest_sub_path = b.fmt("plugins/{s}.{s}", .{ name, plugin_ext }),
        });

        plugins_step.dependOn(&lib.step);
        install_step.dependOn(&inst.step);
    }

    // Optional convenience: zig build run -- <args>
    const run_cmd = b.addRunArtifact(exe);
    if (b.args) |args| run_cmd.addArgs(args);

    const run_step = b.step("run", "Run zc");
    run_step.dependOn(&run_cmd.step);
}
