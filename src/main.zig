const std = @import("std");
const exit = std.process.exit;
const stdfile = std.fs.File;
const sha2 = std.crypto.hash.sha2;

const chunker = @import("chunker.zig");

const help_text =
    \\ zchunk [options] <filename>...
    \\
    \\ Options:
    \\   --chunking=<strategy>    Chunking strategy (fixed, cdc) [default: fixed]
    \\   --hash=<algorithm>       Hash algorithm (sha256, sha1, xxhash) [default: sha256]
    \\   --mode=<mode>            Output mode (summary, json, chunks) [default: summary]
    \\   --output=<file>          Write output to file instead of stdout
    \\   --threads=<n>            Number of processing threads [default: 1]
    \\   --help                   Show this help information
    \\
;

const Hash = enum {
    sha256,
    // TODO: Implement these
    // sha1,
    // xxhash
};
const Mode = enum { summary, json, chunks };

const FIXED_CHUNK_SIZE_BYTES: u32 = 1024 * 1024; // 1024 KiB
const CDC_MIN_CHUNK_SIZE_BYTES: u32 = 256 * 1024; // 256 KiB
const CDC_AVG_CHUNK_SIZE_BYTES: u32 = 1024 * 1024; // 1024 KiB
const CDC_MAX_CHUNK_SIZE_BYTES: u32 = 4096 * 1024; // 4096 KiB
const WINDOW_SIZE: usize = 64;
const WINDOW_MASK = WINDOW_SIZE - 1;

const Args = struct {
    chunking_strategy: chunker.ChunkingStrategy = chunker.ChunkingStrategy.fixed,
    hash: Hash = Hash.sha256,
    mode: Mode = Mode.summary,
    out_fname: ?[]const u8 = null,
    in_fname: []const u8,
    /// if not provided, we will only use a single thread
    /// if 0 is provided we use all available cores
    n_threads: u8 = 1,
};

fn parse_args(args: *std.process.ArgIterator, input_fname_buf: [*]u8, output_fname_buf: [*]u8) ?Args {
    _ = args.skip(); // skip executable
    var mode: Mode = Mode.summary;
    var hash: Hash = Hash.sha256;
    var chunking_strategy: chunker.ChunkingStrategy = chunker.ChunkingStrategy.fixed;
    var n_threads: u8 = 1;
    var in_file: ?[]const u8 = null;
    var out_file: ?[]const u8 = null;

    var fname_arg: ?[]const u8 = null;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--help")) {
            return null;
        }
        var split = std.mem.splitSequence(u8, arg, "=");
        if (std.mem.eql(u8, split.peek().?, arg)) {
            fname_arg = arg;
            break;
        }
        const flag = split.next().?;
        if (std.mem.eql(u8, flag, "--mode")) {
            const mode_str = split.next().?;
            if (std.mem.eql(u8, mode_str, "summary")) {
                mode = Mode.summary;
            } else if (std.mem.eql(u8, mode_str, "json")) {
                mode = Mode.json;
            } else if (std.mem.eql(u8, mode_str, "chunks")) {
                mode = Mode.chunks;
            } else {
                std.debug.print("Unrecognized mode provided: {s}\n\n", .{mode_str});
                return null;
            }
        } else if (std.mem.eql(u8, flag, "--chunking")) {
            const strategy_str = split.next().?;
            if (std.mem.eql(u8, strategy_str, "fixed")) {
                chunking_strategy = chunker.ChunkingStrategy.fixed;
            } else if (std.mem.eql(u8, strategy_str, "cdc")) {
                chunking_strategy = chunker.ChunkingStrategy.cdc;
            } else {
                std.debug.print("Unrecognized chunking strategy provided: {s}\n\n", .{strategy_str});
                return null;
            }
        } else if (std.mem.eql(u8, flag, "--hash")) {
            const hash_str = split.next().?;
            if (std.mem.eql(u8, hash_str, "sha256")) {
                hash = Hash.sha256;
                // } else if (std.mem.eql(u8, hash_str, "sha1")) {
                //     hash = hash_kind.sha1;
                // } else if (std.mem.eql(u8, hash_str, "xxhash")) {
                //     hash = hash_kind.xxhash;
            } else {
                std.debug.print("Unrecognized hashing function provided: {s}\n\n", .{hash_str});
                return null;
            }
        } else if (std.mem.eql(u8, flag, "--output")) {
            const fname = split.next().?;
            if (fname.len > 255) {
                std.debug.print("Output filename is too long. Max supported length: 255. Provided: {d}\n", .{fname.len});
                std.process.exit(1);
            }
            @memcpy(output_fname_buf, fname);
            out_file = output_fname_buf[0..fname.len];
        } else if (std.mem.eql(u8, flag, "--threads")) {
            const n_thread_str = split.next().?;
            n_threads = std.fmt.parseInt(u8, n_thread_str, 0) catch |err| switch (err) {
                std.fmt.ParseIntError.Overflow => {
                    std.debug.print("Provided num threads is too large. Max supported size is 127. Provided: {s}\n", .{n_thread_str});
                    std.process.exit(1);
                },
                std.fmt.ParseIntError.InvalidCharacter => {
                    std.debug.print("Unable to parse num threads. Provided: {s}\n", .{n_thread_str});
                    std.process.exit(1);
                },
            };
        } else {
            std.debug.print("Unrecognized flag provided: {s}\n\n", .{arg});
            return null;
        }
    }

    // if we reach here, it means we have reached the fname.
    if (fname_arg) |fname| {
        if (fname.len > 255) {
            std.debug.print("Input filename is too long. Max supported length: 255. Provided: {d}\n", .{fname.len});
            std.process.exit(1);
        }
        @memcpy(input_fname_buf, fname);
        in_file = input_fname_buf[0..fname.len];
    } else {
        std.debug.print("Expected a filename.\n\n", .{});
        return null;
    }

    var return_args = Args{
        .mode = mode,
        .n_threads = n_threads,
        .hash = hash,
        .chunking_strategy = chunking_strategy,
        .in_fname = in_file.?,
    };
    if (out_file) |f| {
        return_args.out_fname = f;
    }

    return return_args;
}

fn format_size_human_readable(size: u64, buf: []u8) ![]const u8 {
    var fbs = std.io.fixedBufferStream(buf);
    var writer = fbs.writer();

    if (size < 1024) {
        try writer.print("{d} bytes", .{size});
    } else if (size < 1024 * 1024) {
        try writer.print("{d:.2} KB", .{@as(f64, @floatFromInt(size)) / 1024});
    } else if (size < 1024 * 1024 * 1024) {
        try writer.print("{d:.2} MB", .{@as(f64, @floatFromInt(size)) / (1024 * 1024)});
    } else {
        try writer.print("{d:.2} GB", .{@as(f64, @floatFromInt(size)) / (1024 * 1024 * 1024)});
    }

    return fbs.getWritten();
}

fn format_hash_hex(hash: []const u8, buf: []u8) ![]const u8 {
    var fbs = std.io.fixedBufferStream(buf);
    var writer = fbs.writer();

    for (hash) |byte| {
        try writer.print("{x:0>2}", .{byte});
    }

    return fbs.getWritten();
}

fn handle_output(args: Args, allocator: std.mem.Allocator) void {
    const start_time = std.time.milliTimestamp();

    const compute_result = chunker.compute_hashes(args.in_fname, args.n_threads, args.chunking_strategy, allocator);
    const hashes = compute_result.hashes;
    const size = compute_result.file_size;
    defer allocator.free(hashes);

    const nchunks = if (args.chunking_strategy == chunker.ChunkingStrategy.cdc)
        hashes.len / 32 // Each hash is 32 bytes
    else
        (size + FIXED_CHUNK_SIZE_BYTES - 1) / FIXED_CHUNK_SIZE_BYTES;

    const merkle_tree = chunker.construct_merkle_tree(hashes, allocator);
    defer allocator.free(merkle_tree);

    // Calculate processing time and throughput
    const end_time = std.time.milliTimestamp();
    const processing_time_ms = @as(u64, @intCast(end_time - start_time));
    const processing_time_sec = @as(f64, @floatFromInt(processing_time_ms)) / 1000.0;

    // Calculate throughput in MB/s
    const throughput_mb_s = if (processing_time_sec > 0)
        @as(f64, @floatFromInt(size)) / (1024 * 1024) / processing_time_sec
    else
        0;

    // Create output writer based on whether we're writing to a file or stdout
    if (args.out_fname) |out_fname| {
        const out_file = std.fs.cwd().createFile(out_fname, .{ .read = true }) catch |err| {
            chunker.print_and_exit_with_error("Error creating output file '{s}': {}", .{ out_fname, err });
        };
        defer out_file.close();

        const writer = out_file.writer();
        output_by_mode(args, writer, size, nchunks, hashes, merkle_tree, processing_time_ms, throughput_mb_s) catch |err| {
            chunker.print_and_exit_with_error("Error writing to output file: {}", .{err});
        };
    } else {
        const stdout = std.io.getStdOut().writer();
        output_by_mode(args, stdout, size, nchunks, hashes, merkle_tree, processing_time_ms, throughput_mb_s) catch |err| {
            chunker.print_and_exit_with_error("Error writing to stdout: {}", .{err});
        };
    }
}

fn output_by_mode(args: Args, writer: anytype, size: u64, nchunks: u64, hashes: []const u8, merkle_tree: []const u8, processing_time_ms: u64, throughput_mb_s: f64) !void {
    switch (args.mode) {
        .summary => try output_summary(args, writer, size, nchunks, hashes, merkle_tree, processing_time_ms, throughput_mb_s),
        .json => try output_json(args, writer, size, nchunks, hashes, merkle_tree, processing_time_ms, throughput_mb_s),
        .chunks => try output_chunks(args, writer, size, nchunks, hashes, merkle_tree),
    }
}

fn output_summary(args: Args, writer: anytype, size: u64, nchunks: u64, hashes: []const u8, merkle_tree: []const u8, processing_time_ms: u64, throughput_mb_s: f64) !void {
    _ = hashes;
    var size_buf: [32]u8 = undefined;
    const size_str = try format_size_human_readable(size, &size_buf);

    var hash_buf: [64]u8 = undefined;
    const merkle_root_hex = if (merkle_tree.len >= 32)
        try format_hash_hex(merkle_tree[0..32], &hash_buf)
    else
        "(empty file)";

    try writer.print("File: {s} ({s})\n", .{ args.in_fname, size_str });
    // Adjust output based on strategy, using constants
    if (args.chunking_strategy == chunker.ChunkingStrategy.fixed) {
        try writer.print("Chunking: fixed ({d} KB each)\n", .{FIXED_CHUNK_SIZE_BYTES / 1024});
    } else {
        try writer.print("Chunking: cdc (min={d}KB, avg={d}KB, max={d}KB)\n", .{ CDC_MIN_CHUNK_SIZE_BYTES / 1024, CDC_AVG_CHUNK_SIZE_BYTES / 1024, CDC_MAX_CHUNK_SIZE_BYTES / 1024 });
    }
    try writer.print("Chunks: {d}\n", .{nchunks});
    try writer.print("Processing time: {d:.2} seconds\n", .{@as(f64, @floatFromInt(processing_time_ms)) / 1000.0});
    try writer.print("Throughput: {d:.1} MB/s\n", .{throughput_mb_s});
    try writer.print("Overall hash: {s}\n", .{merkle_root_hex});
}

fn output_json(args: Args, writer: anytype, size: u64, nchunks: u64, hashes: []const u8, merkle_tree: []const u8, processing_time_ms: u64, throughput_mb_s: f64) !void {
    var hash_buf: [64]u8 = undefined;
    const merkle_root_hex = if (merkle_tree.len >= 32)
        try format_hash_hex(merkle_tree[0..32], &hash_buf)
    else
        "";

    try writer.print("{{\n", .{});
    try writer.print("  \"filename\": \"{s}\",\n", .{args.in_fname});
    try writer.print("  \"size\": {d},\n", .{size});
    if (args.chunking_strategy == chunker.ChunkingStrategy.fixed) {
        try writer.print("  \"chunking_strategy\": \"fixed\",\n", .{});
        try writer.print("  \"chunk_size\": {d},\n", .{FIXED_CHUNK_SIZE_BYTES});
    } else {
        try writer.print("  \"chunking_strategy\": \"cdc\",\n", .{});
        try writer.print("  \"min_chunk_size\": {d},\n", .{CDC_MIN_CHUNK_SIZE_BYTES});
        try writer.print("  \"max_chunk_size\": {d},\n", .{CDC_MAX_CHUNK_SIZE_BYTES});
        try writer.print("  \"avg_chunk_size\": {d},\n", .{CDC_AVG_CHUNK_SIZE_BYTES});
    }
    try writer.print("  \"chunks\": [\n", .{});

    for (0..nchunks) |chunk_idx| {
        const last_chunk = chunk_idx == nchunks - 1;
        // This calculation is only correct for fixed chunking
        const current_chunk_size = if (args.chunking_strategy == chunker.ChunkingStrategy.fixed) blk: {
            if (last_chunk and size % FIXED_CHUNK_SIZE_BYTES != 0) {
                break :blk size % FIXED_CHUNK_SIZE_BYTES;
            } else {
                break :blk FIXED_CHUNK_SIZE_BYTES;
            }
        } else blk: {
            // Placeholder for CDC - actual size is unknown here without boundary info
            // Outputting 0 as placeholder.
            break :blk @as(usize, 0); // FIXME Placeholder
        };
        // This offset is only correct for fixed chunking
        const current_offset = if (args.chunking_strategy == chunker.ChunkingStrategy.fixed)
            chunk_idx * FIXED_CHUNK_SIZE_BYTES
        else // Placeholder for CDC - actual offset is unknown here
            @as(u64, 0); // FIXME Placeholder

        try writer.print("    {{\n", .{});
        try writer.print("      \"index\": {d},\n", .{chunk_idx});
        try writer.print("      \"offset\": {d},\n", .{current_offset}); // FIXME for CDC
        try writer.print("      \"size\": {d},\n", .{current_chunk_size}); // FIXME for CDC
        try writer.print("      \"hash\": \"", .{});

        if ((chunk_idx + 1) * 32 <= hashes.len) {
            for (hashes[chunk_idx * 32 .. (chunk_idx + 1) * 32]) |byte| {
                try writer.print("{x:0>2}", .{byte});
            }
        }

        try writer.print("\"{s}\n", .{if (chunk_idx == nchunks - 1) "" else ","});
        try writer.print("    }}{s}\n", .{if (chunk_idx == nchunks - 1) "" else ","});
    }

    try writer.print("  ],\n", .{});
    try writer.print("  \"overall_hash\": \"{s}\",\n", .{merkle_root_hex});
    try writer.print("  \"processing_time_ms\": {d},\n", .{processing_time_ms});
    try writer.print("  \"throughput_mb_s\": {d:.1}\n", .{throughput_mb_s});
    try writer.print("}}\n", .{});
}

fn output_chunks(args: Args, writer: anytype, size: u64, nchunks: u64, hashes: []const u8, merkle_tree: []const u8) !void {
    var size_buf: [32]u8 = undefined;
    const size_str = try format_size_human_readable(size, &size_buf);

    try writer.print("File: {s} ({s})\n", .{ args.in_fname, size_str });
    try writer.print("Chunk  Offset      Size        Hash (SHA-256)\n", .{});

    for (0..nchunks) |chunk_idx| {
        const last_chunk = chunk_idx == nchunks - 1;
        // This calculation is only correct for fixed chunking
        const current_chunk_size = if (args.chunking_strategy == chunker.ChunkingStrategy.fixed) blk: {
            if (last_chunk and size % FIXED_CHUNK_SIZE_BYTES != 0) {
                break :blk size % FIXED_CHUNK_SIZE_BYTES;
            } else {
                break :blk FIXED_CHUNK_SIZE_BYTES;
            }
        } else blk: {
            // Placeholder for CDC - actual size is unknown here without boundary info
            // Outputting 0 as placeholder.
            break :blk @as(usize, 0); // FIXME Placeholder
        };
        // This offset is only correct for fixed chunking
        const current_offset = if (args.chunking_strategy == chunker.ChunkingStrategy.fixed)
            chunk_idx * FIXED_CHUNK_SIZE_BYTES
        else // Placeholder for CDC - actual offset is unknown here
            @as(u64, 0); // FIXME Placeholder

        try writer.print("{d:<6} {d:<11} {d:<11} ", .{ chunk_idx, current_offset, current_chunk_size }); // FIXME for CDC

        if ((chunk_idx + 1) * 32 <= hashes.len) {
            for (hashes[chunk_idx * 32 .. (chunk_idx + 1) * 32]) |byte| {
                try writer.print("{x:0>2}", .{byte});
            }
        }
        try writer.print("\n", .{});
    }

    // Add the merkle root hash
    try writer.print("\nOverall hash: ", .{});
    if (merkle_tree.len >= 32) {
        for (merkle_tree[0..32]) |byte| {
            try writer.print("{x:0>2}", .{byte});
        }
    } else {
        try writer.print("(empty file)", .{});
    }
    try writer.print("\n", .{});
}

pub fn main() !void {
    var args = std.process.args();
    var input_fname_buf: [255]u8 = [_]u8{0} ** 255;
    var output_fname_buf: [255]u8 = [_]u8{0} ** 255;
    const parsed_args = parse_args(&args, &input_fname_buf, &output_fname_buf);
    var gpa = std.heap.GeneralPurposeAllocator(.{ .safety = true }){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    if (parsed_args) |a| {
        // Print effective chunk sizes for clarity, especially for CDC
        if (a.chunking_strategy == chunker.ChunkingStrategy.fixed) {
            std.debug.print("Using fixed chunking with size: {d} KB\n", .{FIXED_CHUNK_SIZE_BYTES / 1024});
        } else {
            std.debug.print("Using CDC chunking with min={d}KB, avg={d}KB, max={d}KB\n", .{ CDC_MIN_CHUNK_SIZE_BYTES / 1024, CDC_AVG_CHUNK_SIZE_BYTES / 1024, CDC_MAX_CHUNK_SIZE_BYTES / 1024 });
        }
        handle_output(a, allocator);
    } else {
        std.debug.print("{s}\n\n", .{help_text});
    }
}
