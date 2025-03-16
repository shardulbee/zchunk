const std = @import("std");
const exit = std.process.exit;
const stdfile = std.fs.File;
const sha2 = std.crypto.hash.sha2;

const help_text =
    \\ zchunk [options] <filename>...
    \\
    \\ Options:
    \\   --chunk-size=<size>    Size of each chunk in KB (default: 1024)
    \\   --hash=<algorithm>     Hash algorithm (sha256, sha1, xxhash) [default: sha256]
    \\   --mode=<mode>          Output mode (summary, json, chunks) [default: summary]
    \\   --output=<file>        Write output to file instead of stdout
    \\   --threads=<n>          Number of processing threads [default: 1]
    \\   --help                 Show this help information
    \\
;

const Hash = enum {
    sha256,
    // TODO: Implement these
    // sha1,
    // xxhash
};
const Mode = enum { summary, json, chunks };

const Args = struct {
    // TODO: this is in bytes for now, maybe change to KB?
    chunk_size: u32,
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
    var chunk_size: u32 = 1024 * 1024; // Default 1024 KB (1MB)
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
        } else if (std.mem.eql(u8, flag, "--chunk-size")) {
            const chunk_size_str = split.next().?;
            const kb_size = std.fmt.parseInt(u32, chunk_size_str, 0) catch |err| switch (err) {
                std.fmt.ParseIntError.Overflow => {
                    std.debug.print("Provided chunk-size is too large. Max supported size is 65536 KB. Provided: {s}\n", .{chunk_size_str});
                    std.process.exit(1);
                },
                std.fmt.ParseIntError.InvalidCharacter => {
                    std.debug.print("Unable to parse chunk size. Provided: {s}\n", .{chunk_size_str});
                    std.process.exit(1);
                },
            };
            // Convert KB to bytes
            chunk_size = kb_size * 1024;
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

    var return_args = Args{ .mode = mode, .n_threads = n_threads, .hash = hash, .chunk_size = chunk_size, .in_fname = in_file.? };
    if (out_file) |f| {
        return_args.out_fname = f;
    }

    return return_args;
}

fn handle_read_error(err: stdfile.ReadError) noreturn {
    std.debug.print("Encountered error while reading: {?}\n", .{err});
    exit(1);
}

fn open_file_from_fname(fname: []const u8) std.fs.File {
    if (std.fs.path.isAbsolute(fname)) {
        return std.fs
            .openFileAbsolute(fname, stdfile.OpenFlags{}) catch |err|
            handle_open_error(err, fname);
    } else {
        return std.fs.cwd()
            .openFile(fname, stdfile.OpenFlags{}) catch |err|
            handle_open_error(err, fname);
    }
}

fn handle_open_error(err: stdfile.OpenError, path: []const u8) noreturn {
    switch (err) {
        stdfile.OpenError.FileNotFound => {
            std.debug.print("Unable to find file: {s}\n\n", .{path});
            std.process.exit(1);
        },
        else => exit(1),
    }
}

fn print_and_exit_with_error(comptime msg: []const u8, args: anytype) noreturn {
    std.debug.print(msg, args);
    exit(1);
}

/// Constructs a Merkle tree from a list of leaf node hashes
///
/// Takes an array of SHA-256 hashes (32 bytes each) and constructs a complete
/// Merkle tree. For odd numbers of leaves, the first leaf is duplicated to ensure
/// a balanced tree.
///
/// The tree is constructed bottom-up, with leaves at the end of the tree array.
/// Each parent node contains the hash of its two children concatenated together.
///
/// Example tree structure with 4 leaves [A,B,C,D]:
///
///               Root
///              /    \
///           Hash(AB) Hash(CD)
///           /    \    /    \
///          A      B  C      D
///
/// The returned array contains the entire tree with the root at index 0.
/// For n leaves, tree size is (2n-1) nodes, each node is 32 bytes.
///
/// Example tree structure with 3 leafes [A,B,C]:
///
///               Root
///              /    \
///           Hash(AA) Hash(C)
///           /    \    /    \
///          A      A  C      D
fn construct_merkle_tree(hashes: []u8, allocator: std.mem.Allocator) []const u8 {
    const original_n_leaves = hashes.len / 32;

    // Handle empty file (no leaves)
    if (original_n_leaves == 0) {
        return &[_]u8{};
    }

    // Determine if we need to duplicate the last leaf for odd counts
    const n_leaves = if (original_n_leaves % 2 == 1)
        original_n_leaves + 1
    else
        original_n_leaves;

    // A tree with n leaves has n-1 internal nodes, so 2n-1 nodes total
    const tree_size = 2 * n_leaves - 1;
    var tree = allocator.alloc(u8, tree_size * 32) catch
        print_and_exit_with_error("Unable to allocate memory for tree", .{});

    // The leaves are at the end of the tree, copy them over
    const leaf_start_idx = tree_size - original_n_leaves;
    @memcpy(tree[leaf_start_idx * 32 ..], hashes);

    // If odd number of leaves, duplicate the last leaf
    if (original_n_leaves % 2 == 1) {
        @memcpy(tree[(leaf_start_idx - 1) * 32 .. leaf_start_idx * 32], tree[leaf_start_idx * 32 .. (leaf_start_idx + 1) * 32]);
    }

    // Build the tree bottom-up by hashing pairs of nodes
    var idx: usize = tree_size - 1;
    while (idx > 0) {
        const parent_idx = (idx - 1) / 2;
        const start_idx = (idx - 1) * 32;
        const end_idx = start_idx + 64;

        sha2.Sha256.hash(tree[start_idx..end_idx], @ptrCast(tree[parent_idx * 32 .. (parent_idx + 1) * 32]), .{});

        idx -= 2;
    }
    return tree;
}

fn compute_hashes(args: Args, allocator: std.mem.Allocator) []u8 {
    var file: std.fs.File = open_file_from_fname(args.in_fname);
    defer file.close();

    var pool: std.Thread.Pool = undefined;
    var nthreads: u32 = undefined;
    if (args.n_threads == 0) {
        nthreads = @truncate(std.Thread.getCpuCount() catch print_and_exit_with_error("unable to get cpu count\n", .{}));
    } else {
        nthreads = args.n_threads;
    }

    const stat: stdfile.Stat = file.stat() catch print_and_exit_with_error("Unable to stat file.\n", .{});
    const size: u64 = stat.size;
    const nchunks: u64 = (size + args.chunk_size - 1) / args.chunk_size;

    // need to pre-allocate space where each thread will write their hash
    // each sha256 is 32 bytes so we need 32 bytes * chunks
    const hashes = allocator.alloc(u8, nchunks * 32) catch print_and_exit_with_error("Unable to allocate memory for hashes", .{});

    if (nthreads > 1) {
        pool.init(.{ .allocator = allocator, .n_jobs = @as(u32, nthreads) }) catch print_and_exit_with_error("Unable to init threadpool", .{});
        defer pool.deinit();
        const chunks_per_thread: u64 = (nchunks + nthreads - 1) / nthreads;
        for (0..nthreads) |thread_idx| {
            pool.spawn(work, .{ chunks_per_thread, thread_idx, args, hashes, allocator }) catch |err| print_and_exit_with_error("Unable to spawn thread {d}. err: {any}", .{ thread_idx, err });
        }
    } else {
        work(nchunks, 0, args, hashes, allocator);
    }

    return hashes;
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
    var file = open_file_from_fname(args.in_fname);
    defer file.close();

    const stat: stdfile.Stat = file.stat() catch print_and_exit_with_error("Unable to stat file.\n", .{});
    const size: u64 = stat.size;
    const nchunks: u64 = (size + args.chunk_size - 1) / args.chunk_size;

    // Record start time
    const start_time = std.time.milliTimestamp();

    const hashes = compute_hashes(args, allocator);
    defer allocator.free(hashes);

    const merkle_tree = construct_merkle_tree(hashes, allocator);
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
            print_and_exit_with_error("Error creating output file '{s}': {}\n", .{ out_fname, err });
        };
        defer out_file.close();

        const writer = out_file.writer();
        output_by_mode(args, writer, size, nchunks, hashes, merkle_tree, processing_time_ms, throughput_mb_s) catch |err| {
            print_and_exit_with_error("Error writing to output file: {}\n", .{err});
        };
    } else {
        // Write to stdout
        const stdout = std.io.getStdOut().writer();
        output_by_mode(args, stdout, size, nchunks, hashes, merkle_tree, processing_time_ms, throughput_mb_s) catch |err| {
            print_and_exit_with_error("Error writing to stdout: {}\n", .{err});
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
    try writer.print("Chunks: {d} ({d} KB each)\n", .{ nchunks, args.chunk_size / 1024 });
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
    try writer.print("  \"chunk_size\": {d},\n", .{args.chunk_size});
    try writer.print("  \"chunks\": [\n", .{});

    for (0..nchunks) |chunk_idx| {
        const last_chunk = chunk_idx == nchunks - 1;
        const chunk_size = if (last_chunk and size % args.chunk_size != 0)
            size % args.chunk_size
        else
            args.chunk_size;

        try writer.print("    {{\n", .{});
        try writer.print("      \"index\": {d},\n", .{chunk_idx});
        try writer.print("      \"offset\": {d},\n", .{chunk_idx * args.chunk_size});
        try writer.print("      \"size\": {d},\n", .{chunk_size});
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
        const chunk_size = if (last_chunk and size % args.chunk_size != 0)
            size % args.chunk_size
        else
            args.chunk_size;

        try writer.print("{d:<6} {d:<11} {d:<11} ", .{ chunk_idx, chunk_idx * args.chunk_size, chunk_size });

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

fn work(nchunks: u64, thread_idx: usize, args: Args, hashes: []u8, allocator: std.mem.Allocator) void {
    var file = open_file_from_fname(args.in_fname);
    defer file.close();

    file.seekTo(thread_idx * nchunks * args.chunk_size) catch |err| print_and_exit_with_error("Unable to seek to correct file location for thread {d}. err: {any}\n", .{ thread_idx, err });
    const reader = stdfile.reader(file);
    var buffered_reader = std.io.bufferedReader(reader);

    var fbuffer = allocator.alloc(u8, args.chunk_size) catch |err| print_and_exit_with_error("Unable to allocate fbuffer for thread_idx {d}. err: {any}", .{ thread_idx, err });
    defer allocator.free(fbuffer);

    var bytes_read: u64 = 0;
    var hash_buf: []u8 = undefined;
    for (0..nchunks) |chunk_idx| {
        hash_buf = hashes[(thread_idx * nchunks + chunk_idx) * 32 .. (thread_idx * nchunks + chunk_idx + 1) * 32];
        bytes_read = buffered_reader.read(fbuffer) catch |err| handle_read_error(err);
        sha2.Sha256.hash(fbuffer[0..bytes_read], @ptrCast(hash_buf), .{});
        if (bytes_read != args.chunk_size) {
            break;
        }
    }
}

test "chunk hashing correctly identifies modified chunk" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var tmp_dir = testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const chunk_size: u32 = 1024 * 1024; // 1MB chunks
    const num_chunks: u32 = 100;

    // Create test file with random data
    const test_file = try create_test_file(allocator, &tmp_dir, chunk_size, num_chunks);
    defer {
        allocator.free(test_file.data);
        allocator.free(test_file.path);
    }

    // Get reference to the data for modification later
    var random_data = test_file.data;

    const args = Args{
        .chunk_size = chunk_size,
        .hash = Hash.sha256,
        .mode = Mode.summary,
        .out_fname = null,
        .in_fname = test_file.path,
        .n_threads = 1,
    };

    var random_generator = std.Random.DefaultPrng.init(0); // Use different seed than data generation
    const random = random_generator.random();

    for (0..5) |_| {
        // Write original data to file
        {
            const file = try std.fs.createFileAbsolute(test_file.path, .{});
            defer file.close();
            try file.writeAll(random_data);
        }

        // Compute original hashes
        const original_hashes = compute_hashes(args, allocator);
        defer allocator.free(original_hashes);

        // Randomly select a chunk to modify
        const chunk_to_modify = random.uintLessThan(u32, num_chunks);
        const offset = chunk_to_modify * chunk_size;
        const byte_to_modify = random.uintLessThan(u32, chunk_size);

        // Modify a random byte in the selected chunk
        random_data[offset + byte_to_modify] = ~random_data[offset + byte_to_modify];

        // Write modified data back to file
        {
            const file = try std.fs.createFileAbsolute(test_file.path, .{});
            defer file.close();
            try file.writeAll(random_data);
        }

        // Compute new hashes
        const modified_hashes = compute_hashes(args, allocator);
        defer allocator.free(modified_hashes);

        // Count and track which chunks have different hashes
        var diff_count: u32 = 0;
        var modified_chunk_idx: ?u32 = null;

        for (0..num_chunks) |i| {
            const original_hash = original_hashes[i * 32 .. (i + 1) * 32];
            const modified_hash = modified_hashes[i * 32 .. (i + 1) * 32];

            if (!std.mem.eql(u8, original_hash, modified_hash)) {
                diff_count += 1;
                modified_chunk_idx = @truncate(i);
            }
        }

        try testing.expectEqual(@as(u32, 1), diff_count); // Verify only one chunk is identified as modified
        try testing.expectEqual(chunk_to_modify, modified_chunk_idx.?);

        // Test that the merkle root hash is also different
        const original_tree = construct_merkle_tree(original_hashes, allocator);
        defer allocator.free(original_tree);
        const modified_tree = construct_merkle_tree(modified_hashes, allocator);
        defer allocator.free(modified_tree);
        try testing.expect(!std.mem.eql(u8, original_tree[0..32], modified_tree[0..32]));

        // Restore original data for next iteration
        random_data[offset + byte_to_modify] = ~random_data[offset + byte_to_modify];
    }
}

// Helper function to create a test file with random data
fn create_test_file(allocator: std.mem.Allocator, tmp_dir: *std.testing.TmpDir, chunk_size: u32, num_chunks: u32) !struct { data: []u8, path: []const u8 } {
    const total_size = chunk_size * num_chunks;

    // Generate random data
    const data = try allocator.alloc(u8, total_size);
    errdefer allocator.free(data);

    var rnd = std.Random.DefaultPrng.init(42);
    rnd.random().bytes(data);

    // Create test file path
    const real_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(real_path);

    const test_file_path = try std.fs.path.join(
        allocator,
        &.{ real_path, "test_file.bin" },
    );

    // Write data to file
    {
        const file = try tmp_dir.dir.createFile("test_file.bin", .{});
        defer file.close();
        try file.writeAll(data);
    }

    return .{ .data = data, .path = test_file_path };
}

test "multi-threaded hashing matches single-threaded results" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var tmp_dir = testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    // Create test file with random data
    const chunk_size: u32 = 1024 * 16; // 16KB chunks
    const num_chunks: u32 = 64; // ~1MB total

    const test_file = try create_test_file(allocator, &tmp_dir, chunk_size, num_chunks);
    defer {
        allocator.free(test_file.data);
        allocator.free(test_file.path);
    }

    // Run with single thread (as reference)
    const single_args = Args{
        .chunk_size = chunk_size,
        .hash = Hash.sha256,
        .mode = Mode.summary,
        .out_fname = null,
        .in_fname = test_file.path,
        .n_threads = 1,
    };

    const single_hashes = compute_hashes(single_args, allocator);
    defer allocator.free(single_hashes);

    // Array of thread counts to test
    const thread_counts = [_]u8{ 2, 4, 8, 0 }; // 0 = auto-detect

    for (thread_counts) |n_threads| {
        const thread_args = Args{
            .chunk_size = chunk_size,
            .hash = Hash.sha256,
            .mode = Mode.summary,
            .out_fname = null,
            .in_fname = test_file.path,
            .n_threads = n_threads,
        };

        const thread_hashes = compute_hashes(thread_args, allocator);
        defer allocator.free(thread_hashes);

        // Verify results are identical regardless of thread count
        const results_match = std.mem.eql(u8, single_hashes, thread_hashes);
        try testing.expect(results_match);
    }
}

test "empty file handling" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var tmp_dir = testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const real_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(real_path);
    const empty_file_path = try std.fs.path.join(
        allocator,
        &.{ real_path, "empty.bin" },
    );
    defer allocator.free(empty_file_path);
    {
        const file = try tmp_dir.dir.createFile("empty.bin", .{});
        defer file.close();
    }

    const args = Args{
        .chunk_size = 1024,
        .hash = Hash.sha256,
        .mode = Mode.summary,
        .out_fname = null,
        .in_fname = empty_file_path,
        .n_threads = 1,
    };
    const hashes = compute_hashes(args, allocator);
    defer allocator.free(hashes);
    try testing.expectEqual(@as(usize, 0), hashes.len);

    // Test merkle tree construction with empty hashes
    const merkle_tree = construct_merkle_tree(hashes, allocator);
    defer allocator.free(merkle_tree);
    try testing.expectEqual(@as(usize, 0), merkle_tree.len);
}

test "boundary condition - exact chunk size" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var tmp_dir = testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const chunk_size: u32 = 1024; // 1KB chunk for testing

    // Create a file that is exactly one chunk size
    const data = try allocator.alloc(u8, chunk_size);
    defer allocator.free(data);

    var rnd = std.Random.DefaultPrng.init(42);
    rnd.random().bytes(data);

    const real_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(real_path);

    const file_path = try std.fs.path.join(
        allocator,
        &.{ real_path, "exact_chunk.bin" },
    );
    defer allocator.free(file_path);

    {
        const file = try tmp_dir.dir.createFile("exact_chunk.bin", .{});
        defer file.close();
        try file.writeAll(data);
    }

    const args = Args{
        .chunk_size = chunk_size,
        .hash = Hash.sha256,
        .mode = Mode.summary,
        .out_fname = null,
        .in_fname = file_path,
        .n_threads = 1,
    };

    const hashes = compute_hashes(args, allocator);
    defer allocator.free(hashes);

    // Should have exactly one chunk
    try testing.expectEqual(@as(usize, 32), hashes.len);

    // Test Merkle tree with single chunk
    // For 1 leaf node, we get 3 nodes total due to duplication:
    // 1 leaf node (duplicated to 2) + 1 root = 3 nodes
    const merkle_tree = construct_merkle_tree(hashes, allocator);
    defer allocator.free(merkle_tree);
    try testing.expectEqual(@as(usize, 3 * 32), merkle_tree.len);
}

test "boundary condition - one byte larger than chunk size" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var tmp_dir = testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const chunk_size: u32 = 1024;

    // Create a file that is one byte larger than the chunk size
    const data = try allocator.alloc(u8, chunk_size + 1);
    defer allocator.free(data);

    var rnd = std.Random.DefaultPrng.init(43);
    rnd.random().bytes(data);

    const real_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(real_path);

    const file_path = try std.fs.path.join(
        allocator,
        &.{ real_path, "one_byte_over.bin" },
    );
    defer allocator.free(file_path);

    {
        const file = try tmp_dir.dir.createFile("one_byte_over.bin", .{});
        defer file.close();
        try file.writeAll(data);
    }

    const args = Args{
        .chunk_size = chunk_size,
        .hash = Hash.sha256,
        .mode = Mode.summary,
        .out_fname = null,
        .in_fname = file_path,
        .n_threads = 1,
    };

    const hashes = compute_hashes(args, allocator);
    defer allocator.free(hashes);
    // Should have exactly two chunks
    try testing.expectEqual(@as(usize, 64), hashes.len);

    // Verify merkle tree also has the correct structure for 2 chunks
    // For 2 leaves, we should have 3 nodes in the merkle tree (1 root + 2 leaves)
    const merkle_tree = construct_merkle_tree(hashes, allocator);
    defer allocator.free(merkle_tree);
    try testing.expectEqual(@as(usize, 3 * 32), merkle_tree.len);
}

test "boundary condition - one byte smaller than chunk size" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var tmp_dir = testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const chunk_size: u32 = 1024;

    // Create a file that is one byte smaller than the chunk size
    const data = try allocator.alloc(u8, chunk_size - 1);
    defer allocator.free(data);

    var rnd = std.Random.DefaultPrng.init(44);
    rnd.random().bytes(data);

    const real_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(real_path);

    const file_path = try std.fs.path.join(
        allocator,
        &.{ real_path, "one_byte_under.bin" },
    );
    defer allocator.free(file_path);

    {
        const file = try tmp_dir.dir.createFile("one_byte_under.bin", .{});
        defer file.close();
        try file.writeAll(data);
    }

    const args = Args{
        .chunk_size = chunk_size,
        .hash = Hash.sha256,
        .mode = Mode.summary,
        .out_fname = null,
        .in_fname = file_path,
        .n_threads = 1,
    };

    const hashes = compute_hashes(args, allocator);
    defer allocator.free(hashes);
    try testing.expectEqual(@as(usize, 32), hashes.len);

    // Test Merkle tree with single chunk
    // For 1 leaf node, we get 3 nodes total due to duplication:
    // 1 leaf node (duplicated internally to 2) + 1 root = 3 nodes
    const merkle_tree = construct_merkle_tree(hashes, allocator);
    defer allocator.free(merkle_tree);
    try testing.expectEqual(@as(usize, 3 * 32), merkle_tree.len);
}

test "merkle tree verification" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Create leaf nodes (simulated file chunks)
    const num_leaves = 8;
    var leaves = try allocator.alloc(u8, num_leaves * 32);
    defer allocator.free(leaves);

    // Initialize each leaf with a different value
    for (0..num_leaves) |i| {
        var leaf_hash: [32]u8 = undefined;
        for (0..32) |j| {
            leaf_hash[j] = @truncate((i * 32 + j) % 256);
        }
        @memcpy(leaves[i * 32 .. (i + 1) * 32], &leaf_hash);
    }

    // Construct merkle tree from leaves
    const original_tree = construct_merkle_tree(leaves, allocator);
    defer allocator.free(original_tree);

    // Verify size of the tree
    const expected_size = 2 * num_leaves - 1; // Internal nodes + leaf nodes
    try testing.expectEqual(expected_size * 32, original_tree.len);

    for (0..num_leaves) |leaf_idx| {
        // Modify one byte in the leaf
        leaves[leaf_idx * 32] = ~leaves[leaf_idx * 32];

        // Construct new merkle tree
        const modified_tree = construct_merkle_tree(leaves, allocator);
        defer allocator.free(modified_tree);

        // Root hash should be different
        try testing.expect(!std.mem.eql(u8, original_tree[0..32], modified_tree[0..32]));

        // Check that the verified path to the modified leaf has all different hashes
        var node_idx = leaf_idx + (num_leaves - 1);
        while (node_idx > 0) {
            const parent_idx = (node_idx - 1) / 2;

            // Compare original and modified node hashes
            const original_node = original_tree[parent_idx * 32 .. (parent_idx + 1) * 32];
            const modified_node = modified_tree[parent_idx * 32 .. (parent_idx + 1) * 32];

            // The path from the modified leaf to the root should all have different hashes
            try testing.expect(!std.mem.eql(u8, original_node, modified_node));

            node_idx = parent_idx;
        }
    }
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
        handle_output(a, allocator);
    } else {
        std.debug.print("{s}\n\n", .{help_text});
    }
}
