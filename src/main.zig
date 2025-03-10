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
    var chunk_size: u32 = 64;
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
            chunk_size = std.fmt.parseInt(u32, chunk_size_str, 0) catch |err| switch (err) {
                std.fmt.ParseIntError.Overflow => {
                    std.debug.print("Provided chunk-size is too large. Max supported size is 65536. Provided: {s}\n", .{chunk_size_str});
                    std.process.exit(1);
                },
                std.fmt.ParseIntError.InvalidCharacter => {
                    std.debug.print("Unable to parse chunk size. Provided: {s}\n", .{chunk_size_str});
                    std.process.exit(1);
                },
            };
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

fn construct_merkle_tree(hashes: []u8, allocator: std.mem.Allocator) []const u8 {
    // the rough strategy:
    // - we're given bunch of hashes which correspond to each chunk of the file
    // - these hashes are the leaves of the tree
    // - since it is a binary tree, we can start with the last 2 leaves, concatenate them, and then hash them again
    // - this becomes the parent of the 2 leaves. we store this in the index before the first leaf
    // - then we repeat this process until we reach the first non-leaf and recursively do this for the next level
    // to get the index of the parent, we take the index of the leaf, subtract 1, and divide by 2 (and take the floor but that's the default)

    // a tree with N leaves contains n-1 internal nodes
    // so the whole tree has 2N-1 nodes
    const n_leaves = hashes.len / 32;
    const tree_size = 2 * n_leaves - 1;
    var tree = allocator.alloc(u8, tree_size * 32) catch print_and_exit_with_error("Unable to allocate memory for tree", .{});

    // the leaves are at the end of the tree, copy them over
    @memcpy(tree[(tree_size - n_leaves) * 32 ..], hashes);

    // now we hash 64 bytes bytes from the end of the tree to the beginning, and store that at the parent index
    // we repeat this until we're at the beginning
    //
    // NOTE: this means in the case of an odd number of leaves, the leftmost leaf will not have a sibling
    // this is fine because it will be paired with an internal node from the level above
    // for example, with 3 leaves [A, B, C]:
    // 1. we hash B+C and store it at position 1
    // 2. then we hash A+hash(B+C) and store it at position 0
    // So we are "promoting" the odd leaf to the next level
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
        const chunks_per_thread: u64 = (nchunks + nthreads - 1) / nthreads;
        for (0..nthreads) |thread_idx| {
            pool.spawn(work, .{ chunks_per_thread, thread_idx, args, hashes, allocator }) catch |err| print_and_exit_with_error("Unable to spawn thread {d}. err: {any}", .{ thread_idx, err });
        }
        pool.deinit();
    } else {
        work(nchunks, 0, args, hashes, allocator);
    }

    return hashes;
}

fn handle_stdout(stdout_args: Args, allocator: std.mem.Allocator) void {
    var file = open_file_from_fname(stdout_args.in_fname);
    defer file.close();

    const stat: stdfile.Stat = file.stat() catch print_and_exit_with_error("Unable to stat file.\n", .{});
    const size: u64 = stat.size;
    const nchunks: u64 = (size + stdout_args.chunk_size - 1) / stdout_args.chunk_size;

    const hashes = compute_hashes(stdout_args, allocator);
    defer allocator.free(hashes);

    const merkle_tree = construct_merkle_tree(hashes, allocator);
    defer allocator.free(merkle_tree);

    std.debug.print("File: {s}\n", .{stdout_args.in_fname});
    std.debug.print("Chunk\tOffset\tHash\n", .{});
    for (0..nchunks) |chunk_idx| {
        std.debug.print("{d}\t{d}\t", .{ chunk_idx, chunk_idx * stdout_args.chunk_size });
        for (hashes[chunk_idx * 32 .. (chunk_idx + 1) * 32]) |byte| {
            std.debug.print("{x:0>2}", .{byte});
        }
        std.debug.print("\n", .{});
    }

    std.debug.print("Overall hash: \t", .{});
    for (merkle_tree[0..32]) |byte| {
        std.debug.print("{x:0>2}", .{byte});
    }
    std.debug.print("\n", .{});
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

pub fn main() !void {
    var args = std.process.args();
    var input_fname_buf: [255]u8 = [_]u8{0} ** 255;
    var output_fname_buf: [255]u8 = [_]u8{0} ** 255;
    const parsed_args = parse_args(&args, &input_fname_buf, &output_fname_buf);
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    if (parsed_args) |a| {
        handle_stdout(a, allocator);
    } else {
        std.debug.print("{s}\n\n", .{help_text});
    }
}
