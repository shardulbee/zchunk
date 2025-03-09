const std = @import("std");
const exit = std.process.exit;
const stdfile = std.fs.File;
const sha2 = std.crypto.hash.sha2;

const HELP_TEXT =
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

const HashKind = enum {
    Sha256,
    // TODO: Implement these
    // Sha1,
    // Xxhash
};
const ModeKind = enum { Summary, Json, Chunks };

const HelpArgs = struct {};
const FileArgs = struct {
    // TODO: this is in bytes for now, maybe change to KB?
    chunkSize: u32,
    hash: HashKind = HashKind.Sha256,
    mode: ModeKind = ModeKind.Summary,
    outputFname: []const u8,
    inputFname: []const u8,
    /// if not provided, we will only use a single thread
    /// if 0 is provided we use all available cores
    threads: u8 = 1,
};
const StdoutArgs = struct {
    chunkSize: u32,
    hash: HashKind = HashKind.Sha256,
    mode: ModeKind = ModeKind.Summary,
    outputFname: ?[]const u8 = null,
    inputFname: []const u8,
    /// if not provided, we will only use a single thread
    /// if 0 is provided we use all available cores
    threads: u8 = 1,
};

const Args = union(enum) { help: HelpArgs, file: FileArgs, stdout: StdoutArgs };

fn parseArgs(args: *std.process.ArgIterator, inputFnameBuf: [*]u8, outputFnameBuf: [*]u8) Args {
    _ = args.skip(); // skip executable
    var mode: ModeKind = ModeKind.Summary;
    var hash: HashKind = HashKind.Sha256;
    var chunkSize: u32 = 64;
    var nThreads: u8 = 1;
    var inFile: ?[]const u8 = null;
    var outFile: ?[]const u8 = null;

    var fnameArg: ?[]const u8 = null;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--help")) {
            return Args{ .help = HelpArgs{} };
        }
        var split = std.mem.splitSequence(u8, arg, "=");
        if (std.mem.eql(u8, split.peek().?, arg)) {
            fnameArg = arg;
            break;
        }
        const flag = split.next().?;
        if (std.mem.eql(u8, flag, "--mode")) {
            const modeStr = split.next().?;
            if (std.mem.eql(u8, modeStr, "summary")) {
                mode = ModeKind.Summary;
            } else if (std.mem.eql(u8, modeStr, "json")) {
                mode = ModeKind.Json;
            } else if (std.mem.eql(u8, modeStr, "chunks")) {
                mode = ModeKind.Chunks;
            } else {
                std.debug.print("Unrecognized mode provided: {s}\n\n", .{modeStr});
                return Args{ .help = HelpArgs{} };
            }
        } else if (std.mem.eql(u8, flag, "--hash")) {
            const hashStr = split.next().?;
            if (std.mem.eql(u8, hashStr, "sha256")) {
                hash = HashKind.Sha256;
                // } else if (std.mem.eql(u8, hashStr, "sha1")) {
                //     hash = HashKind.Sha1;
                // } else if (std.mem.eql(u8, hashStr, "xxhash")) {
                //     hash = HashKind.Xxhash;
            } else {
                std.debug.print("Unrecognized hashing function provided: {s}\n\n", .{hashStr});
                return Args{ .help = HelpArgs{} };
            }
        } else if (std.mem.eql(u8, flag, "--chunk-size")) {
            const chunkSizeStr = split.next().?;
            chunkSize = std.fmt.parseInt(u32, chunkSizeStr, 0) catch |err| switch (err) {
                std.fmt.ParseIntError.Overflow => {
                    std.debug.print("Provided chunk-size is too large. Max supported size is 65536. Provided: {s}\n", .{chunkSizeStr});
                    std.process.exit(1);
                },
                std.fmt.ParseIntError.InvalidCharacter => {
                    std.debug.print("Unable to parse chunk size. Provided: {s}\n", .{chunkSizeStr});
                    std.process.exit(1);
                },
            };
        } else if (std.mem.eql(u8, flag, "--output")) {
            const fname = split.next().?;
            if (fname.len > 255) {
                std.debug.print("Output filename is too long. Max supported length: 255. Provided: {d}\n", .{fname.len});
                std.process.exit(1);
            }
            @memcpy(outputFnameBuf, fname);
            outFile = outputFnameBuf[0..fname.len];
        } else if (std.mem.eql(u8, flag, "--threads")) {
            const nThreadStr = split.next().?;
            nThreads = std.fmt.parseInt(u8, nThreadStr, 0) catch |err| switch (err) {
                std.fmt.ParseIntError.Overflow => {
                    std.debug.print("Provided num threads is too large. Max supported size is 127. Provided: {s}\n", .{nThreadStr});
                    std.process.exit(1);
                },
                std.fmt.ParseIntError.InvalidCharacter => {
                    std.debug.print("Unable to parse num threads. Provided: {s}\n", .{nThreadStr});
                    std.process.exit(1);
                },
            };
        } else {
            std.debug.print("Unrecognized flag provided: {s}\n\n", .{arg});
            return Args{ .help = HelpArgs{} };
        }
    }

    // if we reach here, it means we have reached the fname.
    if (fnameArg) |fname| {
        if (fname.len > 255) {
            std.debug.print("Input filename is too long. Max supported length: 255. Provided: {d}\n", .{fname.len});
            std.process.exit(1);
        }
        @memcpy(inputFnameBuf, fname);
        inFile = inputFnameBuf[0..fname.len];
    } else {
        std.debug.print("Expected a filename.\n\n", .{});
        return Args{ .help = HelpArgs{} };
    }

    if (outFile) |f| {
        return Args{ .file = FileArgs{ .mode = mode, .threads = nThreads, .hash = hash, .chunkSize = chunkSize, .outputFname = f, .inputFname = inFile.? } };
    } else {
        return Args{ .stdout = StdoutArgs{ .mode = mode, .threads = nThreads, .hash = hash, .chunkSize = chunkSize, .inputFname = inFile.? } };
    }

    return Args{ .help = HelpArgs{} };
}

fn handleFile(fileArgs: FileArgs) void {
    std.debug.print("Outputting to a file: {?}\n\n", .{fileArgs});
}

fn handleReadError(err: stdfile.ReadError) noreturn {
    std.debug.print("Encountered error while reading: {?}\n", .{err});
    exit(1);
}

fn open_file_from_fname(fname: []const u8) std.fs.File {
    if (std.fs.path.isAbsolute(fname)) {
        return std.fs
            .openFileAbsolute(fname, stdfile.OpenFlags{}) catch |err|
            handleOpenError(err, fname);
    } else {
        return std.fs.cwd()
            .openFile(fname, stdfile.OpenFlags{}) catch |err|
            handleOpenError(err, fname);
    }
}

fn handleOpenError(err: stdfile.OpenError, path: []const u8) noreturn {
    switch (err) {
        stdfile.OpenError.FileNotFound => {
            std.debug.print("Unable to find file: {s}\n\n", .{path});
            std.process.exit(1);
        },
        else => exit(1),
    }
}

fn printAndExitWithError(comptime msg: []const u8, args: anytype) noreturn {
    std.debug.print(msg, args);
    exit(1);
}

// what do i need to not make this stdoutArgs
// 1. nthreads
// 2. file
// 3. chunksize
// 4. allocator
fn compute_hashes(fname: []const u8, stdoutArgs: StdoutArgs, allocator: std.mem.Allocator) []u8 {
    var file: std.fs.File = open_file_from_fname(fname);
    defer file.close();

    var pool: std.Thread.Pool = undefined;
    var nthreads: u32 = undefined;
    if (stdoutArgs.threads == 0) {
        nthreads = @truncate(std.Thread.getCpuCount() catch printAndExitWithError("unable to get cpu count\n", .{}));
    } else {
        nthreads = stdoutArgs.threads;
    }

    const stat: stdfile.Stat = file.stat() catch printAndExitWithError("Unable to stat file.\n", .{});
    const size: u64 = stat.size;
    const nchunks: u64 = (size + stdoutArgs.chunkSize - 1) / stdoutArgs.chunkSize;

    // need to pre-allocate space where each thread will write their hash
    // each sha256 is 32 bytes so we need 32 bytes * chunks
    const hashes = allocator.alloc(u8, nchunks * 32) catch printAndExitWithError("Unable to allocate memory for hashes", .{});

    if (nthreads > 1) {
        pool.init(.{ .allocator = allocator, .n_jobs = @as(u32, nthreads) }) catch printAndExitWithError("Unable to init threadpool", .{});
        const chunks_per_thread: u64 = (nchunks + nthreads - 1) / nthreads;
        for (0..nthreads) |thread_idx| {
            pool.spawn(work, .{ chunks_per_thread, thread_idx, stdoutArgs, hashes, allocator }) catch |err| printAndExitWithError("Unable to spawn thread {d}. err: {any}", .{ thread_idx, err });
        }
        pool.deinit();
    } else {
        work(nchunks, 0, stdoutArgs, hashes, allocator);
    }

    return hashes;
}

fn handleStdout(stdoutArgs: StdoutArgs, allocator: std.mem.Allocator) void {
    var file = open_file_from_fname(stdoutArgs.inputFname);
    defer file.close();

    const stat: stdfile.Stat = file.stat() catch printAndExitWithError("Unable to stat file.\n", .{});
    const size: u64 = stat.size;
    const nchunks: u64 = (size + stdoutArgs.chunkSize - 1) / stdoutArgs.chunkSize;

    const hashes = compute_hashes(stdoutArgs.inputFname, stdoutArgs, allocator);
    defer allocator.free(hashes);

    std.debug.print("File: {s}\n", .{stdoutArgs.inputFname});
    std.debug.print("Chunk\tOffset\tHash\n", .{});
    for (0..nchunks) |chunkIdx| {
        std.debug.print("{d}\t{d}\t", .{ chunkIdx, chunkIdx * stdoutArgs.chunkSize });
        for (hashes[chunkIdx * 32 .. (chunkIdx + 1) * 32]) |byte| {
            std.debug.print("{x:0>2}", .{byte});
        }
        std.debug.print("\n", .{});
    }
}

fn work(nchunks: u64, thread_idx: usize, args: StdoutArgs, hashes: []u8, allocator: std.mem.Allocator) void {
    var file = open_file_from_fname(args.inputFname);
    defer file.close();

    file.seekTo(thread_idx * nchunks * args.chunkSize) catch |err| printAndExitWithError("Unable to seek to correct file location for thread {d}. err: {any}\n", .{ thread_idx, err });
    const reader = stdfile.reader(file);
    var buffered_reader = std.io.bufferedReader(reader);

    var fbuffer = allocator.alloc(u8, args.chunkSize) catch |err| printAndExitWithError("Unable to allocate fbuffer for thread_idx {d}. err: {any}", .{ thread_idx, err });
    defer allocator.free(fbuffer);

    var bytes_read: u64 = 0;
    var hashBuf: []u8 = undefined;
    for (0..nchunks) |chunk_idx| {
        hashBuf = hashes[(thread_idx * nchunks + chunk_idx) * 32 .. (thread_idx * nchunks + chunk_idx + 1) * 32];
        bytes_read = buffered_reader.read(fbuffer) catch |err| handleReadError(err);
        sha2.Sha256.hash(fbuffer[0..bytes_read], @ptrCast(hashBuf), .{});
        if (bytes_read != args.chunkSize) {
            break;
        }
    }
}

pub fn main() !void {
    var args = std.process.args();
    var inputFnameBuf: [255]u8 = [_]u8{0} ** 255;
    var outputFnameBuf: [255]u8 = [_]u8{0} ** 255;
    const zcargs = parseArgs(&args, &inputFnameBuf, &outputFnameBuf);
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    // const allocator = std.heap.c_allocator;
    switch (zcargs) {
        Args.help => std.debug.print("{s}\n\n", .{HELP_TEXT}),
        Args.file => handleFile(zcargs.file),
        Args.stdout => handleStdout(zcargs.stdout, allocator),
    }
}
