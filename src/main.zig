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
    threads: u8 = 0, // -1 means use num cores
};
const StdoutArgs = struct {
    chunkSize: u32,
    hash: HashKind = HashKind.Sha256,
    mode: ModeKind = ModeKind.Summary,
    outputFname: ?[]const u8 = null,
    inputFname: []const u8,
    /// if not provided, we will only use a single thread
    /// if 0 is provided we use all available cores
    threads: u8 = 0,
};

const Args = union(enum) { help: HelpArgs, file: FileArgs, stdout: StdoutArgs };

fn parseArgs(args: *std.process.ArgIterator, inputFnameBuf: [*]u8, outputFnameBuf: [*]u8) Args {
    _ = args.skip(); // skip executable
    var mode: ModeKind = ModeKind.Summary;
    var hash: HashKind = HashKind.Sha256;
    var chunkSize: u32 = 64;
    var nThreads: u8 = 0;
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

fn handleStdout(stdoutArgs: StdoutArgs, allocator: std.mem.Allocator) void {
    var file: stdfile = undefined;
    if (std.fs.path.isAbsolute(stdoutArgs.inputFname)) {
        file = std.fs
            .openFileAbsolute(stdoutArgs.inputFname, stdfile.OpenFlags{}) catch |err|
            handleOpenError(err, stdoutArgs.inputFname);
    } else {
        file = std.fs.cwd()
            .openFile(stdoutArgs.inputFname, stdfile.OpenFlags{}) catch |err|
            handleOpenError(err, stdoutArgs.inputFname);
    }
    defer file.close();

    var pool: std.Thread.Pool = undefined;
    var nthreads: u32 = undefined;
    if (stdoutArgs.threads == 0) {
        nthreads = @truncate(std.Thread.getCpuCount() catch printAndExitWithError("unable to get cpu count\n", .{}));
    } else if (stdoutArgs.threads == 1) {
        const reader = stdfile.reader(file);
        var buffered_reader = std.io.bufferedReader(reader);

        const fBuffer: []u8 = allocator.alloc(u8, stdoutArgs.chunkSize) catch printAndExitWithError("Unable to allocate memory.", .{});
        defer allocator.free(fBuffer);
        //
        // var i: usize = 0;
        // var offset: usize = 0;
        // var resBuf: [sha2.Sha256.digest_length]u8 = undefined;
        //
        // var bytesRead: usize = buffered_reader.read(fBuffer) catch |err| handleReadError(err);
        // if (bytesRead == 0) {
        //     printAndExitWithError("Unable to read file", .{});
        // }
        // var hasher = sha2.Sha256.init(.{});
        // while (bytesRead > 0) : (bytesRead = buffered_reader.read(fBuffer) catch |err| handleReadError(err)) {
        //     std.debug.print("{d}\t{d}\t", .{ i, offset });
        //     sha2.Sha256.hash(fBuffer, &resBuf, .{});
        //     hasher.update(fBuffer);
        //
        //     for (resBuf) |byte| {
        //         std.debug.print("{x:0>2}", .{byte});
        //     }
        //     std.debug.print("\n", .{});
        //
        //     i += 1;
        //     offset += stdoutArgs.chunkSize;
        // }
        //
        // var finalHash: [sha2.Sha256.digest_length]u8 = undefined;
        // hasher.final(&finalHash);
        // std.debug.print("Final hash: ", .{});
        // for (finalHash) |byte| {
        //     std.debug.print("{x:0>2}", .{byte});
        // }
        // std.debug.print("\n", .{});

    } else {
        nthreads = stdoutArgs.threads;
    }

    pool.init(.{ .allocator = allocator, .n_jobs = @as(u32, nthreads) }) catch printAndExitWithError("Unable to init threadpool", .{});

    const stat: stdfile.Stat = file.stat() catch printAndExitWithError("Unable to stat file.\n", .{});
    const size: u64 = stat.size;
    const chunks: u64 = (size + stdoutArgs.chunkSize - 1) / stdoutArgs.chunkSize;

    // need to pre-allocate space where each thread will write their hash
    // each sha256 is 32 bytes so we need 32 bytes * chunks
    const hashes = allocator.alloc(u8, chunks * 32) catch printAndExitWithError("Unable to allocate memory for hashes", .{});
    defer allocator.free(hashes);

    // pre allocate a buffer into which each thread will read its data
    const chunkbuffer = allocator.alloc(u8, nthreads * stdoutArgs.chunkSize) catch printAndExitWithError("Unable to allocate memory for hashes", .{});
    defer allocator.free(chunkbuffer);

    for (0..chunks) |chunkIdx| {
        pool.spawn(hashChunk, .{ chunkIdx, chunkIdx % nthreads, chunkbuffer, hashes, stdoutArgs }) catch printAndExitWithError("unable to spawn thread\n", .{});
    }
    pool.deinit();

    std.debug.print("File: {s}\n", .{stdoutArgs.inputFname});
    std.debug.print("Chunk\tOffset\tHash\n", .{});
    for (0..chunks) |chunkIdx| {
        std.debug.print("{d}\t{d}\t", .{ chunkIdx, chunkIdx * stdoutArgs.chunkSize });
        for (hashes[chunkIdx * 32 .. (chunkIdx + 1) * 32]) |byte| {
            std.debug.print("{x:0>2}", .{byte});
        }
        std.debug.print("\n", .{});
    }
}

fn hashChunk(chunkIdx: usize, chunkbufidx: usize, chunkbuffer: []u8, hashes: []u8, args: StdoutArgs) void {
    var file: stdfile = undefined;
    if (std.fs.path.isAbsolute(args.inputFname)) {
        file = std.fs
            .openFileAbsolute(args.inputFname, stdfile.OpenFlags{}) catch |err|
            handleOpenError(err, args.inputFname);
    } else {
        file = std.fs.cwd()
            .openFile(args.inputFname, stdfile.OpenFlags{}) catch |err|
            handleOpenError(err, args.inputFname);
    }
    defer file.close();

    const fbuffer = chunkbuffer[chunkbufidx * args.chunkSize .. (chunkbufidx + 1) * args.chunkSize];
    file.seekTo(chunkIdx * args.chunkSize) catch printAndExitWithError("unable to seek to chunkIdx: {d}", .{chunkIdx});
    const reader = stdfile.reader(file);
    var buffered_reader = std.io.bufferedReader(reader);
    const bytesRead: usize = buffered_reader.read(fbuffer) catch |err| handleReadError(err);
    if (bytesRead == 0) {
        printAndExitWithError("error reading from file at idx {d}\n", .{chunkIdx});
    }

    var resbuf: [sha2.Sha256.digest_length]u8 = undefined;

    sha2.Sha256.hash(fbuffer[0..bytesRead], &resbuf, .{});
    @memcpy(hashes[chunkIdx * 32 .. (chunkIdx + 1) * 32], &resbuf);
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
