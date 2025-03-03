const std = @import("std");

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

const HashKind = enum { Sha256, Sha1, Xxhash };
const ModeKind = enum { Summary, Json, Chunks };

const ArgsType = enum { help, file, stdout };

const HelpArgs = struct {};
const FileArgs = struct {
    /// User can specify up to 2^16 = 65,536 byte chunks
    chunkSize: u16,
    hash: HashKind = HashKind.Sha256,
    mode: ModeKind = ModeKind.Summary,
    outputFname: []const u8,
    inputFname: []const u8,
    /// if not provided, we will only use a single thread
    /// if -1 is provided we use all available cores
    threads: ?i8, // -1 means use num cores
};
const StdoutArgs = struct {
    chunkSize: u16,
    hash: HashKind = HashKind.Sha256,
    mode: ModeKind = ModeKind.Summary,
    outputFname: ?[]const u8 = null,
    inputFname: []const u8,
    /// if not provided, we will only use a single thread
    /// if -1 is provided we use all available cores
    threads: ?i8,
};

const Args = union(ArgsType) { help: HelpArgs, file: FileArgs, stdout: StdoutArgs };

fn parseArgs(args: *std.process.ArgIterator, inputFnameBuf: [*]u8, outputFnameBuf: [*]u8) Args {
    _ = args.skip(); // skip executable
    var mode: ModeKind = ModeKind.Summary;
    var hash: HashKind = HashKind.Sha256;
    var chunkSize: u16 = 1024;
    var nThreads: i8 = 1;
    var inFile: ?[]const u8 = null;
    var outFile: ?[]const u8 = null;

    var fnameArg: ?[]const u8 = null;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--help")) {
            return Args{ .help = HelpArgs{} };
        }
        var split = std.mem.split(u8, arg, "=");
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
            } else if (std.mem.eql(u8, hashStr, "sha1")) {
                hash = HashKind.Sha1;
            } else if (std.mem.eql(u8, hashStr, "xxhash")) {
                hash = HashKind.Xxhash;
            } else {
                std.debug.print("Unrecognized hashing function provided: {s}\n\n", .{hashStr});
                return Args{ .help = HelpArgs{} };
            }
        } else if (std.mem.eql(u8, flag, "--chunk-size")) {
            const chunkSizeStr = split.next().?;
            chunkSize = std.fmt.parseInt(u16, chunkSizeStr, 0) catch |err| switch (err) {
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
            nThreads = std.fmt.parseInt(i8, nThreadStr, 0) catch |err| switch (err) {
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

fn handleStdout(stdoutArgs: StdoutArgs) void {
    std.debug.print("Outputting to stdout: {?}\n\n", .{stdoutArgs});
}

pub fn main() !void {
    var args = std.process.args();
    var inputFnameBuf: [255]u8 = [_]u8{0} ** 255;
    var outputFnameBuf: [255]u8 = [_]u8{0} ** 255;
    const zcargs = parseArgs(&args, &inputFnameBuf, &outputFnameBuf);
    switch (zcargs) {
        ArgsType.help => std.debug.print("{s}\n\n", .{HELP_TEXT}),
        ArgsType.file => handleFile(zcargs.file),
        ArgsType.stdout => handleStdout(zcargs.stdout),
    }
}
