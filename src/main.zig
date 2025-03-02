const std = @import("std");

// ```
// zchunk [options] <filename>...
//
// Options:
//   --chunk-size=<size>    Size of each chunk in KB (default: 1024)
//   --hash=<algorithm>     Hash algorithm (sha256, sha1, xxhash) [default: sha256]
//   --mode=<mode>          Output mode (summary, json, chunks) [default: summary]
//   --output=<file>        Write output to file instead of stdout
//   --threads=<n>          Number of processing threads [default: 1]
//   --help                 Show this help information
// ```
//

const ZChunkArgs = struct {
    const HashKind = enum { Sha256 };
    const ModeKind = enum { Summary, Json, Chunks };

    chunkSize: usize, // TODO: justify this size?
    hash: HashKind,
    mode: ModeKind,
    outputFname: ?[]const u8, // const prob makes sense
    help: bool,
    threads: ?i4, // -1 means use num cores
};

pub fn main() !void {
    var args = std.process.args();
    var i: usize = 0;
    _ = args.skip(); // skip executable
    var help: bool = false;
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--help")) {
            help = true;
            break;
        } else {
            // this isn't actually a thing, because people are going to provide files that we need to read
            std.debug.print("Provided unexpected flag: {s}\n\nRun `zchunk --help` to show usage.\n", .{arg});
            std.process.exit(1);
        }
        i += 1;
        std.debug.print("Arg number {d}: {s}\n", .{ i, arg });
    }
    const zcargs = ZChunkArgs{ .hash = ZChunkArgs.HashKind.Sha256, .mode = ZChunkArgs.ModeKind.Summary, .chunkSize = 16, .threads = -1, .outputFname = undefined, .help = help };
    std.debug.print("{?}\n", .{zcargs});
}
