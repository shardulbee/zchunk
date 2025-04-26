const std = @import("std");
const exit = std.process.exit;
const sha2 = std.crypto.hash.sha2;
const stdfile = std.fs.File;

pub const ChunkingStrategy = enum { fixed, cdc };

const FIXED_CHUNK_SIZE_BYTES: u32 = 1024 * 1024; // 1024 KiB
const CDC_MIN_CHUNK_SIZE_BYTES: u32 = 256 * 1024; // 256 KiB
const CDC_AVG_CHUNK_SIZE_BYTES: u32 = 1024 * 1024; // 1024 KiB
const CDC_MAX_CHUNK_SIZE_BYTES: u32 = 4096 * 1024; // 4096 KiB
const CDC_AVG_CHUNK_PATTERN_MASK: u32 = CDC_AVG_CHUNK_SIZE_BYTES - 1; // Mask to achieve avg size. Assumes AVG_CHUNK_SIZE is power of 2.
const WINDOW_SIZE: usize = 64;
const WINDOW_MASK = WINDOW_SIZE - 1;
const IRREDUCIBLE_POLYNOMIAL: u64 = 0x42f0e1eba9ea3693;

fn _gp2_mul(x: u64, y: u64) u64 {
    var result: u64 = 0;
    var a = x;
    var b = y;

    while (b != 0) {
        if ((b & 1) != 0) {
            result ^= a;
        }
        const high_bit_set = (a & (1 << 63)) != 0;
        a <<= 1;
        if (high_bit_set) {
            a ^= IRREDUCIBLE_POLYNOMIAL;
        }
        b >>= 1;
    }

    return result;
}

test "_gp2_mul_specific_known_values" {
    const testing = std.testing;
    try testing.expectEqual(@as(u64, 0), _gp2_mul(0, 0));
    try testing.expectEqual(@as(u64, 0), _gp2_mul(0, 1));
    try testing.expectEqual(@as(u64, 0), _gp2_mul(1, 0));
    try testing.expectEqual(@as(u64, 1), _gp2_mul(1, 1));
    // x * x^63 = x^64 = irreducible polynomial
    try testing.expectEqual(IRREDUCIBLE_POLYNOMIAL, _gp2_mul(2, 1 << 63));
}

test "_gp2_mul_property_multiplication_by_zero_and_one" {
    const testing = std.testing;
    const reps = 100;
    var prng = std.Random.DefaultPrng.init(0);
    const random = prng.random();

    for (0..reps) |_| {
        const x = random.int(u64);
        try testing.expectEqual(@as(u64, 0), _gp2_mul(x, 0));
        try testing.expectEqual(@as(u64, 0), _gp2_mul(0, x));
        try testing.expectEqual(x, _gp2_mul(x, 1));
        try testing.expectEqual(x, _gp2_mul(1, x));
    }
}

test "_gp2_mul_property_commutativity" {
    const testing = std.testing;
    const reps = 100;
    var prng = std.Random.DefaultPrng.init(1); // Seed 1
    const random = prng.random();

    for (0..reps) |_| {
        const a = random.int(u64);
        const b = random.int(u64);
        try testing.expectEqual(_gp2_mul(a, b), _gp2_mul(b, a));
    }
}

test "_gp2_mul_property_distributivity_over_xor" {
    const testing = std.testing;
    const reps = 100;
    var prng = std.Random.DefaultPrng.init(2); // Seed 2
    const random = prng.random();

    for (0..reps) |_| {
        const a = random.int(u64);
        const b = random.int(u64);
        const c = random.int(u64);
        try testing.expectEqual(_gp2_mul(a, b ^ c), (_gp2_mul(a, b) ^ _gp2_mul(a, c)));
    }
}

const RabinFingerprintTables = struct {
    mod_table: [256]u64,
    out_table: [256]u64,

    fn init() RabinFingerprintTables {
        return .{
            .mod_table = compute_mod_table(),
            .out_table = compute_out_table(),
        };
    }

    fn compute_mod_table() [256]u64 {
        var inner: [256]u64 = undefined;
        for (0..256) |i| {
            if (i == 0) {
                inner[i] = 0;
            } else if (i == 1) {
                inner[i] = IRREDUCIBLE_POLYNOMIAL; // P' = P without x^64 term
            } else {
                // This uses GF(2) multiplication of i and the irreducible polynomial
                inner[i] = _gp2_mul(@intCast(i), IRREDUCIBLE_POLYNOMIAL);
            }
        }
        return inner;
    }

    /// This computes the value of i * x^((window_size-1)*8) mod P(x) for all i in [0, 255]
    /// This is used to XOR the contribution of the outgoing byte in the fingerprint
    /// this means computing x^504. We can do this by squaring x^63 8 times.
    /// x^63 = 1 << 63, so we just compute _gp2_mul(1 << 63, 1 << 63) 8 times.
    fn compute_out_table() [256]u64 {
        var inner: [256]u64 = undefined;
        var pow_base: u64 = _gp2_mul(1 << 63, 1 << 63);
        for (0..6) |_| {
            pow_base = _gp2_mul(pow_base, 1 << 63);
        }

        for (0..256) |i| {
            inner[i] = _gp2_mul(i, pow_base);
        }
        return inner;
    }
};

// Helper function for testing window rotation
fn rotate_left_test_helper(buffer: [WINDOW_SIZE]u8, count: usize) [WINDOW_SIZE]u8 {
    var rotated_buffer: [WINDOW_SIZE]u8 = undefined;
    const n = WINDOW_SIZE;
    const c = count % n;
    for (0..n) |i| {
        rotated_buffer[i] = buffer[(i + c) % n];
    }
    return rotated_buffer;
}

test "rabin fingerprint window_pos vs buffer rotation equivalence" {
    const testing = std.testing;
    const tables = RabinFingerprintTables.init();
    const reps = 10000; // Number of fuzzing iterations
    var prng = std.Random.DefaultPrng.init(3); // Seed for this test
    const random = prng.random();

    for (0..reps) |_| {
        var original_buffer: [WINDOW_SIZE]u8 = undefined;
        random.bytes(&original_buffer);

        const k = random.intRangeAtMost(usize, 0, WINDOW_SIZE - 1);

        const fp1 = compute_rabin_fingerprint(original_buffer, k, &tables);
        const rotated_buffer = rotate_left_test_helper(original_buffer, k);
        const fp2 = compute_rabin_fingerprint(rotated_buffer, 0, &tables);

        try testing.expectEqual(fp1, fp2);
    }
}

test "rabin fingerprint compute and update consistency" {
    const testing = std.testing;

    // Initialize tables
    const tables = RabinFingerprintTables.init();

    // Create a random number generator
    var rnd = std.Random.DefaultPrng.init(42); // Using a common seed, can be varied
    const random = rnd.random();

    // Run multiple iterations of the test
    const reps = 1000; // Increased iterations for more thorough fuzzing
    for (0..reps) |i| {
        // Generate random initial window
        var window: [WINDOW_SIZE]u8 = undefined;
        random.bytes(&window);

        // Compute initial fingerprint. For this test, we can simplify and always start at window_pos = 0
        // for the initial computation, as the update logic implicitly handles the sliding window effect.
        const fp1 = compute_rabin_fingerprint(window, 0, &tables);

        // Generate a new random byte to slide into the window
        const in_byte = random.int(u8);
        // The byte that will be pushed out is at position 0 of the current `window` array
        // because compute_rabin_fingerprint with window_pos = 0 starts reading from window[0].
        const out_byte = window[0];

        // Update fingerprint using update function
        const fp2_updated = update_rabin_fingerprint(fp1, in_byte, out_byte, &tables);

        // To recompute from scratch for verification:
        // Create the new window state by sliding the window content and adding the in_byte.
        var new_window_for_recompute: [WINDOW_SIZE]u8 = undefined;
        @memcpy(new_window_for_recompute[0 .. WINDOW_SIZE - 1], window[1..WINDOW_SIZE]);
        new_window_for_recompute[WINDOW_SIZE - 1] = in_byte;

        // Compute the fingerprint on this new window state, starting at window_pos = 0.
        const fp2_recomputed = compute_rabin_fingerprint(new_window_for_recompute, 0, &tables);

        // Verify that both methods produce the same fingerprint
        if (fp2_recomputed != fp2_updated) {
            // (Debug printing as before, useful for failures)
            std.debug.print("\nIteration {d}: Mismatch!\n", .{i});
            std.debug.print("  fp1 (initial on old window): {x}\n", .{fp1});
            std.debug.print("  out_byte (from old window[0]): {x}\n", .{out_byte});
            std.debug.print("  in_byte (new byte):            {x}\n", .{in_byte});
            std.debug.print("  fp2_updated (via update_rabin_fingerprint): {x}\n", .{fp2_updated});
            std.debug.print("  fp2_recomputed (on new window state):   {x}\n", .{fp2_recomputed});
            // std.debug.print("  initial window: {any}\n", .{window});
            // std.debug.print("  new_window_for_recompute: {any}\n", .{new_window_for_recompute});
        }
        try testing.expectEqual(fp2_recomputed, fp2_updated);
    }
}

/// Computes the Rabin fingerprint for a given window.
/// The window defines a WINDOW_SIZE * 8 degree polynomial.
/// The irreducible polynomial is defined by IRREDUCIBLE_POLYNOMIAL.
/// The degree of the irreducible polypomial determines the number of bits in the fingerprint.
///
/// In our case WINDOW_SIZE is 64 meaning the window defines a 512 degree polynomial.
/// But all computation is done in GF(2^64) so we need to reduce the degree of the polynomial.
///
/// The degree of the irreducible polynomial is 64.
/// Therefore the fingerprint will be 64 bits long (8 bytes) and is represented as a u64.
///
/// The general algorithm to comptue the fingerprint is as follows:
///
/// 1. Initialize the fingerprint to 0.
/// 2. For each byte in the window:
///    a. Shift the fingerprint left by 8 bits.
///    b. Add the byte to the fingerprint.
///    Adding the byte in GF(2^64) is equivalent to XORing the byte with the fingerprint.
///    c. Reduce the fingerprint modulo the irreducible polynomial.
///    We use the mod_table to look up what should be added to the result to perform the reduction
fn compute_rabin_fingerprint(fbuffer: [WINDOW_SIZE]u8, window_pos: usize, tables: *const RabinFingerprintTables) u64 {
    var pos = window_pos;
    var fp: u64 = 0;
    for (0..WINDOW_SIZE) |_| {
        // Correct logic: Apply modulus correction based on the high byte of fp *before* shift/add
        fp = ((fp << 8) ^ @as(u64, fbuffer[pos])) ^ tables.mod_table[fp >> 56];
        pos = (pos + 1) & WINDOW_MASK;
    }
    return fp;
}

fn update_rabin_fingerprint(fingerprint: u64, in_byte: u8, out_byte: u8, tables: *const RabinFingerprintTables) u64 {
    // 1. Calculate approximate fingerprint without the out_byte's contribution
    const fp_without_out = fingerprint ^ tables.out_table[out_byte];

    // 2. Shift this value and XOR in the new byte
    const shifted_plus_in = (fp_without_out << 8) ^ @as(u64, in_byte);

    // 3. Determine the correction factor based on the high byte of the state *before* shift and add (i.e., fp_without_out)
    const correction = tables.mod_table[fp_without_out >> 56];

    // 4. Apply the correction
    return shifted_plus_in ^ correction;
}

// This function assumes that the min, max, and avg chunk sizes are set.
// The caller is responsible for performing this validation.
fn compute_cdc_hashes(file: std.fs.File, allocator: std.mem.Allocator) ![]u8 {
    // Use the hardcoded constants
    const min_chunk_size_bytes = CDC_MIN_CHUNK_SIZE_BYTES;
    const max_chunk_size_bytes = CDC_MAX_CHUNK_SIZE_BYTES;
    const avg_chunk_size_bytes = CDC_AVG_CHUNK_SIZE_BYTES;

    std.debug.print("CDC: min={d}kb, max={d}kb, avg={d}kb\n", .{ min_chunk_size_bytes / 1024, max_chunk_size_bytes / 1024, avg_chunk_size_bytes / 1024 });

    const stat = try file.stat();
    std.debug.print("File size: {d} bytes\n", .{stat.size});

    // Calculate the maximum number of indices we could need if every chunk was the minimum size
    // This is a safe upper bound because the last chunk may be smaller than the minimum size
    const max_num_indices = (stat.size + min_chunk_size_bytes - 1) / min_chunk_size_bytes;
    std.debug.print("Allocating indices array with max size: {d}\n", .{max_num_indices});
    // We allocate a buffer in which the indices of the chunk boundaries will be stored by find_chunk_boundaries
    const indices = try allocator.alloc(usize, max_num_indices);
    defer allocator.free(indices);

    std.debug.print("Finding chunk boundaries...\n", .{});
    const tables = RabinFingerprintTables.init();
    const final_indices = try find_chunk_boundaries(
        file,
        indices,
        &tables,
    );
    if (final_indices.len == 0) {
        std.debug.print("No chunks found!\n", .{});
        return error.NoChunksFound;
    }
    std.debug.print("Found {d} chunk boundaries\n", .{final_indices.len});
    std.debug.print("First two indices: {d}, {d}\n", .{ final_indices[0], final_indices[1] });

    // We assume that find_chunk_boundaries returns a slice with the last index being the last byte of the file
    const num_chunks = final_indices.len - 1;
    const hash_size = num_chunks * 32;
    std.debug.print("Allocating {d} bytes for {d} chunk hashes\n", .{ hash_size, num_chunks });

    const hashes = try allocator.alloc(u8, num_chunks * 32);
    errdefer allocator.free(hashes);

    // Seek back to the beginning of the file to read chunks
    try file.seekTo(0);
    const reader = std.fs.File.reader(file);
    std.debug.print("Allocating buffer of size {d} for chunk data\n", .{max_chunk_size_bytes});
    var data_buffer = try allocator.alloc(u8, max_chunk_size_bytes);
    var n_bytes: usize = 0;
    defer allocator.free(data_buffer);

    std.debug.print("Computing hashes for {d} chunks\n", .{num_chunks});
    for (0..num_chunks - 1) |chunk_idx| {
        // the difference between successive byte indices is the number of bytes to read
        n_bytes = final_indices[chunk_idx + 1] - final_indices[chunk_idx];
        // std.debug.print("Chunk {d}: Reading {d} bytes\n", .{ chunk_idx, n_bytes });
        _ = reader.read(data_buffer[0..n_bytes]) catch |err| handle_read_error(err);
        sha2.Sha256.hash(data_buffer[0..n_bytes], @ptrCast(hashes[chunk_idx * 32 .. (chunk_idx + 1) * 32]), .{});
    }

    std.debug.print("Finished computing hashes\n", .{});
    return hashes;
}

/// This should return a sequence of indices that correspond to the byte index of boundaries
/// The size of the returned slice is bounded above by the data size / min chunk size
/// The caller is responsible for providing the memory for the returned data
/// This function will return a slice of the provided memory.
/// The first index should always be 0 and the last index should always be the last byte of the file
fn find_chunk_boundaries(
    file: std.fs.File,
    indices: []usize,
    tables: *const RabinFingerprintTables,
) ![]const usize {
    const stat = try file.stat();
    std.debug.print("find_chunk_boundaries: file size = {d}\n", .{stat.size});

    // Handle empty or small files
    if (stat.size == 0) {
        indices[0] = 0;
        return indices[0..1];
    }

    // First chunk always starts at 0
    indices[0] = 0;
    var last_chunk_idx: usize = 0;

    const reader = std.fs.File.reader(file);
    var buffered_reader = std.io.bufferedReader(reader);
    var reader_impl = buffered_reader.reader();

    // Circular buffer for our window
    var window_buffer: [WINDOW_SIZE]u8 = undefined;
    var window_pos: usize = 0;
    var byte_idx: usize = 0;
    var fingerprint: u64 = undefined;

    // Read bytes until EOF
    while (byte_idx < stat.size) : (byte_idx += 1) {
        // Read next byte and update circular buffer
        const in_byte = reader_impl.readByte() catch |err| handle_read_error(err);
        const out_byte = window_buffer[window_pos];
        window_buffer[window_pos] = in_byte;
        window_pos = (window_pos + 1) & WINDOW_MASK;

        // Calculate current chunk size
        const current_chunk_size = byte_idx - indices[last_chunk_idx];

        // If we haven't reached minimum chunk size, just maintain window and continue
        if (current_chunk_size < CDC_MIN_CHUNK_SIZE_BYTES) {
            continue;
        }

        // If we've exceeded max chunk size, force a boundary
        if (current_chunk_size >= CDC_MAX_CHUNK_SIZE_BYTES) {
            last_chunk_idx += 1;
            indices[last_chunk_idx] = byte_idx;
            continue;
        }

        // If this is the first byte after minimum size, compute initial fingerprint
        if (current_chunk_size == CDC_MIN_CHUNK_SIZE_BYTES) {
            fingerprint = compute_rabin_fingerprint(window_buffer, window_pos, tables);
            if ((fingerprint & @as(u64, CDC_AVG_CHUNK_PATTERN_MASK)) == 0) {
                last_chunk_idx += 1;
                indices[last_chunk_idx] = byte_idx;
            }
        } else {
            // Otherwise update existing fingerprint and check for boundary
            fingerprint = update_rabin_fingerprint(fingerprint, in_byte, out_byte, tables);
            if ((fingerprint & @as(u64, CDC_AVG_CHUNK_PATTERN_MASK)) == 0) {
                last_chunk_idx += 1;
                indices[last_chunk_idx] = byte_idx;
            }
        }
    }

    // Add final boundary at EOF if needed
    if (indices[last_chunk_idx] != stat.size) {
        last_chunk_idx += 1;
        indices[last_chunk_idx] = stat.size;
    }

    return indices[0 .. last_chunk_idx + 1];
}

pub fn handle_read_error(err: anytype) noreturn {
    std.debug.print("Encountered error while reading: {?}\n", .{err});
    exit(1);
}

pub fn print_and_exit_with_error(comptime msg: []const u8, args: anytype) noreturn {
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
pub fn construct_merkle_tree(hashes: []u8, allocator: std.mem.Allocator) []const u8 {
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

pub const ComputeHashesResult = struct {
    hashes: []u8,
    file_size: u64,
};

pub fn compute_hashes(args_in_fname: []const u8, args_n_threads: u8, args_chunking_strategy: ChunkingStrategy, allocator: std.mem.Allocator) ComputeHashesResult {
    var file: std.fs.File = open_file_from_fname(args_in_fname);
    defer file.close();

    const stat_val = file.stat() catch print_and_exit_with_error("Unable to stat file.\n", .{}); // Renamed to avoid conflict
    const size: u64 = stat_val.size;

    if (args_chunking_strategy == ChunkingStrategy.cdc) {
        // Pass allocator directly, constants are used inside
        const cdc_hashes = compute_cdc_hashes(file, allocator) catch |err| {
            std.debug.print("Error during CDC processing: {any}\n", .{err});
            std.process.exit(1);
        };
        return .{ .hashes = cdc_hashes, .file_size = size };
    }

    // Original fixed-size chunking implementation
    var pool: std.Thread.Pool = undefined;
    var nthreads: u32 = undefined;
    if (args_n_threads == 0) {
        nthreads = @truncate(std.Thread.getCpuCount() catch print_and_exit_with_error("unable to get cpu count\n", .{}));
    } else {
        nthreads = args_n_threads;
    }

    // Use fixed chunk size constant
    const nchunks: u64 = (size + FIXED_CHUNK_SIZE_BYTES - 1) / FIXED_CHUNK_SIZE_BYTES;

    // need to pre-allocate space where each thread will write their hash
    // each sha256 is 32 bytes so we need 32 bytes * chunks
    const hashes = allocator.alloc(u8, nchunks * 32) catch print_and_exit_with_error("Unable to allocate memory for hashes", .{});

    if (nthreads > 1) {
        pool.init(.{ .allocator = allocator, .n_jobs = @as(u32, nthreads) }) catch print_and_exit_with_error("Unable to init threadpool", .{});
        defer pool.deinit();
        const chunks_per_thread: u64 = (nchunks + nthreads - 1) / nthreads;
        for (0..nthreads) |thread_idx| {
            // Pass necessary info excluding chunk size args
            pool.spawn(work, .{ chunks_per_thread, thread_idx, args_in_fname, hashes, allocator }) catch |err| print_and_exit_with_error("Unable to spawn thread {d}. err: {any}", .{ thread_idx, err });
        }
    } else {
        // Call work directly for single thread
        work(nchunks, 0, args_in_fname, hashes, allocator);
    }

    return .{ .hashes = hashes, .file_size = size };
}

// work function is only used for fixed chunking currently
fn work(nchunks_per_thread: u64, thread_idx: usize, in_fname: []const u8, hashes: []u8, allocator: std.mem.Allocator) void {
    var file = open_file_from_fname(in_fname);
    defer file.close();

    // Use fixed chunk size constant
    const chunk_size = FIXED_CHUNK_SIZE_BYTES;

    file.seekTo(thread_idx * nchunks_per_thread * chunk_size) catch |err| print_and_exit_with_error("Unable to seek to correct file location for thread {d}. err: {any}\n", .{ thread_idx, err });
    const reader = stdfile.reader(file);
    var buffered_reader = std.io.bufferedReader(reader);

    var fbuffer = allocator.alloc(u8, chunk_size) catch |err| print_and_exit_with_error("Unable to allocate fbuffer for thread_idx {d}. err: {any}", .{ thread_idx, err });
    defer allocator.free(fbuffer);

    var bytes_read: u64 = 0;
    var hash_buf: []u8 = undefined;
    for (0..nchunks_per_thread) |chunk_idx| {
        // Calculate the global chunk index this thread is working on
        const global_chunk_idx = thread_idx * nchunks_per_thread + chunk_idx;
        // Ensure we don't write past the allocated hashes slice
        if ((global_chunk_idx + 1) * 32 > hashes.len) {
            // This can happen if the total number of chunks is not perfectly divisible by nthreads
            break;
        }
        hash_buf = hashes[global_chunk_idx * 32 .. (global_chunk_idx + 1) * 32];
        bytes_read = buffered_reader.read(fbuffer) catch |err| handle_read_error(err);
        // Check if zero bytes read, meaning EOF, should only happen if file size is 0 or multiple of chunk size
        if (bytes_read == 0) break;
        sha2.Sha256.hash(fbuffer[0..bytes_read], @ptrCast(hash_buf), .{});
        // Check if we read less than a full chunk, indicating the last chunk
        if (bytes_read != chunk_size) {
            break;
        }
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

// Helper function to create a test file with random data of a specific total size
fn create_random_file(allocator: std.mem.Allocator, tmp_dir: *std.testing.TmpDir, total_size: usize, file_name: []const u8) !struct { data: []u8, path: []const u8 } {
    // Generate random data
    const data = try allocator.alloc(u8, total_size);
    // errdefer allocator.free(data); // Caller owns data

    var rnd = std.Random.DefaultPrng.init(42); // Consistent seed for file content
    rnd.random().bytes(data);

    // Create test file path
    const real_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(real_path);

    const test_file_path = try std.fs.path.join(
        allocator,
        &.{ real_path, file_name },
    );
    // errdefer allocator.free(test_file_path); // Caller owns path

    return .{ .data = data, .path = test_file_path };
}

test "chunk hashing correctly identifies modified chunk" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var tmp_dir = testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    // Use fixed chunk size constant for this test
    const chunk_size = FIXED_CHUNK_SIZE_BYTES;
    const num_chunks: u32 = 100;

    // Create test file with random data (using the fixed chunk size)
    const test_file = try create_test_file(allocator, &tmp_dir, chunk_size, num_chunks);
    defer {
        allocator.free(test_file.data);
        allocator.free(test_file.path);
    }

    // Get reference to the data for modification later
    var random_data = test_file.data;

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
        const original_hashes_result = compute_hashes(test_file.path, 1, ChunkingStrategy.fixed, allocator);
        const original_hashes = original_hashes_result.hashes;
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
        const modified_hashes_result = compute_hashes(test_file.path, 1, ChunkingStrategy.fixed, allocator);
        const modified_hashes = modified_hashes_result.hashes;
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

test "cdc chunk hashing localizes changes" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var tmp_dir = testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const target_file_size = CDC_AVG_CHUNK_SIZE_BYTES * 10 + CDC_AVG_CHUNK_SIZE_BYTES / 2;

    const test_file_info = try create_random_file(allocator, &tmp_dir, @intCast(target_file_size), "cdc_test_file.bin");
    defer {
        allocator.free(test_file_info.data);
        allocator.free(test_file_info.path);
    }

    var file_data = test_file_info.data;
    var random_generator = std.Random.DefaultPrng.init(99);
    const num_iterations = 5;
    for (0..num_iterations) |iter| {
        std.debug.print("CDC Test Iteration: {d}\n", .{iter});
        {
            const file_to_write = try std.fs.createFileAbsolute(test_file_info.path, .{});
            defer file_to_write.close();
            try file_to_write.writeAll(file_data);
        }

        const original_result = compute_hashes(test_file_info.path, 1, ChunkingStrategy.cdc, allocator);
        const original_hashes = original_result.hashes;
        defer allocator.free(original_hashes);

        // Randomly select a byte to modify
        const byte_idx_to_modify = random_generator.random().uintLessThan(usize, file_data.len);
        file_data[byte_idx_to_modify] = ~file_data[byte_idx_to_modify];

        {
            const file_to_write = try std.fs.createFileAbsolute(test_file_info.path, .{});
            defer file_to_write.close();
            try file_to_write.writeAll(file_data);
        }

        const modified_result = compute_hashes(test_file_info.path, 1, ChunkingStrategy.cdc, allocator);
        const modified_hashes = modified_result.hashes;
        defer allocator.free(modified_hashes);

        const num_original_chunks = original_hashes.len / 32;
        const num_modified_chunks = modified_hashes.len / 32;

        // Calculate the number of differing hashes based on the metric discussed
        var differing_hashes_count: usize = 0;
        const min_chunk_count = @min(num_original_chunks, num_modified_chunks);

        for (0..min_chunk_count) |i| {
            const original_hash_ptr = original_hashes[i * 32 .. (i + 1) * 32];
            const modified_hash_ptr = modified_hashes[i * 32 .. (i + 1) * 32];
            if (!std.mem.eql(u8, original_hash_ptr, modified_hash_ptr)) {
                differing_hashes_count += 1;
            }
        }
        differing_hashes_count += @intCast(@abs(num_original_chunks - num_modified_chunks));

        std.debug.print("  Original Chunks: {d}, Modified Chunks: {d}, Differing Hashes (metric: {d}\n", .{ num_original_chunks, num_modified_chunks, differing_hashes_count });
        std.debug.print("  Byte modified at index: {d}\n", .{byte_idx_to_modify});

        // Assertion: The number of "changed" hashes should be very small.
        // Typically 1 (byte change within a chunk, no boundary shift)
        // or 2 (boundary shifts, affecting two chunks; or one chunk split/merged near the modification)
        // Allowing up to 3 to be safe for edge cases with boundary logic.
        try testing.expect(differing_hashes_count <= 3);

        file_data[byte_idx_to_modify] = ~file_data[byte_idx_to_modify];
    }
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
    const single_hashes_result = compute_hashes(test_file.path, 1, ChunkingStrategy.fixed, allocator);
    const single_hashes = single_hashes_result.hashes;
    defer allocator.free(single_hashes);

    // Array of thread counts to test
    const thread_counts = [_]u8{ 2, 4, 8, 0 }; // 0 = auto-detect

    for (thread_counts) |n_threads| {
        const thread_hashes_result = compute_hashes(test_file.path, n_threads, ChunkingStrategy.fixed, allocator);
        const thread_hashes = thread_hashes_result.hashes;
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

    const result = compute_hashes(empty_file_path, 1, ChunkingStrategy.fixed, allocator);
    const hashes = result.hashes;
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

    const result = compute_hashes(file_path, 1, ChunkingStrategy.fixed, allocator);
    const hashes = result.hashes;
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

    const chunk_size: u32 = FIXED_CHUNK_SIZE_BYTES + 1;

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

    const result = compute_hashes(file_path, 1, ChunkingStrategy.fixed, allocator);
    const hashes = result.hashes;
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

    const result = compute_hashes(file_path, 1, ChunkingStrategy.fixed, allocator);
    const hashes = result.hashes;
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
