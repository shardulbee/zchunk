# ZChunk

Split files into chunks. Hash them.

## Features

- [x] File chunking with SHA-256 hashing
- [x] Multi-threaded processing
- [x] Merkle tree construction for integrity verification
- [ ] Output formats (JSON, summary, chunks)
- [ ] Multiple hash algorithms (SHA-1, xxHash)
- [ ] File output redirection
- [ ] Verification mode
- [ ] Content-defined chunking

## Why ZChunk?

ZChunk is primarily a learning project exploring:
- Zig
- Trying to see how close I can get to the upper bounds of performance of some common operations (hashing, disk I/O). Drawing heavily from [napkin-math](https://github.com/sirupsen/napkin-math) and trying to build up the intuition for performance.
- Data integrity verification (Merkle trees)

While the current fixed-size chunking has limitations for real-world file synchronization (adding a byte at the beginning shifts all chunks), it demonstrates core concepts that could be expanded with content-defined chunking.

## Usage

```
zchunk [options] <filename>...

Options:
  --chunk-size=<size>    Size of each chunk in KB (default: 1024)
  --hash=<algorithm>     Hash algorithm (sha256, sha1, xxhash) [default: sha256]
  --mode=<mode>          Output mode (summary, json, chunks) [default: summary]
  --output=<file>        Write output to file instead of stdout
  --threads=<n>          Number of processing threads [default: 1]
  --help                 Show this help information
```

### Examples

```bash
# Process a file with default settings
$ zchunk largefile.iso

# Use smaller chunks and 4 threads
$ zchunk --chunk-size=256 --threads=4 backup.tar

# JSON output to file
$ zchunk --mode=json --output=hashes.json file1.mp4 file2.mp4
```

## Performance

On an M3 Max processor, `zchunk` can hash a 10 GB file (1024KB chunks) in:
- ~2.25 GB/s single-threaded
- ~18 GB/s with all (16) threads

## Status

WIP

## License

MIT
