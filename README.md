# ZChunk: Fast File Chunking & Hashing Utility

ZChunk is a high-performance file processing utility written in Zig that splits files into configurable chunks and computes cryptographic hashes of each chunk. It's designed as a foundational tool for backup systems, file synchronization, content-addressable storage, and data integrity verification.

## Why ZChunk is Useful

ZChunk enables efficient file processing by:

- **Reducing Data Transfer**: Only changed chunks need to be transferred during syncs/backups
- **Enabling Deduplication**: Identify and store identical chunks only once
- **Verifying Data Integrity**: Detect corruption or tampering at the chunk level
- **Facilitating Parallel Processing**: Break large files into independently processable units

### Real-World Applications

- **Backup Systems**: Modern backup tools use chunking to minimize data transfer
- **File Sync Services**: Dropbox, Google Drive, and similar services use chunk-based syncing
- **Content-Addressable Storage**: Systems like Git use content hashing for efficient storage
- **Large File Verification**: Verify integrity of downloaded files or archives

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

Process a single file with default settings:
```bash
$ zchunk largefile.iso
```

Use smaller chunks and multiple threads:
```bash
$ zchunk --chunk-size=256 --threads=4 backup.tar
```

Process multiple files with JSON output:
```bash
$ zchunk --mode=json --output=hashes.json file1.mp4 file2.mp4
```

Verify a previously processed file:
```bash
$ zchunk --mode=verify --input=hashes.json largefile.iso
```

## Sample Output

### Summary Mode (default)
```
File: largefile.iso (1.24 GB)
Chunks: 1,269 (1 MB each)
Processing time: 2.34 seconds
Throughput: 542.7 MB/s
Overall hash: 7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069
```

### JSON Mode
```json
{
  "filename": "largefile.iso",
  "size": 1331634176,
  "chunk_size": 1048576,
  "chunks": [
    {
      "index": 0,
      "offset": 0,
      "size": 1048576,
      "hash": "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12"
    },
    {
      "index": 1,
      "offset": 1048576,
      "size": 1048576,
      "hash": "de9f2c7fd25e1b3afad3e85a0bd17d9b100db4b3"
    },
    ...
  ],
  "overall_hash": "7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
  "processing_time_ms": 2340,
  "throughput_mb_s": 542.7
}
```

### Chunks Mode
```
File: largefile.iso (1.24 GB)
Chunk  Offset      Size        Hash (SHA-256)
0      0           1048576     2fd4e1c67a2d28fced849ee1bb76e7391b93eb12
1      1048576     1048576     de9f2c7fd25e1b3afad3e85a0bd17d9b100db4b3
2      2097152     1048576     6b86b273ff34fce19d6b804eff5a3f5747ada4ea
...
1268   1330585600  1048576     ef2d127de37b942baad06145e54b0c619a1f22e0
```

### Verification Mode
```
File: largefile.iso
Status: VERIFIED (all chunks match)
Changed chunks: 0 of 1,269
Processing time: 2.21 seconds
```

Or if changes are detected:
```
File: largefile.iso
Status: MODIFIED (3 chunks changed)
Changed chunks: 3 of 1,269
  Chunk 42: expected 2fd4e1c..., found 6b86b27...
  Chunk 97: expected de9f2c..., found ef2d127...
  Chunk 843: expected 6b86b2..., found d4735e...
Processing time: 2.19 seconds
```

## License

MIT
