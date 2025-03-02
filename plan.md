## Plan

## High-Level Implementation Plan

- **Phase 1**: Basic single-threaded implementation
- **Phase 2**: Performance optimization and memory efficiency
- **Phase 3**: Multi-threading support
- **Phase 4**: Additional features (verification mode, content-defined chunking)

## Core Requirements

1. **File Processing**
   - Handle files of any size, including those larger than available RAM
   - Process multiple files in sequence
   - Robust error handling for file access issues

2. **Chunking**
   - Split files into fixed-size chunks (configurable)
   - Process each chunk independently
   - Track chunk position within the file

3. **Hashing**
   - Apply cryptographic hash algorithms to each chunk
   - Support multiple hash algorithms (SHA-256, SHA-1, xxHash)
   - Optional calculation of overall file hash

4. **Performance**
   - Minimize memory usage and copying
   - Optional multi-threading support
   - Report processing speed and resource usage

5. **Output Options**
   - Multiple output formats (summary, JSON, detailed)
   - Configurable output destination
   - Machine-readable options for integration


## Immediate Next-Up
- [ ] Implement CLI parsing
- [ ] Read up on hashing API in zig
- [ ] Start basic impl

### Done
- [x] Clean up build.zig


## Open Questions
- file are read as bytes, I imagine?
- do we need to do anything special for binary formats?
- we compute hashes for all the chunks, but how do you compute the hash for the entire thing if you can't store it all in memory. probably worthwhile to read up on how the hashing functions actually work
- do we want to handle piping stuff in?

## Notes

Naive implementation:

- allocate a buffer with length N/chunk_size, where N is the file size in bytes. Each elementin the buffer will contain the hash for the n-th chunk
- read file from disk
  - need to support big files, so need to stream rather than load whole file into memory
  - how much to load at once? there is likely a tradeoff between too big, and it is wasteful because computing the hash is CPU bound while reading the IO bound
  - FUTURE: Can we async this?
- read chunk_size bytes, compute hash, store in buffer, increment index
- output





