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
- [ ] Store hash results in an arraylist so that we can either output or write to file
- [ ] Only print at end, after computing hashes
- [ ] Compute overall hash
- [ ] Actually understand if `final` is doing the right thing
- [ ] Figure out if I can use tagged union for hashes? prob not because not comptime
- [ ] (stretch?) Implement xxhash


### Side-quests
- [ ] have a way to quickly add stuff to a "later" checklist so that I can forget about it
- [ ] add codecompanion and copilot?
- [ ] Figure out how to fix zig autofmt to not have such long lines?

### Done
- [x] unrelated: syntax highlighting for TODO
- [x] Start basic impl
- [x] Read up on hashing API in Zig
- [x] Handle TODOs
- [x] Implement CLI parsing
- [x] Clean up build.zig


## Open Questions
- do we want to handle piping data in?
- for data backup/sync purposes, what if you add 1 byte to the beginning of the file. won't that cause all hashes to be different, meaning you have to probably sync everything?
