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


## Currently Working On
- [ ] special case single-threaded to just use a loop instead of threadpool
  - avoid the overhead. currently we open a fd every time we read a chunk which is unnecessary and also we don't read the data sequentially

## Immediate Next-Up
- [ ] write tests
- [ ] compute an overall hash
  - use a merkle tree?
- [ ] implement writing to file
- [ ] implement implement json mode
- [ ] implement summary mode
- [ ] log file size and time to hash so we can output throughput
- [ ] compute an overall hash always
- [ ] (stretch?) Implement xxhash

### Side-quests
- [ ] Figure out how to fix zig autofmt to not have such long lines?

### Done
- [x] support multi-threading
  - hell yeah, goes from 1.2sec singlethreaded for 2.1gb to 223ms.
  - dropped to 113ms if we pre-allocate the chunk buffer for all threads instead of allocating in each thread
- [x] changing the implementation back to not read the whole file into memory
- [x] read the whole file into memory and see if that helps with performance.
  - initially, this did not work. I wasn't sure why. So I spun up a huge box on vultr and it was faster. So definitely the processor power matters even if single threaded.
  - on my m3 max, the time did not change at all. i.e. ~45 secs to hash my 2gb file even when everything was in memory
  - then i realized that i was compiling in debug mode. changing this to --release=fast took it down to 6sec.
  - I think this must be the bottleneck then? Would be curious to learn if there are better ideas
  - on the xeon box, hashing the file takes ~1.6 sec which is about 1.25 GB/s which lines up with [this comment](https://github.com/sirupsen/napkin-math/pull/32#issue-2501810610)
- [x] figure out why single-threaded hash performance does not line up with [napkin math](https://github.com/sirupsen/napkin-math)
   - hashing a 2GB file takes me 44 secs, when napkin math suggests that throughput is 2GiB/s
   - maybe has to do with sequential SSD read, but that as well is 4GiB/s
   - experiment with (1) reading larger sizes into memory before hashing chunks and (2) reading whole file into memory
     - reading file differently didn't change performance. this makes sense because the hashing bandwidth is the bottleneck, not reading from a (fast) disk.
- [x] actually read the file in specified KB instad of bytes
- [x] unrelated: syntax highlighting for TODO
- [x] Start basic impl
- [x] Read up on hashing API in Zig
- [x] Handle TODOs
- [x] Implement CLI parsing
- [x] Clean up build.zig


## Open Questions
- do we want to handle piping data in?
- for data backup/sync purposes, what if you add 1 byte to the beginning of the file. won't that cause all hashes to be different, meaning you have to probably sync everything?
