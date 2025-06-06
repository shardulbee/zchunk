## PLAN

## NOW

- [ ] implement CDC (content-defined chunking) with Rabin fingerprinting
  - the general idea is to compute a hash over a sliding window a key thing for performance is to "subtract" the first byte from the window and "add" the next byte when the window moves.
  - a key question here is how to implement the rolling hash alongside the streaming read the current implementation reads a chunk into memory then compute the hash but now we don't know where the chunk boundary is to begin with.
  - another key thing to note is that the size of the window is not the same thing as the chunk size. the window provides a sliding window of bytes to compute the hash over, but the chunk is the data between the last boundary and the current boundary.
  - naively, we need to read at least WINDOW_SIZE bytes to compute the hash.
  - [Restic uses a window size of 64 bytes](https://restic.net/blog/2015-09-12/restic-foundation1-cdc/), so let's start with that

## TODO

- [ ] refactor to only use a buffered reader where needed

## DONE

## QUESTIONS

### DONE

- [x] look into content-defined chunking for stuff like syncing
- do we want to handle piping data in?
  - nah, doesn't really help with learning systems/perf really?
- [x] implement writing to file
- [x] implement json mode
- [x] implement summary mode and chunks mode
- for data backup/sync purposes, what if you add 1 byte to the beginning of the file. won't that cause all hashes to be different, meaning you have to probably sync everything?
  - yes, this is why [content-defined chunking](https://joshleeb.com/posts/content-defined-chunking.html) exists
- [x] write test to compute hash
  - randly perturb a chunk and make sure the hash is different
- [x] compute an overall hash
  - this is the merkle root. Building the merkle tree takes around ~8ms for the 2GB file
- [x] make single thread more streamlined
- My original conclusion that the xeon server was more performant because of better processor speed was wrong. with the recent changes to the implementation (removing allocations in a loop), I now get closer to the napkin math numbers even on my personal laptop, and it turns out that it is more performant on my local machine (single threaded and multithreaded) than the xeon machine. Asking claude, M3 max is stronker than E-2388G xeon processor.
- [x] special case single-threaded to just use a loop instead of threadpool
  - avoid the overhead. currently we open a fd every time we read a chunk which is unnecessary and also we don't read the data sequentially
- [x] support multi-threading
  - hell yeah, goes from 1.2sec singlethreaded for 2.1gb to 223ms.
  - dropped to 113ms if we pre-allocate the chunk buffer for all threads instead of allocating in each thread
- [x] changing the implementation back to not read the whole file into memory
- [x] read the whole file into memory and see if that helps with performance.
  - On my m3 max, the perf did not change at all. i.e. ~45 secs to hash my 2gb file even when everything was in memory
  - I wasn't sure why, So I spun up a huge box on vultr and it went faster. So the processor definitely matters even if single threaded. Because processors operate at different frequencies and computing a hash is CPU bound, duh. But still didn't line up with napkin math
  - Then I realized that I was compiling in debug mode. changing this to `--release=fast` took it down to 6sec.
  - On the xeon box, hashing the file takes ~1.6 sec which is about 1.25 GB/s which lines up with [this comment](https://github.com/sirupsen/napkin-math/pull/32#issue-2501810610)
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
