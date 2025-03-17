# Content-Defined Chunking (CDC) Notes

Implementation notes for adding CDC to ZChunk.

## What & Why

CDC chunks files based on content patterns rather than fixed sizes. This solves the "boundary shift problem" mentioned in the README:

> While the current fixed-size chunking has limitations for real-world file synchronization (adding a byte at the beginning shifts all chunks), it demonstrates core concepts that could be expanded with content-defined chunking.

Key benefits:
- Only chunks near modified content change
- Efficient syncing (transfer only modified chunks)
- Better deduplication

## How It Works

1. Sliding window moves through file byte-by-byte
2. Calculate rolling hash at each position
3. Create chunk boundary when hash meets criteria (hash % divisor == target)

That's it. This creates boundaries based on content patterns.

## Algorithm Options

### Rabin-Karp

The classic implementation. For window w[0...n]:

```
H = w[0]*a^(n-1) + w[1]*a^(n-2) + ... + w[n]*a^0
```

Where `a` is a prime base (typically 31 or 101).

When sliding window forward:
```
H_new = (H_old - w_oldest * a^(n-1)) * a + w_new
```

Problem: Computationally expensive due to multiplications.

### Gear Hashing

Faster alternative using bit operations:

```
H_new = (H_old << 1) ^ w_new ^ FINGERPRINT_TABLE[w_oldest]
```

3-10x faster than Rabin-Karp with similar properties.

### FastCDC

Modern optimized approach:
- Uses gear-based hash
- Two-phase chunking approach
- Normalized chunking with rolling hash mask

Implementation sketch:
```
mask_s = (1 << bits_s) - 1  // Normal cutpoint mask
mask_l = (1 << bits_l) - 1  // Large cutpoint mask

for each byte:
  fp = Gear_Hash(window)

  if chunk_size >= max_size:
    cut here (forced boundary)
  else if chunk_size >= min_size:
    if (fp & mask_s) == 0:  // Normal cutpoint
      cut here
  else:
    // Wait until minimum chunk size
    continue
```

## Integration with Merkle Trees

Fits perfectly with ZChunk's existing Merkle tree implementation:

1. CDC creates content-based chunks
2. SHA-256 hash each chunk (separate from rolling hash)
3. Build Merkle tree from chunk hashes
4. Use for verification/comparison

The rolling hash is *only* used for finding chunk boundaries, not for the final hash values.

## Implementation Considerations

### Chunk Size Control
- **Average**: Controlled by divisor/mask (bigger divisor = bigger chunks)
- **Min**: Prevents tiny chunks (typically 2-8KB)
- **Max**: Forces boundaries (typically 64KB-1MB)

### Optimization
- **Window Size**: 32-64 bytes typical
- **Algorithm**: Gear-based hashes > Rabin-Karp
- **Parallelization**: Process separate regions concurrently

## CLI Interface for ZChunk

Proposed update:

```
zchunk [options] <filename>...

Options:
  --chunking=<strategy>    Chunking strategy (fixed, cdc) [default: fixed]
  --chunk-size=<size>      Size of each chunk in KB (for fixed chunking) [default: 1024]
  --min-chunk=<size>       Minimum chunk size in KB (for CDC) [default: 16]
  --max-chunk=<size>       Maximum chunk size in KB (for CDC) [default: 4096]
  --avg-chunk=<size>       Target average chunk size in KB (for CDC) [default: 1024]
  --hash=<algorithm>       Hash algorithm (sha256, sha1, xxhash) [default: sha256]
  --mode=<mode>            Output mode (summary, json, chunks) [default: summary]
  --output=<file>          Write output to file instead of stdout
  --threads=<n>            Number of processing threads [default: 1]
  --help                   Show this help information
```

This separates chunking strategy from hashing algorithm.

## Performance Impact

CDC adds computational overhead:
- Rolling hash at each byte position
- Boundary checking logic
- Chunk metadata management

But this is offset by benefits for sync/deduplication use cases.

## References

1. [Gear Hashing for Content-Defined Chunking](https://joshleeb.com/posts/gear-hashing.html)
2. [Significant Bits | Iota: Rolling Hashes and FastCDC](https://sigpod.dev/1)
3. [Intro to Content-Defined Chunking](https://joshleeb.com/posts/content-defined-chunking.html)
4. [Improving on Gear Hashing with FastCDC](https://joshleeb.com/posts/fastcdc.html)
5. [A Look at RapidCDC and QuickCDC](https://joshleeb.com/posts/quickcdc-rapidcdc.html)
