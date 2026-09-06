# Batched memcpy

This directory reproduces Table 7. It compares the `cudaMemcpyBatchAsync` library baseline with the same CUDA copy kernel launched on the matching domain (`aware`) and the opposite domain (`swapped`). Both source and destination allocations are placed on domain 0.

## Run

```bash
./run.sh --preset quick
./run.sh --preset paper --device auto --output-dir out/paper
```

`quick` copies 64 chunks as a functional check. `paper` uses the five Table 7 batch sizes of 64, 256, 1,024, 4,096, and 16,384 with 4 KiB chunks and a fixed 32-CTA copy kernel. The CSV reports all three times and the two speedups over `cudaMemcpyBatchAsync`; every mode is checked against the source data. Generated files go below `out/` unless `--output-dir` is supplied. See [../README.md](../README.md) or `./run.sh --help` for common options.
