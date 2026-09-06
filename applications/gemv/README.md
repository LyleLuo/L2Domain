# GEMV

This directory reproduces Figure 11 with a locality-oblivious baseline, SideGPU-aware placement, swapped placement, and cuBLAS. The matrix width is K=1024. The SideGPU kernel assigns one 32-lane warp to each row and uses 16 warps, or 512 threads, per block, so one block covers up to 16 rows.

## Run

```bash
./run.sh --preset quick
./run.sh --preset paper --device auto --output-dir out/paper
```

`quick` measures M=2048. `paper` runs the nine Figure 11 values from M=2048 through M=10240. Each case checks the three SideGPU outputs against a CPU reference and fails if the maximum absolute error exceeds `1e-3`. The runner writes its CSV, metadata, and regenerated plot below `out/` unless `--output-dir` is supplied. If CSVs from several paper GPUs are placed in the same output directory, the plot includes each available GPU. See [../README.md](../README.md) or `./run.sh --help` for common options.
