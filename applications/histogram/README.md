# Histogram

This evaluation adapts the two-stage, 256-bin organization of the [NVIDIA CUDA histogram sample](https://github.com/NVIDIA/cuda-samples/tree/master/cpp/2_Concepts_and_Techniques/histogram). It compares a conventional baseline with Green Context and persistent-kernel execution, each under aware and swapped domain assignment.

## Run

```bash
./run.sh --preset quick
./run.sh --preset paper --device auto --output-dir out/paper
```

`quick` runs one 32 MiB correctness point. `paper` measures 32, 64, 128, and 512 MiB inputs with 128-thread blocks, one resident block per SM, ten repetitions, and a cold cache. The reported times cover the first stage, in which CTAs build partial histograms. The common second-stage merge is excluded from timing but is executed before every final histogram is checked. Generated files go below `out/` unless `--output-dir` is supplied. See [../README.md](../README.md) or `./run.sh --help` for common options.
