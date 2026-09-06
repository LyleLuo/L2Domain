# ReLU

This application reproduces Figure 9 and its related occupancy and CUTLASS comparisons. The Figure 9 and occupancy experiments use 128-thread blocks. The default `main` suite compares the locality-oblivious baseline with persistent-kernel and Green Context execution under aware and swapped domain assignment.

## Run

```bash
./run.sh --preset quick
./run.sh --preset paper --device auto
./run.sh --preset paper --suite occupancy --device auto
./run.sh --preset paper --suite cutlass --device auto
```

## Suites

- `main` is the default. The paper preset measures 16, 64, 256, and 1,024 MiB inputs for both execution mechanisms and both domain assignments, for 16 cases in total.
- `occupancy` measures the 1 GiB aware Green Context case with 1, 2, 4, and 8 resident blocks per SM.
- `cutlass` compares the SideGPU persistent predicate implementation with an elementwise CUDA kernel that applies CUTLASS `LinearCombinationRelu`; the paper preset measures 64, 128, 256, 512, 1,024, and 2,048 MiB inputs with 1,024 threads on A100/H100 and 128 threads on B200.
- `all` runs all three suites.

`quick` uses reduced inputs and repetitions for a functional check. Every case validates its output, and generated CSVs, binaries, and metadata go below `out/` unless `--output-dir` is supplied. The CUTLASS suite uses the pinned `thirdparty/cutlass` submodule; initialize it with `git submodule update --init thirdparty/cutlass` before running that suite. See [../README.md](../README.md) or `./run.sh --help` for all options.
