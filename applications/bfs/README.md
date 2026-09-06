# BFS

This evaluation compares the paper's locality-oblivious baseline, aware dispatch, and swapped negative control. The CSV reports `aware_speedup` and `swapped_speedup` from median GPU workflow time, both relative to the baseline.

## Run

```bash
./run.sh --preset quick
./run.sh --preset paper --device auto --output-dir out/paper
```

`quick` runs one 65,536-vertex correctness point. `paper` runs the eight unique Table 10 cases on a 2,000,000-vertex graph: degrees 16/32/64/128, source counts 512/4,096/16,384, and traversal depths 2/4/8. The degree-64, source-4,096, depth-4 default is emitted once, and locality is fixed at 95%. Generated files go below `out/` unless `--output-dir` is supplied. See [../README.md](../README.md) or `./run.sh --help` for common options.
