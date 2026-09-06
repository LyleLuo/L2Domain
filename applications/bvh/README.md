# BVH

This evaluation compares the paper's regular locality-oblivious baseline with aware, adaptive, and swapped traversal. The CSV reports `speedup_vs_regular` from the summed GPU phase time. This preserves the Table 9 measurement: traversal, refit, and GPU worklist rebuild times are summed, while host gaps and device-to-host worklist-count copies are excluded. The CSV records the actual regular and side launch counts together with each Green Context's SM count.

## Run

```bash
./run.sh --preset quick
./run.sh --preset paper --device auto --output-dir out/paper
```

`quick` runs one 65,536-node correctness point. `paper` runs the eight unique Table 9 cases: traversal depths 16/32/64, ray-worklist skews 50/65/80/95%, and matched node/ray sizes of 1,048,576, 2,097,152, and 4,194,304. The default point is emitted once. Generated files go below `out/` unless `--output-dir` is supplied. See [../README.md](../README.md) or `./run.sh --help` for common options.
