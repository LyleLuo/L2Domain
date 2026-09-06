# Dependent hash probe

This application runs the seven Table 8 configurations with the persistent baseline, Aware routing, and Swapped routing. One process builds each table, query stream, and pair of worklists once, then measures all three configurations on that identical input.

## Run

```bash
./run.sh --preset quick
./run.sh --preset paper --device auto --output-dir out/paper
```

`quick` runs a reduced correctness case. `paper` runs the seven Table 8 points with K=32 million keys, Q=8 million queries, and one-factor variations of lookup depth, table capacity, and hot-query percentage. In the CSV, `C` is the physical table capacity, `effective_load` is `100K/C`, `L` is the dependent lookup depth, `h` is the percentage of hot queries, and `H` is the number of hot keys per GPU domain. Both speedups are relative to the persistent baseline. Generated files go below `out/` unless `--output-dir` is supplied. See [../README.md](../README.md) or `./run.sh --help` for common options.
