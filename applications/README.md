# Application evaluations

The seven directories reproduce the paper's ReLU, Histogram, GEMV, Batched Memcpy, dependent Hash Probe, BVH, and BFS evaluations. Each application contains one public `run.sh`, its implementation in `src/`, and a short README describing the measured configurations.

```bash
./applications/<name>/run.sh --device auto --preset quick
./applications/<name>/run.sh --device auto --preset paper --output-dir out/paper
./applications/<name>/run.sh --preset paper --dry-run
```

`quick` builds and runs a small correctness case; `paper` runs the complete paper sweep. Common options are `--device`, `--preset`, `--arch`, `--output-dir`, `--build-only`, `--dry-run`, `--wait-timeout`, and `--poll-interval`. Generated CSVs, binaries, and metadata go to the application's ignored `out/` directory by default. The repository does not keep pre-generated application measurements.
