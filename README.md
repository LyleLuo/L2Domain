# L2Domian

L2Domian is the artifact for the MICRO 2026 paper *Same Cache, Different Latency: Understanding and Exploiting L2 Domain Locality in GPUs*. This repository reproduces GPU architecture Figures 1–5 and the seven application evaluations in the paper. The architecture characterization uses XDom-PChase to expose cross-domain access behavior, MaskProbe to recover locality-domain address bits, and DomainScan to infer chip-wide SM-to-domain affinity.

## Layout

```text
architecture/          Figure 1–5 measurement and plotting
applications/          ReLU, Histogram, GEMV, Batched Memcpy, Hash, BVH, and BFS
scripts/               Shared GPU selection and runner helpers
utility/               Probe primitives and SideGPU support used by the experiments
thirdparty/cutlass/     Pinned CUTLASS submodule used by the ReLU comparison
```



## Requirements

Full reproduction requires:

- Linux with `bash`, `flock`, and `nvidia-smi`
- CUDA Toolkit 13.2 or newer
- Python 3.11 or newer

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
git submodule update --init thirdparty/cutlass
```

Use any newer Python executable in place of `python3.11` when needed.

The CUTLASS submodule is never downloaded automatically.

## Run

```bash
./architecture/run_figure_01.sh --device auto
./applications/bfs/run.sh --device auto --preset quick
./applications/bfs/run.sh --preset paper --dry-run
```

Architecture runs write generated data and plots below `architecture/out/figure_0N/` by default; the repository keeps one H100 CSV and PNG for each architecture figure in `architecture/data/` and `architecture/figures/`. Application runs write CSVs, binaries, and metadata below the application's ignored `out/` directory, and pre-generated application results are not kept in the repository.

Use an application's default `quick` preset for its functional and correctness check, and use `--preset paper` only for the complete sweep. See [architecture/README.md](architecture/README.md) and [applications/README.md](applications/README.md) for the paper parameters.

## Paper

Please refer to this paper for more details.

```bibtex
@inproceedings{luo2026samecache,
  title     = {Same Cache, Different Latency: Understanding and Exploiting L2 Domain Locality in GPUs},
  author    = {Luo, Weile and Ma, Yibin and Chen, Yuhan and Fan, Ruibo and Liu, Hongyuan and Wang, Qiang and Chu, Xiaowen},
  booktitle = {Proceedings of the 59th IEEE/ACM International Symposium on Microarchitecture (MICRO)},
  year      = {2026}
}
```

