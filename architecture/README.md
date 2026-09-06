# GPU architecture characterization

Each public script builds the shared CUDA probe, measures one paper experiment, saves its CSV, and renders its PNG. 


| Figure | Entry point          | Measurement                                                                       |
| ------ | -------------------- | --------------------------------------------------------------------------------- |
| 1      | `./run_figure_01.sh` | Random-stride pointer-chasing latency over 1 KiB–1 GiB working sets               |
| 2      | `./run_figure_02.sh` | XDom-PChase per-element latency with four traces over 2,048 elements              |
| 3      | `./run_figure_03.sh` | DomainScan heatmap for bits 7–31 over real two-SM TPC resources                   |
| 4      | `./run_figure_04.sh` | Bidirectional MaskProbe for bits 5–23                                             |
| 5      | `./run_figure_05.sh` | Dense 128-byte XDom-PChase offset sweep over 0–256 KiB with active-bit boundaries |


CUDA 13.1 or newer is required for Figure 3's Green Context TPC enumeration. Figures 2, 4, and 5 accept `--peer auto|A,B`; auto mode measures the SM-to-domain map and chooses one SM from each locality domain, while an explicit pair runs without calibration. Figure 4 keeps bits 21–23 in the raw curves but does not classify them as stable intra-page mask bits. Figure 5 may begin in either the near or far phase because physical-page placement can invert the trace; the transition relation is the reproduced result.

```bash
./run_figure_01.sh --device auto
./run_figure_04.sh --device auto --peer auto
./run_figure_05.sh --device auto --peer auto --output-dir /tmp/sidedgpu/figure05
./run_figure_03.sh --dry-run --arch sm_90a
```

By default, Figure N writes generated files below `architecture/out/figure_0N/` and leaves `architecture/data/` and `architecture/figures/` unchanged. Any `--output-dir DIR` places `data/`, `figures/`, and build artifacts below `DIR`. Passing `--output-dir .` explicitly updates that Figure's repository CSV and PNG. 