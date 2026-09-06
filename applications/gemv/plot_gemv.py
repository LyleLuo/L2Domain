#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GPU_CSVS = [
    ("A100 PCIe", "final_gemv_A100_PCIe.csv"),
    ("H100 SXM", "final_gemv_H100_SXM.csv"),
    ("B200", "final_gemv_B200.csv"),
]


def load_csv(path: Path):
    with path.open() as f:
        return [
            {
                k: (
                    int(v)
                    if k in {"M", "K", "lanes_per_row", "warps_per_block"}
                    else v
                    if k == "pass"
                    else float(v)
                )
                for k, v in row.items()
            }
            for row in csv.DictReader(f)
        ]


def validate_rows(path: Path, rows, expected_config):
    if not rows:
        raise ValueError(f"{path}: CSV contains no data rows")
    for row_number, row in enumerate(rows, start=2):
        if row["K"] != 1024:
            raise ValueError(f"{path}:{row_number}: expected K=1024, got {row['K']}")
        if row["pass"] != "PASS":
            raise ValueError(f"{path}:{row_number}: refusing to plot pass={row['pass']}")
        config = (row["lanes_per_row"], row["warps_per_block"])
        if expected_config is None:
            expected_config = config
        elif config != expected_config:
            raise ValueError(
                f"{path}:{row_number}: mixed GEMV configurations: "
                f"expected {expected_config[0]} lanes/{expected_config[1]} warps, "
                f"got {config[0]}/{config[1]}"
            )
    return expected_config


def plot_panel(ax, rows, title):
    series = [
        ("aware_ms", "o", "Aware", "-", 3.4, "#1f77b4"),
        ("baseline_ms", "s", "Baseline", "-", 2.8, "#ff7f0e"),
        ("swapped_ms", "^", "Swapped", "-", 2.8, "#2ca02c"),
        ("cublas_ms", "D", "cuBLAS", "--", 2.5, "#111111"),
    ]
    for key, marker, label, style, linewidth, color in series:
        ax.plot(
            [row["M"] for row in rows],
            [row[key] * 1000.0 for row in rows],
            marker=marker,
            markersize=10,
            label=label,
            linestyle=style,
            linewidth=linewidth,
            color=color,
        )

    ax.set_title(title, pad=10, fontsize=28)
    x_values = [row["M"] for row in rows]
    tick_candidates = [2048, 4096, 6144, 8192, 10240]
    ticks = [tick for tick in tick_candidates if min(x_values) <= tick <= max(x_values)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(tick) for tick in ticks])
    for label in ax.get_xticklabels():
        label.set_rotation(35)
        label.set_ha("center")
    ax.tick_params(axis="x", labelsize=30, pad=4)
    ax.tick_params(axis="y", labelsize=30, pad=4)
    ax.grid(True, alpha=0.3)

    values = [row[key] * 1000.0 for key, *_ in series for row in rows]
    padding = (max(values) - min(values)) * 0.10 if max(values) > min(values) else 0.001
    ax.set_ylim(max(0, min(values) - padding), max(values) + padding)


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Plot final GEMV CSVs.")
    parser.add_argument("--data-dir", default=script_dir / "out", help="Directory containing one or more final_gemv_*.csv files")
    parser.add_argument("--output-dir", default=script_dir / "out", help="Directory for generated PNG and PDF files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    plt.rcParams.update(
        {
            "font.size": 28,
            "axes.titlesize": 28,
            "axes.labelsize": 34,
            "legend.fontsize": 30,
        }
    )

    loaded = []
    expected_config = None
    candidates = [(title, data_dir / filename) for title, filename in GPU_CSVS]
    known_names = {filename for _, filename in GPU_CSVS}
    for candidate in sorted(data_dir.glob("final_gemv_*.csv")):
        if candidate.name in known_names:
            continue
        label = candidate.stem.removeprefix("final_gemv_").replace("_", " ")
        candidates.append((label, candidate))

    for title, candidate in candidates:
        if not candidate.exists():
            continue
        rows = load_csv(candidate)
        expected_config = validate_rows(candidate, rows, expected_config)
        loaded.append((title, rows))

    if not loaded:
        parser.error(f"no GEMV CSVs found in {data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes_grid = plt.subplots(1, len(loaded), figsize=(5.6 * len(loaded), 7.4), squeeze=False, sharey=False)
    axes = axes_grid[0]
    for ax, (title, rows) in zip(axes, loaded):
        plot_panel(ax, rows, title)

    legend_ax = next((ax for ax in axes if ax.lines), axes[0])
    fig.legend(
        *legend_ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        fontsize=30,
        columnspacing=0.7,
        handlelength=1.5,
    )
    fig.supxlabel("M (rows), K=1024", fontsize=34, y=-0.015)
    fig.supylabel("Time (us)", fontsize=34, x=0.005)
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.20, top=0.78, wspace=0.20)

    png = output_dir / "figure_11.png"
    pdf = output_dir / "figure_11.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png} and {pdf}")


if __name__ == "__main__":
    main()
