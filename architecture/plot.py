#!/usr/bin/env python3
"""Render the five architecture figures from their self-describing CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
    return rows


def require_columns(rows: list[dict[str, str]], columns: Iterable[str]) -> None:
    missing = set(columns).difference(rows[0])
    if missing:
        raise ValueError(f"CSV is missing columns: {', '.join(sorted(missing))}")


def configure_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "matplotlib is required to render figures; install it in the selected Python environment"
        ) from error

    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 120,
            "savefig.dpi": 180,
        }
    )
    return plt


def plot_figure_1(plt, rows: list[dict[str, str]]):
    require_columns(rows, ("device_name", "region_bytes", "latency_cycles"))
    devices = {row["device_name"] for row in rows}
    if len(devices) != 1:
        raise ValueError("Figure 1 expects measurements from exactly one GPU")
    font_size = 60
    fig, ax = plt.subplots(figsize=(32, 10))
    x = [int(row["region_bytes"]) for row in rows]
    y = [float(row["latency_cycles"]) for row in rows]
    label = next(iter(devices)).removeprefix("NVIDIA ")
    ax.plot(x, y, linewidth=5, color="tab:red", label=label)
    ax.set_xscale("log", base=2)
    ticks = [1 << 10, 1 << 14, 1 << 18, 1 << 24, 1 << 28]
    labels = ["1 KiB", "16 KiB", "256 KiB", "16 MiB", "256 MiB"]
    ax.set_xticks(ticks, labels)
    ax.set_xlabel("Chain Data Volume", fontsize=font_size, labelpad=20)
    ax.set_ylabel("Latency (cycles)", fontsize=font_size, labelpad=20)
    ax.tick_params(labelsize=font_size)
    maximum = max(float(row["latency_cycles"]) for row in rows)
    ax.set_ylim(0, max(950, int((maximum + 49) // 50) * 50))
    legend = ax.legend(fontsize=font_size)
    for line in legend.get_lines():
        line.set_linewidth(5)
    fig.tight_layout()
    return fig


def plot_figure_2(plt, rows: list[dict[str, str]]):
    columns = (
        "index",
        "leader_first_cycles",
        "leader_second_cycles",
        "follower_first_cycles",
        "follower_second_cycles",
    )
    require_columns(rows, columns)
    x = [int(row["index"]) for row in rows]
    series = (
        ("leader_first_cycles", "Leader's first access", "#1f77b4"),
        ("leader_second_cycles", "Leader's second access", "#d62728"),
        ("follower_first_cycles", "Follower's first access", "#ff7f0e"),
        ("follower_second_cycles", "Follower's second access", "#2ca02c"),
    )
    font_size = 34
    fig, ax = plt.subplots(figsize=(18, 10))
    for column, label, color in series:
        ax.plot(x, [float(row[column]) for row in rows], linewidth=3, label=label, color=color)
    ax.set_xlim(0, len(x))
    ax.set_xticks(range(0, len(x) + 1, 256))
    ax.set_xlabel("Index offset (stride 128 bytes)", fontsize=font_size, labelpad=20)
    ax.set_ylabel("Latency (cycles)", fontsize=font_size, labelpad=20)
    ax.tick_params(labelsize=font_size)
    legend = ax.legend(fontsize=font_size, loc="upper left")
    for line in legend.get_lines():
        line.set_linewidth(3)
    fig.tight_layout()
    return fig


def plot_figure_3(plt, rows: list[dict[str, str]]):
    require_columns(rows, ("tpc_id", "follower_sm", "address_bit", "latency_cycles"))
    tpcs = sorted({int(row["tpc_id"]) for row in rows})
    bits = sorted({int(row["address_bit"]) for row in rows})
    values = {
        (int(row["tpc_id"]), int(row["address_bit"])): float(row["latency_cycles"])
        for row in rows
    }
    matrix = [[values[(tpc, bit)] for bit in bits] for tpc in tpcs]

    font_size = 52
    fig, ax = plt.subplots(figsize=(max(12, len(bits) * 0.9), max(10, len(tpcs) * 0.5)))
    image = ax.imshow(matrix, cmap="viridis", aspect="auto", interpolation="nearest")
    x_step = max(1, len(bits) // 10)
    y_step = 2
    ax.set_xticks(range(0, len(bits), x_step), [bits[i] for i in range(0, len(bits), x_step)])
    ax.set_yticks(range(0, len(tpcs), y_step), [tpcs[i] for i in range(0, len(tpcs), y_step)])
    ax.set_xlabel("Index Offset (log2(bytes))", fontsize=font_size, labelpad=20)
    ax.set_ylabel("TPC ID (SM ID / 2)", fontsize=font_size, labelpad=20)
    ax.tick_params(axis="both", labelsize=font_size)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.5, aspect=30)
    colorbar.ax.tick_params(labelsize=font_size)
    colorbar.set_label("Latency (cycles)", fontsize=font_size, labelpad=20)
    fig.tight_layout()
    return fig


def plot_figure_4(plt, rows: list[dict[str, str]]):
    columns = (
        "sm_a",
        "sm_b",
        "address_bit",
        "sm_a_follower_cycles",
        "sm_b_follower_cycles",
    )
    require_columns(rows, columns)
    bits = [int(row["address_bit"]) for row in rows]
    sm_a = rows[0]["sm_a"]
    sm_b = rows[0]["sm_b"]
    a_latency = [float(row["sm_a_follower_cycles"]) for row in rows]
    b_latency = [float(row["sm_b_follower_cycles"]) for row in rows]

    font_size = 36
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(bits, a_latency, marker="o", markersize=13, linewidth=3, label=f"SM {sm_a} Access")
    ax.plot(bits, b_latency, marker="s", markersize=13, linewidth=3, label=f"SM {sm_b} Access")
    ax.fill_between(bits, b_latency, a_latency, alpha=0.2, color="gray")
    ax.set_xticks(bits)
    ax.set_xlabel("Index offset (log2(bytes))", fontsize=font_size, labelpad=20)
    ax.set_ylabel("Latency (cycles)", fontsize=font_size, labelpad=20)
    ax.tick_params(labelsize=font_size)
    legend = ax.legend(fontsize=font_size)
    for line in legend.get_lines():
        line.set_linewidth(3)
    fig.tight_layout()
    return fig


def plot_figure_5(plt, rows: list[dict[str, str]]):
    columns = (
        "leader_sm",
        "follower_sm",
        "round_id",
        "index",
        "offset_bytes",
        "latency_cycles",
        "boundary_bit",
        "boundary_kind",
    )
    require_columns(rows, columns)
    font_size = 34
    round_ids = sorted({int(row["round_id"]) for row in rows})
    if round_ids != [0, 1]:
        raise ValueError("Figure 5 requires exactly two pointer-chase rounds")

    fig, axes_grid = plt.subplots(2, 1, figsize=(28, 6), sharey=True, squeeze=False)
    axes = [axes_grid[0][0], axes_grid[1][0]]
    all_latencies = [float(row["latency_cycles"]) for row in rows]
    for round_id, ax in zip(round_ids, axes):
        selected = [row for row in rows if int(row["round_id"]) == round_id]
        selected.sort(key=lambda row: int(row["index"]))
        x = [int(row["offset_bytes"]) / 1024.0 for row in selected]
        y = [float(row["latency_cycles"]) for row in selected]
        ax.plot(x, y, color="tab:blue", linewidth=3)
        start = 128 * round_id
        ax.set_xlim(start, start + 128)
        ticks = list(range(start, start + 128, 8))
        ax.set_xticks(ticks, [str(tick) for tick in ticks])
        ax.tick_params(labelsize=font_size, labelbottom=True)

    boundaries = sorted(
        (
            int(row["boundary_bit"]),
            row["boundary_kind"],
            int(row["offset_bytes"]) / 1024.0,
        )
        for row in rows
        if row["boundary_bit"]
    )
    active_number = 0
    y_min = min(all_latencies)
    y_max = max(all_latencies)
    label_y = y_min + 0.28 * (y_max - y_min)
    for bit, kind, position in boundaries:
        ax = axes[0] if position <= 128 else axes[1]
        if kind == "active":
            active_number += 1
            ax.axvline(position, color="black", linewidth=2)
            text_x = min(position, 126.5)
            ax.text(
                text_x,
                label_y,
                str(active_number),
                ha="center",
                va="center",
                color="white",
                fontsize=font_size * 0.48,
                bbox={"boxstyle": "circle,pad=0.14", "facecolor": "black", "edgecolor": "none"},
            )
        elif kind == "inactive":
            ax.axvline(position, color="red", linestyle="--", linewidth=2)

    top_axis = axes[0].twiny()
    top_axis.set_xlim(axes[0].get_xlim())
    power_bits = list(range(12, 18))
    top_axis.set_xticks([2**bit / 1024 for bit in power_bits])
    top_axis.set_xticklabels([rf"$2^{{{bit}}}$" for bit in power_bits])
    top_axis.tick_params(axis="x", labelsize=font_size * 0.62, length=0, pad=2)
    fig.supylabel("Latency (cycles)", fontsize=font_size)
    axes[-1].set_xlabel("Index Offset (KiB)", fontsize=font_size, labelpad=20)
    fig.tight_layout(rect=(0.03, 0, 1, 1))
    return fig


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    plt = configure_matplotlib()
    plotters = {
        1: plot_figure_1,
        2: plot_figure_2,
        3: plot_figure_3,
        4: plot_figure_4,
        5: plot_figure_5,
    }
    figure = plotters[args.figure](plt, rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)
    plt.close(figure)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
