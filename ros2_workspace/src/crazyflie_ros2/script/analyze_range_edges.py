# -*- coding: utf-8 -*-
"""
Analyze edges in Crazyflie /range/down logs with a rolling window z-score,
and optionally plot/save figures that highlight detected edges.

Usage examples:
  python analyze_range_edges.py --csv /mnt/data/range_down_log_20251013_130613.csv --plot
  python analyze_range_edges.py --csv /mnt/data/range_down_log_20251013_130613.csv --save-prefix /mnt/data/range_edges
  python analyze_range_edges.py --csv /mnt/data/range_down_log_20251013_130613.csv --window 75 --z-thresh 2.5 --plot --preview 10
  python analyze_range_edges.py --csv range_down_log_20251013_130613.csv --z-col range_m --save-prefix /mnt/data/range_edges --time-axis time

The input CSV is expected to have these columns:
time_ns,time_iso,range_m,height_err_m,delta_m_per_s,zscore_delta,edge_flag
"""

import argparse
import glob
import math
import os
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def compute_rolling_zscore(
    s: pd.Series, window: int, min_periods: Optional[int] = None, eps: float = 1e-9
) -> pd.Series:
    """Compute trailing rolling z-score of a Series."""
    if min_periods is None:
        min_periods = max(3, int(window // 3))
    roll_mean = s.rolling(window=window, min_periods=min_periods).mean()
    roll_std = s.rolling(window=window, min_periods=min_periods).std(ddof=1)
    z = (s - roll_mean) / (roll_std.replace(0.0, eps).fillna(eps))
    return z, roll_mean, roll_std


def make_axes_data(df: pd.DataFrame, use_time: bool, time_col: str, x_label: str):
    if use_time:
        # Convert ISO to pandas datetime; fallback to index if parsing fails
        try:
            x = pd.to_datetime(df[time_col])
            x_label = f"{time_col}"
        except Exception:
            x = df.index
            x_label = "sample"
    else:
        x = df.index
        x_label = "sample"
    return x, x_label


def plot_series_with_edges(x, y, edges_mask, title, ylabel, save_path: Optional[str] = None, show: bool = False):
    # Convert to numpy arrays to avoid pandas Index multidimensional indexing issues
    x = np.asarray(x)
    y = np.asarray(y)
    edges_mask = np.asarray(edges_mask, dtype=bool).ravel()
    # Safety: ensure same length
    n = min(len(x), len(y), len(edges_mask))
    x, y, edges_mask = x[:n], y[:n], edges_mask[:n]

    # Single figure for the series
    fig = plt.figure(figsize=(10, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(x, y, linewidth=1.5)
    # Mark edges as scatter on top
    edge_x = x[edges_mask]
    edge_y = y[edges_mask]
    ax.scatter(edge_x, edge_y, s=20)  # default color/markers
    ax.set_title(title)
    ax.set_xlabel(getattr(x, "name", None) or "x")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show(block=False)


def plot_zscore(x, z, z_thresh, title, save_path: Optional[str] = None, show: bool = False):
    x = np.asarray(x)
    z = np.asarray(z)
    fig = plt.figure(figsize=(10, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(x, z, linewidth=1.5)
    # Threshold guides
    ax.axhline(z_thresh, linestyle='--', linewidth=1.0)
    ax.axhline(-z_thresh, linestyle='--', linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel(getattr(x, "name", None) or "x")
    ax.set_ylabel("rolling z-score")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show(block=False)


def find_latest_range_log():
    """Find the latest range_down_log_*.csv file in the current directory."""
    pattern = "range_down_log_*.csv"
    files = glob.glob(pattern)
    if not files:
        return None
    # Sort by filename (which includes timestamp) to get the latest
    files.sort(reverse=True)
    return files[0]


def main():
    ap = argparse.ArgumentParser(description="Detect edges in Crazyflie range log using rolling z-score, with optional plots.")
    ap.add_argument("--csv", default=None, help="Path to log CSV (default: auto-detect latest range_down_log_*.csv)")
    ap.add_argument("--window", type=int, default=None, help="Rolling window size in rows (default: prompt if not specified)")
    ap.add_argument("--z-thresh", type=float, default=None, help="Edge threshold on |z| (default: prompt if not specified)")
    ap.add_argument(
        "--z-col",
        default="delta_m_per_s",
        choices=["delta_m_per_s", "range_m", "height_err_m"],
        help="Column to compute rolling z-score on (default: delta_m_per_s)",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Output CSV path for annotated results (default: <input>_edges.csv)",
    )
    ap.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Print the first N detected edges to stdout (default: 0 = don't print)",
    )
    ap.add_argument(
        "--center",
        action="store_true",
        help="Use centered rolling window (default: trailing window)",
    )
    ap.add_argument(
        "--save-prefix",
        default=None,
        help="If set, save PNGs with this prefix (e.g., /mnt/data/range_edges).",
    )
    ap.add_argument(
        "--time-axis",
        choices=["index", "time"],
        default="index",
        help="X-axis: 'index' (default) or 'time' (use time_iso).",
    )
    args = ap.parse_args()

    # Auto-detect CSV file if not specified
    if args.csv is None:
        in_path = find_latest_range_log()
        if in_path is None:
            raise SystemExit("No range_down_log_*.csv files found in current directory. Please specify --csv.")
        print(f"Auto-detected latest log file: {in_path}")
    else:
        in_path = args.csv

    if not os.path.exists(in_path):
        raise SystemExit(f"Input file not found: {in_path}")

    # Prompt for window if not specified
    if args.window is None:
        try:
            window_input = input("Enter rolling window size in rows (default 50): ").strip()
            args.window = int(window_input) if window_input else 50
        except ValueError:
            print("Invalid input, using default: 50")
            args.window = 50

    # Prompt for z-thresh if not specified
    if args.z_thresh is None:
        try:
            z_input = input("Enter edge threshold on |z| (default 3.0): ").strip()
            args.z_thresh = float(z_input) if z_input else 3.0
        except ValueError:
            print("Invalid input, using default: 3.0")
            args.z_thresh = 3.0

    df = pd.read_csv(in_path)

    required_cols = [
        "time_ns",
        "time_iso",
        "range_m",
        "height_err_m",
        "delta_m_per_s",
        "zscore_delta",
        "edge_flag",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing required columns: {missing}")

    # Ensure numeric types where needed
    for col in ["range_m", "height_err_m", "delta_m_per_s", "zscore_delta", "edge_flag"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Rolling stats on the chosen column
    target = df[args.z_col]
    if args.center:
        roll_mean = target.rolling(window=args.window, min_periods=max(3, args.window // 3), center=True).mean()
        roll_std = target.rolling(window=args.window, min_periods=max(3, args.window // 3), center=True).std(ddof=1)
        eps = 1e-9
        z = (target - roll_mean) / (roll_std.replace(0.0, eps).fillna(eps))
    else:
        z, roll_mean, roll_std = compute_rolling_zscore(target, args.window)

    df["roll_mean_" + args.z_col] = roll_mean
    df["roll_std_" + args.z_col] = roll_std
    df["zscore_win_" + args.z_col] = z

    # Edge decision
    thresh = float(args.z_thresh)
    df["edge_flag_win"] = (df["zscore_win_" + args.z_col].abs() >= thresh).astype(int)

    # Prepare output path
    if args.output:
        out_path = args.output
    else:
        root, ext = os.path.splitext(in_path)
        out_path = f"{root}_edges.csv"

    df.to_csv(out_path, index=False)

    # Summary
    n_edges = int(df["edge_flag_win"].sum())
    total = len(df)
    first_idx = int(df.index[df["edge_flag_win"] == 1][0]) if n_edges > 0 else -1
    first_time = df.loc[first_idx, "time_iso"] if n_edges > 0 else "N/A"

    print("=== Edge Detection Summary ===")
    print(f"File: {in_path}")
    print(f"Rows: {total}")
    print(f"Window: {args.window}  Z-thresh: {thresh:.3f}  Column: {args.z_col}  Centered: {args.center}")
    print(f"Detected edge rows: {n_edges}")
    print(f"Output: {out_path}")
    if n_edges > 0:
        print(f"First detected @ row {first_idx}  time_iso={first_time}  z={df.loc[first_idx, 'zscore_win_' + args.z_col]:.2f}  "
              f"value={df.loc[first_idx, args.z_col]:.4f}")
        if args.preview > 0:
            print("\n--- First edges ---")
            preview_df = df.loc[df["edge_flag_win"] == 1, ["time_iso", args.z_col, "zscore_win_" + args.z_col]].head(args.preview)
            # Pretty print
            with pd.option_context("display.max_rows", None, "display.width", 120):
                print(preview_df.to_string(index=False))

    # --------- Plotting / Saving Figures ---------
    # Always enable plotting
    do_plot = True
    do_save = bool(args.save_prefix)

    if do_plot or do_save:
        # Choose x-axis
        use_time = args.time_axis == "time"
        x, x_label = None, None
        if use_time:
            try:
                x = pd.to_datetime(df["time_iso"])
                x.name = "time"
            except Exception:
                x = df.index
                x.name = "sample"
        else:
            x = df.index
            x.name = "sample"

        # Figure 1: chosen column with edge markers
        series_title = f"{args.z_col} with detected edges (|z| ≥ {thresh:g})"
        save1 = f"{args.save_prefix}_series.png" if do_save else None
        plot_series_with_edges(x, df[args.z_col], df["edge_flag_win"] == 1, series_title, args.z_col, save_path=save1, show=do_plot)

        # Figure 2: rolling z-score with threshold lines
        z_title = f"Rolling z-score of {args.z_col} (window={args.window})"
        save2 = f"{args.save_prefix}_zscore.png" if do_save else None
        plot_zscore(x, df["zscore_win_" + args.z_col], thresh, z_title, save_path=save2, show=do_plot)

        if do_plot:
            # Keep windows open until closed by user
            plt.show()

        if do_save:
            print(f"Saved figures: {save1}, {save2}")

    # Exit code 0 if success
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
