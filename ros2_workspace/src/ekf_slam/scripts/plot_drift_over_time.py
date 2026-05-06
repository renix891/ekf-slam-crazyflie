#!/usr/bin/env python3
"""Plot belief-vs-truth drift over time, per Run 2 headline bag.

Generates four figures in results/figures/, two per bag:

    drift_per_state_ekf.png  — 4 subplots (dx, dy, dz, dyaw), EKF Run 2
                                only, blue, ~161 s.
    drift_xy_total_ekf.png   — single panel d_xy for EKF Run 2, drift
                                rate in legend.
    drift_per_state_odom.png — 4 subplots (dx, dy, dz, dyaw), Odom Run 2
                                only, red, ~47 s.
    drift_xy_total_odom.png  — single panel d_xy for Odom Run 2, drift
                                rate in legend.

Belief is /ekf_pose, truth is /crazyflie/odom. Each bag has its own time
axis; no overlay between runs. Odom dz is expected to be a flat zero
line because odom_to_pose.py passes z through unchanged.

Usage:
    python3 plot_drift_over_time.py
"""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message


PROJECT_DIR = Path("/home/renix/EKF-SLAM-Autonomous-Crazyflie")
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
HEADLINE_BAGS_DIR = PROJECT_DIR / "analysis" / "headline_bags"
HEADLINE_EKF_BAG  = HEADLINE_BAGS_DIR / "final_ekf_bag_run2"
HEADLINE_ODOM_BAG = HEADLINE_BAGS_DIR / "final_odom_bag_run2"

# Simulator's crazyflie/body link sits 17.425 mm above the model origin
# (see model.sdf line 17). The EKF publishes belief in the body frame, so
# subtract this offset from belief_z before differencing against truth.
BODY_LINK_Z_OFFSET = 0.01743


def quat_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def read_bag(bag_dir):
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag_dir), storage_id="mcap"),
        ConverterOptions("", ""),
    )
    msg_types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}

    bel_t, bel_x, bel_y, bel_z, bel_yaw = [], [], [], [], []
    tru_t, tru_x, tru_y, tru_z, tru_yaw = [], [], [], [], []

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        t = t_ns * 1e-9
        msg = deserialize_message(data, msg_types[topic])
        if topic == "/ekf_pose":
            bel_t.append(t)
            bel_x.append(msg.pose.position.x)
            bel_y.append(msg.pose.position.y)
            bel_z.append(msg.pose.position.z)
            q = msg.pose.orientation
            bel_yaw.append(quat_to_yaw(q.x, q.y, q.z, q.w))
        elif topic == "/crazyflie/odom":
            tru_t.append(t)
            tru_x.append(msg.pose.pose.position.x)
            tru_y.append(msg.pose.pose.position.y)
            tru_z.append(msg.pose.pose.position.z)
            q = msg.pose.pose.orientation
            tru_yaw.append(quat_to_yaw(q.x, q.y, q.z, q.w))

    if not bel_t:
        raise RuntimeError(f"no /ekf_pose messages in {bag_dir}")
    if not tru_t:
        raise RuntimeError(f"no /crazyflie/odom messages in {bag_dir}")

    t0 = bel_t[0]
    return {
        "belief": {
            "t":   np.asarray(bel_t,   dtype=float) - t0,
            "x":   np.asarray(bel_x,   dtype=float),
            "y":   np.asarray(bel_y,   dtype=float),
            "z":   np.asarray(bel_z,   dtype=float),
            "yaw": np.asarray(bel_yaw, dtype=float),
        },
        "truth": {
            "t":   np.asarray(tru_t,   dtype=float) - t0,
            "x":   np.asarray(tru_x,   dtype=float),
            "y":   np.asarray(tru_y,   dtype=float),
            "z":   np.asarray(tru_z,   dtype=float),
            "yaw": np.asarray(tru_yaw, dtype=float),
        },
    }


def compute_drift(belief, truth, z_offset=0.0):
    """Interpolate truth onto belief.t and return signed drift signals.

    All angle math stays in radians here; caller converts to degrees at
    plot time. Yaw is interpolated component-wise via sin/cos to avoid
    ±π wraparound corruption.
    """
    t = belief["t"]
    truth_x_on_t   = np.interp(t, truth["t"], truth["x"])
    truth_y_on_t   = np.interp(t, truth["t"], truth["y"])
    truth_z_on_t   = np.interp(t, truth["t"], truth["z"])
    truth_sin_on_t = np.interp(t, truth["t"], np.sin(truth["yaw"]))
    truth_cos_on_t = np.interp(t, truth["t"], np.cos(truth["yaw"]))
    truth_yaw_on_t = np.arctan2(truth_sin_on_t, truth_cos_on_t)

    dx   = belief["x"]   - truth_x_on_t
    dy   = belief["y"]   - truth_y_on_t
    dz   = (belief["z"] - z_offset) - truth_z_on_t
    dyaw = np.arctan2(np.sin(belief["yaw"] - truth_yaw_on_t),
                      np.cos(belief["yaw"] - truth_yaw_on_t))
    d_xy = np.hypot(dx, dy)
    return t, dx, dy, dz, dyaw, d_xy


def plot_per_state(drift, color, label, out_path, title_note=""):
    t, dx, dy, dz, dyaw, _ = drift

    dx_cm   = dx * 100.0
    dy_cm   = dy * 100.0
    dz_cm   = dz * 100.0
    dyaw_d  = np.degrees(dyaw)

    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    title = f"Belief-vs-truth drift over time — {label}"
    if title_note:
        title += f" {title_note}"
    fig.suptitle(title)

    panels = [
        (axs[0], "dx [cm]",   dx_cm),
        (axs[1], "dy [cm]",   dy_cm),
        (axs[2], "dz [cm]",   dz_cm),
        (axs[3], "dyaw [deg]", dyaw_d),
    ]
    for ax, ylabel, y in panels:
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.plot(t, y, color=color, label=label)
        ax.annotate(f"{y[-1]:.2f}", xy=(t[-1], y[-1]),
                    xytext=(5, 0), textcoords="offset points",
                    color=color, ha="left", va="center")
        ax.set_ylabel(ylabel)
        ax.grid(True)

    axs[0].legend(loc="best")
    axs[-1].set_xlabel("time [s]")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_xy_total(drift, color, label, out_path):
    t, _, _, _, _, d_xy = drift
    d_xy_cm = d_xy * 100.0
    rate = d_xy_cm[-1] / t[-1] if t[-1] > 0 else float("nan")

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.suptitle(f"Total xy drift magnitude — {label}")

    ax.plot(t, d_xy_cm, color=color,
            label=f"{label}: {rate:.3f} cm/s")
    ax.annotate(f"{d_xy_cm[-1]:.2f} cm @ {t[-1]:.1f} s",
                xy=(t[-1], d_xy_cm[-1]),
                xytext=(5, 0), textcoords="offset points",
                color=color, ha="left", va="center")

    ax.set_xlabel("time [s]")
    ax.set_ylabel("|belief − truth| xy [cm]")
    ax.grid(True)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    runs = [
        (HEADLINE_EKF_BAG,  "blue", "EKF (Run 2)",  "ekf",
            BODY_LINK_Z_OFFSET, "(body-link offset removed)"),
        (HEADLINE_ODOM_BAG, "red",  "Odom (Run 2)", "odom",
            0.0, ""),
    ]

    for bag_dir, *_ in runs:
        if not bag_dir.is_dir():
            raise SystemExit(f"bag directory not found: {bag_dir}")

    for bag_dir, color, label, suffix, z_offset, title_note in runs:
        print(f"reading {bag_dir}")
        data = read_bag(bag_dir)
        print(f"  /ekf_pose       n={len(data['belief']['t'])}")
        print(f"  /crazyflie/odom n={len(data['truth']['t'])}")

        drift = compute_drift(data["belief"], data["truth"],
                              z_offset=z_offset)
        plot_per_state(drift, color, label,
                       FIGURES_DIR / f"drift_per_state_{suffix}.png",
                       title_note=title_note)
        plot_xy_total(drift,  color, label,
                       FIGURES_DIR / f"drift_xy_total_{suffix}.png")


if __name__ == "__main__":
    main()
