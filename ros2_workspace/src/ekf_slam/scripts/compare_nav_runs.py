#!/usr/bin/env python3
"""Compare EKF-corrected vs raw-odometry autonomous navigation runs.

Reads:
    results/gazebo_full_nav_ekf_bag/        (EKF run, /ekf_pose = EKF output)
    results/gazebo_full_nav_odom_only_bag/  (baseline, /ekf_pose = raw odom adapter)

Outputs:
    results/nav_comparison_trajectories.png  — both /ekf_pose XY paths overlaid
    results/nav_comparison_summary.txt       — landing precision side-by-side
and prints the same summary to stdout.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py


PROJECT_DIR = '/home/renix/EKF-SLAM-Autonomous-Crazyflie'
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
EKF_BAG  = os.path.join(RESULTS_DIR, 'gazebo_full_nav_ekf_bag')
ODOM_BAG = os.path.join(RESULTS_DIR, 'gazebo_full_nav_odom_only_bag')

POSE_TOPIC = '/ekf_pose'
GOAL = (0.8, 0.0)  # matches launch files


def read_pose_xy(bag_path):
    """Return (times_s, xs, ys) for /ekf_pose in chronological order."""
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if POSE_TOPIC not in type_map:
        raise RuntimeError(f"{POSE_TOPIC} not found in {bag_path}")

    MsgType = get_message(type_map[POSE_TOPIC])
    ts, xs, ys = [], [], []
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic != POSE_TOPIC:
            continue
        msg = deserialize_message(data, MsgType)
        p = msg.pose.position
        ts.append(t_ns * 1e-9)
        xs.append(p.x)
        ys.append(p.y)

    return np.array(ts), np.array(xs), np.array(ys)


def landing_stats(label, ts, xs, ys, goal):
    """Compute final position, distance to goal, and tail jitter."""
    if len(xs) == 0:
        return {
            'label': label, 'samples': 0, 'final_x': float('nan'),
            'final_y': float('nan'), 'goal_dist': float('nan'),
            'tail_std': float('nan'), 'duration_s': 0.0,
        }
    final_x, final_y = float(xs[-1]), float(ys[-1])
    dist = float(np.hypot(final_x - goal[0], final_y - goal[1]))

    # Tail jitter: stddev of last 1 s of poses (tells us if the drone settled
    # vs. is still drifting at bag end).
    if len(ts) > 1:
        cutoff = ts[-1] - 1.0
        mask = ts >= cutoff
        tail_x = xs[mask]
        tail_y = ys[mask]
        tail_std = float(np.hypot(np.std(tail_x), np.std(tail_y))) if mask.sum() > 1 else 0.0
    else:
        tail_std = 0.0

    return {
        'label': label,
        'samples': len(xs),
        'final_x': final_x,
        'final_y': final_y,
        'goal_dist': dist,
        'tail_std': tail_std,
        'duration_s': float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0,
    }


def fmt_row(stats):
    return (f"  samples       : {stats['samples']}\n"
            f"  duration      : {stats['duration_s']:.2f} s\n"
            f"  final pos     : ({stats['final_x']:.3f}, {stats['final_y']:.3f}) m\n"
            f"  dist to goal  : {stats['goal_dist']:.3f} m\n"
            f"  tail jitter   : {stats['tail_std']:.4f} m (last 1 s)\n")


def main():
    missing = [p for p in (EKF_BAG, ODOM_BAG) if not os.path.isdir(p)]
    if missing:
        print('ERROR: missing bag directories:', file=sys.stderr)
        for p in missing:
            print(f'  {p}', file=sys.stderr)
        return 1

    ekf_t,  ekf_x,  ekf_y  = read_pose_xy(EKF_BAG)
    odom_t, odom_x, odom_y = read_pose_xy(ODOM_BAG)

    ekf_stats  = landing_stats('EKF',          ekf_t,  ekf_x,  ekf_y,  GOAL)
    odom_stats = landing_stats('Odom-only',    odom_t, odom_x, odom_y, GOAL)

    summary_lines = [
        'Autonomous navigation — EKF vs raw odometry',
        f'Goal pose: ({GOAL[0]:.2f}, {GOAL[1]:.2f}) m',
        '',
        '[EKF run]   ' + EKF_BAG,
        fmt_row(ekf_stats),
        '[Odom run]  ' + ODOM_BAG,
        fmt_row(odom_stats),
    ]
    if not (np.isnan(ekf_stats['goal_dist']) or np.isnan(odom_stats['goal_dist'])):
        delta = odom_stats['goal_dist'] - ekf_stats['goal_dist']
        winner = 'EKF' if delta > 0 else ('Odom' if delta < 0 else 'tie')
        summary_lines.append(
            f"Landing-error delta (odom - ekf): {delta:+.3f} m  →  {winner} closer to goal"
        )
    summary = '\n'.join(summary_lines)
    print(summary)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'nav_comparison_summary.txt'), 'w') as f:
        f.write(summary + '\n')

    fig, ax = plt.subplots(figsize=(8, 8))
    if len(ekf_x):
        ax.plot(ekf_x, ekf_y, '-', color='C0', linewidth=1.5, label='EKF /ekf_pose')
        ax.plot(ekf_x[0], ekf_y[0], 'o', color='C0', markersize=8, label='EKF start')
        ax.plot(ekf_x[-1], ekf_y[-1], 's', color='C0', markersize=10, label='EKF final')
    if len(odom_x):
        ax.plot(odom_x, odom_y, '-', color='C3', linewidth=1.5, label='Odom-only /ekf_pose')
        ax.plot(odom_x[0], odom_y[0], 'o', color='C3', markersize=8, label='Odom start')
        ax.plot(odom_x[-1], odom_y[-1], 's', color='C3', markersize=10, label='Odom final')
    ax.plot(GOAL[0], GOAL[1], '*', color='gold', markersize=20,
            markeredgecolor='k', label=f'Goal ({GOAL[0]}, {GOAL[1]})')

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    title = 'Autonomous navigation trajectories\nEKF-corrected vs raw odometry'
    if not np.isnan(ekf_stats['goal_dist']) and not np.isnan(odom_stats['goal_dist']):
        title += (f"\nLanding error — EKF: {ekf_stats['goal_dist']:.3f} m  | "
                  f"Odom: {odom_stats['goal_dist']:.3f} m")
    ax.set_title(title)
    fig.tight_layout()
    out_png = os.path.join(RESULTS_DIR, 'nav_comparison_trajectories.png')
    fig.savefig(out_png, dpi=150)
    print(f'\nSaved plot:    {out_png}')
    print(f'Saved summary: {os.path.join(RESULTS_DIR, "nav_comparison_summary.txt")}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
