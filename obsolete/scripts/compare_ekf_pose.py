#!/usr/bin/env python3
"""Compare /ekf_pose between two ROS 2 bags.

Reads PoseStamped from each bag, aligns them on time relative to the bag's
first message, and emits:

  * XY trajectory plot, both runs overlaid
  * |Δpose| over time (current minus pose-only, time-aligned)
  * Final-approach close-up (last 5 s)
  * Console summary: per-axis std, max abs deviation, final landing point

Defaults expect both bags under EKF-SLAM-Autonomous-Crazyflie/results/.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def read_pose_stream(bag_path: str, topic: str = '/ekf_pose'):
    storage = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
    converter = rosbag2_py.ConverterOptions('', '')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage, converter)

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in type_map:
        raise RuntimeError(f"Topic {topic!r} not found in {bag_path}. "
                           f"Available: {list(type_map)}")
    msg_type = get_message(type_map[topic])

    times: List[float] = []
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    yaws: List[float] = []
    while reader.has_next():
        t, data, ts = reader.read_next()
        if t != topic:
            continue
        msg = deserialize_message(data, msg_type)
        h = msg.header.stamp
        times.append(h.sec + h.nanosec * 1e-9)
        p = msg.pose.position
        q = msg.pose.orientation
        # Yaw from quaternion (z-axis)
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaws.append(np.arctan2(siny, cosy))
        xs.append(p.x)
        ys.append(p.y)
        zs.append(p.z)

    if not times:
        raise RuntimeError(f"No /ekf_pose messages in {bag_path}")
    return (
        np.array(times),
        np.array(xs), np.array(ys), np.array(zs),
        np.array(yaws),
    )


def stats(label: str, t, x, y, z, yaw):
    print(f"--- {label} ---")
    print(f"  duration       : {t[-1] - t[0]:.2f} s")
    print(f"  msgs           : {len(t)}")
    print(f"  final pose     : ({x[-1]:+.3f}, {y[-1]:+.3f}, {z[-1]:+.3f})  "
          f"yaw={np.degrees(yaw[-1]):+.1f} deg")
    # last 1 s tail
    mask = t >= (t[-1] - 1.0)
    print(f"  last-1s mean   : ({x[mask].mean():+.3f}, {y[mask].mean():+.3f})  "
          f"std=({x[mask].std():.3f}, {y[mask].std():.3f})")


def per_step_jumps(label, t, x, y):
    dx = np.diff(x); dy = np.diff(y)
    d = np.hypot(dx, dy)
    if d.size == 0:
        return
    top = np.argsort(d)[-5:][::-1]
    print(f"--- {label}: top-5 single-step XY jumps ---")
    for i in top:
        print(f"    t={t[i+1]-t[0]:6.2f}s  d={d[i]*100:6.2f} cm  "
              f"@({x[i+1]:+.3f},{y[i+1]:+.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--current',
                    default='/home/renix/EKF-SLAM-Autonomous-Crazyflie/results/gazebo_full_nav_ekf_bag',
                    help='current ekf_slam (post-Stage-3) bag path')
    ap.add_argument('--baseline',
                    default='/home/renix/EKF-SLAM-Autonomous-Crazyflie/results/gazebo_full_nav_ekf_pose_only_bag',
                    help='frozen scan-to-map (v1) bag path')
    ap.add_argument('--goal-x', type=float, default=0.8)
    ap.add_argument('--goal-y', type=float, default=0.0)
    ap.add_argument('--no-plot', action='store_true', help='skip matplotlib output')
    ap.add_argument('--out',
                    default='/home/renix/EKF-SLAM-Autonomous-Crazyflie/results/ekf_pose_compare.png')
    args = ap.parse_args()

    print(f"Reading current : {args.current}")
    print(f"Reading baseline: {args.baseline}")

    cur = read_pose_stream(args.current)
    bas = read_pose_stream(args.baseline)
    tc, xc, yc, zc, ywc = cur
    tb, xb, yb, zb, ywb = bas

    # Normalize time to start-of-bag for each.
    tc_rel = tc - tc[0]
    tb_rel = tb - tb[0]

    print()
    stats("CURRENT (post-Stage-3 ekf_slam)",   tc_rel, xc, yc, zc, ywc)
    print()
    stats("BASELINE (frozen ekf_slam_pose_only)", tb_rel, xb, yb, zb, ywb)
    print()

    # Distance to goal over time
    cur_d_goal = np.hypot(xc - args.goal_x, yc - args.goal_y)
    bas_d_goal = np.hypot(xb - args.goal_x, yb - args.goal_y)

    def goal_summary(label, t, d):
        idx_min = int(np.argmin(d))
        print(f"--- {label}: distance-to-goal ({args.goal_x:+.2f},{args.goal_y:+.2f}) ---")
        print(f"    closest approach: {d[idx_min]*100:6.2f} cm at t={t[idx_min]:6.2f}s")
        print(f"    final           : {d[-1]*100:6.2f} cm")
        # Did we ever come close, then drift away?
        # Only consider 'drift' if closest was below 30cm and final is >50cm
        if d[idx_min] < 0.30 and d[-1] > 0.50:
            print(f"    !! DRIFT: closest {d[idx_min]*100:.1f}cm @ t={t[idx_min]:.2f}s "
                  f"-> final {d[-1]*100:.1f}cm @ t={t[-1]:.2f}s")
    goal_summary("CURRENT",  tc_rel, cur_d_goal)
    goal_summary("BASELINE", tb_rel, bas_d_goal)
    print()

    per_step_jumps("CURRENT",  tc, xc, yc)
    per_step_jumps("BASELINE", tb, xb, yb)
    print()

    if args.no_plot:
        return 0

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return 0

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # XY trajectories
    ax = axes[0, 0]
    ax.plot(xb, yb, color='tab:blue', label='baseline (pose-only v1)', alpha=0.85)
    ax.plot(xc, yc, color='tab:orange', label='current (Stage 3)', alpha=0.85)
    ax.scatter([args.goal_x], [args.goal_y], color='red', marker='x', s=80, label='goal')
    ax.scatter([xc[-1]], [yc[-1]], color='tab:orange', marker='*', s=100,
               edgecolors='k', label='cur final')
    ax.scatter([xb[-1]], [yb[-1]], color='tab:blue',   marker='*', s=100,
               edgecolors='k', label='bas final')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title('XY trajectory')
    ax.legend(fontsize=8)

    # Distance to goal vs time
    ax = axes[0, 1]
    ax.plot(tb_rel, bas_d_goal * 100, color='tab:blue',   label='baseline')
    ax.plot(tc_rel, cur_d_goal * 100, color='tab:orange', label='current')
    ax.set_xlabel('t (s)'); ax.set_ylabel('|pose - goal|  (cm)')
    ax.set_title(f'Distance to goal ({args.goal_x:+.2f}, {args.goal_y:+.2f})')
    ax.grid(alpha=0.3); ax.legend()

    # Per-step XY jump magnitude
    ax = axes[1, 0]
    db = np.hypot(np.diff(xb), np.diff(yb))
    dc = np.hypot(np.diff(xc), np.diff(yc))
    ax.plot(tb_rel[1:], db * 100, color='tab:blue',   label='baseline', alpha=0.7)
    ax.plot(tc_rel[1:], dc * 100, color='tab:orange', label='current',  alpha=0.7)
    ax.set_xlabel('t (s)'); ax.set_ylabel('|Δpose| step (cm)')
    ax.set_title('Per-message XY step (look for spikes)')
    ax.grid(alpha=0.3); ax.legend()

    # Final-approach close-up: last 5 s of each, recentered on goal
    ax = axes[1, 1]
    def tail(t_rel, x, y, secs=5.0):
        m = t_rel >= (t_rel[-1] - secs)
        return x[m], y[m]
    xb_t, yb_t = tail(tb_rel, xb, yb)
    xc_t, yc_t = tail(tc_rel, xc, yc)
    ax.plot(xb_t, yb_t, color='tab:blue',   label='baseline last 5s')
    ax.plot(xc_t, yc_t, color='tab:orange', label='current last 5s')
    ax.scatter([args.goal_x], [args.goal_y], color='red', marker='x', s=80)
    ax.scatter([xc[-1]], [yc[-1]], color='tab:orange', marker='*', s=100, edgecolors='k')
    ax.scatter([xb[-1]], [yb[-1]], color='tab:blue',   marker='*', s=100, edgecolors='k')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title('Final-approach close-up (last 5 s)')
    ax.legend(fontsize=8)

    fig.suptitle('EKF /ekf_pose: current (Stage 3) vs frozen pose-only baseline',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"Wrote plot: {args.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
