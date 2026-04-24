#!/usr/bin/env python3
"""Read the Gazebo test bag, compare /crazyflie/odom vs /ekf_pose, and plot.

Produces:
    results/trajectory_comparison.png
    results/ekf_vs_odom_error.png
and prints summary stats to stdout.
"""

import os
import bisect
import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py


BAG_DIR = '/home/renix/EKF-SLAM-Autonomous-Crazyflie/results/gazebo_test_bag'
RESULTS_DIR = '/home/renix/EKF-SLAM-Autonomous-Crazyflie/results'
ODOM_TOPIC = '/crazyflie/odom'
EKF_TOPIC = '/ekf_pose'

WALL_HALF = 2.0       # walls at ±2 m
WALL_THICKNESS = 0.1
LANDING_PAD = (0.8, 0.0, 0.3, 0.3)    # x, y, width, height


def read_bag_xy(bag_path):
    """Return {topic: (times_ns, xs, ys)} for the two topics of interest."""
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}

    out = {ODOM_TOPIC: ([], [], []), EKF_TOPIC: ([], [], [])}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic not in out:
            continue
        MsgType = get_message(type_map[topic])
        msg = deserialize_message(data, MsgType)
        if topic == ODOM_TOPIC:
            p = msg.pose.pose.position
        else:
            p = msg.pose.position
        out[topic][0].append(t_ns)
        out[topic][1].append(p.x)
        out[topic][2].append(p.y)

    return {k: (np.array(v[0]), np.array(v[1]), np.array(v[2])) for k, v in out.items()}


def nearest_match(ref_t, other_t, other_x, other_y):
    """For each ref_t, find the nearest other_t; return aligned (x, y) arrays."""
    if len(other_t) == 0:
        return np.full_like(ref_t, np.nan, dtype=float), np.full_like(ref_t, np.nan, dtype=float)
    sorted_t = other_t   # already chronological from reader
    xs = np.empty(len(ref_t))
    ys = np.empty(len(ref_t))
    for i, t in enumerate(ref_t):
        idx = bisect.bisect_left(sorted_t, t)
        if idx == 0:
            nearest = 0
        elif idx == len(sorted_t):
            nearest = len(sorted_t) - 1
        else:
            nearest = idx if abs(sorted_t[idx] - t) < abs(sorted_t[idx - 1] - t) else idx - 1
        xs[i] = other_x[nearest]
        ys[i] = other_y[nearest]
    return xs, ys


def plot_trajectory(odom_xy, ekf_xy, out_path):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Walls (black rectangles). Each is 4 m long in one dimension, 0.1 m thick.
    walls = [
        (WALL_HALF - WALL_THICKNESS / 2, -WALL_HALF, WALL_THICKNESS, 2 * WALL_HALF),  # east
        (-WALL_HALF - WALL_THICKNESS / 2, -WALL_HALF, WALL_THICKNESS, 2 * WALL_HALF),  # west
        (-WALL_HALF, WALL_HALF - WALL_THICKNESS / 2, 2 * WALL_HALF, WALL_THICKNESS),  # north
        (-WALL_HALF, -WALL_HALF - WALL_THICKNESS / 2, 2 * WALL_HALF, WALL_THICKNESS),  # south
    ]
    for (x, y, w, h) in walls:
        ax.add_patch(Rectangle((x, y), w, h, color='black'))

    # Landing pad (yellow square).
    px, py, pw, ph = LANDING_PAD
    ax.add_patch(Rectangle((px - pw / 2, py - ph / 2), pw, ph,
                           color='gold', ec='darkgoldenrod', label='Landing pad'))

    ax.plot(odom_xy[0], odom_xy[1], 'r--', linewidth=1.5, label='Raw Odometry')
    ax.plot(ekf_xy[0], ekf_xy[1], 'b-', linewidth=1.5, label='EKF Corrected')
    ax.plot(0, 0, 'go', markersize=10, label='Start (0,0)')

    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-2.3, 2.3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Trajectory: Raw Odometry vs EKF-Corrected')
    ax.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_error(times_s, distances, out_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times_s, distances, 'b-', linewidth=1.2)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('|odom - ekf| (m)')
    ax.set_title('EKF vs Odometry Position Difference Over Time')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    if not os.path.isdir(BAG_DIR):
        raise SystemExit(f'bag not found at {BAG_DIR}')

    data = read_bag_xy(BAG_DIR)
    odom_t, odom_x, odom_y = data[ODOM_TOPIC]
    ekf_t, ekf_x, ekf_y = data[EKF_TOPIC]

    if len(odom_t) == 0 or len(ekf_t) == 0:
        raise SystemExit(
            f'no messages on one of the topics: '
            f'odom={len(odom_t)} ekf={len(ekf_t)}')

    # Align EKF samples to odom timestamps via nearest-neighbor.
    ekf_x_aligned, ekf_y_aligned = nearest_match(odom_t, ekf_t, ekf_x, ekf_y)

    dx = odom_x - ekf_x_aligned
    dy = odom_y - ekf_y_aligned
    dist = np.sqrt(dx * dx + dy * dy)

    t0_ns = min(odom_t[0], ekf_t[0])
    duration_s = (max(odom_t[-1], ekf_t[-1]) - t0_ns) * 1e-9
    rel_t_s = (odom_t - t0_ns) * 1e-9

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_trajectory((odom_x, odom_y), (ekf_x, ekf_y),
                    os.path.join(RESULTS_DIR, 'trajectory_comparison.png'))
    plot_error(rel_t_s, dist,
               os.path.join(RESULTS_DIR, 'ekf_vs_odom_error.png'))

    ate_rms = math.sqrt(float(np.mean(dist * dist)))

    print(f'Flight duration:        {duration_s:.3f} s')
    print(f'/crazyflie/odom msgs:   {len(odom_t)}')
    print(f'/ekf_pose       msgs:   {len(ekf_t)}')
    print(f'ATE RMS:                {ate_rms:.4f} m')
    print(f'Max   |odom - ekf|:     {float(np.max(dist)):.4f} m')
    print(f'Mean  |odom - ekf|:     {float(np.mean(dist)):.4f} m')
    print(f'Saved: {os.path.join(RESULTS_DIR, "trajectory_comparison.png")}')
    print(f'Saved: {os.path.join(RESULTS_DIR, "ekf_vs_odom_error.png")}')


if __name__ == '__main__':
    main()
