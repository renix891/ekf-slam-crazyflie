#!/usr/bin/env python3
"""Render the EKF-SLAM joint state at flight end.

Reads the most recent /ekf_slam/debug/landmark_lines and
/ekf_slam/debug/landmark_corners MarkerArray messages from the
gazebo_full_nav_ekf_bag and overlays the persisted landmarks on the
C-space dot grid built from the same flight's scans (lightblue=free,
red=occupied). EKF trajectory drawn in black.

Output: results/ekf_slam_landmarks.png
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# Reuse helpers and constants from analyze_map.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_map import read_bag, classify_dot_grid, _draw_dot_grid, GOAL


def read_final_landmarks(bag_path: str):
    """Return (last_lines_marker, last_corners_marker) MarkerArrays — or None
    for a topic if no messages of that topic were recorded."""
    storage = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
    converter = rosbag2_py.ConverterOptions('', '')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage, converter)

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    line_topic = '/ekf_slam/debug/landmark_lines'
    corner_topic = '/ekf_slam/debug/landmark_corners'

    lines_type = (get_message(type_map[line_topic])
                  if line_topic in type_map else None)
    corners_type = (get_message(type_map[corner_topic])
                    if corner_topic in type_map else None)

    last_lines = None
    last_corners = None

    while reader.has_next():
        topic, data, _t = reader.read_next()
        if topic == line_topic and lines_type is not None:
            last_lines = deserialize_message(data, lines_type)
        elif topic == corner_topic and corners_type is not None:
            last_corners = deserialize_message(data, corners_type)

    return last_lines, last_corners


def line_segments_from_markers(marker_array) -> List[Tuple[float, float, float, float]]:
    """Pull (x1, y1, x2, y2) from each LINE_STRIP marker that's an ADD.
    Skips DELETEALL / DELETE / empty markers."""
    if marker_array is None:
        return []
    segs = []
    for m in marker_array.markers:
        # action: 0=ADD, 1=MODIFY, 2=DELETE, 3=DELETEALL
        if m.action == 3 or m.action == 2:
            continue
        # type: LINE_STRIP=4, LINE_LIST=5
        if len(m.points) < 2:
            continue
        p1 = m.points[0]
        p2 = m.points[-1]
        segs.append((p1.x, p1.y, p2.x, p2.y))
    return segs


def corner_points_from_markers(marker_array) -> List[Tuple[float, float]]:
    """Pull (x, y) from each non-DELETE marker. Corners are CUBE markers
    whose pose.position is the corner's world coordinate."""
    if marker_array is None:
        return []
    pts = []
    for m in marker_array.markers:
        if m.action == 3 or m.action == 2:
            continue
        # CUBE / SPHERE marker: pose.position is the placement.
        pts.append((m.pose.position.x, m.pose.position.y))
    return pts


def main():
    bag = '/home/renix/EKF-SLAM-Autonomous-Crazyflie/results/gazebo_full_nav_ekf_bag'
    out = '/home/renix/EKF-SLAM-Autonomous-Crazyflie/results/ekf_slam_landmarks.png'

    print(f"Reading bag : {bag}")
    last_map, poses, scans = read_bag(bag)
    print(f"  poses : {len(poses['x'])}")
    print(f"  scans : {len(scans['t'])}")

    last_lines, last_corners = read_final_landmarks(bag)
    line_segs = line_segments_from_markers(last_lines)
    corner_pts = corner_points_from_markers(last_corners)
    print(f"  final line landmarks   : {len(line_segs)}")
    print(f"  final corner landmarks : {len(corner_pts)}")

    grid = classify_dot_grid(poses, scans)
    if grid is None:
        print("WARNING: no scan endpoints; using empty grid background.")
        free_xy = np.empty((0, 2))
        occ_xy = np.empty((0, 2))
    else:
        free_xy, occ_xy, *_ = grid

    fig, ax = plt.subplots(figsize=(10, 9))
    _draw_dot_grid(ax, free_xy, occ_xy, free_size=6, occ_size=14, zorder=1)

    # EKF trajectory
    if poses['x'].size:
        ax.plot(poses['x'], poses['y'], '-', color='black',
                lw=1.6, alpha=0.85, zorder=4, label='EKF trajectory')
        ax.plot(poses['x'][0], poses['y'][0], 'o', color='green',
                markersize=10, markeredgecolor='black', zorder=6, label='Start')

    # Goal pad
    ax.plot(GOAL[0], GOAL[1], '*', color='gold',
            markersize=18, markeredgecolor='black', zorder=6, label='Goal')

    # Line landmarks (blue thick segments)
    first = True
    for (x1, y1, x2, y2) in line_segs:
        ax.plot([x1, x2], [y1, y2], '-', color='royalblue',
                lw=2.4, alpha=0.95, zorder=5,
                label='Line landmark (EKF state)' if first else None)
        first = False

    # Corner landmarks (orange dots)
    if corner_pts:
        cx = [p[0] for p in corner_pts]
        cy = [p[1] for p in corner_pts]
        ax.scatter(cx, cy, color='darkorange', s=120, edgecolors='black',
                   linewidths=1.2, zorder=7,
                   label='Corner landmark (EKF state)')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(alpha=0.25)
    ax.set_title(
        f'EKF-SLAM joint state — {len(line_segs)} line + {len(corner_pts)} corner landmarks'
    )
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"Wrote: {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
