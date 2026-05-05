#!/usr/bin/env python3
"""Quantitative EKF-vs-odom comparison for the autonomous-nav report.

Reads:
    results/gazebo_full_nav_ekf_bag/
    results/gazebo_full_nav_odom_only_bag/

Computes per-run:
    1. Obstacle-localization error  : centroid of scan hits near obstacle_4
                                      vs ground-truth (0.40, 0.00).
    2. Path-efficiency ratio        : path_length / straight_line(start, goal).
    3. C-obs violations per metre   : pose samples in inflated obstacle space
                                      (robot radius 0.10 m) per metre flown.
    4. Time to goal                 : duration of /ekf_pose stream.
    5. Landing error                : final pose distance to goal.

Outputs a single comparison table to stdout and to
results/slam_error_report.txt.
"""

import os
import sys
from typing import Tuple

import numpy as np

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py


PROJECT_DIR = '/home/renix/EKF-SLAM-Autonomous-Crazyflie'
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
EKF_BAG = os.path.join(RESULTS_DIR, 'gazebo_full_nav_ekf_bag')
ODOM_BAG = os.path.join(RESULTS_DIR, 'gazebo_full_nav_odom_only_bag')

GOAL = (0.8, 0.0)
START = (0.0, 0.0)

# Ground-truth obstacle_4 from crazyflie_world.sdf.
OBS_GT = (0.4, 0.0)
OBS_HALF = 0.15           # 0.3 m box, half-extent
HIT_RADIUS = 0.40         # window for collecting scan hits attributable to obstacle_4

# C-space inflation parameters — must match compare_nav_runs.py.
ROBOT_RADIUS = 0.10
CSPACE_RES = 0.05
CSPACE_BOUNDS = (-2.2, 2.2, -2.2, 2.2)

# 4-beam multiranger bearings, body frame: back / right / front / left.
BEAM_BEARINGS = np.array([np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0])

# Obstacles as in compare_nav_runs.py — needed to build the inflated C-obs grid.
WORLD_OBSTACLES = [
    (1.95,  2.05, -2.05,  2.05),   # wall_east
    (-2.05, -1.95, -2.05, 2.05),   # wall_west
    (-2.05, 2.05,  1.95,  2.05),   # wall_north
    (-2.05, 2.05, -2.05, -1.95),   # wall_south
    (0.4,  0.6,   0.4,  0.6),      # obstacle_1
    (-0.6, -0.4,  0.2,  0.4),      # obstacle_2
    (0.1,  0.3,  -0.7, -0.5),      # obstacle_3
    (0.25, 0.55, -0.15, 0.15),     # obstacle_4
]


def open_reader(bag_path: str):
    so = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
    co = rosbag2_py.ConverterOptions('', '')
    r = rosbag2_py.SequentialReader()
    r.open(so, co)
    return r, {t.name: t.type for t in r.get_all_topics_and_types()}


def read_bag(bag_path: str):
    """Single pass over the bag — returns (poses dict, scans dict)."""
    reader, type_map = open_reader(bag_path)

    needed = ('/ekf_pose', '/crazyflie/scan')
    for topic in needed:
        if topic not in type_map:
            raise RuntimeError(f"{topic} missing in {bag_path}")

    PoseT = get_message(type_map['/ekf_pose'])
    ScanT = get_message(type_map['/crazyflie/scan'])

    pt, px, py, pyaw = [], [], [], []
    st, sranges = [], []

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic == '/ekf_pose':
            msg = deserialize_message(data, PoseT)
            q = msg.pose.orientation
            yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            pt.append(t_ns * 1e-9)
            px.append(msg.pose.position.x)
            py.append(msg.pose.position.y)
            pyaw.append(yaw)
        elif topic == '/crazyflie/scan':
            msg = deserialize_message(data, ScanT)
            st.append(t_ns * 1e-9)
            sranges.append(np.asarray(msg.ranges, dtype=float))

    poses = {'t': np.array(pt), 'x': np.array(px),
             'y': np.array(py), 'yaw': np.array(pyaw)}
    scans = {'t': np.array(st), 'ranges': sranges}
    return poses, scans


def beam_endpoints(poses, scans):
    """Project each scan's 4 beams to world-frame endpoints using the
    nearest pose in time. Drops max-range / NaN beams."""
    if poses['t'].size == 0 or scans['t'].size == 0:
        return np.empty((0, 2))

    pt = poses['t']
    out_x, out_y = [], []
    for s_t, ranges in zip(scans['t'], scans['ranges']):
        idx = int(np.searchsorted(pt, s_t))
        if idx >= len(pt):
            idx = len(pt) - 1
        elif idx > 0 and abs(pt[idx - 1] - s_t) < abs(pt[idx] - s_t):
            idx -= 1
        x, y, yaw = poses['x'][idx], poses['y'][idx], poses['yaw'][idx]
        if len(ranges) < 4:
            continue
        for i in range(4):
            r = ranges[i]
            if not np.isfinite(r) or r <= 0.05 or r > 3.4:
                continue
            ang = yaw + BEAM_BEARINGS[i]
            out_x.append(x + r * np.cos(ang))
            out_y.append(y + r * np.sin(ang))
    return np.column_stack([out_x, out_y])


def obstacle_surface_rms(endpoints: np.ndarray) -> Tuple[float, int]:
    """For each scan hit within HIT_RADIUS of OBS_GT, compute the distance
    to the nearest face of the ground-truth box, then return (rms, n_hits).

    A hit on the box surface itself has distance 0. Drift biases each hit
    away from the surface, so RMS surface-distance is a direct measure of
    pose accuracy — independent of how many hits the run accumulated."""
    if endpoints.size == 0:
        return float('nan'), 0
    d_centre = np.hypot(endpoints[:, 0] - OBS_GT[0],
                        endpoints[:, 1] - OBS_GT[1])
    mask = d_centre < HIT_RADIUS
    if mask.sum() == 0:
        return float('nan'), 0
    pts = endpoints[mask]
    xmin, xmax = OBS_GT[0] - OBS_HALF, OBS_GT[0] + OBS_HALF
    ymin, ymax = OBS_GT[1] - OBS_HALF, OBS_GT[1] + OBS_HALF
    # Distance from each point to the rectangle (0 inside, exterior distance
    # outside — symmetric: a hit just outside the wall is treated like a hit
    # just inside, both measure surface-error magnitude).
    dx = np.maximum.reduce([np.zeros(len(pts)),
                            xmin - pts[:, 0], pts[:, 0] - xmax])
    dy = np.maximum.reduce([np.zeros(len(pts)),
                            ymin - pts[:, 1], pts[:, 1] - ymax])
    surf = np.hypot(dx, dy)
    rms = float(np.sqrt(np.mean(surf ** 2)))
    return rms, int(mask.sum())


def path_length(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size < 2:
        return 0.0
    return float(np.hypot(np.diff(xs), np.diff(ys)).sum())


def build_cobs_keys(resolution: float):
    """Return a set of (ix, iy) cell keys covered by the inflated obstacles —
    same Minkowski-sum logic as compare_nav_runs.build_cspace_grid, kept
    in keys for fast trajectory-membership tests."""
    x_min, x_max, y_min, y_max = CSPACE_BOUNDS
    xs = np.arange(x_min, x_max + resolution, resolution)
    ys = np.arange(y_min, y_max + resolution, resolution)
    cx, cy = np.meshgrid(xs, ys)
    occ_mask = np.zeros_like(cx, dtype=bool)
    for (xmin, xmax, ymin, ymax) in WORLD_OBSTACLES:
        dx = np.maximum.reduce([np.zeros_like(cx), xmin - cx, cx - xmax])
        dy = np.maximum.reduce([np.zeros_like(cy), ymin - cy, cy - ymax])
        occ_mask |= (np.hypot(dx, dy) <= ROBOT_RADIUS)
    keys = set()
    for x, y in zip(cx[occ_mask], cy[occ_mask]):
        keys.add((round(x / resolution), round(y / resolution)))
    return keys


def cobs_violations(xs, ys, occ_keys, resolution):
    if xs.size == 0:
        return 0
    return sum(1 for x, y in zip(xs, ys)
               if (round(x / resolution), round(y / resolution)) in occ_keys)


def cobs_time_fraction(ts, xs, ys, occ_keys, resolution):
    """Time-fraction of the trajectory spent inside inflated C-obs.

    Each sample 'owns' the interval to the next sample (trapezoidal in time).
    A sample inside C-obs adds its dt to the numerator. Robust to
    EKF (10 Hz) vs odom (50 Hz) sample-rate differences — only the wall-clock
    seconds inside C-obs matter, not the count of samples."""
    if ts.size < 2:
        return float('nan'), 0.0, 0.0
    dts = np.diff(ts)
    inside = np.array([
        (round(x / resolution), round(y / resolution)) in occ_keys
        for x, y in zip(xs[:-1], ys[:-1])
    ], dtype=bool)
    t_inside = float(dts[inside].sum())
    t_total = float(dts.sum())
    frac = (t_inside / t_total) if t_total > 0 else float('nan')
    return frac, t_inside, t_total


def analyze(label: str, bag_path: str, occ_keys):
    poses, scans = read_bag(bag_path)
    endpoints = beam_endpoints(poses, scans)
    obs_rms, n_hits = obstacle_surface_rms(endpoints)

    plen = path_length(poses['x'], poses['y'])
    straight = float(np.hypot(GOAL[0] - START[0], GOAL[1] - START[1]))
    eff = (plen / straight) if straight > 0 else float('nan')

    cobs_frac, t_inside, t_total = cobs_time_fraction(
        poses['t'], poses['x'], poses['y'], occ_keys, CSPACE_RES)

    duration = float(poses['t'][-1] - poses['t'][0]) if poses['t'].size > 1 else 0.0

    if poses['x'].size:
        fx, fy = float(poses['x'][-1]), float(poses['y'][-1])
        landing_err = float(np.hypot(fx - GOAL[0], fy - GOAL[1]))
    else:
        landing_err = float('nan')

    return {
        'label': label,
        'obstacle_surface_rms_m': obs_rms,
        'obstacle_hits': n_hits,
        'path_length_m': plen,
        'efficiency_ratio': eff,
        'cobs_time_fraction': cobs_frac,
        'cobs_time_inside_s': t_inside,
        'duration_s': duration,
        'landing_error_m': landing_err,
        'samples': int(poses['t'].size),
    }


def fmt(stats_e, stats_o):
    def cell(val, fmt_str):
        if isinstance(val, float) and np.isnan(val):
            return 'NaN'
        return fmt_str.format(val)

    rows = [
        ('Obstacle surface RMS',
         cell(stats_e['obstacle_surface_rms_m'], '{:.3f} m'),
         cell(stats_o['obstacle_surface_rms_m'], '{:.3f} m')),
        ('  (n hits inside window)',
         str(stats_e['obstacle_hits']),
         str(stats_o['obstacle_hits'])),
        ('Path efficiency ratio',
         cell(stats_e['efficiency_ratio'], '{:.2f}'),
         cell(stats_o['efficiency_ratio'], '{:.2f}')),
        ('  (path length)',
         cell(stats_e['path_length_m'], '{:.2f} m'),
         cell(stats_o['path_length_m'], '{:.2f} m')),
        ('Time-fraction in C-obs',
         cell(stats_e['cobs_time_fraction'], '{:.1%}'),
         cell(stats_o['cobs_time_fraction'], '{:.1%}')),
        ('  (seconds inside)',
         cell(stats_e['cobs_time_inside_s'], '{:.1f} s'),
         cell(stats_o['cobs_time_inside_s'], '{:.1f} s')),
        ('Time to goal',
         cell(stats_e['duration_s'], '{:.1f} s'),
         cell(stats_o['duration_s'], '{:.1f} s')),
        ('Landing error',
         cell(stats_e['landing_error_m'], '{:.3f} m'),
         cell(stats_o['landing_error_m'], '{:.3f} m')),
    ]

    lines = []
    lines.append(
        f"Ground truth obstacle_4 centre: ({OBS_GT[0]:.2f}, {OBS_GT[1]:.2f}) m, "
        f"size 0.30 m\n"
        f"Goal: ({GOAL[0]:.2f}, {GOAL[1]:.2f}) m   "
        f"Robot radius: {ROBOT_RADIUS:.2f} m\n"
    )
    lines.append(f"  {'Metric':<28s}{'EKF':<18s}{'Odom-Only':<18s}")
    lines.append('  ' + '-' * 64)
    for label, ekf_v, odom_v in rows:
        lines.append(f"  {label:<28s}{ekf_v:<18s}{odom_v:<18s}")
    return '\n'.join(lines)


def main():
    for p in (EKF_BAG, ODOM_BAG):
        if not os.path.isdir(p):
            print(f'ERROR: bag directory missing: {p}', file=sys.stderr)
            return 1

    occ_keys = build_cobs_keys(CSPACE_RES)

    ekf = analyze('EKF', EKF_BAG, occ_keys)
    odom = analyze('Odom-Only', ODOM_BAG, occ_keys)

    table = fmt(ekf, odom)
    print('SLAM error report — EKF-corrected vs raw odometry\n')
    print(table)

    out = os.path.join(RESULTS_DIR, 'slam_error_report.txt')
    with open(out, 'w') as f:
        f.write('SLAM error report — EKF-corrected vs raw odometry\n\n')
        f.write(table + '\n')
    print(f'\nSaved: {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
