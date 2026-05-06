#!/usr/bin/env python3
"""Quantitative EKF-vs-odom comparison for the two-leg headline run.

Reads the Run-2 headline bags:
    analysis/headline_bags/final_ekf_bag_run2/
    analysis/headline_bags/final_odom_bag_run2/

Each bag contains:
    /ekf_pose         — belief (EKF output, or odom+drift baseline)
    /crazyflie/odom   — Gazebo physics ground truth
    /crazyflie/scan   — 4-beam multiranger

Two-leg mission: outbound to (0.8, 0.0), return to (0.0, 0.0). Legs are
detected from the truth z trace (rising-edge above TAKEOFF_Z, then drop
below LAND_Z).

Computes:

  Bag-wide
    * Obstacle surface RMS — scan hits within HIT_RADIUS of obstacle_4
      projected via belief pose, RMS distance to nearest GT box face.
      Belief-frame is intentional: this measures localization error.

  Per leg (outbound, return)
    * Path efficiency      — straight(start→goal) / actual_path_length,
                             truth-frame, ≤1 (1.0 = perfect, < = worse).
    * C-obs time-fraction  — wall-clock seconds inside the inflated
                             obstacle grid, truth-frame.
    * Time to goal         — truth duration of the leg.
    * Landing error, truth — ‖truth_xy[leg_end] − leg_goal‖.
    * Landing error, belief — ‖belief_xy(at truth_t[leg_end]) − leg_goal‖.

Outputs a comparison table to stdout and to results/slam_error_report.txt.
"""

import os
import sys
from typing import List, Tuple

import numpy as np

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py


PROJECT_DIR = '/home/renix/EKF-SLAM-Autonomous-Crazyflie'
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
HEADLINE_BAGS_DIR = os.path.join(PROJECT_DIR, 'analysis', 'headline_bags')
EKF_BAG  = os.path.join(HEADLINE_BAGS_DIR, 'final_ekf_bag_run2')
ODOM_BAG = os.path.join(HEADLINE_BAGS_DIR, 'final_odom_bag_run2')

OUTBOUND_GOAL = (0.8, 0.0)
RETURN_GOAL   = (0.0, 0.0)
START         = (0.0, 0.0)

# Leg-detection thresholds, mirrors compare_nav_runs.py.
LAND_Z    = 0.10
TAKEOFF_Z = 0.20

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
    """Single pass — returns (belief, truth, scans) dicts."""
    reader, type_map = open_reader(bag_path)

    needed = ('/ekf_pose', '/crazyflie/odom', '/crazyflie/scan')
    for topic in needed:
        if topic not in type_map:
            raise RuntimeError(f"{topic} missing in {bag_path}")

    PoseT  = get_message(type_map['/ekf_pose'])
    OdomT  = get_message(type_map['/crazyflie/odom'])
    ScanT  = get_message(type_map['/crazyflie/scan'])

    bt, bx, by, bz, byaw = [], [], [], [], []
    tt, tx, ty, tz       = [], [], [], []
    st, sranges          = [], []

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic == '/ekf_pose':
            msg = deserialize_message(data, PoseT)
            q = msg.pose.orientation
            yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            bt.append(t_ns * 1e-9)
            bx.append(msg.pose.position.x)
            by.append(msg.pose.position.y)
            bz.append(msg.pose.position.z)
            byaw.append(yaw)
        elif topic == '/crazyflie/odom':
            msg = deserialize_message(data, OdomT)
            tt.append(t_ns * 1e-9)
            tx.append(msg.pose.pose.position.x)
            ty.append(msg.pose.pose.position.y)
            tz.append(msg.pose.pose.position.z)
        elif topic == '/crazyflie/scan':
            msg = deserialize_message(data, ScanT)
            st.append(t_ns * 1e-9)
            sranges.append(np.asarray(msg.ranges, dtype=float))

    truth_order = np.argsort(tt)
    belief = {'t':   np.array(bt),   'x': np.array(bx),
              'y':   np.array(by),   'z': np.array(bz),
              'yaw': np.array(byaw)}
    truth  = {'t': np.array(tt)[truth_order],
              'x': np.array(tx)[truth_order],
              'y': np.array(ty)[truth_order],
              'z': np.array(tz)[truth_order]}
    scans  = {'t': np.array(st), 'ranges': sranges}
    return belief, truth, scans


def detect_legs(t: np.ndarray, z: np.ndarray) -> List[Tuple[int, int]]:
    """Identify (start_idx, land_idx) pairs for each landing leg.

    Walks the z trace: rising-edge above TAKEOFF_Z = leg start, first
    sample dropping below LAND_Z = leg end. Up to 2 legs.
    """
    legs = []
    in_air = False
    leg_start = 0
    for i in range(len(t)):
        if not in_air and z[i] >= TAKEOFF_Z:
            in_air = True
            leg_start = i
        elif in_air and z[i] < LAND_Z:
            legs.append((leg_start, i))
            in_air = False
            if len(legs) == 2:
                break
    if in_air and len(legs) < 2:
        legs.append((leg_start, len(t) - 1))
    return legs


def beam_endpoints(belief, scans):
    """Project each scan's 4 beams to world-frame endpoints using the
    nearest-time belief pose. Drops max-range / NaN beams."""
    if belief['t'].size == 0 or scans['t'].size == 0:
        return np.empty((0, 2))

    pt = belief['t']
    out_x, out_y = [], []
    for s_t, ranges in zip(scans['t'], scans['ranges']):
        idx = int(np.searchsorted(pt, s_t))
        if idx >= len(pt):
            idx = len(pt) - 1
        elif idx > 0 and abs(pt[idx - 1] - s_t) < abs(pt[idx] - s_t):
            idx -= 1
        x, y, yaw = belief['x'][idx], belief['y'][idx], belief['yaw'][idx]
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
    """RMS distance from each near-obstacle_4 scan hit to the GT box surface."""
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
    """(ix, iy) cell keys covered by the inflated obstacles."""
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


def cobs_time_fraction(ts, xs, ys, occ_keys, resolution):
    """Wall-clock fraction of the trajectory inside inflated C-obs."""
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


def leg_metrics(belief, truth, leg_idx, leg_start_xy, leg_goal, occ_keys):
    """Per-leg metrics in truth-frame, plus belief landing error.

    leg_idx is a (start, end) pair into the truth arrays. Returns a dict
    of NaNs if leg_idx is None.
    """
    nan_dict = {
        'path_eff':         float('nan'),
        'cobs_frac':        float('nan'),
        'cobs_t_inside_s':  float('nan'),
        'cobs_t_total_s':   float('nan'),
        'duration_s':       float('nan'),
        'landing_truth_m':  float('nan'),
        'landing_belief_m': float('nan'),
        'truth_final_xy':   (float('nan'), float('nan')),
        'belief_final_xy':  (float('nan'), float('nan')),
    }
    if leg_idx is None:
        return nan_dict

    s, e = leg_idx
    if e <= s + 1:
        return nan_dict

    leg_t = truth['t'][s:e + 1]
    leg_x = truth['x'][s:e + 1]
    leg_y = truth['y'][s:e + 1]

    actual_len = path_length(leg_x, leg_y)
    straight = float(np.hypot(leg_goal[0] - leg_start_xy[0],
                              leg_goal[1] - leg_start_xy[1]))
    eff = (straight / actual_len) if actual_len > 1e-6 else float('nan')

    cobs_frac, t_inside, t_total = cobs_time_fraction(
        leg_t, leg_x, leg_y, occ_keys, CSPACE_RES)

    duration = float(leg_t[-1] - leg_t[0])

    tx_end, ty_end = float(leg_x[-1]), float(leg_y[-1])
    landing_truth = float(np.hypot(tx_end - leg_goal[0], ty_end - leg_goal[1]))

    t_at = float(leg_t[-1])
    if belief['t'].size:
        bx_at = float(np.interp(t_at, belief['t'], belief['x']))
        by_at = float(np.interp(t_at, belief['t'], belief['y']))
        landing_belief = float(np.hypot(bx_at - leg_goal[0], by_at - leg_goal[1]))
    else:
        bx_at, by_at = float('nan'), float('nan')
        landing_belief = float('nan')

    return {
        'path_eff':         eff,
        'cobs_frac':        cobs_frac,
        'cobs_t_inside_s':  t_inside,
        'cobs_t_total_s':   t_total,
        'duration_s':       duration,
        'landing_truth_m':  landing_truth,
        'landing_belief_m': landing_belief,
        'truth_final_xy':   (tx_end, ty_end),
        'belief_final_xy':  (bx_at, by_at),
    }


def analyze(label: str, bag_path: str, occ_keys):
    belief, truth, scans = read_bag(bag_path)

    endpoints = beam_endpoints(belief, scans)
    obs_rms, n_hits = obstacle_surface_rms(endpoints)

    legs = detect_legs(truth['t'], truth['z'])

    def leg_or_none(i):
        return legs[i] if i < len(legs) else None

    out_metrics = leg_metrics(belief, truth, leg_or_none(0),
                              START, OUTBOUND_GOAL, occ_keys)
    ret_metrics = leg_metrics(belief, truth, leg_or_none(1),
                              OUTBOUND_GOAL, RETURN_GOAL, occ_keys)

    return {
        'label': label,
        'obstacle_surface_rms_m': obs_rms,
        'obstacle_hits': n_hits,
        'n_legs': len(legs),
        'outbound': out_metrics,
        'return':   ret_metrics,
    }


def fmt(stats_e, stats_o):
    def cell(val, fmt_str):
        if isinstance(val, float) and np.isnan(val):
            return 'NaN'
        return fmt_str.format(val)

    lines = []
    lines.append(
        f"Ground truth obstacle_4 centre: ({OBS_GT[0]:.2f}, {OBS_GT[1]:.2f}) m, "
        f"size 0.30 m"
    )
    lines.append(
        f"Outbound goal: {OUTBOUND_GOAL}    Return goal: {RETURN_GOAL}    "
        f"Robot radius: {ROBOT_RADIUS:.2f} m"
    )
    lines.append('')

    # Bag-wide table
    lines.append('Bag-wide metrics')
    lines.append(f"  {'Metric':<32s}{'EKF':<18s}{'Odom-Only':<18s}")
    lines.append('  ' + '-' * 68)
    lines.append(
        f"  {'Obstacle surface RMS':<32s}"
        f"{cell(stats_e['obstacle_surface_rms_m'], '{:.3f} m'):<18s}"
        f"{cell(stats_o['obstacle_surface_rms_m'], '{:.3f} m'):<18s}"
    )
    lines.append(
        f"  {'  (n hits inside window)':<32s}"
        f"{str(stats_e['obstacle_hits']):<18s}"
        f"{str(stats_o['obstacle_hits']):<18s}"
    )
    lines.append(
        f"  {'  (legs detected, truth z)':<32s}"
        f"{str(stats_e['n_legs']):<18s}"
        f"{str(stats_o['n_legs']):<18s}"
    )
    lines.append('')

    # Per-leg table
    def leg_block(title, key):
        block = []
        block.append(title)
        block.append(f"  {'Metric':<32s}{'EKF':<18s}{'Odom-Only':<18s}")
        block.append('  ' + '-' * 68)
        e = stats_e[key]
        o = stats_o[key]
        rows = [
            ('Path efficiency (straight/actual)',
             cell(e['path_eff'], '{:.3f}'),
             cell(o['path_eff'], '{:.3f}')),
            ('Time-fraction in C-obs',
             cell(e['cobs_frac'], '{:.1%}'),
             cell(o['cobs_frac'], '{:.1%}')),
            ('  (seconds inside)',
             cell(e['cobs_t_inside_s'], '{:.1f} s'),
             cell(o['cobs_t_inside_s'], '{:.1f} s')),
            ('Duration (truth)',
             cell(e['duration_s'], '{:.1f} s'),
             cell(o['duration_s'], '{:.1f} s')),
            ('Landing error — truth',
             cell(e['landing_truth_m'], '{:.3f} m'),
             cell(o['landing_truth_m'], '{:.3f} m')),
            ('Landing error — belief',
             cell(e['landing_belief_m'], '{:.3f} m'),
             cell(o['landing_belief_m'], '{:.3f} m')),
        ]
        for name, ev, ov in rows:
            block.append(f"  {name:<32s}{ev:<18s}{ov:<18s}")
        return block

    lines.extend(leg_block('Outbound leg  — start (0,0) → goal (0.8, 0.0)',
                           'outbound'))
    lines.append('')
    lines.extend(leg_block('Return leg    — start (0.8, 0) → goal (0.0, 0.0)',
                           'return'))
    lines.append('')
    lines.append('Notes:')
    lines.append('  Obstacle RMS uses belief-frame scan projection — that is the '
                 'localization-error signal.')
    lines.append('  Path efficiency, C-obs fraction, duration, and landing-truth '
                 'are truth-frame.')
    lines.append('  Landing-belief = ‖belief_xy(at truth-leg-end-t) − leg_goal‖, '
                 'controller-view.')
    return '\n'.join(lines)


def main():
    for p in (EKF_BAG, ODOM_BAG):
        if not os.path.isdir(p):
            print(f'ERROR: bag directory missing: {p}', file=sys.stderr)
            return 1

    occ_keys = build_cobs_keys(CSPACE_RES)

    ekf  = analyze('EKF',       EKF_BAG,  occ_keys)
    odom = analyze('Odom-Only', ODOM_BAG, occ_keys)

    table = fmt(ekf, odom)
    print('SLAM error report — EKF-corrected vs raw odometry (Run 2 headline bags)\n')
    print(table)

    out = os.path.join(RESULTS_DIR, 'slam_error_report.txt')
    with open(out, 'w') as f:
        f.write('SLAM error report — EKF-corrected vs raw odometry (Run 2 headline bags)\n\n')
        f.write(table + '\n')
    print(f'\nSaved: {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
