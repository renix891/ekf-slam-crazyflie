#!/usr/bin/env python3
"""Compare EKF-SLAM vs noisy-odometry runs of the two-leg final mission.

Two-leg mission:
  Leg 1 (outbound): takeoff → fly to (0.8, 0) → box-landing on the pad.
  Leg 2 (return):   takeoff → fly back to (0, 0) → straight-descend.

Reads the bags produced by the final-experiment launch files:
    results/final_ekf_bag/    (EKF-SLAM run, /ekf_pose = EKF output)
    results/final_odom_bag/   (baseline, /ekf_pose = odom + Brownian drift)

Outputs:
    results/nav_comparison_trajectories.png  — trajectories colour-coded by leg
    results/nav_comparison_summary.txt       — table (EKF vs Odom)

Primary metric: RETURN-LEG landing distance vs (0, 0). The phantom-centroid
proof — odometry's accumulated drift means the "I'm back at the origin"
estimate misses by tens of cm; EKF's scan-to-map + landmark corrections keep
it within a few cm.
"""

import os
import sys
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py


PROJECT_DIR = '/home/renix/EKF-SLAM-Autonomous-Crazyflie'
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
EKF_BAG  = os.path.join(RESULTS_DIR, 'final_ekf_bag')
ODOM_BAG = os.path.join(RESULTS_DIR, 'final_odom_bag')

POSE_TOPIC   = '/ekf_pose'
OUTBOUND_GOAL = (0.8, 0.0)
RETURN_GOAL   = (0.0, 0.0)
LAND_Z   = 0.05    # m — anything below this is "landed"
TAKEOFF_Z = 0.20   # m — anything above this is "in the air"

# C-space (mirrors the existing visualisation).
ROBOT_RADIUS = 0.10
CSPACE_RES = 0.05
CSPACE_BOUNDS = (-2.2, 2.2, -2.2, 2.2)

WORLD_OBSTACLES = [
    (1.95,  2.05, -2.05,  2.05),
    (-2.05, -1.95, -2.05, 2.05),
    (-2.05, 2.05,  1.95,  2.05),
    (-2.05, 2.05, -2.05, -1.95),
    (0.4,  0.6,   0.4,  0.6),
    (-0.6, -0.4,  0.2,  0.4),
    (0.1,  0.3,  -0.7, -0.5),
    (0.25, 0.55, -0.15, 0.15),
]


def build_cspace_grid(obstacles, robot_radius, resolution, bounds):
    x_min, x_max, y_min, y_max = bounds
    xs = np.arange(x_min, x_max + resolution, resolution)
    ys = np.arange(y_min, y_max + resolution, resolution)
    cx, cy = np.meshgrid(xs, ys)
    occ_mask = np.zeros_like(cx, dtype=bool)
    for (xmin, xmax, ymin, ymax) in obstacles:
        dx = np.maximum.reduce([np.zeros_like(cx), xmin - cx, cx - xmax])
        dy = np.maximum.reduce([np.zeros_like(cy), ymin - cy, cy - ymax])
        d = np.hypot(dx, dy)
        occ_mask |= (d <= robot_radius)
    free_mask = ~occ_mask
    return (np.column_stack([cx[free_mask], cy[free_mask]]),
            np.column_stack([cx[occ_mask],  cy[occ_mask]]))


def trajectory_clips(xs, ys, occ_xy, resolution):
    if len(xs) == 0 or occ_xy.size == 0:
        return 0
    occ_keys = {(round(p[0] / resolution), round(p[1] / resolution))
                for p in occ_xy}
    return sum(
        1 for x, y in zip(xs, ys)
        if (round(x / resolution), round(y / resolution)) in occ_keys
    )


def read_pose_xyz(bag_path):
    """Return (times_s, xs, ys, zs) for /ekf_pose, sorted by time."""
    storage = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
    converter = rosbag2_py.ConverterOptions('', '')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage, converter)

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if POSE_TOPIC not in type_map:
        raise RuntimeError(f"{POSE_TOPIC} not found in {bag_path}")
    MsgType = get_message(type_map[POSE_TOPIC])

    ts, xs, ys, zs = [], [], [], []
    while reader.has_next():
        topic, data, _t_ns = reader.read_next()
        if topic != POSE_TOPIC:
            continue
        msg = deserialize_message(data, MsgType)
        h = msg.header.stamp
        ts.append(h.sec + h.nanosec * 1e-9)
        p = msg.pose.position
        xs.append(p.x)
        ys.append(p.y)
        zs.append(p.z)
    return np.array(ts), np.array(xs), np.array(ys), np.array(zs)


def detect_legs(t, z) -> List[Tuple[int, int]]:
    """Identify (start_idx, land_idx) pairs for each landing leg.

    A leg ends when z falls and stays below LAND_Z. Walks the z trace,
    finds rising-edge above TAKEOFF_Z (= leg start) followed by the first
    sample dropping below LAND_Z (= leg end). Up to 2 legs.
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
        # Drone hasn't landed by bag end — close the leg at the last sample.
        legs.append((leg_start, len(t) - 1))
    return legs


def leg_landing_stats(label, t, x, y, z, leg_idx, goal):
    """Stats for one leg: final pose, distance to its goal, settling jitter."""
    if leg_idx is None:
        return {'label': label, 'final_x': float('nan'), 'final_y': float('nan'),
                'goal_dist': float('nan'), 'tail_std_m': float('nan'),
                'duration_s': 0.0, 'samples': 0}

    s, e = leg_idx
    fx = float(x[e])
    fy = float(y[e])
    dist = float(np.hypot(fx - goal[0], fy - goal[1]))

    # Tail jitter over the last 1 s of the leg.
    leg_t = t[s:e + 1]
    leg_x = x[s:e + 1]
    leg_y = y[s:e + 1]
    if len(leg_t) > 1:
        cutoff = leg_t[-1] - 1.0
        m = leg_t >= cutoff
        if m.sum() > 1:
            tail_std = float(np.hypot(np.std(leg_x[m]), np.std(leg_y[m])))
        else:
            tail_std = 0.0
    else:
        tail_std = 0.0
    return {
        'label': label,
        'final_x': fx, 'final_y': fy,
        'goal_dist': dist,
        'tail_std_m': tail_std,
        'duration_s': float(t[e] - t[s]) if e > s else 0.0,
        'samples': e - s + 1,
    }


def path_efficiency(t, x, y, leg_idx, goal_start, goal_end):
    """Ratio of straight-line distance / actual path length over a leg.

    1.0 = perfectly straight; <1 = inefficient. We use straight-line(start, goal)
    over actual-path-length so that EKF's smoother trajectory scores higher.
    """
    if leg_idx is None:
        return float('nan')
    s, e = leg_idx
    if e <= s + 1:
        return float('nan')
    dx = np.diff(x[s:e + 1])
    dy = np.diff(y[s:e + 1])
    actual_len = float(np.sum(np.hypot(dx, dy)))
    if actual_len < 1e-6:
        return float('nan')
    straight = float(np.hypot(goal_end[0] - goal_start[0],
                              goal_end[1] - goal_start[1]))
    return straight / actual_len


def main():
    missing = [p for p in (EKF_BAG, ODOM_BAG) if not os.path.isdir(p)]
    if missing:
        print('ERROR: missing bag directories:', file=sys.stderr)
        for p in missing:
            print(f'  {p}', file=sys.stderr)
        return 1

    print(f"Reading EKF  : {EKF_BAG}")
    ekf_t, ekf_x, ekf_y, ekf_z = read_pose_xyz(EKF_BAG)
    print(f"Reading Odom : {ODOM_BAG}")
    odom_t, odom_x, odom_y, odom_z = read_pose_xyz(ODOM_BAG)

    ekf_legs  = detect_legs(ekf_t,  ekf_z)
    odom_legs = detect_legs(odom_t, odom_z)
    print(f"  EKF legs detected  : {len(ekf_legs)}")
    print(f"  Odom legs detected : {len(odom_legs)}")

    def leg_or_none(legs, i):
        return legs[i] if i < len(legs) else None

    ekf_out  = leg_landing_stats('EKF outbound',  ekf_t,  ekf_x,  ekf_y,  ekf_z,
                                 leg_or_none(ekf_legs,  0), OUTBOUND_GOAL)
    ekf_ret  = leg_landing_stats('EKF return',    ekf_t,  ekf_x,  ekf_y,  ekf_z,
                                 leg_or_none(ekf_legs,  1), RETURN_GOAL)
    odom_out = leg_landing_stats('Odom outbound', odom_t, odom_x, odom_y, odom_z,
                                 leg_or_none(odom_legs, 0), OUTBOUND_GOAL)
    odom_ret = leg_landing_stats('Odom return',   odom_t, odom_x, odom_y, odom_z,
                                 leg_or_none(odom_legs, 1), RETURN_GOAL)

    ekf_eff_out  = path_efficiency(ekf_t,  ekf_x,  ekf_y,  leg_or_none(ekf_legs,  0),
                                   (0.0, 0.0), OUTBOUND_GOAL)
    ekf_eff_ret  = path_efficiency(ekf_t,  ekf_x,  ekf_y,  leg_or_none(ekf_legs,  1),
                                   OUTBOUND_GOAL, RETURN_GOAL)
    odom_eff_out = path_efficiency(odom_t, odom_x, odom_y, leg_or_none(odom_legs, 0),
                                   (0.0, 0.0), OUTBOUND_GOAL)
    odom_eff_ret = path_efficiency(odom_t, odom_x, odom_y, leg_or_none(odom_legs, 1),
                                   OUTBOUND_GOAL, RETURN_GOAL)

    free_xy, occ_xy = build_cspace_grid(
        WORLD_OBSTACLES, ROBOT_RADIUS, CSPACE_RES, CSPACE_BOUNDS)
    ekf_clips  = trajectory_clips(ekf_x,  ekf_y,  occ_xy, CSPACE_RES)
    odom_clips = trajectory_clips(odom_x, odom_y, occ_xy, CSPACE_RES)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def fmt_m(v): return f"{v*100:7.2f} cm" if not np.isnan(v) else '   N/A   '
    def fmt_eff(v): return f"{v:6.3f}" if not np.isnan(v) else '  N/A '
    def fmt_s(v): return f"{v:6.2f} s" if not np.isnan(v) else '   N/A  '

    summary = []
    summary.append("Final two-leg mission — EKF-SLAM vs noisy odometry baseline")
    summary.append(f"Outbound goal: {OUTBOUND_GOAL}    Return goal: {RETURN_GOAL}")
    summary.append("=" * 70)
    summary.append(f"{'Metric':<22}{'EKF':>14}{'Odom':>14}{'Δ (Odom-EKF)':>20}")
    summary.append("-" * 70)

    def row(name, ekf_v, odom_v, fmt, sign_label='cm'):
        de = (odom_v - ekf_v) * 100 if sign_label == 'cm' else (odom_v - ekf_v)
        de_str = (f"{de:+8.2f} {sign_label}"
                  if not (np.isnan(ekf_v) or np.isnan(odom_v)) else '    N/A    ')
        summary.append(f"{name:<22}{fmt(ekf_v):>14}{fmt(odom_v):>14}{de_str:>20}")

    row("Outbound landing",   ekf_out['goal_dist'],   odom_out['goal_dist'], fmt_m)
    row("Return landing **",  ekf_ret['goal_dist'],   odom_ret['goal_dist'], fmt_m)
    row("Outbound path eff.", ekf_eff_out, odom_eff_out, fmt_eff, sign_label='')
    row("Return path eff.",   ekf_eff_ret, odom_eff_ret, fmt_eff, sign_label='')
    row("Outbound time",      ekf_out['duration_s'], odom_out['duration_s'],
        fmt_s, sign_label='s')
    row("Return time",        ekf_ret['duration_s'], odom_ret['duration_s'],
        fmt_s, sign_label='s')

    summary.append("-" * 70)
    summary.append("** RETURN LANDING is the primary metric — phantom-centroid proof.")
    summary.append("")
    summary.append(f"Final EKF return pose:   ({ekf_ret['final_x']:.3f}, "
                   f"{ekf_ret['final_y']:.3f}) m")
    summary.append(f"Final Odom return pose:  ({odom_ret['final_x']:.3f}, "
                   f"{odom_ret['final_y']:.3f}) m")
    summary.append("")
    summary.append("Trajectory C-space clip count "
                   f"(robot_radius={ROBOT_RADIUS} m):")
    summary.append(f"  EKF  : {ekf_clips} / {len(ekf_x)} samples inside inflated obstacles")
    summary.append(f"  Odom : {odom_clips} / {len(odom_x)} samples inside inflated obstacles")

    text = "\n".join(summary)
    print(text)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_txt = os.path.join(RESULTS_DIR, 'nav_comparison_summary.txt')
    with open(out_txt, 'w') as f:
        f.write(text + '\n')

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(free_xy[:, 0], free_xy[:, 1], c='lightblue', s=4, zorder=1,
               label='Free Space ($C_{free}$)')
    ax.scatter(occ_xy[:, 0],  occ_xy[:, 1],  c='red',       s=4, zorder=2,
               label='Inflated Obstacles ($C_{obs}$)')

    def plot_legs(label, x, y, legs, color):
        for i, (s, e) in enumerate(legs):
            tag = 'outbound' if i == 0 else 'return'
            style = '-' if i == 0 else '--'
            ax.plot(x[s:e + 1], y[s:e + 1], style, color=color, lw=1.6,
                    alpha=0.95, zorder=3, label=f'{label} {tag}')
            ax.plot(x[e], y[e], 's', color=color, markersize=10,
                    markeredgecolor='black', zorder=5)

    plot_legs('EKF',  ekf_x,  ekf_y,  ekf_legs,  'C0')
    plot_legs('Odom', odom_x, odom_y, odom_legs, 'C3')

    ax.plot(*OUTBOUND_GOAL, '*', color='gold', markersize=22,
            markeredgecolor='black', zorder=6, label='Outbound goal (pad)')
    ax.plot(*RETURN_GOAL, 'P', color='lime', markersize=18,
            markeredgecolor='black', zorder=6, label='Return goal (origin)')

    ax.set_xlim(*CSPACE_BOUNDS[0:2])
    ax.set_ylim(*CSPACE_BOUNDS[2:4])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    title = ('Two-leg mission — EKF-SLAM vs noisy odometry\n'
             f"Return landing — EKF: {fmt_m(ekf_ret['goal_dist']).strip()}, "
             f"Odom: {fmt_m(odom_ret['goal_dist']).strip()}")
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    out_png = os.path.join(RESULTS_DIR, 'nav_comparison_trajectories.png')
    fig.savefig(out_png, dpi=150)
    print(f"\nSaved plot:    {out_png}")
    print(f"Saved summary: {out_txt}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
