#!/usr/bin/env python3
"""Render the headline-run map figures for both Run 2 bags.

Inputs (Run 2 headline bags, each contains /map, /ekf_pose, /crazyflie/scan,
and the EKF bag also contains /ekf_slam/debug/landmark_lines):
    analysis/headline_bags/final_ekf_bag_run2/
    analysis/headline_bags/final_odom_bag_run2/

Outputs (all in results/figures/):
    map_occupancy_ekf.png        — dot-grid occupancy + trajectory, EKF Run 2
    map_occupancy_odom.png       — dot-grid occupancy + trajectory, Odom Run 2
    map_scan_endpoints_ekf.png   — scan endpoints + GT overlay, EKF Run 2
    map_scan_endpoints_odom.png  — scan endpoints + GT overlay, Odom Run 2
    map_landmarks_ekf.png        — final EKF tracked landmarks + GT overlay

All bag-comparison figures share COMMON_BOUNDS so wall thickness / smear is
directly eyeball-comparable. Ground-truth obstacle/wall outlines (from the
SDF) overlay the scan-endpoint and landmark figures so "is dispersion real
or drift?" reads visually.
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
HEADLINE_BAGS_DIR = os.path.join(PROJECT_DIR, 'analysis', 'headline_bags')
HEADLINE_EKF_BAG  = os.path.join(HEADLINE_BAGS_DIR, 'final_ekf_bag_run2')
HEADLINE_ODOM_BAG = os.path.join(HEADLINE_BAGS_DIR, 'final_odom_bag_run2')
FIGURES_DIR = os.path.join(PROJECT_DIR, 'results', 'figures')

OUTBOUND_GOAL = (0.8, 0.0)
RETURN_GOAL   = (0.0, 0.0)

# Shared axis window for the two occupancy maps and the two scan-endpoint
# maps so wall-thickness / smear is visually comparable across bags.
COMMON_BOUNDS = (-2.2, 2.2, -2.2, 2.2)

# Ground-truth boxes from crazyflie_world.sdf (4 walls + 4 obstacles).
# Format: (xmin, xmax, ymin, ymax, kind). Landing pad is intentionally
# absent — it sits at z=0–0.05 m, below the multiranger scan plane, so
# drawing it would mis-imply scans should hit it.
WORLD_GEOMETRY_BOXES = [
    (1.95,  2.05, -2.0,   2.0,  'wall'),    # wall_east
    (-2.05, -1.95, -2.0,  2.0,  'wall'),    # wall_west
    (-2.0,  2.0,   1.95,  2.05, 'wall'),    # wall_north
    (-2.0,  2.0,  -2.05, -1.95, 'wall'),    # wall_south
    (0.4,   0.6,   0.4,   0.6,  'obstacle'),  # obstacle_1
    (-0.6, -0.4,   0.2,   0.4,  'obstacle'),  # obstacle_2
    (0.1,   0.3,  -0.7,  -0.5,  'obstacle'),  # obstacle_3
    (0.25,  0.55, -0.15,  0.15, 'obstacle'),  # obstacle_4
]

# Matches the EKF/mapper convention: [back, right, front, left]
BEAM_BEARINGS = np.array([np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0])
BEAM_NAMES = ['back', 'right', 'front', 'left']


def open_reader(bag_path):
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    return reader, type_map


def read_bag(bag_path):
    """Single pass over the bag — returns last map, all poses, all scans."""
    reader, type_map = open_reader(bag_path)

    needed = ['/map', '/ekf_pose', '/crazyflie/scan']
    for topic in needed:
        if topic not in type_map:
            raise RuntimeError(f"{topic} not found in {bag_path}")

    MapType = get_message(type_map['/map'])
    PoseType = get_message(type_map['/ekf_pose'])
    ScanType = get_message(type_map['/crazyflie/scan'])

    last_map = None
    pose_t, pose_x, pose_y, pose_yaw = [], [], [], []
    scan_t, scan_ranges = [], []

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic == '/map':
            last_map = deserialize_message(data, MapType)
        elif topic == '/ekf_pose':
            msg = deserialize_message(data, PoseType)
            p = msg.pose.position
            q = msg.pose.orientation
            yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            pose_t.append(t_ns * 1e-9)
            pose_x.append(p.x)
            pose_y.append(p.y)
            pose_yaw.append(yaw)
        elif topic == '/crazyflie/scan':
            msg = deserialize_message(data, ScanType)
            scan_t.append(t_ns * 1e-9)
            scan_ranges.append(np.asarray(msg.ranges, dtype=float))

    poses = {
        't':   np.array(pose_t),
        'x':   np.array(pose_x),
        'y':   np.array(pose_y),
        'yaw': np.array(pose_yaw),
    }
    scans = {
        't':      np.array(scan_t),
        'ranges': scan_ranges,
    }
    return last_map, poses, scans


# ---------------------------------------------------------------------------
# Reconstructed occupancy grid (not the bag's /map)
#
# The bag's /map has scattered "partial occupancy" cells where the drone flew
# (artefacts of the ROS mapper's accumulator) and a 40x20 m origin offset
# that bloats the visualization. Build our own grid from the same raw scans
# the EKF and mapper saw, using the log-odds Bayesian update from
# Downloads/MIDTERM/Problem 4/Problem4_mapping.py:
#   - Hit cell      : += L_OCC  = log(0.70 / 0.30)
#   - Ray-traced    : += L_FREE = log(0.35 / 0.65)   (cells the beam passed through)
# Probability is sigmoid(log_odds). Visualized with the same midterm style:
# a 'Reds' heatmap masked below threshold so free space stays clean.
# ---------------------------------------------------------------------------

OCC_RES = 0.05               # m, finer cells than the bag's /map (5 cm)
L_OCC = np.log(0.70 / 0.30)  # +0.847
L_FREE = np.log(0.35 / 0.65) # -0.619
MAX_LOG_ODDS = 30.0


def build_occupancy_from_scans(poses, scans, resolution=OCC_RES, margin=0.5):
    """Reconstruct an occupancy grid from the trajectory and laser scans."""
    sx, sy, _, st = beam_endpoints_world(poses, scans)
    if sx.size == 0:
        return None

    all_x = np.concatenate([poses['x'], sx])
    all_y = np.concatenate([poses['y'], sy])
    x_min = float(all_x.min()) - margin
    x_max = float(all_x.max()) + margin
    y_min = float(all_y.min()) - margin
    y_max = float(all_y.max()) + margin

    nx = int(np.ceil((x_max - x_min) / resolution))
    ny = int(np.ceil((y_max - y_min) / resolution))
    log_odds = np.zeros((ny, nx))

    pt = poses['t']
    px = poses['x']
    py = poses['y']

    def _pose_at(t):
        idx = int(np.searchsorted(pt, t))
        if idx >= len(pt):
            idx = len(pt) - 1
        elif idx > 0 and abs(pt[idx - 1] - t) < abs(pt[idx] - t):
            idx -= 1
        return px[idx], py[idx]

    for hx, hy, t in zip(sx, sy, st):
        rx, ry = _pose_at(t)

        ray_len = np.hypot(hx - rx, hy - ry)
        n_samples = max(1, int(ray_len / (resolution * 0.8)))
        for k in range(n_samples):
            t_ = k / n_samples
            fx = rx + t_ * (hx - rx)
            fy = ry + t_ * (hy - ry)
            jx = int((fx - x_min) / resolution)
            jy = int((fy - y_min) / resolution)
            if 0 <= jx < nx and 0 <= jy < ny:
                log_odds[jy, jx] += L_FREE
                log_odds[jy, jx]  = max(log_odds[jy, jx], -MAX_LOG_ODDS)

        ix = int((hx - x_min) / resolution)
        iy = int((hy - y_min) / resolution)
        if 0 <= ix < nx and 0 <= iy < ny:
            log_odds[iy, ix] += L_OCC
            log_odds[iy, ix]  = min(log_odds[iy, ix], MAX_LOG_ODDS)

    prob = 1.0 / (1.0 + np.exp(-log_odds))
    return prob, x_min, y_min, resolution


def classify_dot_grid(poses, scans, resolution=OCC_RES):
    """Return (free_xy, occ_xy, x_min, y_min, res, nx, ny)."""
    result = build_occupancy_from_scans(poses, scans, resolution=resolution)
    if result is None:
        return None
    prob, x_min, y_min, res = result
    ny, nx = prob.shape

    iy, ix = np.indices((ny, nx))
    cx = x_min + (ix + 0.5) * res
    cy = y_min + (iy + 0.5) * res

    touched = np.abs(prob - 0.5) > 1e-9
    occ_mask = touched & (prob > 0.5)
    free_mask = touched & (prob < 0.5)

    free_xy = np.column_stack([cx[free_mask], cy[free_mask]])
    occ_xy = np.column_stack([cx[occ_mask], cy[occ_mask]])
    return free_xy, occ_xy, x_min, y_min, res, nx, ny


def _draw_dot_grid(ax, free_xy, occ_xy, *, free_size=6, occ_size=14,
                   free_alpha=1.0, occ_alpha=1.0, zorder=1, with_label=True):
    """Lightblue free dots, red occupied dots — midterm Problem 3 style."""
    free_lbl = 'Free Space ($C_{free}$)' if with_label else None
    occ_lbl = 'Obstacle Space ($C_{obs}$)' if with_label else None
    if free_xy.size:
        ax.scatter(free_xy[:, 0], free_xy[:, 1], c='lightgray',
                   s=free_size, alpha=free_alpha, zorder=zorder,
                   label=free_lbl)
    if occ_xy.size:
        ax.scatter(occ_xy[:, 0], occ_xy[:, 1], c='red',
                   s=occ_size, alpha=occ_alpha, zorder=zorder + 1,
                   label=occ_lbl)


def _draw_goals(ax):
    """Both leg goals — filled green squares; return has a heavier black
    border to read as the final/landing target."""
    ax.plot(OUTBOUND_GOAL[0], OUTBOUND_GOAL[1], 's',
            markerfacecolor='green', markeredgecolor='black',
            markeredgewidth=0.8, markersize=8, zorder=6,
            label='Outbound goal (pad)')
    ax.plot(RETURN_GOAL[0], RETURN_GOAL[1], 's',
            markerfacecolor='green', markeredgecolor='black',
            markeredgewidth=2.5, markersize=8, zorder=6,
            label='Return goal (origin)')


def _draw_world_geometry(ax, *, alpha=0.9, zorder=4):
    """Overlay GT room outlines from the SDF.

    Walls: solid black, lw=1.5. Obstacles: dashed gray, lw=1.0.
    A single legend entry "Ground truth (SDF)" is attached to the first
    wall edge drawn; subsequent edges share styling but no label.
    """
    labelled = False
    for xmin, xmax, ymin, ymax, kind in WORLD_GEOMETRY_BOXES:
        xs = [xmin, xmax, xmax, xmin, xmin]
        ys = [ymin, ymin, ymax, ymax, ymin]
        if kind == 'wall':
            color, ls, lw = 'black', '-', 1.5
        else:
            color, ls, lw = 'dimgray', '--', 1.0
        lbl = 'Ground truth (SDF)' if not labelled else None
        labelled = True
        ax.plot(xs, ys, color=color, linestyle=ls, linewidth=lw,
                alpha=alpha, zorder=zorder, label=lbl)


def read_final_landmark_lines(bag_path):
    """Single pass — return the final /ekf_slam/debug/landmark_lines
    MarkerArray as a list of (p1, p2) numpy-pair tuples in world frame.

    Skips the DELETEALL sentinel marker the EKF emits at index 0, and
    any non-LINE_STRIP markers. Multi-segment LINE_STRIPs are walked as
    consecutive pairs (today the EKF emits 2-point strips, but this is
    defensive)."""
    reader, type_map = open_reader(bag_path)
    topic = '/ekf_slam/debug/landmark_lines'
    if topic not in type_map:
        raise RuntimeError(f"{topic} not found in {bag_path}")
    MarkerArray = get_message(type_map[topic])

    last_msg = None
    while reader.has_next():
        topic_, data, _t_ns = reader.read_next()
        if topic_ == topic:
            last_msg = deserialize_message(data, MarkerArray)

    if last_msg is None:
        return []

    LINE_STRIP, ADD = 4, 0
    segments = []
    for m in last_msg.markers:
        if m.type != LINE_STRIP or m.action != ADD or len(m.points) < 2:
            continue
        for i in range(len(m.points) - 1):
            p1 = np.array([m.points[i].x,     m.points[i].y])
            p2 = np.array([m.points[i + 1].x, m.points[i + 1].y])
            segments.append((p1, p2))
    return segments


def read_final_corner_markers(bag_path):
    """Single pass — return the final /ekf_slam/debug/landmark_corners
    MarkerArray as a list of (x, y) tuples in world frame.

    Skips the DELETEALL sentinel (action=3); MODIFY (2) and ADD (0) both
    pass through, since the EKF emits MODIFY for the persistent corner set."""
    reader, type_map = open_reader(bag_path)
    topic = '/ekf_slam/debug/landmark_corners'
    if topic not in type_map:
        return []
    MarkerArray = get_message(type_map[topic])

    last_msg = None
    while reader.has_next():
        topic_, data, _t_ns = reader.read_next()
        if topic_ == topic:
            last_msg = deserialize_message(data, MarkerArray)

    if last_msg is None:
        return []

    DELETEALL = 3
    corners = []
    for m in last_msg.markers:
        if m.action == DELETEALL:
            continue
        corners.append((m.pose.position.x, m.pose.position.y))
    return corners


def plot_occupancy_map(label, poses, scans, out_path):
    """Discrete C-space dot grid + trajectory + both leg goals."""
    grid = classify_dot_grid(poses, scans, resolution=OCC_RES)
    if grid is None:
        print(f"  WARNING ({label}): no scan endpoints; skipping occupancy plot.")
        return
    free_xy, occ_xy, *_ = grid

    fig, ax = plt.subplots(figsize=(10, 8))
    _draw_dot_grid(ax, free_xy, occ_xy, free_size=6, occ_size=14, zorder=1)

    if poses['x'].size:
        ax.plot(poses['x'], poses['y'], '-', color='rebeccapurple',
                lw=1.2, alpha=0.9, zorder=3, label='Trajectory')
        ax.plot(poses['x'][0], poses['y'][0], 'o', color='rebeccapurple',
                markersize=6, zorder=5, label='Start')
    _draw_goals(ax)

    ax.set_xlim(COMMON_BOUNDS[0], COMMON_BOUNDS[1])
    ax.set_ylim(COMMON_BOUNDS[2], COMMON_BOUNDS[3])
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(f'Discretized Occupancy Grid — {label} '
                 f'({OCC_RES:.2f} m cells)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}  (free={len(free_xy)}, occupied={len(occ_xy)})")


def beam_endpoints_world(poses, scans):
    """For each scan, look up the nearest pose in time and project the four
    beam endpoints into the world frame. Returns:
        endpoints_x, endpoints_y, beam_index (0..3), scan_time
    Beams that hit max-range or come back NaN are dropped."""
    if poses['t'].size == 0 or scans['t'].size == 0:
        return (np.array([]), np.array([]), np.array([], dtype=int),
                np.array([]))

    pt = poses['t']
    sx, sy, si, st = [], [], [], []

    for s_t, ranges in zip(scans['t'], scans['ranges']):
        idx = int(np.searchsorted(pt, s_t))
        if idx >= len(pt):
            idx = len(pt) - 1
        elif idx > 0 and abs(pt[idx - 1] - s_t) < abs(pt[idx] - s_t):
            idx -= 1

        x = poses['x'][idx]
        y = poses['y'][idx]
        yaw = poses['yaw'][idx]

        if len(ranges) < 4:
            continue
        for i in range(4):
            r = ranges[i]
            if not np.isfinite(r) or r <= 0.05 or r > 3.4:
                continue
            ang = yaw + BEAM_BEARINGS[i]
            sx.append(x + r * np.cos(ang))
            sy.append(y + r * np.sin(ang))
            si.append(i)
            st.append(s_t)

    return np.array(sx), np.array(sy), np.array(si, dtype=int), np.array(st)


def plot_scan_endpoints(label, poses, scans, out_path):
    sx, sy, _, st = beam_endpoints_world(poses, scans)

    fig, ax = plt.subplots(figsize=(9, 9))

    if sx.size:
        t0 = st.min()
        t_norm = (st - t0) / max(st.max() - t0, 1e-6)
        sc = ax.scatter(sx, sy, c=t_norm, cmap='coolwarm', s=8, alpha=0.8)
        cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label('time (early → late)')

    if poses['x'].size:
        ax.plot(poses['x'], poses['y'], '-', color='rebeccapurple',
                lw=1.2, label='Trajectory')
        ax.plot(poses['x'][0], poses['y'][0], 'o', color='rebeccapurple',
                markersize=6, label='Start')
    _draw_world_geometry(ax, alpha=0.9, zorder=4)
    _draw_goals(ax)

    ax.set_xlim(COMMON_BOUNDS[0], COMMON_BOUNDS[1])
    ax.set_ylim(COMMON_BOUNDS[2], COMMON_BOUNDS[3])
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'Raw Laser Scan Endpoints in World Frame — {label}')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_ekf_landmarks(label, bag_path, out_path):
    """Final state of /ekf_slam/debug/landmark_lines, plus GT overlay.

    Each landmark is a LINE_STRIP with 2 points; rendered as a red line
    with small dots at each endpoint so the reader can see where the EKF
    thinks each tracked surface starts and ends relative to the GT walls.
    """
    segments = read_final_landmark_lines(bag_path)
    corners  = read_final_corner_markers(bag_path)

    fig, ax = plt.subplots(figsize=(9, 9))
    _draw_world_geometry(ax, alpha=0.4, zorder=2)

    if not segments:
        ax.text(0.5, 0.5, 'no EKF landmarks in final MarkerArray',
                transform=ax.transAxes, ha='center', va='center',
                color='dimgray', fontsize=12)
        print(f"  WARNING ({label}): final landmark_lines MarkerArray was empty.")

    for i, (p1, p2) in enumerate(segments):
        line_lbl = 'EKF tracked landmark' if i == 0 else None
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color='red', linewidth=2, zorder=4, label=line_lbl)

    for i, (cx, cy) in enumerate(corners):
        corner_lbl = 'EKF corner landmark' if i == 0 else None
        ax.plot(cx, cy, marker='o', color='orange', markersize=8,
                markeredgecolor='black', markeredgewidth=0.8,
                linestyle='None', zorder=6, label=corner_lbl)

    _draw_goals(ax)

    ax.set_xlim(COMMON_BOUNDS[0], COMMON_BOUNDS[1])
    ax.set_ylim(COMMON_BOUNDS[2], COMMON_BOUNDS[3])
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'EKF tracked landmarks (lines + corners) — final state — {label}')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}  ({len(segments)} lines, {len(corners)} corners)")


def main():
    bags = [
        (HEADLINE_EKF_BAG,  'EKF Run 2',  'ekf'),
        (HEADLINE_ODOM_BAG, 'Odom Run 2', 'odom'),
    ]

    for bag_path, _label, _suffix in bags:
        if not os.path.isdir(bag_path):
            print(f"ERROR: bag directory not found: {bag_path}", file=sys.stderr)
            return 1

    os.makedirs(FIGURES_DIR, exist_ok=True)

    for bag_path, label, suffix in bags:
        print(f"Reading bag ({label}): {bag_path}")
        _last_map, poses, scans = read_bag(bag_path)
        print(f"  poses : {poses['t'].size}")
        print(f"  scans : {scans['t'].size}")

        if scans['t'].size == 0:
            print(f"ERROR ({label}): no /crazyflie/scan messages; skipping.",
                  file=sys.stderr)
            continue

        plot_occupancy_map(
            label, poses, scans,
            os.path.join(FIGURES_DIR, f'map_occupancy_{suffix}.png'))
        plot_scan_endpoints(
            label, poses, scans,
            os.path.join(FIGURES_DIR, f'map_scan_endpoints_{suffix}.png'))

    print(f"Reading EKF landmarks: {HEADLINE_EKF_BAG}")
    plot_ekf_landmarks(
        'EKF Run 2', HEADLINE_EKF_BAG,
        os.path.join(FIGURES_DIR, 'map_landmarks_ekf.png'))

    return 0


if __name__ == '__main__':
    sys.exit(main())
