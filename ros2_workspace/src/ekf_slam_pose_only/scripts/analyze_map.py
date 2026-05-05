#!/usr/bin/env python3
"""Visualize the latest EKF-run bag: occupancy map + raw scans + extracted lines.

Inputs:
    results/gazebo_full_nav_ekf_bag/   (must contain /map, /ekf_pose, /crazyflie/scan)

Outputs:
    results/occupancy_map.png
    results/scan_endpoints.png
    results/line_extraction.png
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py


PROJECT_DIR = '/home/renix/EKF-SLAM-Autonomous-Crazyflie'
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
BAG = os.path.join(RESULTS_DIR, 'gazebo_full_nav_ekf_bag')

GOAL = (0.8, 0.0)

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
    """Reconstruct an occupancy grid from the trajectory and laser scans.

    Mirrors OceanMapper.update_from_lidar: each beam endpoint marks its cell
    occupied (+L_OCC), and cells along the ray from drone to endpoint are
    marked free (+L_FREE) using a Bresenham-style step. Returns
    (prob_map, x_min, y_min, resolution)."""
    sx, sy, _, st = beam_endpoints_world(poses, scans)
    if sx.size == 0:
        return None

    # Tight grid bounds: span the trajectory and the endpoint cloud, plus a
    # small margin so we don't clip walls right at the edge.
    all_x = np.concatenate([poses['x'], sx])
    all_y = np.concatenate([poses['y'], sy])
    x_min = float(all_x.min()) - margin
    x_max = float(all_x.max()) + margin
    y_min = float(all_y.min()) - margin
    y_max = float(all_y.max()) + margin

    nx = int(np.ceil((x_max - x_min) / resolution))
    ny = int(np.ceil((y_max - y_min) / resolution))
    log_odds = np.zeros((ny, nx))

    # Pose lookup by timestamp — same nearest-neighbour scheme as
    # beam_endpoints_world; we need the drone position for ray tracing.
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

        # Mark ray cells as free (cells the beam passed through). Step about
        # 0.8 of a cell at a time so we don't skip cells on diagonal beams.
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

        # Mark the hit cell as occupied — done after the ray pass so an
        # endpoint is never overwritten by its own ray's free update.
        ix = int((hx - x_min) / resolution)
        iy = int((hy - y_min) / resolution)
        if 0 <= ix < nx and 0 <= iy < ny:
            log_odds[iy, ix] += L_OCC
            log_odds[iy, ix]  = min(log_odds[iy, ix], MAX_LOG_ODDS)

    prob = 1.0 / (1.0 + np.exp(-log_odds))
    return prob, x_min, y_min, resolution


def classify_dot_grid(poses, scans, resolution=OCC_RES):
    """Return (free_xy, occ_xy, x_min, y_min, res) — the C-space dot grid
    in the midterm Problem 3 / Problem 4 style. A cell is 'occupied' if its
    log-odds ended above 0 (more endpoints than ray-throughs) and 'free' if
    below 0. Cells that no ray ever touched stay 'unknown' and get no dot."""
    result = build_occupancy_from_scans(poses, scans, resolution=resolution)
    if result is None:
        return None
    prob, x_min, y_min, res = result
    ny, nx = prob.shape

    iy, ix = np.indices((ny, nx))
    cx = x_min + (ix + 0.5) * res
    cy = y_min + (iy + 0.5) * res

    # Cells where any update happened: prob != 0.5 (log_odds != 0).
    touched = np.abs(prob - 0.5) > 1e-9
    occ_mask = touched & (prob > 0.5)
    free_mask = touched & (prob < 0.5)

    free_xy = np.column_stack([cx[free_mask], cy[free_mask]])
    occ_xy = np.column_stack([cx[occ_mask], cy[occ_mask]])
    return free_xy, occ_xy, x_min, y_min, res, nx, ny


def _draw_dot_grid(ax, free_xy, occ_xy, *, free_size=6, occ_size=14,
                   free_alpha=1.0, occ_alpha=1.0, zorder=1, with_label=True):
    """Scatter the discrete C-space grid. Matches midterm Part1.visualize_c_space:
       lightblue free dots, red occupied dots."""
    free_lbl = 'Free Space ($C_{free}$)' if with_label else None
    occ_lbl = 'Obstacle Space ($C_{obs}$)' if with_label else None
    if free_xy.size:
        ax.scatter(free_xy[:, 0], free_xy[:, 1], c='lightblue',
                   s=free_size, alpha=free_alpha, zorder=zorder,
                   label=free_lbl)
    if occ_xy.size:
        ax.scatter(occ_xy[:, 0], occ_xy[:, 1], c='red',
                   s=occ_size, alpha=occ_alpha, zorder=zorder + 1,
                   label=occ_lbl)


def plot_occupancy_map(_unused_occ, poses, scans, out_path):
    """Discrete C-space dot grid (midterm Problem 3 style): each cell is a
    coloured dot — lightblue=free (rays passed through), red=occupied (a beam
    endpoint landed there). Trajectory overlaid as a blue line."""
    grid = classify_dot_grid(poses, scans, resolution=OCC_RES)
    if grid is None:
        print("  WARNING: no scan endpoints; skipping occupancy plot.")
        return
    free_xy, occ_xy, x_min, y_min, res, nx, ny = grid

    fig, ax = plt.subplots(figsize=(10, 8))
    _draw_dot_grid(ax, free_xy, occ_xy, free_size=6, occ_size=14, zorder=1)

    if poses['x'].size:
        ax.plot(poses['x'], poses['y'], '-', color='blue',
                lw=1.5, alpha=0.85, zorder=3, label='EKF trajectory')
        ax.plot(poses['x'][0], poses['y'][0], 'go',
                markersize=10, zorder=5, label='Start')
    ax.plot(GOAL[0], GOAL[1], '*', color='gold',
            markersize=18, markeredgecolor='black', zorder=5, label='Goal')

    ax.set_xlim(x_min, x_min + nx * res)
    ax.set_ylim(y_min, y_min + ny * res)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(f'Discretized Occupancy Grid (C-space style, {res:.2f} m cells)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}  (grid: {nx}x{ny} cells, "
          f"free={len(free_xy)}, occupied={len(occ_xy)})")


def beam_endpoints_world(poses, scans):
    """For each scan, look up the nearest pose in time and project the four
    beam endpoints into the world frame. Returns arrays of equal length:
        endpoints_x, endpoints_y, beam_index (0..3), scan_time
    Beams that hit max-range or come back NaN are dropped."""
    if poses['t'].size == 0 or scans['t'].size == 0:
        return (np.array([]), np.array([]), np.array([], dtype=int),
                np.array([]))

    pt = poses['t']
    sx, sy, si, st = [], [], [], []

    for s_t, ranges in zip(scans['t'], scans['ranges']):
        # nearest-neighbour pose
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


def plot_scan_endpoints(poses, scans, out_path):
    sx, sy, _, st = beam_endpoints_world(poses, scans)

    fig, ax = plt.subplots(figsize=(9, 9))

    if sx.size:
        # Color by elapsed time — early=blue, late=red
        t0 = st.min()
        t_norm = (st - t0) / max(st.max() - t0, 1e-6)
        sc = ax.scatter(sx, sy, c=t_norm, cmap='coolwarm', s=8, alpha=0.8)
        cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label('time (early → late)')

    if poses['x'].size:
        ax.plot(poses['x'], poses['y'], '-', color='black',
                lw=2.0, label='EKF trajectory')
        ax.plot(poses['x'][0], poses['y'][0], 'o', color='green',
                markersize=8, label='start')
    ax.plot(GOAL[0], GOAL[1], '*', color='gold',
            markersize=14, markeredgecolor='black', label='goal')

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Raw Laser Scan Endpoints in World Frame')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Line extraction: cluster_points -> split_and_merge (IEPF) -> TLS -> segment
#
# Direct port of midterm Problem 2 (Downloads/MIDTERM/Problem 2/Problem2.py).
# Points are processed in the sequential order they were generated (one scan
# at a time, beams in fixed angular order) so that consecutive distance is a
# meaningful proxy for "this point and the next likely came off the same
# surface". A point is just a point — beams don't matter.
# ---------------------------------------------------------------------------

CLUSTER_DIST = 0.30      # m, sequential gap that breaks a cluster
IEPF_THRESH = 0.08       # m, max perpendicular distance before splitting at a corner
MIN_CLUSTER_PTS = 3      # min points to bother fitting a line


def cluster_points(points, distance_threshold):
    """Walk the points in order and split a new cluster whenever consecutive
    Euclidean distance exceeds distance_threshold."""
    if len(points) == 0:
        return []
    clusters = []
    current = [points[0]]
    for i in range(1, len(points)):
        d = np.linalg.norm(points[i] - points[i - 1])
        if d > distance_threshold:
            clusters.append(np.array(current))
            current = []
        current.append(points[i])
    if current:
        clusters.append(np.array(current))
    return clusters


def split_and_merge(cluster, threshold):
    """Iterative End-Point Fit: connect the cluster's first and last points
    with a line, find the point with the largest orthogonal distance to that
    line, and if it exceeds the threshold split the cluster there and
    recurse. Otherwise the cluster is one straight segment."""
    if len(cluster) <= 2:
        return [cluster]

    start_p = cluster[0]
    end_p = cluster[-1]
    line_vec = end_p - start_p
    line_len = np.linalg.norm(line_vec)
    if line_len == 0.0:
        return [cluster]

    # Cross-product magnitude divided by line length = perpendicular distance.
    point_vecs = cluster - start_p
    cross = np.abs(point_vecs[:, 0] * line_vec[1] - point_vecs[:, 1] * line_vec[0])
    distances = cross / line_len

    max_idx = int(np.argmax(distances))
    max_dist = float(distances[max_idx])

    if max_dist > threshold:
        # Both halves share the corner point so neither side loses information.
        left = split_and_merge(cluster[:max_idx + 1], threshold)
        right = split_and_merge(cluster[max_idx:], threshold)
        return left + right
    return [cluster]


def total_least_squares(cluster):
    """TLS line fit via eigendecomposition. Returns (n, n0) so n . p = n0
    is the line. Smallest eigenvector of the centred covariance is the
    normal."""
    x = cluster[:, 0]
    y = cluster[:, 1]
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))

    A = np.array([
        [np.mean(x * x), np.mean(x * y)],
        [np.mean(x * y), np.mean(y * y)],
    ])
    b = np.array([[mean_x], [mean_y]])
    D = A - b @ b.T
    eigvals, eigvecs = np.linalg.eigh(D)
    n = eigvecs[:, int(np.argmin(eigvals))]
    n0 = float((b.T @ n)[0])
    return n, n0


def get_line_segment(cluster, n, n0):
    """Project the cluster onto the line direction; the extreme projections
    give a bounded segment that doesn't extrapolate beyond the data."""
    direction = np.array([n[1], -n[0]])
    projections = cluster @ direction
    min_idx = int(np.argmin(projections))
    max_idx = int(np.argmax(projections))

    def project(p):
        dist = float(p @ n - n0)
        return p - dist * n

    return project(cluster[min_idx]), project(cluster[max_idx])


def plot_line_extraction(poses, scans, out_path):
    sx, sy, si, st = beam_endpoints_world(poses, scans)

    fig, ax = plt.subplots(figsize=(9, 9))

    # Discretized environment underneath so the extracted lines visibly
    # correspond to walls. Lower opacity so the red lines stay dominant.
    grid = classify_dot_grid(poses, scans, resolution=OCC_RES)
    if grid is not None:
        free_xy, occ_xy, *_ = grid
        _draw_dot_grid(ax, free_xy, occ_xy, free_size=4, occ_size=8,
                       free_alpha=0.35, occ_alpha=0.45, zorder=1)

    if sx.size:
        ax.scatter(sx, sy, c='dimgray', s=4, alpha=0.5, zorder=2,
                   label='Noisy Point Cloud')

        # The midterm's cluster_points assumes consecutive points came off
        # the same surface — true for a single dense angular sweep, but our
        # data is 377 sparse 4-beam scans where adjacent entries in the
        # array typically point at different walls. To recover meaningful
        # sequential order we feed each beam direction as its own ordered
        # stream (one beam slides continuously along one wall as the drone
        # moves). All resulting clusters from all beams pool together for
        # IEPF/TLS — this is NOT fitting per-beam, it's just using beam
        # index to recover the "consecutive == same surface" property the
        # midterm relies on. A single wall seen by two beams will produce
        # two clusters that both fit the same line — fine.
        points = np.column_stack([sx, sy])
        initial = []
        for beam in np.unique(si):
            mask = si == beam
            order = np.argsort(st[mask])
            stream = points[mask][order]
            initial.extend(cluster_points(stream, CLUSTER_DIST))

        refined = []
        for c in initial:
            refined.extend(split_and_merge(c, IEPF_THRESH))

        segments = []
        for c in refined:
            if len(c) < MIN_CLUSTER_PTS:
                continue
            n, n0 = total_least_squares(c)
            p1, p2 = get_line_segment(c, n, n0)
            segments.append((p1, p2))

        for i, (p1, p2) in enumerate(segments):
            lbl = 'Reconstructed Segment' if i == 0 else None
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color='red', linewidth=3, zorder=4, label=lbl)

        print(f"  initial clusters: {len(initial)}  "
              f"after IEPF: {len(refined)}  fitted segments: {len(segments)}")

    if poses['x'].size:
        ax.plot(poses['x'], poses['y'], '-', color='black', lw=1.5,
                zorder=3, label='EKF trajectory')
        ax.plot(poses['x'][0], poses['y'][0], 'o', color='green',
                markersize=8, zorder=5, label='start')
    ax.plot(GOAL[0], GOAL[1], '*', color='gold',
            markersize=14, markeredgecolor='black', zorder=5, label='goal')

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Extracted Wall/Obstacle Lines (over discretized C-space)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    if not os.path.isdir(BAG):
        print(f"ERROR: bag directory not found: {BAG}", file=sys.stderr)
        return 1

    print(f"Reading bag: {BAG}")
    last_map, poses, scans = read_bag(BAG)
    print(f"  poses : {poses['t'].size}")
    print(f"  scans : {scans['t'].size}")
    print(f"  map   : {'present' if last_map is not None else 'MISSING'}")

    if scans['t'].size == 0:
        print("ERROR: no /crazyflie/scan messages in bag; cannot render "
              "occupancy plot", file=sys.stderr)
        return 1

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_occupancy_map(last_map, poses, scans,
                       os.path.join(RESULTS_DIR, 'occupancy_map.png'))
    plot_scan_endpoints(poses, scans,
                        os.path.join(RESULTS_DIR, 'scan_endpoints.png'))
    plot_line_extraction(poses, scans,
                         os.path.join(RESULTS_DIR, 'line_extraction.png'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
