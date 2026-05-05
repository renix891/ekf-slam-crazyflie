#!/usr/bin/env python3
"""Side-by-side reconstructed occupancy maps for EKF vs odometry runs.

Reuses the dot-grid classifier and renderer from analyze_map.py so both
sides share resolution, colour scheme, and inflation conventions. The
visual difference (wall thickness, smear) is the point of the figure.

Output: results/ekf_vs_odom_map_comparison.png
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# analyze_map.py lives next to this script and already opens bags, builds
# the log-odds grid, and renders the midterm-style dot grid.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_map import (
    read_bag, classify_dot_grid, _draw_dot_grid, GOAL, OCC_RES,
)

PROJECT_DIR = '/home/renix/EKF-SLAM-Autonomous-Crazyflie'
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
EKF_BAG = os.path.join(RESULTS_DIR, 'gazebo_full_nav_ekf_bag')
ODOM_BAG = os.path.join(RESULTS_DIR, 'gazebo_full_nav_odom_only_bag')
OUT = os.path.join(RESULTS_DIR, 'ekf_vs_odom_map_comparison.png')

# Force a common axis window so the two subplots are visually comparable —
# without this, each subplot would auto-scale to its own data extent and
# wall thickness would be hard to eyeball-compare.
COMMON_BOUNDS = (-2.2, 2.2, -2.2, 2.2)


def panel(ax, bag_path, title):
    if not os.path.isdir(bag_path):
        ax.set_title(f'{title}\n(missing: {os.path.basename(bag_path)})')
        ax.set_aspect('equal')
        return
    _, poses, scans = read_bag(bag_path)
    grid = classify_dot_grid(poses, scans, resolution=OCC_RES)
    if grid is None:
        ax.set_title(f'{title}\n(no scan endpoints)')
        ax.set_aspect('equal')
        return
    free_xy, occ_xy, *_ = grid

    _draw_dot_grid(ax, free_xy, occ_xy, free_size=4, occ_size=10, zorder=1)
    if poses['x'].size:
        ax.plot(poses['x'], poses['y'], '-', color='blue',
                lw=1.4, alpha=0.85, zorder=3, label='Trajectory')
        ax.plot(poses['x'][0], poses['y'][0], 'go',
                markersize=9, zorder=5, label='Start')
    ax.plot(GOAL[0], GOAL[1], '*', color='gold', markersize=16,
            markeredgecolor='black', zorder=5, label='Goal')

    ax.set_xlim(COMMON_BOUNDS[0], COMMON_BOUNDS[1])
    ax.set_ylim(COMMON_BOUNDS[2], COMMON_BOUNDS[3])
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'{title}\n'
                 f'free={len(free_xy)}, occupied={len(occ_xy)}')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right', fontsize=8)


def main():
    fig, (ax_e, ax_o) = plt.subplots(1, 2, figsize=(16, 8))
    panel(ax_e, EKF_BAG, 'EKF-corrected pose')
    panel(ax_o, ODOM_BAG, 'Raw odometry')

    fig.suptitle(f'Scan Map Quality: EKF vs Odometry-Only  '
                 f'({OCC_RES:.2f} m cells)', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f'wrote {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
