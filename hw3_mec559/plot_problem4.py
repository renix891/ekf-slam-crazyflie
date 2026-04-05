#!/usr/bin/env python3
"""
Visualization script for Problem 4: Occupancy Grid Mapping
Creates plots of point clouds and occupancy grids.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_pointcloud(test_num):
    """
    Create scatter plot of point cloud data.
    """
    data_path = Path("build") / f"test{test_num}_pointcloud.csv"
    points = pd.read_csv(data_path)

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(points['x'], points['y'], c='blue', s=1, alpha=0.5)

    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title(f'Point Cloud Test {test_num} ({len(points)} points)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    output_path = f"pointcloud_{test_num}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_occupancy_grid(test_num):
    """
    Create occupancy grid visualization.
    """
    data_path = Path("build") / f"test{test_num}_grid.csv"
    grid = pd.read_csv(data_path, header=None).values

    rows, cols = grid.shape

    fig, ax = plt.subplots(figsize=(12, 10))

    # Display grid (origin=lower to match coordinate system)
    # cmap=binary: white=0 (free), black=1 (occupied)
    im = ax.imshow(grid, cmap='binary', origin='lower', interpolation='none')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Occupancy (0=free, 1=occupied)', fontsize=11)

    # Calculate resolution and dimensions (from C++ code)
    resolution = 0.1  # meters
    occupied = np.sum(grid)
    total = rows * cols
    occupancy_pct = 100.0 * occupied / total

    ax.set_xlabel('Grid Column', fontsize=12)
    ax.set_ylabel('Grid Row', fontsize=12)
    ax.set_title(f'Occupancy Grid Test {test_num}\n'
                 f'{rows}×{cols} cells (resolution={resolution}m, '
                 f'occupancy={occupancy_pct:.1f}%)',
                 fontsize=14, fontweight='bold')

    output_path = f"occupancy_grid_{test_num}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    

    for test_num in [1, 2]:
        print(f"Test {test_num}:")
        plot_pointcloud(test_num)
        plot_occupancy_grid(test_num)
        print()


