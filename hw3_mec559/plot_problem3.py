#!/usr/bin/env python3
"""
Analysis and Visualization script for Problem 3: Scan Matching
Reads transformation matrices, analyzes them, and creates plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def extract_rotation_angle(T):
    """
    Extract rotation angle in degrees from 3x3 transformation matrix.
    Uses arctan2(T[1,0], T[0,0]) to get angle.
    """
    angle_rad = np.arctan2(T[1, 0], T[0, 0])
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def extract_translation_magnitude(T):
    """
    Extract translation magnitude from 3x3 transformation matrix.
    Magnitude = sqrt(T[0,2]^2 + T[1,2]^2)
    """
    tx = T[0, 2]
    ty = T[1, 2]
    magnitude = np.sqrt(tx**2 + ty**2)
    return magnitude

def analyze_scan_results():
    """
    Read scan_results.csv and analyze transformation parameters.
    """
    # Read the CSV file from build directory
    data_path = Path("build/scan_results.csv")
    df = pd.read_csv(data_path)

    # Initialize storage for results
    results = []

    # Process each scan (group by scan_number)
    for scan_num in range(1, 6):
        scan_data = df[df['scan_number'] == scan_num]

        # Get noiseless and noisy rows
        noiseless_row = scan_data[scan_data['type'] == 'noiseless'].iloc[0]
        noisy_row = scan_data[scan_data['type'] == 'noisy'].iloc[0]

        # Reconstruct 3x3 transformation matrices
        T_noiseless = np.array([
            [noiseless_row['T00'], noiseless_row['T01'], noiseless_row['T02']],
            [noiseless_row['T10'], noiseless_row['T11'], noiseless_row['T12']],
            [noiseless_row['T20'], noiseless_row['T21'], noiseless_row['T22']]
        ])

        T_noisy = np.array([
            [noisy_row['T00'], noisy_row['T01'], noisy_row['T02']],
            [noisy_row['T10'], noisy_row['T11'], noisy_row['T12']],
            [noisy_row['T20'], noisy_row['T21'], noisy_row['T22']]
        ])

        # Extract parameters
        angle_noiseless = extract_rotation_angle(T_noiseless)
        angle_noisy = extract_rotation_angle(T_noisy)

        trans_noiseless = extract_translation_magnitude(T_noiseless)
        trans_noisy = extract_translation_magnitude(T_noisy)

        results.append({
            'scan': scan_num,
            'angle_noiseless': angle_noiseless,
            'angle_noisy': angle_noisy,
            'trans_noiseless': trans_noiseless,
            'trans_noisy': trans_noisy
        })

    return results

def print_summary_table(results):
    
    print("=" * 90)
    print("SCAN MATCHING TRANSFORMATION ANALYSIS")
    print("=" * 90)
    print()
    print(f"{'Scan':<6} {'Noiseless':<12} {'Noisy':<12} {'Noiseless':<14} {'Noisy':<14}")
    print(f"{'#':<6} {'Angle (°)':<12} {'Angle (°)':<12} {'Translation Mag':<14} {'Translation Mag':<14}")
    print("-" * 90)

    for res in results:
        print(f"{res['scan']:<6} "
              f"{res['angle_noiseless']:>11.3f}  "
              f"{res['angle_noisy']:>11.3f}  "
              f"{res['trans_noiseless']:>13.6f}  "
              f"{res['trans_noisy']:>13.6f}")

    print("-" * 90)
    print()

    # Print observations
    print("OBSERVATIONS:")
    print("-" * 90)

    for res in results:
        angle_diff = abs(res['angle_noisy'] - res['angle_noiseless'])
        trans_diff = abs(res['trans_noisy'] - res['trans_noiseless'])

        print(f"Scan {res['scan']}:")
        print(f"  Angle difference:       {angle_diff:>8.3f}°")
        print(f"  Translation difference: {trans_diff:>8.6f}")
        print()

    # Compute statistics
    angle_diffs = [abs(r['angle_noisy'] - r['angle_noiseless']) for r in results]
    trans_diffs = [abs(r['trans_noisy'] - r['trans_noiseless']) for r in results]

    print("STATISTICS:")
    print("-" * 90)
    print(f"Mean angle difference:       {np.mean(angle_diffs):.3f}°")
    print(f"Max angle difference:        {np.max(angle_diffs):.3f}°")
    print(f"Mean translation difference: {np.mean(trans_diffs):.6f}")
    print(f"Max translation difference:  {np.max(trans_diffs):.6f}")
    print("=" * 90)

def load_scan_data(scan_num):
    """
    Load point cloud data for a given scan number.
    Returns: points1, points2_noiseless_aligned, points2_noisy_aligned
    """
    data_dir = Path("build")

    points1 = pd.read_csv(data_dir / f"scan_{scan_num}_points1.csv")
    points2_noiseless = pd.read_csv(data_dir / f"scan_{scan_num}_points2_noiseless_aligned.csv")
    points2_noisy = pd.read_csv(data_dir / f"scan_{scan_num}_points2_noisy_aligned.csv")

    return points1, points2_noiseless, points2_noisy

def plot_single_scan(scan_num, angle, save_dir="."):
    """
    Create a scatter plot for a single scan showing all three point sets.
    """
    # Load data
    points1, points2_noiseless, points2_noisy = load_scan_data(scan_num)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot points in specific order for visibility
    # 1. Red noisy first (background, transparent)
    ax.scatter(points2_noisy['x'], points2_noisy['y'], c='red', s=20, alpha=0.3,
               label='Noisy aligned', zorder=1)
    # 2. Blue frame1 second (middle layer)
    ax.scatter(points1['x'], points1['y'], c='blue', s=25, alpha=0.8,
               label='Frame 1 reference', zorder=3)
    # 3. Green noiseless last on top (X markers, highly visible)
    ax.scatter(points2_noiseless['x'], points2_noiseless['y'], c='green', s=10,
               marker='x', linewidths=1.0, alpha=1.0,
               label='Noiseless aligned', zorder=5)

    # Formatting
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'Scan {scan_num} — Point Alignment Verification (θ = {angle:.1f}°)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    # Save figure
    output_path = Path(save_dir) / f"scan_{scan_num}_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_all_scans_summary(results, save_dir="."):
    """
    Create a 2x3 grid showing all 5 scans (6th cell empty).
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, res in enumerate(results):
        scan_num = res['scan']
        angle = res['angle_noiseless']
        ax = axes[idx]

        # Load data
        points1, points2_noiseless, points2_noisy = load_scan_data(scan_num)

        # Plot points in specific order for visibility
        # Red noisy first, blue frame1 second, green X markers on top
        ax.scatter(points2_noisy['x'], points2_noisy['y'], c='red', s=10, alpha=0.3,
                   label='Noisy', zorder=1)
        ax.scatter(points1['x'], points1['y'], c='blue', s=12, alpha=0.8,
                   label='Frame 1', zorder=3)
        ax.scatter(points2_noiseless['x'], points2_noiseless['y'], c='green', s=2.5,
                   marker='x', linewidths=1.0, alpha=1.0,
                   label='Noiseless', zorder=5)

        # Formatting
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)
        ax.set_title(f'Scan {scan_num} (θ = {angle:.1f}°)', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    # Hide the 6th subplot (empty)
    axes[5].axis('off')

    plt.tight_layout()

    # Save figure
    output_path = Path(save_dir) / "scan_all_plots.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    
    results = analyze_scan_results()
    print_summary_table(results)

    # Create individual plots for each scan
    print("Creating individual scan plots:")
    for res in results:
        plot_single_scan(res['scan'], res['angle_noiseless'])

    plot_all_scans_summary(results)

    
