#!/usr/bin/env python3
"""ATE/RPE evaluation for the Run 2 headline bags.

Reads:
    analysis/headline_bags/final_ekf_bag_run2/
    analysis/headline_bags/final_odom_bag_run2/

For each bag:
    belief = /ekf_pose            (filter output)
    truth  = /crazyflie/odom      (Gazebo physics ground truth — the
                                   un-noised version; /crazyflie/odom_noisy
                                   carries the noisy variant in the EKF bag)

ATE (Absolute Trajectory Error, translation-only):
    ATE_i = ‖belief_xy(t_i) − truth_xy(t_i)‖
    Truth is linearly interpolated onto belief timestamps.

RPE (Relative Pose Error, translation-only, fixed Δ window):
    RPE_i = ‖(belief_xy(t_i+Δ) − belief_xy(t_i))
            − (truth_xy(t_i+Δ) − truth_xy(t_i))‖
    Δ = 1.0 s.

Outputs (results/figures/):
    ate_timeseries.png   — ATE vs time, EKF (top) and odom (bottom)
                           stacked, shared y-axis. Horizontal RMSE line per
                           subplot.
    rpe_histogram.png    — overlaid RPE distributions, one alpha-blended
                           histogram per bag, with vertical RMSE lines.
    ate_rpe_summary.txt  — RMSE/mean/max per bag, sample counts noted.
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
EKF_BAG  = os.path.join(HEADLINE_BAGS_DIR, 'final_ekf_bag_run2')
ODOM_BAG = os.path.join(HEADLINE_BAGS_DIR, 'final_odom_bag_run2')
FIGURES_DIR = os.path.join(PROJECT_DIR, 'results', 'figures')

RPE_DELTA_S = 1.0   # seconds — the standard TUM-RGBD value

# Re-use the analyze_map.py palette so figures sit in the same family.
COLOR_EKF  = 'red'
COLOR_ODOM = 'rebeccapurple'


def open_reader(bag_path):
    so = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
    co = rosbag2_py.ConverterOptions('', '')
    r = rosbag2_py.SequentialReader()
    r.open(so, co)
    return r, {t.name: t.type for t in r.get_all_topics_and_types()}


def read_belief_and_truth(bag_path):
    """Single pass — return belief and truth as (t, x, y) arrays each.

    Belief from /ekf_pose. Truth from /crazyflie/odom. Truth sorted by
    timestamp because rosbag2 doesn't guarantee chronological order
    across topics."""
    reader, type_map = open_reader(bag_path)
    for topic in ('/ekf_pose', '/crazyflie/odom'):
        if topic not in type_map:
            raise RuntimeError(f"{topic} missing in {bag_path}")

    PoseT = get_message(type_map['/ekf_pose'])
    OdomT = get_message(type_map['/crazyflie/odom'])

    bt, bx, by = [], [], []
    tt, tx, ty = [], [], []

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic == '/ekf_pose':
            msg = deserialize_message(data, PoseT)
            bt.append(t_ns * 1e-9)
            bx.append(msg.pose.position.x)
            by.append(msg.pose.position.y)
        elif topic == '/crazyflie/odom':
            msg = deserialize_message(data, OdomT)
            tt.append(t_ns * 1e-9)
            tx.append(msg.pose.pose.position.x)
            ty.append(msg.pose.pose.position.y)

    order = np.argsort(tt)
    truth  = (np.array(tt)[order], np.array(tx)[order], np.array(ty)[order])
    belief = (np.array(bt),        np.array(bx),        np.array(by))
    return belief, truth


def compute_ate(belief, truth):
    """Translation-only ATE: per-belief-timestamp Euclidean distance after
    interpolating truth onto belief timestamps. Returns (t_rel, ate_m).

    Belief samples outside the truth time range are discarded so we don't
    extrapolate. t_rel is seconds since the first valid sample."""
    bt, bx, by = belief
    tt, tx, ty = truth
    if bt.size == 0 or tt.size == 0:
        return np.array([]), np.array([])

    mask = (bt >= tt[0]) & (bt <= tt[-1])
    bt_v, bx_v, by_v = bt[mask], bx[mask], by[mask]
    if bt_v.size == 0:
        return np.array([]), np.array([])

    tx_at = np.interp(bt_v, tt, tx)
    ty_at = np.interp(bt_v, tt, ty)
    ate = np.hypot(bx_v - tx_at, by_v - ty_at)
    return bt_v - bt_v[0], ate


def compute_rpe(belief, truth, delta_s=RPE_DELTA_S):
    """Translation-only RPE at fixed time offset.

    For each belief sample at t_i, find the belief sample closest to
    t_i + delta_s (within delta_s/2 tolerance — otherwise drop). Truth
    is interpolated onto both endpoints. Returns the RPE array."""
    bt, bx, by = belief
    tt, tx, ty = truth
    if bt.size < 2 or tt.size < 2:
        return np.array([])

    span = (tt[0], tt[-1])
    rpe_vals = []
    for i in range(bt.size):
        t0 = bt[i]
        t1 = t0 + delta_s
        if t0 < span[0] or t1 > span[1]:
            continue
        j = int(np.searchsorted(bt, t1))
        if j >= bt.size:
            continue
        if abs(bt[j] - t1) > delta_s * 0.5:
            continue

        dbx = bx[j] - bx[i]
        dby = by[j] - by[i]

        tx0 = np.interp(t0,    tt, tx); ty0 = np.interp(t0,    tt, ty)
        tx1 = np.interp(bt[j], tt, tx); ty1 = np.interp(bt[j], tt, ty)
        dtx = tx1 - tx0
        dty = ty1 - ty0

        rpe_vals.append(np.hypot(dbx - dtx, dby - dty))

    return np.array(rpe_vals)


def stats(arr):
    if arr.size == 0:
        return {'rmse': float('nan'), 'mean': float('nan'),
                'max': float('nan'), 'n': 0}
    return {
        'rmse': float(np.sqrt(np.mean(arr ** 2))),
        'mean': float(np.mean(arr)),
        'max':  float(np.max(arr)),
        'n':    int(arr.size),
    }


def plot_ate_timeseries(ekf_ate, odom_ate, out_path):
    """Stacked vertical: EKF on top (longer flight), odom below.
    Shared y-axis (meters)."""
    (te, ae) = ekf_ate
    (to, ao) = odom_ate

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharey=True)
    ax_e, ax_o = axes

    if ae.size:
        ax_e.plot(te, ae, '-', color=COLOR_EKF, lw=1.2, label='ATE')
        rmse_e = float(np.sqrt(np.mean(ae ** 2)))
        ax_e.axhline(rmse_e, color=COLOR_EKF, lw=1.0, ls='--', alpha=0.7,
                     label=f'RMSE = {rmse_e:.3f} m')
        ax_e.set_title(f'EKF Run 2 — ATE vs time ({te[-1]:.1f} s flight)')
    else:
        ax_e.set_title('EKF Run 2 — ATE vs time')
    ax_e.set_ylabel('ATE [m]')
    ax_e.grid(True, alpha=0.3)
    ax_e.legend(loc='upper right', fontsize=9)

    if ao.size:
        ax_o.plot(to, ao, '-', color=COLOR_ODOM, lw=1.2, label='ATE')
        rmse_o = float(np.sqrt(np.mean(ao ** 2)))
        ax_o.axhline(rmse_o, color=COLOR_ODOM, lw=1.0, ls='--', alpha=0.7,
                     label=f'RMSE = {rmse_o:.3f} m')
        ax_o.set_title(f'Odom Run 2 — ATE vs time ({to[-1]:.1f} s flight)')
    else:
        ax_o.set_title('Odom Run 2 — ATE vs time')
    ax_o.set_xlabel('time since flight start [s]')
    ax_o.set_ylabel('ATE [m]')
    ax_o.grid(True, alpha=0.3)
    ax_o.legend(loc='upper right', fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_rpe_histogram(ekf_rpe, odom_rpe, out_path):
    """Overlaid alpha-blended histograms with shared bins, two RMSE lines."""
    fig, ax = plt.subplots(figsize=(9, 6))

    if ekf_rpe.size and odom_rpe.size:
        combined = np.concatenate([ekf_rpe, odom_rpe])
    elif ekf_rpe.size:
        combined = ekf_rpe
    else:
        combined = odom_rpe

    if combined.size > 1:
        q1, q3 = np.percentile(combined, [25, 75])
        iqr = q3 - q1
        if iqr > 0:
            h = 2.0 * iqr / (combined.size ** (1.0 / 3.0))
            n_bins = max(8, int((combined.max() - combined.min()) / h))
        else:
            n_bins = 20
        bin_edges = np.linspace(0.0, combined.max() * 1.05, n_bins)
    else:
        bin_edges = 20

    if ekf_rpe.size:
        ax.hist(ekf_rpe, bins=bin_edges, alpha=0.55, color=COLOR_EKF,
                edgecolor='black', linewidth=0.5,
                label=f'EKF Run 2  (n={ekf_rpe.size})')
        rmse_e = float(np.sqrt(np.mean(ekf_rpe ** 2)))
        ax.axvline(rmse_e, color=COLOR_EKF, lw=1.5, ls='--',
                   label=f'EKF RMSE = {rmse_e:.3f} m')

    if odom_rpe.size:
        ax.hist(odom_rpe, bins=bin_edges, alpha=0.55, color=COLOR_ODOM,
                edgecolor='black', linewidth=0.5,
                label=f'Odom Run 2 (n={odom_rpe.size})')
        rmse_o = float(np.sqrt(np.mean(odom_rpe ** 2)))
        ax.axvline(rmse_o, color=COLOR_ODOM, lw=1.5, ls='--',
                   label=f'Odom RMSE = {rmse_o:.3f} m')

    ax.set_xlabel(f'RPE magnitude over Δ={RPE_DELTA_S:.1f} s window [m]')
    ax.set_ylabel('count')
    ax.set_title(f'Relative Pose Error distribution — Run 2 (Δ={RPE_DELTA_S:.1f} s)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def write_summary(ekf_ate_arr, odom_ate_arr, ekf_rpe, odom_rpe, out_path):
    se_a = stats(ekf_ate_arr)
    so_a = stats(odom_ate_arr)
    se_r = stats(ekf_rpe)
    so_r = stats(odom_rpe)

    lines = [
        f"ATE/RPE summary — Run 2, Δ={RPE_DELTA_S:.1f}s, translation-only",
        "",
        "EKF Run 2 (final_ekf_bag_run2/)",
        f"  ATE  : RMSE = {se_a['rmse']:.3f} m   "
        f"mean = {se_a['mean']:.3f} m   "
        f"max = {se_a['max']:.3f} m   (n={se_a['n']} samples)",
        f"  RPE  : RMSE = {se_r['rmse']:.3f} m   "
        f"mean = {se_r['mean']:.3f} m   "
        f"max = {se_r['max']:.3f} m   (n={se_r['n']} windows)",
        "",
        "Odom Run 2 (final_odom_bag_run2/)",
        f"  ATE  : RMSE = {so_a['rmse']:.3f} m   "
        f"mean = {so_a['mean']:.3f} m   "
        f"max = {so_a['max']:.3f} m   (n={so_a['n']} samples)",
        f"  RPE  : RMSE = {so_r['rmse']:.3f} m   "
        f"mean = {so_r['mean']:.3f} m   "
        f"max = {so_r['max']:.3f} m   (n={so_r['n']} windows)",
        "",
        "Notes:",
        "  ATE = ‖belief_xy(t) − truth_xy(t)‖ at every belief sample,",
        "        truth interpolated onto belief timestamps.",
        f"  RPE = ‖Δbelief − Δtruth‖ over a {RPE_DELTA_S:.1f}s window;",
        "        only windows fully inside the truth time range are used.",
        "  Translation-only — orientation error not included.",
        "  Sample-count asymmetry: the EKF flight is much longer than the",
        "  odom flight, so the EKF distribution is estimated from more",
        "  windows. Compare RMSE/mean for cross-bag conclusions; raw",
        "  histogram counts are not directly comparable.",
        "",
    ]
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  wrote {out_path}")


def main():
    for p in (EKF_BAG, ODOM_BAG):
        if not os.path.isdir(p):
            print(f"ERROR: bag directory not found: {p}", file=sys.stderr)
            return 1
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print(f"Reading EKF bag : {EKF_BAG}")
    ekf_belief, ekf_truth = read_belief_and_truth(EKF_BAG)
    print(f"  belief samples: {ekf_belief[0].size}   "
          f"truth samples: {ekf_truth[0].size}")

    print(f"Reading Odom bag: {ODOM_BAG}")
    odom_belief, odom_truth = read_belief_and_truth(ODOM_BAG)
    print(f"  belief samples: {odom_belief[0].size}   "
          f"truth samples: {odom_truth[0].size}")

    ekf_t,  ekf_ate  = compute_ate(ekf_belief, ekf_truth)
    odom_t, odom_ate = compute_ate(odom_belief, odom_truth)

    ekf_rpe  = compute_rpe(ekf_belief,  ekf_truth)
    odom_rpe = compute_rpe(odom_belief, odom_truth)

    print(f"  EKF  ATE samples: {ekf_ate.size}    RPE windows: {ekf_rpe.size}")
    print(f"  Odom ATE samples: {odom_ate.size}    RPE windows: {odom_rpe.size}")

    plot_ate_timeseries((ekf_t, ekf_ate), (odom_t, odom_ate),
                        os.path.join(FIGURES_DIR, 'ate_timeseries.png'))
    plot_rpe_histogram(ekf_rpe, odom_rpe,
                       os.path.join(FIGURES_DIR, 'rpe_histogram.png'))
    write_summary(ekf_ate, odom_ate, ekf_rpe, odom_rpe,
                  os.path.join(FIGURES_DIR, 'ate_rpe_summary.txt'))

    return 0


if __name__ == '__main__':
    sys.exit(main())
