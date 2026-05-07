#!/usr/bin/env python3
"""EKF innovation timeseries — z (down-range) and yaw (odom) channels.

Reads:
    analysis/headline_bags/final_ekf_bag_run2/

Reconstructs the two innovations the EKF computes that are recoverable
from the bag:

    z innovation   = r_down(t) − ekf_pose.z(at t)
                     where r_down comes from /crazyflie/range/down (single
                     beam ToF, ranges[0]). Gated in/out by the live EKF
                     rule (ekf_core.cpp updateZ):
                       μ_z < 0.10           : |nu| < 0.50
                       |commanded_vz| > 0.05: |nu| < 0.40
                       hover                : |nu| < 0.10

    yaw innovation = wrap(odom_yaw(t) − ekf_pose.yaw(at t))
                     no live magnitude gate; every odom tick fires.

Pre-update rejections also mirrored: r non-finite, r outside
[range_min, range_max] or [0.05, 2.0], r < z_est − 0.10. The last is
approximated using interpolated belief z (live z_est is not in the bag).

Outputs:
    results/figures/innovation_timeseries.png  — two stacked subplots
    results/figures/innovation_summary.txt     — per-channel RMSE/mean/max
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
EKF_BAG = os.path.join(PROJECT_DIR, 'analysis', 'headline_bags',
                       'final_ekf_bag_run2')
FIGURES_DIR = os.path.join(PROJECT_DIR, 'results', 'figures')

# Mirror ekf_slam_node.cpp pre-update rejection envelope.
RANGE_MIN_FLOOR = 0.05
RANGE_MAX_CEIL  = 2.0
FLOOR_DROP_TOL  = 0.10   # r < z_est - 0.10 → likely obstacle on floor
NEAR_GROUND_Z   = 0.10
ACTIVE_VZ_THR   = 0.05

# Live |nu| outlier gates (ekf_core.cpp updateZ).
GATE_Z_NEAR_GROUND = 0.50
GATE_Z_ACTIVE      = 0.40
GATE_Z_HOVER       = 0.10

COLOR_KEEP = 'tab:blue'
COLOR_DROP = 'red'


def open_reader(bag_path):
    so = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
    co = rosbag2_py.ConverterOptions('', '')
    r = rosbag2_py.SequentialReader()
    r.open(so, co)
    return r, {t.name: t.type for t in r.get_all_topics_and_types()}


def quat_to_yaw(q):
    return float(np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                            1.0 - 2.0 * (q.y * q.y + q.z * q.z)))


def wrap(a):
    """Wrap to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def read_bag(bag_path):
    """Single pass — return arrays for ekf_pose, odom yaw, range/down, cmd_vel.z."""
    reader, type_map = open_reader(bag_path)
    needed = ('/ekf_pose', '/crazyflie/odom',
              '/crazyflie/range/down', '/cmd_vel')
    for topic in needed:
        if topic not in type_map:
            raise RuntimeError(f"{topic} missing in {bag_path}")

    PoseT = get_message(type_map['/ekf_pose'])
    OdomT = get_message(type_map['/crazyflie/odom'])
    LaserT = get_message(type_map['/crazyflie/range/down'])
    TwistT = get_message(type_map['/cmd_vel'])

    # belief (ekf_pose)
    bt, bx, by, bz, byaw = [], [], [], [], []
    # odom (truth) for yaw observation
    ot, oyaw = [], []
    # down-range readings (raw, before any gating)
    dt, dr, drmin, drmax = [], [], [], []
    # commanded vz (for outlier-gate selection)
    ct, cvz = [], []

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        t = t_ns * 1e-9
        if topic == '/ekf_pose':
            m = deserialize_message(data, PoseT)
            bt.append(t)
            bx.append(m.pose.position.x)
            by.append(m.pose.position.y)
            bz.append(m.pose.position.z)
            byaw.append(quat_to_yaw(m.pose.orientation))
        elif topic == '/crazyflie/odom':
            m = deserialize_message(data, OdomT)
            ot.append(t)
            oyaw.append(quat_to_yaw(m.pose.pose.orientation))
        elif topic == '/crazyflie/range/down':
            m = deserialize_message(data, LaserT)
            if len(m.ranges) == 0:
                continue
            dt.append(t)
            dr.append(float(m.ranges[0]))
            drmin.append(float(m.range_min))
            drmax.append(float(m.range_max))
        elif topic == '/cmd_vel':
            m = deserialize_message(data, TwistT)
            ct.append(t)
            cvz.append(float(m.linear.z))

    # Sort odom by time (rosbag2 isn't guaranteed chronological across topics).
    o_order = np.argsort(ot)
    return {
        'bt': np.array(bt), 'bx': np.array(bx), 'by': np.array(by),
        'bz': np.array(bz), 'byaw': np.array(byaw),
        'ot': np.array(ot)[o_order], 'oyaw': np.array(oyaw)[o_order],
        'dt': np.array(dt), 'dr': np.array(dr),
        'drmin': np.array(drmin), 'drmax': np.array(drmax),
        'ct': np.array(ct), 'cvz': np.array(cvz),
    }


def select_z_gate(mu_z, cmd_vz):
    """Return the active |nu| threshold for updateZ at this tick."""
    if mu_z < NEAR_GROUND_Z:
        return GATE_Z_NEAR_GROUND
    if abs(cmd_vz) > ACTIVE_VZ_THR:
        return GATE_Z_ACTIVE
    return GATE_Z_HOVER


def compute_z_innovation(d):
    """Returns dict with arrays:
        t        — flight-time-relative seconds (length n)
        nu       — z innovation [m]            (length n)
        keep     — boolean, True = applied     (length n)
        gate     — active threshold at tick    (length n)
        rejected_pre — boolean, True = pre-update reject (range envelope or
                       floor-obstacle); excluded from arrays returned (these
                       values were never even passed to updateZ)."""
    if d['dt'].size == 0 or d['bt'].size == 0:
        return {'t': np.array([]), 'nu': np.array([]),
                'keep': np.array([], dtype=bool),
                'gate': np.array([]), 'n_pre_rejected': 0}

    # Interpolate belief z and commanded vz onto each down-range timestamp.
    bz_at = np.interp(d['dt'], d['bt'], d['bz'])
    if d['ct'].size:
        cvz_at = np.interp(d['dt'], d['ct'], d['cvz'])
    else:
        cvz_at = np.zeros_like(d['dt'])

    n_pre = 0
    keep_t, keep_nu, keep_keep, keep_gate = [], [], [], []
    for i in range(d['dt'].size):
        r = d['dr'][i]
        if not np.isfinite(r):
            n_pre += 1
            continue
        if r < d['drmin'][i] or r > d['drmax'][i]:
            n_pre += 1
            continue
        if r < RANGE_MIN_FLOOR or r > RANGE_MAX_CEIL:
            n_pre += 1
            continue
        if r < bz_at[i] - FLOOR_DROP_TOL:
            n_pre += 1
            continue

        nu = r - bz_at[i]
        gate = select_z_gate(bz_at[i], cvz_at[i])
        keep = abs(nu) < gate

        keep_t.append(d['dt'][i])
        keep_nu.append(nu)
        keep_keep.append(keep)
        keep_gate.append(gate)

    t = np.array(keep_t)
    if t.size:
        t_rel = t - d['bt'][0]
    else:
        t_rel = t
    return {
        't':     t_rel,
        'nu':    np.array(keep_nu),
        'keep':  np.array(keep_keep, dtype=bool),
        'gate':  np.array(keep_gate),
        'n_pre_rejected': n_pre,
    }


def compute_yaw_innovation(d):
    """Returns dict with arrays:
        t   — flight-time-relative seconds (length m)
        nu  — yaw innovation [rad], wrapped to (-pi, pi]"""
    if d['ot'].size == 0 or d['bt'].size == 0:
        return {'t': np.array([]), 'nu': np.array([])}

    # Belief yaw needs unwrapped interpolation to avoid jumps at ±π.
    byaw_unwrapped = np.unwrap(d['byaw'])
    byaw_at = np.interp(d['ot'], d['bt'], byaw_unwrapped)

    nu = wrap(d['oyaw'] - byaw_at)
    t_rel = d['ot'] - d['bt'][0]
    return {'t': t_rel, 'nu': nu}


def stats(arr):
    if arr.size == 0:
        return {'rmse': float('nan'), 'mean': float('nan'),
                'max': float('nan'), 'n': 0}
    return {
        'rmse': float(np.sqrt(np.mean(arr ** 2))),
        'mean': float(np.mean(arr)),
        'max':  float(np.max(np.abs(arr))),
        'n':    int(arr.size),
    }


def plot_timeseries(z, yaw, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax_z, ax_y = axes

    if z['t'].size:
        keep_mask = z['keep']
        drop_mask = ~keep_mask
        if keep_mask.any():
            ax_z.plot(z['t'][keep_mask], z['nu'][keep_mask],
                      'o', color=COLOR_KEEP, markersize=2.5, alpha=0.7,
                      linestyle='None',
                      label=f'gated-in (n={int(keep_mask.sum())})')
        if drop_mask.any():
            ax_z.plot(z['t'][drop_mask], z['nu'][drop_mask],
                      'x', color=COLOR_DROP, markersize=7,
                      markeredgewidth=1.5, linestyle='None',
                      label=f'gated-out (n={int(drop_mask.sum())})')
        # Threshold envelopes — draw all three since the active one varies.
        ax_z.axhline( GATE_Z_HOVER, color='gray', ls='--', lw=0.8, alpha=0.6)
        ax_z.axhline(-GATE_Z_HOVER, color='gray', ls='--', lw=0.8, alpha=0.6,
                     label=f'hover gate ±{GATE_Z_HOVER:.2f} m')
        ax_z.axhline( GATE_Z_ACTIVE, color='gray', ls=':', lw=0.8, alpha=0.5)
        ax_z.axhline(-GATE_Z_ACTIVE, color='gray', ls=':', lw=0.8, alpha=0.5,
                     label=f'active-vz gate ±{GATE_Z_ACTIVE:.2f} m')
    ax_z.axhline(0.0, color='black', lw=0.6, alpha=0.5)
    ax_z.set_ylabel(r'z innovation $r_{\rm down} - \mu_z$ [m]')
    ax_z.set_title('EKF Run 2 — z innovation (down-range vs belief z)')
    ax_z.grid(True, alpha=0.3)
    ax_z.legend(loc='upper right', fontsize=8)

    if yaw['t'].size:
        ax_y.plot(yaw['t'], yaw['nu'], '-', color=COLOR_KEEP, lw=0.6,
                  alpha=0.8, label='yaw innovation')
    ax_y.axhline(0.0, color='black', lw=0.6, alpha=0.5)
    ax_y.set_xlabel('time since flight start [s]')
    ax_y.set_ylabel(r'yaw innovation $\mathrm{wrap}(\psi_{\rm odom} - \mu_\psi)$ [rad]')
    ax_y.set_title('EKF Run 2 — yaw innovation (odom yaw vs belief yaw)')
    ax_y.grid(True, alpha=0.3)
    ax_y.legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def write_summary(z, yaw, out_path):
    keep_mask = z['keep'] if z['t'].size else np.array([], dtype=bool)
    z_in  = z['nu'][keep_mask] if keep_mask.size else np.array([])
    z_out = z['nu'][~keep_mask] if keep_mask.size else np.array([])
    y_all = yaw['nu']

    s_zin  = stats(z_in)
    s_zout = stats(z_out)
    s_yaw  = stats(y_all)

    lines = [
        "EKF innovation summary — Run 2 (EKF bag only)",
        "",
        "z innovation (down-range vs belief z) [m]",
        f"  pre-update rejected (envelope, NaN, floor-obstacle) : {z['n_pre_rejected']}",
        f"  gated-in  (applied to filter): RMSE = {s_zin['rmse']:.4f}   "
        f"mean = {s_zin['mean']:+.4f}   max|nu| = {s_zin['max']:.4f}   "
        f"n = {s_zin['n']}",
        f"  gated-out (live |nu|>thresh) : RMSE = {s_zout['rmse']:.4f}   "
        f"mean = {s_zout['mean']:+.4f}   max|nu| = {s_zout['max']:.4f}   "
        f"n = {s_zout['n']}",
        "",
        "yaw innovation (odom yaw vs belief yaw) [rad]",
        f"  every-tick (no live gate)    : RMSE = {s_yaw['rmse']:.5f}   "
        f"mean = {s_yaw['mean']:+.5f}   max|nu| = {s_yaw['max']:.5f}   "
        f"n = {s_yaw['n']}",
        "",
        "Notes:",
        "  z thresholds vary by drone state (ekf_core.cpp updateZ):",
        f"    μ_z < {NEAR_GROUND_Z:.2f} m              → |nu| < {GATE_Z_NEAR_GROUND:.2f}",
        f"    |commanded vz| > {ACTIVE_VZ_THR:.2f}    → |nu| < {GATE_Z_ACTIVE:.2f}",
        f"    hover                       → |nu| < {GATE_Z_HOVER:.2f}",
        "  z innovation reconstructs the live z_est test approximately by",
        "  interpolating belief z onto the down-range timestamp.",
        "  yaw is updated every odom tick (~50 Hz) with no magnitude gate.",
        "  Sample rates: down-range ~10 Hz, odom ~50 Hz — yaw subplot is",
        "  much denser than the z subplot.",
        "",
    ]
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  wrote {out_path}")


def main():
    if not os.path.isdir(EKF_BAG):
        print(f"ERROR: bag not found: {EKF_BAG}", file=sys.stderr)
        return 1
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print(f"Reading EKF bag: {EKF_BAG}")
    d = read_bag(EKF_BAG)
    print(f"  ekf_pose samples : {d['bt'].size}")
    print(f"  odom samples     : {d['ot'].size}")
    print(f"  down-range msgs  : {d['dt'].size}")
    print(f"  cmd_vel msgs     : {d['ct'].size}")

    z   = compute_z_innovation(d)
    yaw = compute_yaw_innovation(d)

    print(f"  z   innovations    : {z['t'].size} valid, "
          f"{z['n_pre_rejected']} pre-rejected, "
          f"{int((~z['keep']).sum()) if z['keep'].size else 0} live-gated-out")
    print(f"  yaw innovations    : {yaw['t'].size}")

    plot_timeseries(z, yaw,
                    os.path.join(FIGURES_DIR, 'innovation_timeseries.png'))
    write_summary(z, yaw,
                  os.path.join(FIGURES_DIR, 'innovation_summary.txt'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
