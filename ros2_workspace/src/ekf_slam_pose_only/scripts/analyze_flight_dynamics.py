#!/usr/bin/env python3
"""Plot position, orientation, and command-velocity over time from a flight bag.

Reads results/gazebo_full_nav_ekf_bag/ (or path passed as argv[1]) and writes
three PNGs to results/:
    flight_xyz.png       — /crazyflie/odom (red dashed) vs /ekf_pose (blue) for
                           x, y, z, with goal/flight_height refs in black dotted
    flight_rpy.png       — odom roll/pitch/yaw (red); on the yaw subplot, also
                           overlay /ekf_pose yaw (blue) to show EKF tracking
    flight_cmdvel.png    — /cmd_vel linear.x/y/z and angular.z

Usage:
    python3 analyze_flight_dynamics.py [bag_dir]
"""

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message


PROJECT_DIR = Path("/home/renix/EKF-SLAM-Autonomous-Crazyflie")
RESULTS_DIR = PROJECT_DIR / "results"
DEFAULT_BAG = RESULTS_DIR / "gazebo_full_nav_ekf_bag"

GOAL_X = 0.8
GOAL_Y = 0.0
FLIGHT_HEIGHT = 0.3


def quat_to_rpy(qx, qy, qz, qw):
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def read_bag(bag_dir):
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag_dir), storage_id="mcap"),
        ConverterOptions("", ""),
    )
    msg_types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}

    ekf_t, ekf_x, ekf_y, ekf_z, ekf_yaw = [], [], [], [], []
    odom_t, odom_x, odom_y, odom_z = [], [], [], []
    odom_roll, odom_pitch, odom_yaw = [], [], []
    cmd_t, cmd_vx, cmd_vy, cmd_vz, cmd_wz = [], [], [], [], []

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        t = t_ns * 1e-9
        msg = deserialize_message(data, msg_types[topic])
        if topic == "/ekf_pose":
            ekf_t.append(t)
            ekf_x.append(msg.pose.position.x)
            ekf_y.append(msg.pose.position.y)
            ekf_z.append(msg.pose.position.z)
            q = msg.pose.orientation
            _, _, y_ = quat_to_rpy(q.x, q.y, q.z, q.w)
            ekf_yaw.append(math.degrees(y_))
        elif topic == "/crazyflie/odom":
            odom_t.append(t)
            odom_x.append(msg.pose.pose.position.x)
            odom_y.append(msg.pose.pose.position.y)
            odom_z.append(msg.pose.pose.position.z)
            q = msg.pose.pose.orientation
            r, p, y_ = quat_to_rpy(q.x, q.y, q.z, q.w)
            odom_roll.append(math.degrees(r))
            odom_pitch.append(math.degrees(p))
            odom_yaw.append(math.degrees(y_))
        elif topic == "/cmd_vel":
            cmd_t.append(t)
            cmd_vx.append(msg.linear.x)
            cmd_vy.append(msg.linear.y)
            cmd_vz.append(msg.linear.z)
            cmd_wz.append(msg.angular.z)

    return {
        "ekf": (ekf_t, ekf_x, ekf_y, ekf_z, ekf_yaw),
        "odom_xyz": (odom_t, odom_x, odom_y, odom_z),
        "odom_rpy": (odom_t, odom_roll, odom_pitch, odom_yaw),
        "cmd": (cmd_t, cmd_vx, cmd_vy, cmd_vz, cmd_wz),
    }


def rebase(times, t0):
    return [t - t0 for t in times]


def bag_t0(data):
    candidates = []
    for key in ("ekf", "odom_xyz", "cmd"):
        ts = data[key][0]
        if ts:
            candidates.append(ts[0])
    return min(candidates) if candidates else 0.0


def plot_xyz(data, t0, out_path):
    ekf_t, ekf_x, ekf_y, ekf_z, _ = data["ekf"]
    odom_t, odom_x, odom_y, odom_z = data["odom_xyz"]
    if not ekf_t and not odom_t:
        print("  no pose data — skipping flight_xyz.png")
        return

    et = rebase(ekf_t, t0)
    ot = rebase(odom_t, t0)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Position Over Time — odom vs EKF")

    if odom_t:
        axs[0].plot(ot, odom_x, color="red", linestyle="--", label="odom x")
    if ekf_t:
        axs[0].plot(et, ekf_x, color="blue", label="ekf x")
    axs[0].axhline(GOAL_X, color="black", linestyle=":", label=f"goal x={GOAL_X}")
    axs[0].set_ylabel("x [m]")
    axs[0].grid(True)
    axs[0].legend(loc="best")

    if odom_t:
        axs[1].plot(ot, odom_y, color="red", linestyle="--", label="odom y")
    if ekf_t:
        axs[1].plot(et, ekf_y, color="blue", label="ekf y")
    axs[1].axhline(GOAL_Y, color="black", linestyle=":", label=f"goal y={GOAL_Y}")
    axs[1].set_ylabel("y [m]")
    axs[1].grid(True)
    axs[1].legend(loc="best")

    if odom_t:
        axs[2].plot(ot, odom_z, color="red", linestyle="--", label="odom z")
    if ekf_t:
        axs[2].plot(et, ekf_z, color="blue", label="ekf z")
    axs[2].axhline(FLIGHT_HEIGHT, color="black", linestyle=":",
                   label=f"flight_height={FLIGHT_HEIGHT}")
    axs[2].set_ylabel("z [m]")
    axs[2].set_xlabel("time [s]")
    axs[2].grid(True)
    axs[2].legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_rpy(data, t0, out_path):
    odom_t, roll, pitch, odom_yaw = data["odom_rpy"]
    ekf_t, _, _, _, ekf_yaw = data["ekf"]
    if not odom_t:
        print("  no /crazyflie/odom messages — skipping flight_rpy.png")
        return
    ot = rebase(odom_t, t0)
    et = rebase(ekf_t, t0)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Orientation Over Time — odom (red) with EKF yaw overlay (blue)")

    axs[0].plot(ot, roll, color="red", label="odom roll")
    axs[0].set_ylabel("roll [deg]")
    axs[0].grid(True)
    axs[0].legend(loc="best")

    axs[1].plot(ot, pitch, color="red", label="odom pitch")
    axs[1].set_ylabel("pitch [deg]")
    axs[1].grid(True)
    axs[1].legend(loc="best")

    axs[2].plot(ot, odom_yaw, color="red", label="odom yaw")
    if ekf_t:
        axs[2].plot(et, ekf_yaw, color="blue", label="ekf yaw")
    axs[2].set_ylabel("yaw [deg]")
    axs[2].set_xlabel("time [s]")
    axs[2].grid(True)
    axs[2].legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_cmd(data, t0, out_path):
    cmd_t, cmd_vx, cmd_vy, cmd_vz, cmd_wz = data["cmd"]
    if not cmd_t:
        print("  no /cmd_vel messages — skipping flight_cmdvel.png")
        return
    t = rebase(cmd_t, t0)

    fig, axs = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    fig.suptitle("Command Velocity Over Time")

    axs[0].plot(t, cmd_vx, color="tab:blue")
    axs[0].set_ylabel("linear.x [m/s]")
    axs[0].grid(True)

    axs[1].plot(t, cmd_vy, color="tab:orange")
    axs[1].set_ylabel("linear.y [m/s]")
    axs[1].grid(True)

    axs[2].plot(t, cmd_vz, color="tab:green")
    axs[2].set_ylabel("linear.z [m/s]")
    axs[2].grid(True)

    axs[3].plot(t, cmd_wz, color="tab:red")
    axs[3].set_ylabel("angular.z [rad/s]")
    axs[3].set_xlabel("time [s]")
    axs[3].grid(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    bag_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BAG
    if not bag_dir.is_dir():
        print(f"bag directory not found: {bag_dir}")
        sys.exit(1)

    print(f"reading {bag_dir}")
    data = read_bag(bag_dir)
    t0 = bag_t0(data)

    print(f"  /ekf_pose       n={len(data['ekf'][0])}")
    print(f"  /crazyflie/odom n={len(data['odom_xyz'][0])}")
    print(f"  /cmd_vel        n={len(data['cmd'][0])}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_xyz(data, t0, RESULTS_DIR / "flight_xyz.png")
    plot_rpy(data, t0, RESULTS_DIR / "flight_rpy.png")
    plot_cmd(data, t0, RESULTS_DIR / "flight_cmdvel.png")


if __name__ == "__main__":
    main()
