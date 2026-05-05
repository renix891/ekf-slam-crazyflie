#!/bin/bash
# Run this in a SECOND terminal while a flight is going.
# Tells you whether each topic is publishing and at what rate.
#
# usage:
#   source /opt/ros/jazzy/setup.bash
#   source /home/renix/EKF-SLAM-Autonomous-Crazyflie/ros2_workspace/install/setup.bash
#   bash live_topic_check.sh

set -u

echo "==============================================="
echo "Topic publish rates (5 s sample each):"
echo "==============================================="
for topic in /ekf_pose /map /crazyflie/scan \
             /ekf_slam/debug/lines /ekf_slam/debug/corners \
             /ekf_slam/debug/landmark_lines /ekf_slam/debug/landmark_corners; do
    printf "%-45s " "$topic"
    timeout 5 ros2 topic hz "$topic" 2>&1 | grep "average rate" | head -1 \
        || echo "(no messages in 5 s)"
done

echo ""
echo "==============================================="
echo "/map info — most recent message:"
echo "==============================================="
timeout 3 ros2 topic echo /map --once --field info 2>&1 | head -20

echo ""
echo "==============================================="
echo "Non-zero cells in /map:"
echo "==============================================="
timeout 3 ros2 topic echo /map --once --field data 2>&1 \
    | tr ',' '\n' \
    | grep -v "^$" \
    | awk 'BEGIN{neg=0;zero=0;weight=0;occ=0}
           /-1/ {neg++; next}
           /100/ {occ++; next}
           /^- 0$|^0$/ {zero++; next}
           {weight++}
           END{print "  unknown(-1): " neg "\n  free(0):     " zero "\n  weighted:    " weight "\n  occupied(100):" occ}'

echo ""
echo "==============================================="
echo "How many EKF state landmarks right now?"
echo "==============================================="
timeout 3 ros2 topic echo /ekf_slam/debug/landmark_lines --once 2>&1 \
    | grep -c "ns: ekf_slam_landmark_lines"
echo "  ^-- line landmark markers"
timeout 3 ros2 topic echo /ekf_slam/debug/landmark_corners --once 2>&1 \
    | grep -c "ns: ekf_slam_landmark_corners"
echo "  ^-- corner landmark markers"
