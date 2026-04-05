# Crazyflie ROS2 Package

ROS2 Humble package for Crazyflie drone control using the Crazyflie Python library (cflib).

## Topics

### Published Topics

**Crazyflie Node:**
- `/crazyflie/pose` (geometry_msgs/PoseStamped) - Drone position (x, y, z) and orientation (quaternion)
- `/crazyflie/range/front` (sensor_msgs/Range) - Front range sensor
- `/crazyflie/range/back` (sensor_msgs/Range) - Back range sensor
- `/crazyflie/range/left` (sensor_msgs/Range) - Left range sensor
- `/crazyflie/range/right` (sensor_msgs/Range) - Right range sensor
- `/crazyflie/range/up` (sensor_msgs/Range) - Up range sensor
- `/crazyflie/range/down` (sensor_msgs/Range) - Down range sensor

**OctoMap Mapper Node:**
- `/octomap/point_cloud` (sensor_msgs/PointCloud2) - Aggregated obstacle point cloud

**OctoMap Server Node:**
- `/projected_map` (nav_msgs/OccupancyGrid) - 2D occupancy grid for navigation
- `/octomap_binary` (octomap_msgs/Octomap) - 3D OctoMap

**Autonomous Navigation Node:**
- `/autonomous_path` (nav_msgs/Path) - Planned path visualization
- `/goal_marker` (visualization_msgs/Marker) - Goal position marker

### Subscribed Topics

**Crazyflie Node:**
- `/cmd_vel` (geometry_msgs/Twist) - Velocity commands
  - `linear.x` - Forward/backward velocity
  - `linear.y` - Left/right velocity
  - `linear.z` - Up/down velocity (adjusts hover height)
  - `angular.z` - Yaw rate

**Autonomous Navigation Node:**
- `/goal_pose` (geometry_msgs/PoseStamped) - Target position for autonomous navigation
- `/projected_map` (nav_msgs/OccupancyGrid) - Occupancy grid from OctoMap server
- `/crazyflie/pose` (geometry_msgs/PoseStamped) - Current drone position

### Services

- `/enable_autonomous` (std_srvs/SetBool) - Enable/disable autonomous navigation

## Parameters

### Crazyflie Node Parameters

- `uri` (string, default: 'radio://0/80/2M/E7E7E7E7E2') - Crazyflie radio URI
- `hover_height` (double, default: 0.3) - Default hover height in meters
- `speed_factor` (double, default: 0.3) - Speed multiplier for velocity commands
- `frame_id` (string, default: 'crazyflie') - TF frame ID for published messages
- `logging_only` (bool, default: false) - Enable logging-only mode (no flight commands sent)

### OctoMap Mapper Parameters

- `resolution` (double, default: 0.05) - OctoMap voxel resolution in meters
- `max_range` (double, default: 3.0) - Maximum sensor range to trust
- `min_range` (double, default: 0.02) - Minimum sensor range to trust
- `publish_point_cloud` (bool, default: true) - Publish aggregated point cloud
- `point_cloud_history` (int, default: 100) - Number of measurements to keep

### Autonomous Navigation Parameters

- `waypoint_tolerance` (double, default: 0.2) - Distance to waypoint before moving to next (meters)
- `planning_frequency` (double, default: 2.0) - Path replanning frequency (Hz)

### Launch Parameters

- `log_level` (string, default: 'debug') - Logging level for autonomous navigation node (debug, info, warn, error)


## Usage

### Basic Launch (Manual Control Only)

```bash
ros2 launch crazyflie_ros2 crazyflie.launch.py
```

Or with custom parameters:

```bash
ros2 launch crazyflie_ros2 crazyflie.launch.py uri:=radio://0/80/2M/E7E7E7E7E3 hover_height:=0.5
```

### Launch with OctoMap and Autonomous Navigation

**Normal flight mode:**
```bash
ros2 launch crazyflie_ros2 crazyflie_with_octomap.launch.py
```

**Logging-only mode (no flight commands sent):**
```bash
ros2 launch crazyflie_ros2 crazyflie_with_octomap.launch.py logging_only:=true
```

**With debug logging enabled:**
```bash
ros2 launch crazyflie_ros2 crazyflie_with_octomap.launch.py log_level:=debug
```

Debug logging provides detailed pathfinding diagnostics including:
- Grid coordinate conversions (world → grid)
- Bounds checking for start/goal positions
- Path planning worker details
- Weight array values at start/goal
- Detailed failure reasons when no path is found

**With custom parameters:**
```bash
ros2 launch crazyflie_ros2 crazyflie_with_octomap.launch.py \
  uri:=radio://0/80/2M/E7E7E7E7E3 \
  hover_height:=0.5 \
  octomap_resolution:=0.10 \
  max_range:=2.0 \
  log_level:=info \
  logging_only:=false
```

### Autonomous Navigation Commands

**1. Enable autonomous mode:**
```bash
ros2 service call /enable_autonomous std_srvs/srv/SetBool "{data: true}"
```

**2. Send goal position:**
```bash
ros2 topic pub /goal_pose geometry_msgs/msg/PoseStamped "{
  header: {frame_id: 'world'},
  pose: {
    position: {x: 1.0, y: 1.0, z: 0.3}
  }
}"
```

**3. Disable autonomous mode:**
```bash
ros2 service call /enable_autonomous std_srvs/srv/SetBool "{data: false}"
```

### Manual Control with teleop_twist_keyboard

In a separate terminal:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

Controls:
- `i/k` - Forward/backward
- `j/l` - Left/right
- `u/m` - Increase/decrease height
- `o/.` - Yaw left/right

### Monitor Topics

```bash
# View pose
ros2 topic echo /crazyflie/pose

# View range sensors
ros2 topic echo /crazyflie/range/front

# View occupancy grid
ros2 topic echo /projected_map

# View planned path
ros2 topic echo /autonomous_path

# List all topics
ros2 topic list
```

## Architecture

### Basic Mode (crazyflie.launch.py)

The crazyflie_node:
1. Connects to Crazyflie via radio using cflib
2. Sets up logging for position and sensor data at 10Hz
3. Publishes received data to ROS2 topics
4. Listens for `/cmd_vel` messages
5. Sends velocity commands to Crazyflie at 10Hz

### Autonomous Navigation Mode (crazyflie_with_octomap.launch.py)

The system consists of four nodes working together:

**1. crazyflie_node**
- Connects to Crazyflie and streams sensor data
- Publishes pose and range sensor data
- Accepts velocity commands from autonomous navigation or manual control

**2. octomap_mapper_node**
- Subscribes to range sensors and pose
- Transforms sensor readings to world coordinates
- Publishes aggregated point cloud of obstacles

**3. octomap_server_node** (from octomap_server package)
- Subscribes to point cloud
- Builds 3D OctoMap
- Publishes 2D projected occupancy grid for navigation

**4. autonomous_navigation_node**
- Subscribes to occupancy grid and current pose
- Uses A* path planning to navigate to goal positions
- Replans paths dynamically to avoid obstacles
- Publishes velocity commands to crazyflie_node

### Logging-Only Mode

When `logging_only:=true` is set:
- All nodes run normally
- Sensor data is published
- Occupancy maps are built
- Paths are planned
- **Flight commands are NOT sent to the drone**
- Useful for testing and debugging without risk

## Quick Reference

### Launch Commands

```bash
# Basic manual control
ros2 launch crazyflie_ros2 crazyflie.launch.py

# Full autonomous system
ros2 launch crazyflie_ros2 crazyflie_with_octomap.launch.py

# Testing mode (no flight commands)
ros2 launch crazyflie_ros2 crazyflie_with_octomap.launch.py logging_only:=true
```

### Autonomous Navigation Workflow

```bash
# 1. Launch the system
ros2 launch crazyflie_ros2 crazyflie_with_octomap.launch.py

# 2. Enable autonomous mode
ros2 service call /enable_autonomous std_srvs/srv/SetBool "{data: true}"

# 3. Send goal position
ros2 topic pub /goal_pose geometry_msgs/msg/PoseStamped "{
  header: {frame_id: 'world'},
  pose: {position: {x: 1.0, y: 1.0, z: 0.3}}
}"

# 4. Monitor progress
ros2 topic echo /autonomous_path
ros2 topic echo /crazyflie/pose

# 5. Disable when done
ros2 service call /enable_autonomous std_srvs/srv/SetBool "{data: false}"
```

### Useful Monitoring Commands

```bash
# Check all active topics
ros2 topic list

# Monitor drone position
ros2 topic echo /crazyflie/pose

# Check occupancy grid
ros2 topic echo /projected_map

# View planned path
ros2 topic echo /autonomous_path

# Monitor range sensors
ros2 topic echo /crazyflie/range/front
```
