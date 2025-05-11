#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from tf2_msgs.msg import TFMessage
from movement import unpack
import time

start_time = None
move_time = 5
sort_time = 5

current_rot = 0.0
odom_received = False
start_x = 0.0
start_y = 0.0
current_x = 0.0
current_y = 0.0
robot_pose_x = 0.0
robot_pose_y = 0.0

rospy.init_node("perception", anonymous=True)


def tf_callback(msg):
    global current_rot, odom_received, current_x, current_y
    frame_id = msg.transforms[-1].header.frame_id

    if frame_id == "/odom":
        orientation_q = msg.transforms[-1].transform.rotation
        _, _, rot = euler_from_quaternion(
            [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        )

        current_rot = rot
        odom_received = True

        current_x = msg.transforms[-1].transform.translation.x
        current_y = msg.transforms[-1].transform.translation.y


def pose_callback(msg):
    global robot_pose_x, robot_pose_y
    robot_pose_x = msg.pose.pose.position.x
    robot_pose_y = msg.pose.pose.position.y
    rospy.loginfo(f"Robot pose: x={robot_pose_x}, y={robot_pose_y}")
    

vel_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
tf_subscriber = rospy.Subscriber("/tf", TFMessage, tf_callback)
# Use Adaptive Monte Carlo Localization to retrieve robot pose estimation
acml_subscriber = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, pose_callback)  

def normalize_angle(angle):
    """Normalize angle to [-pi, pi]"""
    return np.arctan2(np.sin(angle), np.cos(angle))


def move(msg):
    global current_rot, odom_received, start_x, start_y, current_x, current_y
    start_x = current_x
    start_y = current_y
    color, x_center, center_depth, rotation_angle_deg, dist = unpack(msg)

    rate = rospy.Rate(10)

    while not odom_received and not rospy.is_shutdown():
        print(odom_received)
        rospy.loginfo("Waiting for odom...")
        rate.sleep()

    start_rot = current_rot
    # Fix weird issues cause by arctan2: rotate left when rotation_angle_deg is positive, and vice versa
    if rotation_angle_deg > 0:
        # Positive angle means rotate right (clockwise)
        target_rot = normalize_angle(start_rot - np.deg2rad(abs(rotation_angle_deg)))
    else:
        # Negative angle means rotate left (counterclockwise)
        target_rot = normalize_angle(start_rot + np.deg2rad(abs(rotation_angle_deg)))

    # This also seems to flip the sign of target_rot
    rospy.loginfo(
        f"[{color}] Rotating from {np.rad2deg(start_rot):.2f}° to {-np.rad2deg(target_rot):.2f}°"
    )

    while not rospy.is_shutdown():
        angle_diff = normalize_angle(target_rot - current_rot)

        if abs(angle_diff) < 0.01:
            break

        twist = Twist()
        twist.angular.z = 0.5 * angle_diff
        # This looks like is what cause the move forward always seem off by some cm
        # twist.linear.x = 0.1
        vel_publisher.publish(twist)
        rate.sleep()

    vel_publisher.publish(Twist())
    rospy.sleep(0.5)

    twist = Twist()
    move_speed = 0.1  # m/s
    # move_duration = dist / move_speed
    # start_time = rospy.Time.now().to_sec()

    rospy.loginfo(f"[{color}] Moving forward {dist:.2f} meters...")

    while not rospy.is_shutdown():
        """elapsed = rospy.Time.now().to_sec() - start_time
        if elapsed >= move_duration:
            break"""
        distance_moved = np.sqrt(
            (current_x**2 - start_x**2) + (current_y**2 - start_y**2)
        )

        distance_diff = abs(dist - distance_moved)

        if distance_diff < 0.01:
            break

        twist.linear.x = move_speed
        vel_publisher.publish(twist)
        rate.sleep()

    vel_publisher.publish(Twist())
    rospy.loginfo(f"[{color}] Move complete.")


def sort(obj_info):
    global current_rot, odom_received, start_x, start_y, current_x, current_y, robot_pose_x, robot_pose_y
    start_x = current_x
    start_y = current_y
    color, x_center, center_depth, rotation_angle_deg, dist = unpack(obj_info)
    
    rate = rospy.Rate(10)
    
    while not odom_received and not rospy.is_shutdown():
        print(odom_received)
        rospy.loginfo("Waiting for odom...")
        rate.sleep()

    start_rot = current_rot
    bin_pose_x, bin_pose_y, bin_pose_z = bin_poses[color]
    dx = bin_pose_x - robot_pose_x 
    dy = bin_pose_y - robot_pose_y 
    target_rot = np.arctan2(dy, dx)

    rospy.loginfo(f"[{color}] Rotating from {np.rad2deg(start_rot):.2f}° to {np.rad2deg(target_rot):.2f}°")

    while not rospy.is_shutdown():
        angle_diff = normalize_angle(target_rot - current_rot)

        if abs(angle_diff) < 0.01:
            break  

        twist = Twist()
        twist.angular.z = 0.5 * angle_diff
        vel_publisher.publish(twist)
        rate.sleep()

    vel_publisher.publish(Twist())
    rospy.sleep(0.5)
    
    dist = np.sqrt(
        (bin_pose_x**2 - robot_pose_x**2) + (bin_pose_y**2 - robot_pose_y**2)
    )
    twist = Twist()
    move_speed = 0.1  # m/s
    
    rospy.loginfo(f"[{color}] Moving forward {dist:.2f} meters...")
    
    while not rospy.is_shutdown():
        distance_moved = np.sqrt(
            (current_x**2 - start_x**2) + (current_y**2 - start_y**2)
        )

        distance_diff = abs(dist - distance_moved)

        if distance_diff < 0.01:
            break

        twist.linear.x = move_speed
        vel_publisher.publish(twist)
        rate.sleep()

    vel_publisher.publish(Twist())
    rospy.loginfo(f"[{color}] Sort complete.")
