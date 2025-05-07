#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from tf2_msgs.msg import TFMessage
from movement import unpack
import time

start_time = None
move_time = 5
sort_time = 5

current_rot = 0.0
start_x = 0.0
start_y = 0.0
current_x = 0.0
current_y = 0.0
odom_received = False

def tf_callback(msg):
    global current_rot, current_x, current_y, odom_received
    frame_id = msg.transforms[-1].header.frame_id
    
    if frame_id == "/odom":
        orientation_q = msg.transforms[-1].transform.rotation
        _, _, rot = euler_from_quaternion([
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w
        ])
        
        current_rot = rot
        odom_received = True
        
        current_x = msg.transforms[-1].transform.translation.x
        current_y = msg.transforms[-1].transform.translation.y
        
#rospy.init_node("move_robot")
vel_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
tf_subscriber = rospy.Subscriber("/tf", TFMessage, tf_callback)

def normalize_angle(angle):
    """ Normalize angle to [-pi, pi] """
    return np.arctan2(np.sin(angle), np.cos(angle))

def move(msg):
    global current_rot, start_x, start_y, current_x, current_y, odom_received
    start_x = current_x 
    start_y = current_y
    color, x_center, center_depth, rotation_angle_deg, dist = unpack(msg)
    
    rate = rospy.Rate(10)
    
    while not odom_received and not rospy.is_shutdown():
        print(odom_received)
        rospy.loginfo("Waiting for odom...")
        rate.sleep()

    start_rot = current_rot
    target_rot = normalize_angle(start_rot + np.deg2rad(rotation_angle_deg))

    rospy.loginfo(f"[{color}] Rotating from {np.rad2deg(start_rot):.2f}° to {np.rad2deg(target_rot):.2f}°")

    while not rospy.is_shutdown():
        angle_diff = normalize_angle(target_rot - current_rot)

        if abs(angle_diff) < 0.01:
            break  

        twist = Twist()
        twist.angular.z = 0.5 * angle_diff
        twist.linear.x = 0.1
        vel_publisher.publish(twist)
        rate.sleep()

    vel_publisher.publish(Twist())
    rospy.sleep(0.5)

    twist = Twist()
    move_speed = 0.1  # m/s
    #move_duration = dist / move_speed
    #start_time = rospy.Time.now().to_sec()

    rospy.loginfo(f"[{color}] Moving forward {dist:.2f} meters...")

    while not rospy.is_shutdown():
        """elapsed = rospy.Time.now().to_sec() - start_time
        if elapsed >= move_duration:
            break"""
        distance_moved = np.sqrt((current_x**2 - start_x**2) + (current_y**2 - start_y**2))
        
        distance_diff = abs(dist - distance_moved)
        
        if distance_diff < 0.01:
            break
        
        twist.linear.x = move_speed
        vel_publisher.publish(twist)
        rate.sleep()

    vel_publisher.publish(Twist())
    rospy.loginfo(f"[{color}] Move complete.")
    
    return True

def sort(obj_info):
    if not start_time:
        start_time = time.time()
    if time.time() - start_time >= sort_time:
        start_time = None
        return "finished_sorting"