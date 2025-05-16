#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from tf2_msgs.msg import TFMessage
import time

start_time = None

current_rot = 0.0
odom_received = False
# pose info
start_x = 0.0
start_y = 0.0
current_x = 0.0
current_y = 0.0

rospy.init_node("perception", anonymous=True)

# corner areas as "bins"
color_to_bin = {
    "green cube": "top_left",
    "grey cube": "bottom_right",
    "purple cube": "top_right",
    "yellow cube": "bottom_left",
}

bin_poses = {} # will contain entries such as "top_left": (x, y) after move_square is called by perception

# unpacks object info dict into tuple
def unpack(info):
    return info["color"], info["x_center"], info["center_depth"], info["rotation_angle"], info["dist"]

# get the estimated pose from odometry
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

# move the robot along a square's perimeter specified by the top-left corner's horizontal and vertical distances from origin
def move_square(top_left_horizontal, top_left_vertical):
    global bin_poses
    left_additional_deg = 5 # because it doesnt rotate left enough
    rate = rospy.Rate(10)
    global current_x, current_y, bin_poses
    bin_poses["bottom_right"] = (current_x, current_y)
    move_forward(top_left_vertical)
    bin_poses["top_right"] = (current_x, current_y)
    rotate(-90 - left_additional_deg)
    move_forward(top_left_horizontal)
    bin_poses["top_left"] = (current_x, current_y)
    rotate(-90 - left_additional_deg)
    move_forward(top_left_vertical)
    bin_poses["bottom_left"] = (current_x, current_y)
    rotate(-90 - left_additional_deg)
    move_forward(top_left_horizontal)
    rotate(-90 - left_additional_deg)

# turn the robot around 180 degrees
def turn_around():
    rotate(-180)
    
vel_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
tf_subscriber = rospy.Subscriber("/tf", TFMessage, tf_callback)

# normalize angle to [-pi, pi]
def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

# rotates the robot by the specified degrees amount (left is negative, right is positive)
def rotate(rotation_angle_deg):
    rate = rospy.Rate(10) 
    global current_rot
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
        f"Rotating from {np.rad2deg(start_rot):.2f}° to {-np.rad2deg(target_rot):.2f}°"
    )

    while not rospy.is_shutdown():
        angle_diff = normalize_angle(target_rot - current_rot)

        if abs(angle_diff) < 0.01: # close enough
            break

        twist = Twist()
        twist.angular.z = 0.5 * angle_diff
        # This looks like is what cause the move forward always seem off by some cm
        # twist.linear.x = 0.1
        vel_publisher.publish(twist)
        rate.sleep()

    vel_publisher.publish(Twist())

# moves robot backwards by dist
def move_backwards(dist):
    move_straight(dist, -0.1)

# moves robot forward by dist
def move_forward(dist):
    move_straight(dist, 0.1)

def move_straight(dist, move_speed):
    global current_x, current_y
    start_x = current_x
    start_y = current_y

    rate = rospy.Rate(10)

    twist = Twist()
    move_speed = move_speed # m/s

    while not rospy.is_shutdown():
        distance_moved = np.sqrt(
            (current_x - start_x) ** 2 + (current_y - start_y) ** 2
        )

        distance_diff = abs(dist - distance_moved)

        if distance_diff < 0.01: # close enough
            break

        twist.linear.x = move_speed
        vel_publisher.publish(twist)
        rate.sleep()

    vel_publisher.publish(Twist())
    rospy.loginfo(f"Move complete.")

# wait until odometry info is received
def wait_for_odom():
    global odom_received
    rate = rospy.Rate(10)

    while not odom_received and not rospy.is_shutdown():
        rospy.loginfo("Waiting for odom...")
        rate.sleep() 

# move phase: rotate to align with cube and move in straight line towards cube
def move(msg):
    global current_rot, odom_received, start_x, start_y, current_x, current_y
    start_x = current_x
    start_y = current_y
    color, x_center, center_depth, rotation_angle_deg, dist = unpack(msg)

    rotation_angle_deg -= 8.7 # slight adjustment needed due to depth calculation not being right in the center of the camera

    wait_for_odom()

    rotate(rotation_angle_deg)
	
    rospy.sleep(0.5)
    
    dist -= 0.0254 # about an inch to account for arm handle
    rospy.loginfo(f"[{color}] Moving forward {dist:.2f} meters...")
    move_forward(dist)

# returns target rotation (deg) from np.arctan2 angle (rad)
# angle (rad) is positive from the positive x-axis, x and y are bin's coordinates
# we are assuming all cartesian coordinates
def target_rot_from_arctan2(angle, x, y, current_rot):
    # arctan2 returns a positive angle from the positive x-axis, so we need to calculate the angle the robot needs to turn
    turn_right_extra = 90 # deg, to account current_rot misalignment
    angle = np.rad2deg(angle)
    current_rot = np.rad2deg(current_rot) # account for the robot's current angle alignment for turn amount
    if x >= 0 and y >= 0: # quadrant 1
        return 90 - angle - current_rot + turn_right_extra
    elif (x < 0 and y > 0) or (x <=0 and y <= 0): # quadrant 2 or 3
        return -(angle - 90) - current_rot + turn_right_extra
    else: # quadrant 4
        return 360 - angle + 90 - current_rot + turn_right_extra

# sort phase: rotate to the bin, move straight-line to the bin
def sort(obj_info):
    global current_rot, odom_received, start_x, start_y, current_x, current_y
    start_x = current_x
    start_y = current_y
    color, x_center, center_depth, rotation_angle_deg, dist = unpack(obj_info)
    
    wait_for_odom()

    bin_pose_x, bin_pose_y = bin_poses[color_to_bin[color]] # x, y NOT in cartesian coordinates
    # the calculated odom pose assumes that positive x is the standard positive y ("vertically up"), and the positive y is the standard negative x ("horizontally left")
    dx = bin_pose_x - current_x
    dy = bin_pose_y - current_y
    angle_from_positive_axis = np.arctan2(dx, -dy) # arctan2 assumes cartesian coordinates
    angle_to_turn = target_rot_from_arctan2(angle_from_positive_axis, -dy, dx, current_rot) # pass in cartesian coordinates
    rotate(angle_to_turn)

    rospy.sleep(0.5)

    dist = np.sqrt(
        (bin_pose_x - current_x)**2 + (bin_pose_y - current_y)**2
    )
    move_forward(dist)
    rospy.loginfo(f"[{color}] Sort complete.")