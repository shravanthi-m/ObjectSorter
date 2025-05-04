#!/usr/bin/env python3
from geometry_msgs.msg import Twist, Pose
import rospy
import numpy as np
from tf.transformations import euler_from_quaternion


pose = Pose()
def pose_callback(msg):
    global pose
    pose = msg

vel_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
pose_sub = rospy.Subscriber("/pose", Pose, pose_callback)

def move(color, x_center, center_depth, rotation_angle, dist):
    global pose
    while not rospy.is_shutdown():
        print(euler_from_quaternion((pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)))
        angle_diff = np.deg2rad(rotation_angle) - euler_from_quaternion((pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w))[2]
        twist = Twist()
        print(angle_diff)
        if abs(angle_diff) <= 0.01:
            while dist > 0.1:
                twist.linear.x = 0.2
                twist.angular.z = 0
            twist.linear.x = 0
        else:    
            twist.linear.x = 0.1
            twist.linear.y = 0
            twist.angular.z = -angle_diff * 0.5

        vel_publisher.publish(twist)
    return True
