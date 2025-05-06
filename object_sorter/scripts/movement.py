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

# receives the CURRENT information about the object being tracked
# returns status "finished_moving", "finished_sorting", or None TODO
def move(obj_info):
    color, x_center, center_depth, rotation_angle, dist = unpack(obj_info)




    # note: left the below unchanged from our last meeting
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

def unpack(info):
    return info["color"], info["x_center"], info["center_depth"], info["rotation_angle"], info["dist"]