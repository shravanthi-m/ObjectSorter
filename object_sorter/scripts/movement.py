from geometry_msgs.msg import Twist
import rospy

vel_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

def move(color, x_center, center_depth, rotation_angle, dist):
    while not rospy.is_shutdown():
        twist = Twist()
        if rotation_angle == 0.0:
            while dist > 0.1:
                twist.linear.x = 0.2
                twist.angular.z = 0
            twist.linear.x = 0
        else:    
            twist.linear.x = 0
            twist.linear.y = 0
            twist.angular.z = rotation_angle

    print("helooo")
    vel_publisher.publish(twist)