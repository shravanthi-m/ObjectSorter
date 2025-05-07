#!usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import random

#random walk and obstacle avoidance
#set ranges for left, front, right
#define action space and randomly pick actions from it 
#check if there is obstacle in the direction
#if no obstacle, move in that direction


#subscribe to a topic included in perception.py - assuming it is /obejct_detected
#if object detected according to the topic
#then stop random walk and go to movement_odom


class RandomController:
    def __init__(self):
        rospy.init_node('random_walk_controller')
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        rospy.Subscriber('/object_detected', Bool, self.inference_callback)

        self.dist = 0.5
        self.threshold = 0.2
        self.actions = ["turn_left", "move_forward", "turn_right"]
        self.ready_for_next_action = True
        self.inference_received = False
        self.left_dist = self.front_dist = self.right_dist = 999.0

    def scan_callback(self, scan):
        if self.object_detected:
            return
        self.left_dist = min(scan.ranges[30:90])
        self.front_dist = min(scan.ranges[0:30] + scan.ranges[-30:])
        self.right_dist = min(scan.ranges[-90:-30])

        if self.ready_for_next_action:
            self.ready_for_next_action = False
            self.inference_received = False
            self.action_execute()

    def inference_callback(self, msg):
        self.object_detected = msg.data  # pause walk if True
        if msg.data:
            rospy.loginfo("Object detected — pausing random walk.")
        else:
            rospy.loginfo("No object detected — resuming random walk.")

    def check_obstacle(self):
        if self.right_dist > 1 and self.front_dist > 1 and self.left_dist > 1:
            return "clear"
        return "blocked"

    def pick_action(self):
        return random.choice(self.actions)

    def action_execute(self):
        twist = Twist()
        status = self.check_obstacle()

        if status == "clear":
            action = self.pick_action()
            rospy.loginfo(f"No Obstacle - Action: {action}")
            if action == "move_forward":
                twist.linear.x = 0.5
            elif action == "turn_left":
                twist.linear.x = 0.1
                twist.angular.z = 0.5
            elif action == "turn_right":
                twist.linear.x = 0.1
                twist.angular.z = -0.5
        else:
            rospy.loginfo("Obstacle detected - Avoiding")
            if self.front_dist < 1 and self.left_dist < 1 and self.right_dist < 1:
                twist.angular.z = 0.8
            elif self.front_dist < 1:
                twist.angular.z = -0.5 if self.left_dist > self.right_dist else 0.5
            elif self.left_dist < 1:
                twist.angular.z = -0.5
            elif self.right_dist < 1:
                twist.angular.z = 0.5
            else:
                twist.linear.x = 0.2

        self.pub.publish(twist)
        rospy.sleep(2.5) 

        stop_msg = Twist()
        self.pub.publish(stop_msg)

        self.wait_for_inference_or_timeout(50)

    def wait_for_inference_or_timeout(self, timeout_sec=10):
        rate = rospy.Rate(timeout_sec)
        elapsed = 0
        while not rospy.is_shutdown() and elapsed < timeout_sec and not self.inference_received:
            rate.sleep()
            elapsed += 0.1

        self.ready_for_next_action = True

if __name__ == '__main__':
    try:
        RandomController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
