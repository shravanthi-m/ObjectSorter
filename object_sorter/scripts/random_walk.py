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

class RandomController:
    def __init__(self):
        rospy.init_node('random_walk_controller')
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.dist = 0.5
        self.threshold = 0.2
        self.actions = ["turn_left", "move_forward", "turn_right"]
        self.left_dist = min(scan.ranges[30:90])
        self.front_dist = min(scan.ranges[0:30] + scan.ranges[-30:])
        self.right_dist = min(scan.ranges[-90:-30])
        self.case = ''
        self.msg = Twist()

    def scan_callback(self, scan):
        # left_dist = min(scan.ranges[30:90])
        # front_dist = min(scan.ranges[0:30] + scan.ranges[-30:])
        # right_dist = min(scan.ranges[-90:-30])

        s_left = self.discretize_distance(self.left_dist)
        s_front = self.discretize_distance(self.front_dist)
        s_right = self.discretize_distance(self.right_dist)
        
        state = (s_left, s_front, s_right)

        action = self.pick_action(state)

        range={
            "right" : self.right_dist,
            "center" : self.front_dist,
            "left" : self.left_dist
        }
        check_obstacle(range)
       
        # self.action_execute(action)
        rospy.spin()

    def discretize_distance(self, distance):
        if distance < self.dist - self.threshold:
            return "close"
        elif self.dist - self.threshold <= distance <= self.dist + self.threshold:
            return "desired"
        else:
            return "far"

    def pick_action(self):
        return random.choice(self.actions)

    # def action_execute(self, action):
    #     twist = Twist()
    #     if action == "move_forward":
    #         twist.linear.x = 0.5
    #         twist.angular.z = 0.0
    #     elif action == "turn_left":
    #         twist.linear.x = 0.1
    #         twist.angular.z = 0.5
    #     elif action == "turn_right":
    #         twist.linear.x = 0.1
    #         twist.angular.z = -0.5
    #     elif action == "move_backward":
    #         twist.linear.x = -0.5
    #         twist.angular.z = 0.0
    #     self.pub.publish(twist)

    def check_obstacle(self, range):
        if (range["right"] >1  and range["center"] > 1 and range["left"] >1):
            self.case = 'NO OBSTACLE!'
            linearx=0.6
            angularz=0
        elif (range["right"] > 1  and range["center"] < 1 and range["left"] > 1 ):
            self.case = 'OBSTACLE CENTER!'
            linearx=0
            angularz=-0.5
        elif ( range["right"] < 1  and range["center"] > 1 and range["left"] > 1 ):
            self.case = 'OBSTACLE RIGHT!'
            linearx=0
            angularz=0.5
        elif ( range["right"] > 1  and range["center"] > 1 and range["left"] < 1 ):
            self.case = 'OBSTACLE LEFT!'
            linearx=0
            angularz=-0.5
        elif ( range["right"] < 1  and range["center"] > 1 and range["left"] < 1 ):
            self.case = 'OBSTACLE RIGHT AND LEFT!'
            linearx=0.6
            angularz=0
        elif ( range["right"] > 1  and range["center"] < 1 and range["left"] < 1 ):
            self.case = 'OBSTACLE CENTER AND LEFT!'
            linearx=0
            angularz=-0.5
        elif ( range["right"] < 1  and range["center"] < 1 and range["left"] > 1 ):
            self.case = 'OBSTACLE CENTER AND RIGHT!'
            linearx=0
            angularz=0.5
        elif ( range["right"] < 1  and range["center"] < 1 and range["left"] < 1 ):
            self.case = 'OBSTACLE AHEAD!'
            linearx=0
            angularz=0.8

        rospy.loginfo(self.case)
        self.msg.linear.x = linearx
        self.msg.angular.z = angularz
        self.pub.publish(msg)

if __name__ == '__main__':
    RandomController()
