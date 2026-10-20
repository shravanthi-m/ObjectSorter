#!/usr/bin/env python3
from random_walk import RandomController
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
import ros_numpy
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np

from movement_odom import move, sort, move_square, turn_around, move_backwards, rotate, move_forward, wait_for_odom
import os
import time

MODEL_IMG_WIDTH = 640
IMG_HEIGHT = 480
IMG_CENTER = MODEL_IMG_WIDTH / 2
COLORS = [
    "green cube",
    "grey cube",
    "purple cube",
    "yellow cube",
]
FOV_DEGREES = 69  # RGB field of view in degrees (horizontal)
PIXELS_PER_DEGREE = MODEL_IMG_WIDTH / FOV_DEGREES

# Initialize a YOLO-World model
model = YOLO("/catkin_ws/src/object_sorter/best.onnx")

class Perception:
    def __init__(self):
        # user-specified width and height of rectangular space
        self.top_left_horizontal = float(rospy.get_param("width")) 
        self.top_left_vertical = float(rospy.get_param("height"))
        print(f"Space set to {self.top_left_horizontal} x {self.top_left_vertical} m rectangle")

        move_square(self.top_left_horizontal, self.top_left_vertical) # trace area and get bin poses

        self.bridge = (
            CvBridge()
        )  # converts between ROS Image messages and OpenCV images
        
        self.depth_image = None  # in model size
        self.rgb_image = None  # in model size
        self.d_image = None  # in model size

        # subscribers
        self.color_image_susbcriber = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.image_callback
        )  # rgb image from realsense camera
        self.depth_image_subscriber = rospy.Subscriber(
            "/camera/aligned_depth_to_color/image_raw", Image, self.depth_image_callback
        ) # 640 x 480

        # publishers
        # self.object_detected_publisher = rospy.Publisher(
        #     "/object_detected", Bool, queue_size=10
        # )

        self.random_walk_controller = RandomController()
        self.tracked_objects = {}
        self.cur_tracked_obj = None

        self.last_process_time = time.time()
        self.process_interval = 0.5  # sleep time between image process (s)

        rospy.loginfo("Running Perception node.")

    # resize to width needed for model
    def resize_to_model_size(self, img):
        img = self.bridge.imgmsg_to_cv2(img, desired_encoding="passthrough")
        return cv2.resize(
            img, (MODEL_IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR
        )

    # track objects and initiate movement
    def image_callback(self, msg):
        if time.time() - self.last_process_time < self.process_interval:  # skip frame
            return

        if self.depth_image is None:
            rospy.logerr("!! Matching depth image not available, exiting !!")
            return
        
        resized_img = self.resize_to_model_size(msg)
        self.rgb_image = resized_img
        self.d_image = self.depth_image

        self.last_process_time = time.time()

    # perception-action loop: from camera images, classify objects and initiate moving and sorting behaviors
    def run(self):
        # align to the center of the space and get closer
        wait_for_odom()
        rotate(-45)
        wait_for_odom()
        move_forward(0.15)

        # keep trying to track items
        while not rospy.is_shutdown():
            if self.rgb_image is None:
                continue
            rgb_image = self.rgb_image
            depth_image = self.d_image
            print("=====> START TRACKING...")
            result = model.track(rgb_image, persist=True)
            print("=====> DONE TRACKING!")

            # if there are no tracked objects, return
            contains_tracked_items = result[0].boxes.id is not None
            if not contains_tracked_items:
                print("!! No tracked objects, random walk !!")
               #  self.random_walk_controller.action_execute()
                continue

            objs_info, highest_conf_id = self.get_objects_info(
                result, depth_image
            )  # color, x_center, center_depth, rotation_angle, dist from robot, conf, box for each obj (filters out dist = 0) + obj id with highest confidence
            if not objs_info:
                print("!! All detected objects have distance = 0, exiting !!")
                continue

            self.tracked_objects = objs_info  # store internally

            self.cur_tracked_obj = (
                highest_conf_id  # "lock onto" object with the highest confidence
            )

            detected_msg = Bool()
            detected_msg.data = True

            move(self.tracked_objects[self.cur_tracked_obj])
            sort(self.tracked_objects[self.cur_tracked_obj])

            # move away and turn around to face center of space again
            wait_for_odom()
            move_backwards(0.15)
            wait_for_odom()
            turn_around()
            
           #  self.random_walk_controller.action_execute()

    # returns the minimum depth (m) from all pixels in the box
    def get_center_depth(self, box, depth_image):
        x1, y1, x2, y2 = (
            box  # NOT normalized (top-left and bottom-right corners of bounding box)
        )
        depth_sub_box = depth_image[int(y1):int(y2), int(x1):int(x2)]
        return float(np.min(depth_sub_box[np.nonzero(depth_sub_box)])) * 0.001

    # returns {obj id: (color, x_center, center_depth, rotation_angle, dist from robot, confidence, box} dict for tracked objects from yolo result
    # Note: filters out dist = 0
    def get_objects_info(self, yolo_result, depth_image):
        bounding_boxes = yolo_result[0].boxes.xyxy.cpu()
        classes = yolo_result[0].boxes.cls.cpu().numpy().astype(int)
        obj_ids = yolo_result[0].boxes.id.int().cpu().tolist()
        confs = yolo_result[0].boxes.conf.cpu().numpy().astype(float)

        info = {}
        for box, class_id, obj_id, conf in zip(bounding_boxes, classes, obj_ids, confs):
            # color
            color = COLORS[class_id]

            # x_center
            x1, y1, x2, y2 = (
                box  # NOT normalized (top-left and bottom-right corners of bounding box)
            )
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)

            # center_depth
            #dist = self.get_center_depth(box, depth_image) # in meters
            dist = float(depth_image[y_center, x_center]) * 0.001 # meters

            # rotation angle
            # zero when object is centered, negative if object is to the left, positive if to the right
            pixels_from_center = x_center - IMG_CENTER
            rotation_angle = (
                pixels_from_center / PIXELS_PER_DEGREE
            )  # rot angle in degrees

            if dist == 0:
                rospy.loginfo(
                    f"Ignoring detected object with id = {obj_id} because dist = 0"
                )
            elif conf <= 0.8:
                rospy.loginfo(
                    f"Ignoring detected object with id = {obj_id} because conf = {conf} <= 0.8"
                )
            else:
                rospy.loginfo(
                    f"Detected {color} object with id = {obj_id}, rotation angle = {rotation_angle} degrees, distance = {dist} m"
                )
                info[obj_id] = {
                    "color": color,
                    "x_center": x_center,
                    "center_depth": dist,
                    "rotation_angle": rotation_angle,
                    "dist": dist,
                    "conf": conf,
                    "box": box,
                }
        return info, obj_ids[np.argmax(confs)]

    # sets the depth image
    def depth_image_callback(self, msg):
        self.depth_image = self.resize_to_model_size(msg)

if __name__ == "__main__":
    try:
        perception = Perception()
        perception.run()
        # rospy.spin()
    except rospy.ROSInterruptException:
        pass
