#!/usr/bin/env python3

# For getting and processing input data from realsense camera

import rospy
from sensor_msgs.msg import Image
import ros_numpy
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
from movement import move, sort
import os
import time

RGB_IMG_WIDTH = 1920  # pixels
DEPTH_IMG_WIDTH = 1280
MODEL_IMG_WIDTH = 640
IMG_CENTER = MODEL_IMG_WIDTH / 2
COLORS = [
    "blue cuboid",
    "blue cylinder",
    "green cuboid",
    "grey cuboid",
    "purple cuboid",
    "red cuboid",
]
FOV_DEGREES = 69  # RGB field of view in degrees (horizontal)
PIXELS_PER_DEGREE = MODEL_IMG_WIDTH / FOV_DEGREES

print(os.getcwd())
# Initialize a YOLO-World model
model = YOLO("/catkin_ws/src/object_sorter/best.onnx")
# model.set_classes(COLORS)

class Perception:
    def __init__(self):
        rospy.init_node("perception", anonymous=True)
        self.bridge = (
            CvBridge()
        )  # converts between ROS Image messages and OpenCV images
        self.depth_image = None # in model size

        # subscribers
        self.color_image_susbcriber = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.image_callback
        )  # rgb image from realsense camera
        self.depth_image_subscriber = rospy.Subscriber(
            "/camera/aligned_depth_to_color/image_raw", Image, self.depth_image_callback
        )

        self.tracked_objects = {}
        self.cur_tracked_obj = None
        self.sorting_mode = False
        self.moving_mode = False
        self.idle_mode = False

        self.last_process_time = time.time()
        self.process_interval = 0.5 # sleep time between image process (s)

        rospy.loginfo("Running Perception node.")
    
    # resize to width needed for model
    def resize_to_model_size(img):
        img = self.bridge.imgmsg_to_cv2(img, desired_encoding="passthrough")
        return cv2.resize(img, (MODEL_IMG_WIDTH, img.shape[0]), interpolation=cv2.INTER_LINEAR)

    # track objects and initiate movement
    def image_callback(self, msg):
        if time.time() - self.last_process_time < self.process_interval: # skip frame
            return
        
        print("Entered image callback...")
        if self.depth_image is None:
            rospy.logerr("!! Matching depth image not available, exiting !!")
            return
        # Note: use below code if you want to see the actual image
        # try:
        #     img = self.bridge.imgmsg_to_cv2(
        #         msg, desired_encoding="passthrough"
        #     )  # desired_encoding arg for preserving img data type
        # except Exception as e:
        #     rospy.logerr("Can't convert Image to OpenCv image")
        #     return

        # show the actual image
        # cv2.imshow("Realsense image feed", img)
        # cv2.waitKey(1) # wait 1 ms to let the window refresh, waits for key press
        # cv2.destroyAllWindows()

        resized_img = self.resize_to_model_size(msg)

        # as_np_arr = ros_numpy.numpify(resized_img)
        print("=====> START TRACKING...")
        result = model.track(resized_img, persist=True)
        print("=====> DONE TRACKING!")
        # result[0].show()

        # if there are no tracked objects, return
        contains_tracked_items = result[0].boxes.id is not None
        if not contains_tracked_items:
            print("!! No tracked objects, exiting !!")
            return
        
        objs_info, highest_conf_id = self.get_objects_info(result) # color, x_center, center_depth, rotation_angle, dist from robot, conf, box for each obj (filters out dist = 0) + obj id with highest confidence
        if not objs_info:
            print("!! All detected objects have distance = 0, exiting !!")
            return

        self.tracked_objects = objs_info # store internally

        self.last_process_time = time.time()

        if self.idle_mode:
            self.cur_tracked_obj = highest_conf_id # "lock onto" object with the highest confidence
        
        if self.cur_tracked_obj not in self.tracked_objects:
            print(f"!! Lost tracked object {self.cur_tracked_obj}, exiting and starting over !!")
            self.idle_mode = True
            self.moving_mode = False
            self.sorting_mode = False
            return
        
        self.moving_or_sorting()
    
    # handles logic of whether to keep moving or sorting (or go into idle status again)
    def moving_or_sorting():
        status = None
        if self.idle_mode:
            rospy.loginfo(f"Moving object {self.cur_tracked_obj}...")
            status = move(self.tracked_objects[self.cur_tracked_obj]) # moves a bit and returns whether done moving to object (but has not started sorting behavior yet)
            self.idle_mode = False
            self.moving_mode = True
        elif self.moving_mode:
            status = move(self.tracked_objects[self.cur_tracked_obj])
        elif self.sorting_mode:
            status = sort(self.tracked_objects[self.cur_tracked_obj])

        if status == "finished_moving":
            rospy.loginfo(f"Finished moving tracked object {self.cur_tracked_obj}, Now sorting...")
            self.sorting_mode = True # now we want to sort
            self.moving_mode = False
            self.idle_mode = False
        elif status == "finished_sorting":
            rospy.loginfo(f"Finished sorting tracked object {self.cur_tracked_obj}!")
            self.sorting_mode = False
            self.moving_mode = False
            self.idle_mode = True

    # returns {obj id: (color, x_center, center_depth, rotation_angle, dist from robot, confidence, box} dict for tracked objects from yolo result
    # Note: filters out dist = 0
    def get_objects_info(self, yolo_result):
        bounding_boxes = yolo_result[0].boxes.xyxy.cpu()
        classes = yolo_result[0].boxes.cls.cpu().numpy().astype(int)
        obj_ids = yolo_result[0].boxes.id.int().cpu().tolist()
        confs = yolo_result[0].boxes.conf.cpu().numpy().astype(int)

        info = {}
        for box, class_id, obj_id, conf in zip(bounding_boxes, classes, obj_ids, confs):
            # color
            color = COLORS[class_id]

            # x_center
            x1, y1, x2, y2 = box # NOT normalized (top-left and bottom-right corners of bounding box)
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)

            # center_depth
            center_depth = (
                float(self.depth_image[y_center, x_center]) * 0.001
            )  # mm to m probably
        
            # rotation angle
            # zero when object is centered, negative if object is to the left, positive if to the right
            pixels_from_center = x_center - IMG_CENTER
            rotation_angle = pixels_from_center / PIXELS_PER_DEGREE  # rot angle in degrees

            # dist from robot
            x_offset = center_depth * np.tan(
                np.deg2rad(rotation_angle)
            )  # dist from object center to center of image (m)
            dist = np.sqrt(
                x_offset**2 + center_depth**2
            )  # actual straight-line dist (m) from camera to object

            if dist == 0:
                rospy.loginfo(f"Ignoring detected object with id = {obj_id} because dist = 0")
            else:
                rospy.loginfo(f"Detected {color} object with id = {obj_id}, rotation angle = {rotation_angle} degrees, distance = {dist} m")
                info[obj_id] = {
                    "color": color,
                    "x_center": x_center,
                    "center_depth": center_depth,
                    "rotation_angle": rotation_angle,
                    "dist": dist,
                    "conf": conf,
                    "box": box}
        return info, obj_ids[np.argmax(confs)]

    # sets the depth image (numpy array)
    def depth_image_callback(self, msg):
        self.depth_image = self.resize_to_model_size(msg)

if __name__ == "__main__":
    try:
        perception = Perception()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
