#!/usr/bin/env python3

# For getting and processing input data from realsense camera

from random_walk import RandomController
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
import ros_numpy
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np

# from movement import move
from movement_odom import move, sort
import os
import time

RGB_IMG_WIDTH = 1920  # pixels
DEPTH_IMG_WIDTH = 1280
MODEL_IMG_WIDTH = 640
IMG_CENTER = MODEL_IMG_WIDTH / 2
COLORS = [
    "green cube",
    "grey cube",
    "purple cube",
    "yellow cube",
]
FOV_DEGREES = 69  # RGB field of view in degrees (horizontal)
PIXELS_PER_DEGREE = MODEL_IMG_WIDTH / FOV_DEGREES

print(os.getcwd())
# Initialize a YOLO-World model
model = YOLO("/catkin_ws/src/object_sorter/best.onnx")
# model.set_classes(COLORS)


class Perception:
    def __init__(self):
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
        self.sorting_mode = False
        self.moving_mode = False
        self.idle_mode = True

        self.last_process_time = time.time()
        self.process_interval = 0.5  # sleep time between image process (s)

        rospy.loginfo("Running Perception node.")

    # resize to width needed for model
    def resize_to_model_size(self, img):
        img = self.bridge.imgmsg_to_cv2(img, desired_encoding="passthrough")
        return cv2.resize(
            img, (MODEL_IMG_WIDTH, MODEL_IMG_WIDTH), interpolation=cv2.INTER_LINEAR
        )

    # track objects and initiate movement
    def image_callback(self, msg):
        # Note: use below code if you want to see the actual image
        if time.time() - self.last_process_time < self.process_interval:  # skip frame
            return

        # print("Entered image callback...")
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

        # as_np_arr = ros_numpy.numpify(msg)
        # print("1")
        # result = model.predict(as_np_arr)
        # # result[0].show()
        # print("2")
        # color, x_center, center_depth, rotation_angle, dist = self.get_object_info(
        #     result
        # )
        # move(color, x_center, center_depth, rotation_angle, dist)
        resized_img = self.resize_to_model_size(msg)
        self.rgb_image = resized_img
        self.d_image = self.depth_image

        self.last_process_time = time.time()

    def run_simple(self):
        while not rospy.is_shutdown():
            if self.rgb_image is None:
                continue
            rgb_image = self.rgb_image
            depth_image = self.d_image
            # as_np_arr = ros_numpy.numpify(resized_img)
            print("=====> START TRACKING...")
            result = model.track(rgb_image, persist=True)
            print("=====> DONE TRACKING!")
            # result[0].show()

            # if there are no tracked objects, return
            contains_tracked_items = result[0].boxes.id is not None
            if not contains_tracked_items:
                print("!! No tracked objects, random walk !!")
                self.random_walk_controller.action_execute()
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

    def run(self):
        while not rospy.is_shutdown():
            if self.rgb_image is None:
                continue
            rgb_image = self.rgb_image
            depth_image = self.d_image
            # as_np_arr = ros_numpy.numpify(resized_img)
            if self.idle_mode:
                print("=====> START TRACKING...")
                result = model.track(rgb_image, persist=True)
                print("=====> DONE TRACKING!")
                # result[0].show()

                # if there are no tracked objects, return
                contains_tracked_items = result[0].boxes.id is not None
                if not contains_tracked_items:
                    print("!! No tracked objects, random walk !!")
                    self.random_walk_controller.action_execute()
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
                # self.object_detected_publisher.publish(bool_msg)
            else: # we already locked onto an object
                self.moving_or_sorting()

    # handles logic of whether to keep moving or sorting (or go into idle status again)
    def moving_or_sorting(self):
        status = None
        if self.idle_mode:
            rospy.loginfo(f"Moving object {self.cur_tracked_obj}...")
            status = move(
                self.tracked_objects[self.cur_tracked_obj]
            )  # moves a bit and returns whether done moving to object (but has not started sorting behavior yet)
            self.idle_mode = False
            self.moving_mode = True
            self.sorting_mode = False
        elif self.moving_mode:
            status = move(self.tracked_objects[self.cur_tracked_obj])
        elif self.sorting_mode:
            status = sort(self.tracked_objects[self.cur_tracked_obj])

        if status == "finished_moving":
            rospy.loginfo(
                f"Finished moving tracked object {self.cur_tracked_obj}, Now sorting..."
            )
            self.sorting_mode = True  # now we want to sort
            self.moving_mode = False
            self.idle_mode = False
        elif status == "finished_sorting":
            rospy.loginfo(f"Finished sorting tracked object {self.cur_tracked_obj}!")
            self.sorting_mode = False
            self.moving_mode = False
            self.idle_mode = True

    # returns the minimum depth from all pixels in the box
    def get_center_depth(box, depth_image):
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
            center_depth = self.get_center_depth(box, depth_image) # in meters

            # rotation angle
            # zero when object is centered, negative if object is to the left, positive if to the right
            pixels_from_center = x_center - IMG_CENTER
            rotation_angle = (
                pixels_from_center / PIXELS_PER_DEGREE
            )  # rot angle in degrees

            # dist from robot
            x_offset = center_depth * np.tan(
                np.deg2rad(rotation_angle)
            )  # dist from object center to center of image (m)
            dist = np.sqrt(
                x_offset**2 + center_depth**2
            )  # actual straight-line dist (m) from camera to object

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
                    "center_depth": center_depth,
                    "rotation_angle": rotation_angle,
                    "dist": dist,
                    "conf": conf,
                    "box": box,
                }
        return info, obj_ids[np.argmax(confs)]

    # sets the depth image (numpy array)
    def depth_image_callback(self, msg):
        self.depth_image = self.resize_to_model_size(msg)


if __name__ == "__main__":
    try:
        perception = Perception()
        perception.run_simple()
        # rospy.spin()
    except rospy.ROSInterruptException:
        pass
