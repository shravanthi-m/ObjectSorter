#!/usr/bin/env python3

# For getting and processing input data from realsense camera

import rospy
from sensor_msgs.msg import Image
import ros_numpy
from cv_bridge import CvBridge
import cv2
from yolo import RunYOLO 
from movement import move

class Perception:
    def __init__(self):
        rospy.init_node("perception", anonymous=True)
        self.bridge = CvBridge() # converts between ROS Image messages and OpenCV images
        self.depth_image = None

        # subscribers
        self.color_image_susbcriber = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback) # rgb image from realsense camera
        self.depth_image_subscriber = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_image_callback)

        rospy.loginfo("Running Perception node.")

    # passes classification result and spatial info to movement function
    def image_callback(self, msg):
        # Note: use below code if you want to see the actual image
        # try:
        #     img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough") # desired_encoding arg for preserving img data type
        # except Exception as e:
        #     rospy.logerr("Can't convert Image to OpenCv image")
        #     return

        # show the actual image
        # cv2.imshow("Realsense image feed", img)
        # cv2.waitKey(1) # wait 1 ms to let the window refresh, waits for key press
        # cv2.destroyAllWindows()

        as_np_arr = ros_numpy.numpify(msg)
        result = RunYOLO.get_result(as_np_arr)  # TODO: sending and getting result from yolo
        x_center, center_depth = get_spatial_info(result) # TODO: get x center and its depth
        color = result # TODO: get the color
        move(color, x_center, center_depth)
    
    # returns x center and center depth
    def get_spatial_info(self, yolo_result):
        x1, y1, x2, y2 = yolo_result[0].boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)[0] # get coordinates in xyxy format from bounding boxes TODO: test this out
        x_center = int((x1 + x2) / 2)
        y_center = int((y1 + y2) / 2)

        center_depth = self.depth_image[y_center, x_center] # (0, 0) is at top-left corner of image
        return x_center, float(center_depth) * 0.001 # TODO: idk the units for the depth and if we need to convert -- this is currently doing mm -> m

    # sets the depth image (numpy array)
    def depth_image_callback(self, msg):
        self.depth_image = ros_numpy.numpify(msg)

if __name__ == "__main__":
    try:
        perception = Perception()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
