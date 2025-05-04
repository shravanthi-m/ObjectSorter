#!/usr/bin/env python3

# For getting and processing input data from realsense camera

import rospy
from sensor_msgs.msg import Image
import ros_numpy
# from cv_bridge import CvBridge
# import cv2
from ultralytics import YOLOWorld 
from movement import move

IMG_WIDTH = None
IMG_CENTER = IMG_WIDTH / 2
COLORS = ["purple cuboid", "grey cuboid", "purple cylinder"]
FOV_DEGREES = None # camera field of view in degrees
PIXELS_PER_DEGREE = IMG_WIDTH / FOV_DEGREES

# Initialize a YOLO-World model
model = YOLOWorld(
    "yolov8x-worldv2.pt"
)
model.set_classes(COLORS)

class Perception:
    def __init__(self):
        rospy.init_node("perception", anonymous=True)
        # self.bridge = CvBridge() # converts between ROS Image messages and OpenCV images
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
        result = model.predict(as_np_arr)
        color, x_center, center_depth, rotation_angle, dist = self.get_object_info(result)
        move(color, x_center, center_depth, rotation_angle, dist)

    # returns color, x_center, center_depth, rotation_angle, dist from robot from yolo result (for the first detected object only)
    def get_object_info(self, yolo_result):
        boxes = yolo_result[0].boxes # get bounding boxes
        
        # color
        class_id = boxes.cls.cpu().numpy().astype(int)[0]
        color = COLORS[class_id]
        
        x1, y1, x2, y2 = boxes.xyxy[0].cpu().numpy() # NOT normalized (top-left and bottom-right corners of bounding box)
        
        # center points
        x_center = int((x1 + x2) / 2)
        y_center = int((y1 + y2) / 2)
        
        # depth at center point
        if self.depth_image is None:
            rospy.logerr("Depth image not available")
            return None
        center_depth = float(self.depth_image[y_center, x_center]) * 0.001  # mm to m i think...
        
        # rotation angle
        # zero when object is centered, negative if object is to the left, positive if to the right
        pixels_from_center = x_center - IMG_CENTER
        rotation_angle = pixels_from_center / PIXELS_PER_DEGREE # rot angle in degrees
        
        # dist to move
        x_offset = center_depth * np.tan(np.deg2rad(rotation_angle)) # dist from object center to center of image (m)
        dist = np.sqrt(x_offset ** 2 + center_depth ** 2) # actual straight-line dist (m) from camera to object
        
        rospy.loginfo(f"Detected {color} object, rotation angle = {rotation_angle:.2f} degrees, movement distance {dist:.2f} m")
                        
        return color, x_center, center_depth, rotation_angle, dist

    # sets the depth image (numpy array)
    def depth_image_callback(self, msg):
        self.depth_image = ros_numpy.numpify(msg)

if __name__ == "__main__":
    try:
        perception = Perception()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
