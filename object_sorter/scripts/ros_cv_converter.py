#!/usr/bin/env python3
import numpy as np
import cv2
from sensor_msgs.msg import Image

# ROS Image message encodings
# Reference: http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/Image.html
# Added here for convenience
class Encodings:
    RGB8 = "rgb8"
    RGBA8 = "rgba8"
    RGB16 = "rgb16"
    RGBA16 = "rgba16"
    BGR8 = "bgr8"
    BGRA8 = "bgra8"
    BGR16 = "bgr16"
    BGRA16 = "bgra16"
    MONO8 = "mono8"
    MONO16 = "mono16"
    TYPE_16UC1 = "16UC1"
    BAYER_RGGB8 = "bayer_rggb8"
    BAYER_BGGR8 = "bayer_bggr8"
    BAYER_GBRG8 = "bayer_gbrg8"
    BAYER_GRBG8 = "bayer_grbg8"
    YUV422 = "yuv422"

class CvBridge:
    """
    Custom implementation of cv_bridge using only OpenCV for Python 3.8.
    Converts between ROS Image messages and OpenCV images.
    """

    # Mapping between ROS image encodings and OpenCV formats
    ENCODINGS = {
        # RGB encodings
        "rgb8": (cv2.COLOR_RGB2BGR, np.uint8, 3),
        "rgba8": (cv2.COLOR_RGBA2BGRA, np.uint8, 4),
        "rgb16": (cv2.COLOR_RGB2BGR, np.uint16, 3),
        "rgba16": (cv2.COLOR_RGBA2BGRA, np.uint16, 4),

        # BGR encodings (OpenCV default)
        "bgr8": (None, np.uint8, 3),
        "bgra8": (None, np.uint8, 4),
        "bgr16": (None, np.uint16, 3),
        "bgra16": (None, np.uint16, 4),

        # Mono encodings
        "mono8": (None, np.uint8, 1),
        "mono16": (None, np.uint16, 1),

        # 16-bit depth encodings
        "16UC1": (None, np.uint16, 1),

        # Bayer encodings
        "bayer_rggb8": (cv2.COLOR_BayerBG2BGR, np.uint8, 1),
        "bayer_bggr8": (cv2.COLOR_BayerRG2BGR, np.uint8, 1),
        "bayer_gbrg8": (cv2.COLOR_BayerGR2BGR, np.uint8, 1),
        "bayer_grbg8": (cv2.COLOR_BayerGB2BGR, np.uint8, 1),

        # YUV encodings
        "yuv422": (cv2.COLOR_YUV2BGR_YUYV, np.uint8, 2),
    }

    @staticmethod
    def imgmsg_to_cv2(ros_image, desired_encoding="passthrough"):
        """
        Convert a ROS Image message to an OpenCV image.

        Args:
            ros_image (sensor_msgs.msg.Image): ROS Image message
            desired_encoding (str): The encoding of the output image (default: same as input)

        Returns:
            numpy.ndarray: OpenCV image
        """
        # Get source encoding
        src_encoding = ros_image.encoding

        # Check if encoding is known
        if src_encoding not in CvBridge.ENCODINGS:
            raise ValueError(f"Unsupported encoding: {src_encoding}")

        # Extract image data
        dtype = CvBridge.ENCODINGS[src_encoding][1]
        channels = CvBridge.ENCODINGS[src_encoding][2]

        # Create NumPy array from byte array
        if channels > 1:
            img = np.frombuffer(ros_image.data, dtype=dtype).reshape(
                ros_image.height, ros_image.width, channels)
        else:
            img = np.frombuffer(ros_image.data, dtype=dtype).reshape(
                ros_image.height, ros_image.width)

        # Apply color conversion if needed
        conversion = CvBridge.ENCODINGS[src_encoding][0]
        if conversion is not None:
            img = cv2.cvtColor(img, conversion)

        # Handle desired encoding if not passthrough
        if desired_encoding != "passthrough" and desired_encoding != src_encoding:
            if desired_encoding not in CvBridge.ENCODINGS:
                raise ValueError(f"Unsupported desired encoding: {desired_encoding}")

            # Convert to desired encoding
            if desired_encoding.startswith("mono") and not src_encoding.startswith("mono"):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif src_encoding.startswith("mono") and not desired_encoding.startswith("mono"):
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif desired_encoding.startswith("rgb") and not src_encoding.startswith("rgb"):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif desired_encoding.startswith("bgr") and not src_encoding.startswith("bgr"):
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    @staticmethod
    def cv2_to_imgmsg(cv_image, encoding="bgr8", header=None):
        """
        Convert an OpenCV image to a ROS Image message.

        Args:
            cv_image (numpy.ndarray): OpenCV image
            encoding (str): Encoding of the image
            header (std_msgs.msg.Header): Optional header to use in the ROS Image message

        Returns:
            sensor_msgs.msg.Image: ROS Image message
        """
        # Check if encoding is known
        if encoding not in CvBridge.ENCODINGS:
            raise ValueError(f"Unsupported encoding: {encoding}")

        # Create ROS Image message
        ros_image = Image()

        # Set the header if provided
        if header is not None:
            ros_image.header = header

        # Set image properties
        ros_image.height = cv_image.shape[0]
        ros_image.width = cv_image.shape[1]
        ros_image.encoding = encoding

        # Set step (row length in bytes)
        if len(cv_image.shape) == 3:
            ros_image.step = cv_image.shape[1] * cv_image.shape[2]
            if encoding.startswith("rgb") and cv_image.shape[2] == 3:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        else:
            ros_image.step = cv_image.shape[1] * 2 if encoding == "16UC1" or encoding == "mono16" else cv_image.shape[1]

        # Ensure correct data type for the encoding
        dtype = CvBridge.ENCODINGS[encoding][1]
        if cv_image.dtype != dtype:
            cv_image = cv_image.astype(dtype)

        # Convert to byte array
        ros_image.data = cv_image.tobytes()

        return ros_image


# This class can be used as a drop-in replacement for cv_bridge
# with support for 16UC1 and other common ROS image encodings

if __name__ == '__main__':
    print("CvBridge: Custom OpenCV-based implementation of cv_bridge")
