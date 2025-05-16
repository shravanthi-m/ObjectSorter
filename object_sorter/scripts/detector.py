#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

class YoloTRT:
    def __init__(self, engine_path):
        # Load TensorRT engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        trt.init_libnvinfer_plugins(None, "")  
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Allocate memory for inputs/outputs
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for binding in range(self.engine.num_bindings):
            size = trt.volume(self.engine.get_binding_shape(binding)) * \
                   self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Append to the lists
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({"host": host_mem, "device": device_mem, "size": size})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem, "size": size})
        
        # Get input and output shapes
        self.input_shape = self.engine.get_binding_shape(0)
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        
        # Class names - replace with your model's classes
        self.classes = [
            "blue cuboid",
            "blue cylinder",
            "green cuboid",
            "grey cuboid",
            "purple cuboid",
            "red cuboid",
        ]
        
    def preprocess(self, img):
        # Resize and normalize image
        input_img = cv2.resize(img, (self.input_width, self.input_height))
        input_img = input_img.astype(np.float32)
        input_img = input_img / 255.0  # Normalize to [0,1]
        input_img = np.transpose(input_img, (2, 0, 1))  # HWC to CHW
        input_img = np.expand_dims(input_img, axis=0)  # Add batch dimension
        input_img = np.ascontiguousarray(input_img)  # Make contiguous
        return input_img
    
    def infer(self, img):
        # Preprocess
        input_img = self.preprocess(img)
        
        # Copy input to device
        np.copyto(self.inputs[0]["host"], input_img.ravel())
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        
        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Copy output back to host
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        
        self.stream.synchronize()
        
        # Process output based on your model's output format
        # This will vary depending on YOLOv5 vs YOLOv8 and output format
        # For simplicity, we'll assume the output is in the standard YOLOv5 format
        output = self.outputs[0]["host"]
        return self.process_output(output, img.shape)
    
    def process_output(self, output, orig_shape):
        """
        Process YOLOv11 model output to get detections
        """
        # Get original image dimensions
        orig_h, orig_w = orig_shape[0:2]
        
        # Reshape output based on YOLOv11 format
        # YOLOv11 typically uses a format with:
        # - First 4 values: box coordinates (x, y, w, h) or (x1, y1, x2, y2)
        # - Next value: object confidence
        # - Remaining values: class probabilities
        
        # This is where we need to adapt for YOLOv11's specific output format
        # The exact number of classes may vary based on your model
        num_classes = len(self.classes)
        
        # YOLOv11 may output in a different shape, adjust as needed
        # This assumes batch size of 1, with shape [1, num_detections, 5+num_classes]
        detections = output.reshape(-1, 5 + num_classes)
        
        boxes = []
        scores = []
        class_ids = []
        
        # Confidence threshold
        conf_threshold = 0.25
        # NMS threshold
        nms_threshold = 0.45
        
        for detection in detections:
            # Extract values
            x, y, w, h = detection[0:4]  # YOLOv11 may use center_x, center_y, width, height format
            obj_conf = detection[4]
            class_probs = detection[5:5+num_classes]
            
            # Get class with highest probability
            class_id = np.argmax(class_probs)
            class_conf = class_probs[class_id]
            
            # Calculate final confidence
            final_conf = obj_conf * class_conf
            
            # Filter by confidence
            if final_conf >= conf_threshold:
                # Convert to corner coordinates (x1, y1, x2, y2)
                # YOLOv11 might use different box formats, adjust as needed
                
                # If using center_x, center_y, width, height:
                x1 = int((x - w/2) * orig_w)
                y1 = int((y - h/2) * orig_h)
                x2 = int((x + w/2) * orig_w)
                y2 = int((y + h/2) * orig_h)
                
                # If already using corner coordinates:
                # x1 = int(x * orig_w)
                # y1 = int(y * orig_h)
                # x2 = int(w * orig_w) 
                # y2 = int(h * orig_h)
                
                # Add to lists
                boxes.append([x1, y1, x2, y2])
                scores.append(float(final_conf))
                class_ids.append(int(class_id))
        
        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
        
        # Extract final detections after NMS
        final_boxes = []
        final_scores = []
        final_class_ids = []
        
        if len(indices) > 0:
            # OpenCV 4.x returns different format than 3.x
            if isinstance(indices, tuple):
                indices = indices[0]
            
            for i in indices.flatten():
                final_boxes.append(boxes[i])
                final_scores.append(scores[i])
                final_class_ids.append(class_ids[i])
        
        return final_boxes, final_scores, final_class_ids

class YoloDetectorNode:
    def __init__(self):
        rospy.init_node('yolo_detector_node')
        
        # Get parameters
        engine_path = "/home/triton/cs603-final-project/catkin_ws/src/object_sorter/best.engine"
        self.visualize = rospy.get_param('~visualize', False)
        
        if not engine_path:
            rospy.logerr("No engine path provided!")
            return
        
        # Initialize detector
        self.detector = YoloTRT(engine_path)
        self.bridge = CvBridge()
        
        # Publishers and subscribers
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.detections_pub = rospy.Publisher('/yolo/detections', Float32MultiArray, queue_size=1)
        
        if self.visualize:
            self.image_pub = rospy.Publisher('/yolo/visualization', Image, queue_size=1)
        
        rospy.loginfo("YOLO TensorRT node initialized")
    
    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            
            # Run inference
            boxes, scores, class_ids = self.detector.infer(cv_image)
            
            # Publish detections
            detection_msg = Float32MultiArray()
            for box, score, class_id in zip(boxes, scores, class_ids):
                detection_msg.data.extend([class_id, score, box[0], box[1], box[2], box[3]])
            
            self.detections_pub.publish(detection_msg)
            
            # Visualize if needed
            if self.visualize:
                vis_image = cv_image.copy()
                for box, score, class_id in zip(boxes, scores, class_ids):
                    x1, y1, x2, y2 = box
                    label = f"{self.detector.classes[class_id]}: {score:.2f}"
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis_image, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(vis_image, "passthrough"))
                
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

if __name__ == "__main__":
    try:
        node = YoloDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
