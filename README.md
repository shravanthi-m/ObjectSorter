# Object Sorter

This is a ROS package for automatically sorting objects by color and shape in a user-defined rectangular area.

## Setup

1. Download the object_sorter ROS package. Move the package into your catkin workspace.
2. Download model and (optionally, if you want to try training a model) datasets from here: https://drive.google.com/drive/folders/1c6dV42xMO3MF4OMLY3cvzJqUx-Qwl1d_?usp=sharing
3. Move best.onnx and best.pt into catkin_ws/src/object_sorter
4. Change width and height parameters in launch/object_sorter.launch to your desired dimensions
5. Attach an arm to the base of the robot.
6. Physically position the robot so its center is at the bottom right corner of the rectangular space, camera facing straight towards the upper right corner.
7. In scripts/movement_odom.py, edit color_to_bin to specify the designated bins / corners for each class. Modify COLORS in scripts/perception.py if needed to reflect the object classes.

## Dependencies
Can install using pip:
- numpy
- opencv-python
- ultralytics
- ros_numpy
- cv_bridge

Make sure these ROS dependencies are also installed (for ROS noetic):
- rospy
- geometry-msgs
- nav-msgs
- sensor-msgs
- std-msgs
- tf
- tf2-msgs
- cv-bridge

Using this ROS package also requires the Triton noetic dockerfile setup.

## Running the program

After all of the above is set up:
1. Start the container
2. Within the container, execute these commands:
3. cd into catkin_ws if not already there
4. source devel/setup.bash
5. roslaunch object_sorter object_sorter.launch

The robot will start tracing the perimeter of the space. Change the dimensions if needed, then run again.

After tracing the perimeter, the robot will automatically align with the center and begin tracking, classifying, and sorting objects to their designated bins.
