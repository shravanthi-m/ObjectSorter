from ultralytics import YOLOWorld
import os
import numpy as np

# Initialize a YOLO-World model
model = YOLOWorld(
    "custom_yolo8_world.pt"
)  # or select yolov8m/l-world.pt for different sizes


# model.set_classes(["purple box", "red box", "blue box", "yellow box"])
# model.set_classes(["purple cuboid", "grey cuboid", "purple cylinder"])
# Execute inference with the YOLOv8s-world model on the specified image
# model.save("custom_yolo8_world.pt")
def get_file_names(directory):
    filenames = []
    if not os.path.exists(directory) or not os.path.isdir(directory):
        return filenames

    for filename in os.listdir(directory):
        name, ext = os.path.splitext(filename)
        if os.path.isfile(os.path.join(directory, filename)):
            filenames.append(name)
    return filenames


# for file_name in get_file_names("../datasets/objects/images/train/"):
#     results = model.predict(f"../datasets/objects/images/train/{file_name}.jpg")
#     path_save = f"../datasets/objects/labels/train/{file_name}.txt"
#
#     classes = results[0].boxes.cls.cpu().numpy().astype(int)
#     # names = [results[0].names[i] for i in classes]
#     # print(names)
#     boxes = results[0].boxes.xywhn.cpu().numpy()
#     boxes = np.insert(boxes, 0, classes, axis=1)
#     np.savetxt(path_save, boxes, fmt="%d %f %f %f %f")
#
#     # Show results
#     results[0].show()

file_name = "IMG_1091"
results = model.predict(f"../datasets/objects/images/train/{file_name}.jpg")
path_save = f"../datasets/objects/labels/train/{file_name}.txt"

classes = results[0].boxes.cls.cpu().numpy().astype(int)
names = [results[0].names[i] for i in classes]
print(names)
boxes = results[0].boxes.xywhn.cpu().numpy()
boxes = np.insert(boxes, 0, classes, axis=1)
np.savetxt(path_save, boxes, fmt="%d %f %f %f %f")

# Show results
results[0].show()
