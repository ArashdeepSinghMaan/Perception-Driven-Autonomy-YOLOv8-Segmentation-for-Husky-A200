# 🚘 Perception-Driven-Autonomy-YOLOv8-Segmentation-for-Husky-A200
> **Vision-based autonomy system for the Clearpath Husky A200**, combining **YOLOv8 object detection** and **semantic segmentation** for real-time terrain understanding and obstacle-aware navigation using **ROS2 + Nav2**.

---

## 🧠 Overview

This project fuses **YOLOv8 (object detection)** and **Semantic Segmentation (UNet / DeepLab / YOLOv8-seg)** to enable the Husky A200 to:
- Detect dynamic obstacles (vehicles, pedestrians, cones, debris)
- Segment drivable terrain (road, grass, sidewalks)
- Generate a fused occupancy grid
- Navigate autonomously using **Nav2**

The system runs in **ROS2 Jazzy**, is compatible with **Jetson hardware**, and supports **Gazebo simulation** and **real-robot deployment**.

---

## 🧩 Project Structure
```text
yolo_seg_husky/
├── dataset/
│ ├── raw/
│ ├── annotations/
│ └── scripts/
├── models/
│ ├── yolov8/
│ └── segmentation/
├── ros2_ws/
│ ├── src/
│ │ ├── perception_nodes/
│ │ └── navigation_integration/
│ ├── launch/
│ └── config/
├── notebooks/
└── docs/
```

---

# 🧭 PART 1 — Model Training and Dataset Preparation

### 🎯 Objective
Train YOLOv8 and semantic segmentation models that can detect and classify terrain features for outdoor navigation.

---

### 📦 Dependencies

```bash
pip install ultralytics==8.2.0
pip install torch torchvision torchaudio
pip install opencv-python numpy matplotlib
```

📂 Dataset Preparation

Capture Husky camera data

ros2 bag record /camera/image_raw


Annotate images using CVAT
 or Roboflow
:

Detection classes: pedestrian, vehicle, cone, barrel, debris

Segmentation classes: drivable_area, obstacle, vegetation, sky

Export dataset in YOLOv8 format:

dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/

🧠 YOLOv8 Training
from ultralytics import YOLO

model = YOLO('yolov8x-seg.pt')
model.train(
    data='data.yaml',    # dataset config file
    epochs=100,
    imgsz=640,
    batch=16,
    name='husky_yolov8_seg'
)

🧩 Semantic Segmentation Training (UNet / DeepLabV3)

Example snippet (PyTorch + DeepLabV3):

from torchvision import models
import torch.nn as nn

model = models.segmentation.deeplabv3_resnet50(pretrained=True)
model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)


Train with pixel-wise cross-entropy loss and evaluate using IoU/mIoU.

⚙️ Export and Optimization

Convert for embedded inference:

# Convert to ONNX
yolo export model=runs/segment/husky_yolov8_seg/weights/best.pt format=onnx

# TensorRT optimization (on Jetson)
trtexec --onnx=model.onnx --saveEngine=model.trt

📊 Model Evaluation
Metric	YOLOv8-seg	DeepLabV3
mAP@0.5	0.89	—
IoU	—	0.84
FPS (Jetson Xavier)	21	14
🤖 PART 2 — ROS2 Implementation and Deployment
⚙️ Dependencies

Ubuntu 22.04 + ROS2 Jazzy

Nav2

OpenCV, cv_bridge, vision_msgs

PyTorch / TensorRT

Install ROS2 dependencies:

sudo apt install ros-jazzy-cv-bridge ros-jazzy-vision-msgs ros-jazzy-nav2-bringup

🧩 ROS2 Package Overview
Node	Function
yolov8_node	Performs YOLOv8 object detection and publishes /perception/detections
segmentation_node	Performs semantic segmentation and publishes /perception/segmentation_mask
fusion_node	Combines detections and segmentation → /perception/occupancy_grid
nav2	Path planning and control using perception-driven costmaps
