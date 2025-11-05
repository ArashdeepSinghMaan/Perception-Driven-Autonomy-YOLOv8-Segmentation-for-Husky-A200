🚘 YOLOv8 + DeepLabV3 Based Terrain-Aware Segmentation and Perception System for Autonomous Robot (Husky A200)

A fully integrated computer vision + ROS2 perception pipeline combining YOLOv8 object detection/instance segmentation and DeepLabV3 semantic segmentation to enable terrain-aware autonomous navigation on the Clearpath Husky A200 robot.

This system is designed for real-time outdoor autonomy, including drivable-terrain detection, dynamic obstacle avoidance, fused occupancy grid generation, and Nav2-based motion planning.
The framework supports deployment on Jetson hardware, Gazebo simulation, and ROS2 Jazzy.

Key Capabilities

Multi-class object detection using YOLOv8

Dense semantic segmentation using DeepLabV3

Terrain classification: drivable area, vegetation, obstacles, sky

Dynamic obstacle detection: humans, vehicles, cones, debris

Occupancy grid fusion for navigation

Real-time deployment on embedded GPU (TensorRT)

Full integration with Nav2 for autonomous navigation

1. System Architectue
2.                  +-----------------------------+
                 |   Husky RGB Camera Feed     |
                 +-----------------------------+
                                 |
                     +-----------------------+
                     |      Preprocessing    |
                     +-----------------------+
                         |             |
            +----------------+   +----------------------+
            | YOLOv8-seg     |   |  DeepLabV3 (SegNet)  |
            | Detection+Seg  |   | Terrain Segmentation |
            +----------------+   +----------------------+
                         |             |
                     +------------------------------+
                     |   Perception Fusion Node     |
                     | (Detections + Segmentation)  |
                     +------------------------------+
                                 |
                    +-------------------------------+
                    |      Fused Occupancy Grid     |
                    +-------------------------------+
                                 |
                       +----------------------+
                       |     Nav2 Stack       |
                       +----------------------+
                           Global + Local Planner
                           Controller Server
                           Behavior Tree Navigator
2. Project Structure
   yolo_deeplab_husky/
├── dataset/
│   ├── raw/                
│   ├── annotations/        
│   └── scripts/            
├── models/
│   ├── yolov8/            
│   ├── deeplabv3/          
├── ros2_ws/
│   ├── src/
│   │   ├── perception_nodes/
│   │   └── navigation_integration/
│   ├── launch/
│   └── config/
├── notebooks/
└── docs/
3. Dataset Preparation
3.1 Data Collection

Record camera images from Husky simulation or real hardware:

ros2 bag record /camera/image_raw

3.2 Annotation Tools

CVAT

Roboflow

LabelStudio

Detection Classes (YOLO)

pedestrian

vehicle

cone

barrel

debris

Segmentation Classes (DeepLabV3)

drivable_area

vegetation

sky

obstacle

3.3 Dataset Format
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── seg_masks/

4. Model Training
4.1 YOLOv8 (Detection + Segmentation)
from ultralytics import YOLO

model = YOLO("yolov8x-seg.pt")
model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="husky_yolov8_seg"
)

✅ Template: YOLOv8 Training Results
Metric	Value
mAP@0.5	XX.XX
mAP@0.5:0.95	XX.XX
Segmentation mAP	XX.XX
Precision	XX.XX
Recall	XX.XX
F1 Score	XX.XX
FPS on Jetson (INT8)	XX FPS
Model Size	XX MB
Training Duration	XX hours

Include (Add Images):

Bounding box outputs

Segmentation mask overlays

Confusion matrix

PR curves

4.2 DeepLabV3 Semantic Segmentation

Using TorchVision DeepLabV3 (ResNet-50 / ResNet-101):

from torchvision import models
import torch.nn as nn

model = models.segmentation.deeplabv3_resnet50(pretrained=True)
model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)


Train using:

CrossEntropyLoss

Dice Loss (optional)

Adam / SGD optimizer

IoU & mIoU metrics

✅ Template: DeepLabV3 Training Results
Metric	Value
IoU (drivable area)	XX.XX
IoU (vegetation)	XX.XX
IoU (sky)	XX.XX
IoU (obstacle)	XX.XX
Mean IoU (mIoU)	XX.XX
Pixel Accuracy	XX.XX%
FPS Jetson (FP16)	XX FPS
Model Size	XX MB

Include (Add Images):

Color-coded segmentation overlays

Prediction vs Ground Truth comparisons

Failure cases
 References

Ultralytics YOLOv8


TorchVision DeepLabV3


ROS2 Nav2 Documentation


Clearpath Husky A200


CVAT Annotation Platform



 Author
Arashdeep Singh
Robotics Engineer — Perception | ROS2 | Navigation

