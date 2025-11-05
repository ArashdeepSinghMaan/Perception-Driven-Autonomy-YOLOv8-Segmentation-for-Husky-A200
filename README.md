# 🚘 YOLOv8 + DeepLabV3 Based Terrain-Aware Segmentation and Perception System for Autonomous Robot (Husky A200)

A fully integrated **computer vision + ROS2 perception pipeline** combining **YOLOv8 object detection/instance segmentation** and **DeepLabV3 semantic segmentation** to enable **terrain-aware autonomous navigation** on the Clearpath **Husky A200** robot.

This system enables **real-time outdoor perception**, including drivable-terrain detection, dynamic obstacle avoidance, fused occupancy grid generation, and Nav2-based autonomous motion planning.  
The framework supports deployment on **Jetson hardware**, **Gazebo simulation**, and **ROS2 Jazzy**.

---

## ✅ Key Capabilities

- Multi-class object detection using **YOLOv8**
- Dense semantic segmentation using **DeepLabV3**
- Terrain classification: drivable area, vegetation, sky, obstacles
- Dynamic obstacle detection: humans, vehicles, cones, barrels, debris
- Occupancy grid fusion for navigation
- Real-time deployment using **TensorRT**
- Full integration with **Nav2** for autonomous planning and control

---

# 1. System Architecture

```
                 +-----------------------------+
                 |     Husky RGB Camera Feed   |
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
                       +----------------------------+
                       |        Nav2 Stack          |
                       +----------------------------+
                        Global Planner + Local Planner
                        Controller Server + BT Navigator
```

---

# 2. Project Structure

```
yolo_deeplab_husky/
├── dataset/
│   ├── raw/                
│   ├── annotations/        
│   └── scripts/            
├── models/
│   ├── yolov8/            
│   └── deeplabv3/          
├── ros2_ws/
│   ├── src/
│   │   ├── perception_nodes/
│   │   └── navigation_integration/
│   ├── launch/
│   └── config/
├── notebooks/
└── docs/
```

---

# 3. Dataset Preparation

## 3.1 Data Collection

Collect real or simulated Husky camera data:

```bash
ros2 bag record /camera/image_raw
```

## 3.2 Annotation Tools

- **CVAT**
- **Roboflow**
- **Label Studio**

### Detection Classes (YOLO)

- pedestrian  
- vehicle  
- cone  
- barrel  
- debris  

### Segmentation Classes (DeepLabV3)

- drivable_area  
- vegetation  
- sky  
- obstacle  

## 3.3 Dataset Format (YOLOv8 + Semantic Segmentation)

```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── seg_masks/
```

---

# 4. Model Training

---

## 4.1 YOLOv8 (Detection + Instance Segmentation)

```python
from ultralytics import YOLO

model = YOLO("yolov8x-seg.pt")
model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="husky_yolov8_seg"
)
```

### ✅ YOLOv8 Training Results (Template)

| Metric | Value |
|--------|--------|
| mAP@0.5 | XX.XX |
| mAP@0.5:0.95 | XX.XX |
| Segmentation mAP | XX.XX |
| Precision | XX.XX |
| Recall | XX.XX |
| F1 Score | XX.XX |
| FPS on Jetson (INT8) | XX FPS |
| Model Size | XX MB |
| Training Duration | XX hours |

**Visual Outputs (Recommended):**
- Detection samples  
- Segmentation mask overlays  
- Confusion matrix  
- PR curves  

---

## 4.2 DeepLabV3 Semantic Segmentation

### Model Definition

```python
from torchvision import models
import torch.nn as nn

model = models.segmentation.deeplabv3_resnet50(pretrained=True)
model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
```

### Training Setup

- Loss: CrossEntropyLoss (+ optional Dice Loss)
- Optimizer: Adam or SGD
- Metrics:
  - IoU  
  - mIoU  
  - Pixel Accuracy  

### ✅ DeepLabV3 Training Results (Template)

| Metric | Value |
|--------|--------|
| IoU (drivable area) | XX.XX |
| IoU (vegetation) | XX.XX |
| IoU (sky) | XX.XX |
| IoU (obstacle) | XX.XX |
| Mean IoU (mIoU) | XX.XX |
| Pixel Accuracy | XX.XX% |
| FPS Jetson (FP16) | XX FPS |
| Model Size | XX MB |

### Visual Outputs

- Color-coded segmentation results  
- Prediction vs ground truth comparison  
- Failure case examples  

---

# 5. Model Optimization for Deployment

## 5.1 Export YOLOv8 → ONNX

```bash
yolo export model=runs/segment/husky_yolov8_seg/weights/best.pt format=onnx
```

## 5.2 Convert YOLOv8 ONNX → TensorRT

```bash
trtexec --onnx=model.onnx --saveEngine=model.trt
```

## 5.3 DeepLabV3 Optimization

- TorchScript Export  
- ONNX Export  
- TensorRT FP16 / INT8 Conversion  

---

# 6. ROS2 Implementation

## 6.1 Dependencies

```bash
sudo apt install ros-jazzy-cv-bridge \
                 ros-jazzy-vision-msgs \
                 ros-jazzy-nav2-bringup
```

## 6.2 ROS2 Node Graph

| Node | Function |
|------|----------|
| `yolov8_node` | YOLOv8 inference → `/perception/detections` |
| `deeplab_node` | DeepLabV3 segmentation → `/perception/segmentation_mask` |
| `fusion_node` | Fuses YOLO + segmentation → occupancy grid |
| `nav2` | Navigation planning + control |

---

# 7. Navigation Integration (Nav2)

## 7.1 Costmap Interpretation

| Perception Class | Costmap Value |
|------------------|----------------|
| drivable area | free space |
| vegetation | soft obstacle |
| obstacles / debris | lethal |
| humans / cars | dynamic inflated |

## 7.2 Nav2 Modules Used

- Global Planner (NavFn, Smac)
- Local Planner (DWB or Regulated Pure Pursuit)
- Controller Server
- Behavior Tree Navigator

---

# 8. Evaluation Framework

## 8.1 Perception Metrics

| Category | Metric | Target |
|----------|--------|--------|
| YOLOv8 | mAP@0.5 | > 85% |
| DeepLabV3 | mIoU | > 80% |
| Runtime | FPS | ≥ 15 FPS |

## 8.2 Navigation Metrics

| Scenario | Target |
|----------|--------|
| Straight Navigation | > 95% Success |
| Cluttered Navigation | > 90% Success |
| Dynamic Obstacle Avoidance | > 85% Success |

---

# 9. References

- Ultralytics YOLOv8  
- TorchVision DeepLabV3  
- ROS2 Nav2 Documentation  
- Clearpath Husky A200  
- CVAT Annotation Tool  

---

# 10. Author

**Arashdeep Singh**  
Robotics Engineer — ROS2 • Perception • Autonomous Navigation
