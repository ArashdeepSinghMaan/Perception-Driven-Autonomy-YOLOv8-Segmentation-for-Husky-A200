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
