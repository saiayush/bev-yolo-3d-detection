# LiDAR → BEV → YOLOv8 for Object Detection

## Overview  
This project builds a lightweight 3D detection pipeline by converting LiDAR point clouds into Bird’s Eye View (BEV) images and applying a 2D detector.  

The pipeline transforms raw point clouds into a structured HID (Height, Intensity, Density) BEV representation and trains a YOLOv8 model to detect vehicles in BEV space.

---

## Dataset  
- KITTI 3D Object Detection dataset  
- Used:
  - `velodyne/` (LiDAR point clouds)  
  - `label_2/` (3D annotations)  
  - `calib/` (calibration files)  
- Focus: **Car class**

---

## Training Pipeline  
- Load LiDAR point clouds and apply ROI filtering  
- Discretize x–y plane into BEV grid  
- Generate 3-channel HID BEV images:
  - Height  
  - Intensity  
  - Density  
- Convert KITTI 3D boxes → 2D BEV boxes (YOLO format)  
- Train YOLOv8 on BEV images (single-class detection)  

---

## Inference Pipeline  
- Convert input point cloud → BEV HID image  
- Run YOLOv8 inference  
- Apply confidence threshold + NMS  
- Output BEV bounding boxes  
- Visualize predictions on BEV grid  

---

## Results  

| Metric        | Value        |
|--------------|-------------|
| Precision     | ~89%        |
| Recall        | ~90%        |
| mAP@0.50      | ~0.90       |
| mAP@0.50:0.95 | ~0.69       |
| Inference     | ~20–30 FPS  |

*Evaluation performed in BEV (2D) space.*

---

## Visualizations  

### BEV Predictions  
<!-- Replace the image paths below with your actual outputs -->
![BEV Prediction 1](images/pred1.png)
![BEV Prediction 2](images/pred2.png)
![BEV Prediction 3](images/pred3.png)

- Ground truth vs predicted boxes overlay  
- Clear alignment in dense regions  
- Failure cases in clutter/occlusion  

---

## Future Work  
- Oriented (rotated) BEV bounding boxes  
- Full 3D box prediction (x, y, z, l, w, h, yaw)  
- Evaluation using KITTI 3D metrics  
- Multi-class detection (Car, Pedestrian, Cyclist)  
- Fusion with camera data  

---
