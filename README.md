# Project Write-up: LiDAR → HID BEV → YOLO → BEV Detections (and the path to 3D)

## Overview

This project explores a simple but practical idea:

1. Start with raw **LiDAR point clouds** (KITTI Velodyne `.bin`).
2. Convert each point cloud into a **Bird’s-Eye-View (BEV) image**.
3. Encode the BEV as a **3-channel HID image** (so a standard 2D CNN detector can be trained).
4. Train a **YOLO** model on these BEV images to detect cars.
5. Evaluate performance using **YOLO’s 2D detection metrics** in BEV space.

The motivation is to reuse strong, fast 2D detectors (YOLO) while still leveraging LiDAR geometry through a BEV representation.

---

## Why BEV + YOLO?

Standard 3D detectors (PointPillars, SECOND, CenterPoint, etc.) are powerful but involve:
- custom voxel/pillar operations
- specialized heads for 3D regression
- more complex training code

This project aims to build a **lightweight baseline**:
- Take LiDAR → render it into an image-like representation
- Apply a well-established 2D detector
- Validate whether BEV image detection can reliably localize objects in ground-plane coordinates

The result is a pipeline that is:
- easy to visualize
- easy to train
- fast to iterate on

---

## Dataset

I used the **KITTI 3D Object Detection** dataset.

Required KITTI folders:
- `training/velodyne` (LiDAR point clouds)
- `training/label_2` (3D annotations)
- `training/calib` (calibration files)

I focused on the **Car** class (common for baseline 3D detection experiments).

---

## Step 1 — Load KITTI LiDAR point clouds

Each KITTI Velodyne scan is a `.bin` file containing a float32 array of shape `(N, 4)`:

- `x, y, z` coordinates (in LiDAR frame)
- `intensity`

From each file:
- read point cloud
- filter to a region-of-interest (ROI) in x/y (and sometimes z) so BEV covers a consistent area

Typical ROI concepts:
- forward range in meters (x)
- left/right range in meters (y)
- optionally restrict z to remove ground/sky noise

---

## Step 2 — Convert point cloud into a BEV grid

To create an image-like input, the x/y plane is discretized into a 2D grid:

- Choose BEV resolution, e.g. `0.1m/pixel` or `0.2m/pixel`
- Define grid shape:
  - `W = (X_MAX - X_MIN) / resolution`
  - `H = (Y_MAX - Y_MIN) / resolution`

Then every point is projected into a pixel coordinate:
- `col = floor((x - X_MIN) / res)`
- `row = floor((y - Y_MIN) / res)`

This produces an image coordinate system where every pixel corresponds to a fixed patch of ground-plane area.

---

## Step 3 — Build a 3-channel HID BEV image

Instead of a single grayscale BEV, I used a **3-channel HID encoding** so the detector has richer cues.

Typical BEV features per cell include:
- **Height** statistics (max z / mean z / normalized height)
- **Intensity** statistics (max/mean reflectance)
- **Density** (how many points fall into that cell)

These become three channels:
- Channel 1: Height feature
- Channel 2: Intensity feature
- Channel 3: Density feature

Then I normalize and save as an image (PNG/JPG).

This makes the LiDAR scan “look like” an image tensor to YOLO:
- shape: `H x W x 3`
- values scaled into `[0, 255]` or `[0,1]`

---

## Step 4 — Convert KITTI 3D labels to 2D BEV bounding boxes

KITTI labels provide 3D boxes in camera coordinates, but the project goal is BEV detection in LiDAR space.

The conversion process conceptually involves:
1. Parse label lines (class, dimensions, location, yaw).
2. Use calibration to transform box center/orientation into the LiDAR coordinate frame (or directly compute BEV corners).
3. Compute the 4 ground-plane corners of the 3D box.
4. Project those corners onto the BEV grid (x/y → pixel coords).
5. Create a **2D BEV box** that YOLO can learn.

Since YOLO expects axis-aligned 2D boxes in image space, the BEV target becomes:
- `xmin, ymin, xmax, ymax` in BEV pixel coordinates
- converted to YOLO format: `(class, x_center, y_center, width, height)` normalized by image size

Important note:
- If the underlying car is rotated, an axis-aligned BEV box includes extra empty area.
- This is a limitation of standard YOLO box format.

---

## Step 5 — Prepare a YOLO dataset

I generated a YOLO-style dataset structure:

dataset/
images/
train/
val/
labels/
train/
val/
kitti_bev.yaml


Where:
- Each BEV image has a matching `.txt` file with YOLO labels
- The YAML file points YOLO to train/val folders and class names

---

## Step 6 — Train YOLO on BEV images

I trained a YOLOv8 model using Ultralytics:

- input size (`imgsz`) chosen to match BEV image resolution
- batch size tuned for GPU memory
- trained on car-only (single class) or multi-class (if desired)

Because the input is already an image-like BEV tensor, YOLO training works without architecture changes.

---

## Step 7 — Evaluate performance (what “performance” means here)

The evaluation reported by `model.val()` is:

**2D detection metrics in BEV image space**
- Precision
- Recall
- `mAP@0.50`
- `mAP@0.50:0.95`

This measures how well YOLO’s predicted BEV boxes overlap the BEV ground truth boxes.

### What it does NOT measure (yet)
 Official KITTI metrics:
- KITTI BEV AP
- KITTI 3D AP
- orientation similarity
- full 3D box quality

Those require a true 3D output and KITTI eval tooling.

---

## Step 8 — Visualization

To make the pipeline interpretable, I visualized:

- HID BEV image
- ground-truth boxes drawn on BEV
- predicted YOLO boxes drawn on BEV
- side-by-side comparisons

This makes it easy to see:
- whether boxes align correctly with point density
- whether orientation mismatch causes over-large boxes
- where false positives typically appear (e.g., dense clutter)

---

## “YOLO → back to 3D”: what’s missing vs what’s possible

A key point of intellectual honesty:

### Current pipeline output
Right now, YOLO outputs:
- axis-aligned 2D BEV boxes: `(x, y, w, l)` in pixel space → meters

That gives you **ground-plane localization**, but not a complete KITTI 3D box.

### To truly convert YOLO outputs to 3D boxes, you need extra information
A KITTI 3D bounding box requires:
- center `(x, y, z)`
- dimensions `(l, w, h)`
- yaw (orientation)

A plain 2D BEV axis-aligned box does not uniquely determine:
- yaw
- height
- z center
- true length vs width under rotation

### Practical upgrade paths
If I extend this project to real 3D:
1. **Multi-head regression**
   - Keep YOLO for `(x,y,w,l)`
   - Add heads to regress:
     - yaw (sin/cos)
     - height `h`
     - z center
     - true `(l,w)` if rotated
2. **Two-stage model**
   - YOLO generates proposals in BEV
   - a small MLP/CNN regresses full 3D box attributes per proposal
3. **Hybrid baseline**
   - Use a standard 3D detector for 3D AP
   - Keep BEV-YOLO as a fast 2D BEV baseline and visualization tool

---

## What I learned / key takeaways

- Rendering LiDAR into BEV images makes it easy to apply strong 2D detectors.
- HID channel design (height/intensity/density) matters for performance.
- Axis-aligned boxes in BEV are a limitation for rotated vehicles.
- Reporting “3D detection performance” requires careful definition; without true 3D outputs, the honest result is **2D BEV detection performance**.

---

## Future work

- Add oriented BEV boxes (rotated boxes) using a detector that supports them
- Predict yaw + dimensions + z for true 3D reconstruction
- Evaluate with KITTI official metrics for BEV AP and 3D AP
- Compare against PointPillars baseline for a fair reference

---

## Outputs

This project produces:
- BEV HID images for KITTI frames
- YOLO dataset folders + YAML config
- YOLO training runs (`best.pt`, logs)
- BEV overlay visualizations (GT vs Pred)
- A demo video showing BEV inference results

---

## Summary

This project is a clean baseline demonstrating:

**LiDAR point clouds can be turned into BEV images and used to train YOLO for BEV object detection**, producing interpretable results and measurable 2D metrics — and it sets up a clear roadmap for upgrading to full 3D detection metrics with additional predicted box attributes.
