# BEV-based 3D Object Detection using YOLO

This project explores using **Bird's Eye View (BEV) representations of LiDAR point clouds** for 3D object detection.  
The pipeline converts raw LiDAR point clouds into BEV images with multiple feature channels and trains a YOLO model to detect vehicles.

---

## Project Motivation

LiDAR sensors generate sparse 3D point clouds that are difficult for traditional CNNs to process directly.

A common approach is to convert point clouds into **Bird's Eye View (BEV)** representations.

BEV preserves:

- spatial layout
- metric distances
- object footprints

This makes object detection significantly easier.

---

## Pipeline

The full pipeline used in this project:
