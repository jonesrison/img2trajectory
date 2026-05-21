# IMG2TRAJECTORY

Convert AI-generated line art into continuous robotic drawing trajectories for polar-coordinate sand art tables.

This project transforms an input image into a smooth ordered motion path that can later be converted into motor commands for a 2-motor polar sand drawing system.

---

# Overview

Pipeline:

```text
Input Image
↓
Binarization
↓
Skeletonization
↓
Graph Extraction
↓
Eulerization
↓
Continuous Path Traversal
↓
Trajectory Smoothing
↓
JSON Export
↓
Polar Conversion (hardware side)
```

The software side generates clean continuous `(x, y)` drawing coordinates.

The hardware/firmware side converts those coordinates into:

- radial motion
- rotational motion
- motor steps
- real-world movement

for the sand table.

---

# Features

- AI-image-compatible preprocessing
- Automatic binarization using Otsu thresholding
- Topology-preserving skeletonization
- Graph extraction from skeleton paths
- Eulerian traversal generation
- Continuous trajectory reconstruction
- Trajectory smoothing and simplification
- JSON export for hardware systems
- Full debug visualization pipeline

---

# Project Structure

```text
image_to_trajectory/
│
├── pipeline.py
├── config.py
│
├── preprocessing/
│   ├── binarize.py
│   └── skeletonize.py
│
├── graph/
│   ├── extractor.py
│   └── cleaner.py
│
├── traversal/
│   ├── eulerize.py
│   └── traverser.py
│
├── smoothing/
│   └── smoother.py
│
├── exports/
│   ├── json_export.py
│   └── svg_export.py
│
├── visualization/
│   └── debugger.py
│
└── input.png
```

---

# Installation

## 1. Clone Repository

```bash
git clone <repo-url>
cd IMG2TRAJECTORY
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

OR manually:

```bash
pip install \
opencv-python \
numpy \
scikit-image \
scipy \
networkx \
matplotlib
```

---

# Running the Full Pipeline

From the project root:

```bash
python -m image_to_trajectory.pipeline \
--input image_to_trajectory/input.png \
--debug
```

---

# Output

The pipeline generates:

## JSON trajectory

```text
output/trajectory.json
```

Contains:

- ordered `(x,y)` path
- metadata
- path statistics
- trajectory information

Example:

```json
{
  "x": 125.37,
  "y": 482.11
}
```

---

## Debug Visualizations

```text
debug_outputs/
```

Includes:

- binarized image
- skeleton
- extracted graph
- trajectory visualization

---

# Coordinate System

Coordinates are exported in:

```text
image pixel coordinates
```

Convention:

```text
Origin: top-left
x → right
y ↓ down
```

Example:

```json
{
  "x": 250,
  "y": 400
}
```

means:

- 250 pixels from the left
- 400 pixels from the top

---

# Hardware Integration

The exported trajectory is intended for a:

- 2-motor polar sand table
- polar-coordinate drawing robot
- pen plotter
- robotic drawing arm

The hardware side converts:

```text
(x, y)
```

into:

```text
r = √(x² + y²)
θ = atan2(y, x)
```

which are then mapped to:

- radial motor movement
- rotational motor movement

---

# Current Limitations

The project is still experimental.

Known issues:

- graph node explosion on noisy skeletons
- traversal duplication artifacts
- occasional teleport edges during reconstruction
- excessive Euler duplication on highly branched drawings
- spline overshoot on invalid traversal jumps

---

# Planned Improvements

- Edge-key-aware Euler traversal
- Better graph simplification
- Stroke segmentation instead of forced full Eulerization
- Polar-coordinate export module
- Real-time trajectory preview
- Motor-aware velocity planning
- G-code style export
- GPU-accelerated processing

---

# Example Pipeline Visualization

```text
AI Image
↓
Binary Image
↓
1px Skeleton
↓
Graph Topology
↓
Eulerian Traversal
↓
Smooth Robotic Path
```

---

# Technologies Used

- Python
- OpenCV
- NumPy
- SciPy
- scikit-image
- NetworkX
- Matplotlib

---

# Research Areas Involved

This project combines concepts from:

- Computer Vision
- Computational Geometry
- Graph Theory
- Robotics
- Path Planning
- Motion Smoothing
- Topology Reconstruction

---

# Notes

This project currently focuses on:

```text
image → trajectory generation
```

NOT direct motor control.

Hardware firmware and kinematic conversion are handled separately.

---

# License

MIT License