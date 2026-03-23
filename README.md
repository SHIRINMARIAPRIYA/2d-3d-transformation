# 2D-to-3D Image Reconstruction with Python

This repository provides two methods for generating 3D structures from 2D images:

    Deep Learning Approach: Uses a pretrained Monodepth2 model to predict high-accuracy depth maps.

    Intensity-Based Approach: A lightweight script using OpenCV and Matplotlib to generate 3D surface plots based on pixel intensity (grayscale values).

## 🚀 Features

    Monodepth2 Integration: Predicts depth from a single RGB image using self-supervised monocular depth estimation.

    3D Point Cloud Generation: Converts depth maps into .ply files and visualizes them using Open3D.

    Surface Mapping: Generates interactive 3D surface plots using Matplotlib's mplot3d.

    Preprocessing: Automated image resizing and normalization for model compatibility.

## 📸 Results

The following image demonstrates the 3D surface reconstruction where pixel intensity is mapped to the Z-axis (height):

## 🛠️ Installation

Ensure you have Python 3.8+ installed. You will need to install the following dependencies:
Bash

pip install torch torchvision torchaudio
pip install open3d opencv-python numpy matplotlib Pillow

Note: If you plan to use the Monodepth2 model, ensure you have the model_weights.pth file in the appropriate directory.
## 📖 Usage
Method 1: Deep Learning Depth Estimation

This script uses a neural network to infer depth and generates a point cloud.
Python

## Update image_path and model_path in the script
python monodepth_reconstruction.py

Method 2: Grayscale Intensity Surface Plot

This script provides a quick visualization by treating brightness as height.
Python

## Update image_path in the script
python simple_3d_plot.py

## 📂 Project Structure

    monodepth_reconstruction.py: Script for deep learning-based depth and Open3D visualization.

    simple_3d_plot.py: Script for OpenCV-based surface plotting.

    models/: Directory to store Monodepth2 weights.

    output/: Directory where .ply point cloud files are saved.

## 🧪 How it Works

    Depth Prediction: The image is processed to create a Depth Map (Z). In the deep learning version, the model predicts the distance of objects from the camera.

    Coordinate Mapping: We use a meshgrid of X and Y coordinates corresponding to pixel locations.

    Projection:

        For the surface plot: Z is scaled by a factor (e.g., 20) to emphasize depth.

        For the point cloud: We use the camera intrinsic parameters (fx​,fy​,cx​,cy​) to project 2D pixels into 3D space.

## 🤝 Contributing

Feel free to fork this project, submit PRs, or report issues. If you'd like to improve the depth estimation accuracy or add support for stereo-rectified images, contributions are welcome!
