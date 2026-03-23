# Import necessary libraries
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
from PIL import Image

# Load the pretrained Monodepth2 model
def load_model(model_path):
    model = torch.load(model_path + "/model_weights.pth")
    model.eval()
    return model

# Load and preprocess the image
def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((640, 192))  # Resize for the model
    image = np.array(image).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W)
    image = torch.from_numpy(image).unsqueeze(0)  # Add batch dimension
    return image

# Predict depth map
def predict_depth(model, image):
    with torch.no_grad():
        depth_map = model(image)
    return depth_map.squeeze().cpu().numpy()

# Normalize depth map for visualization
def normalize_depth(depth_map):
    return (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))

# Generate 3D point cloud from depth map
def create_point_cloud(depth_map, fx=525.0, fy=525.0):
    h, w = depth_map.shape
    cx, cy = w / 2, h / 2

    # Generate pixel coordinates
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    zs = depth_map

    # Compute 3D coordinates
    x = (xs - cx) * zs / fx
    y = (ys - cy) * zs / fy

    points = np.stack((x, y, zs), axis=-1)

    # Filter out zero depth points
    valid_points = points[depth_map > 0]
    return valid_points

# Main function to run everything
def main(image_path, model_path):
    model = load_model(model_path)
    image = load_and_preprocess_image(image_path)

    # Predict the depth map
    depth_map = predict_depth(model, image)

    # Normalize depth for visualization
    normalized_depth = normalize_depth(depth_map)

    # Visualize original image and depth map
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(np.array(Image.open(image_path)))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Estimated Depth Map')
    plt.imshow(normalized_depth, cmap='plasma')
    plt.axis('off')
    plt.show()

    # Create and visualize the point cloud
    point_cloud = create_point_cloud(depth_map)

    # Convert to Open3D point cloud format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd], window_name='3D Model from Depth Map', width=800, height=600)

    # Optionally save the point cloud
    o3d.io.write_point_cloud("3d_model.ply", pcd)

# Run the code
if __name__ == "__main__":
    image_path = '/content/wow-removebg-preview.jpg'  # specify your input image here
    model_path = 'models/mono_640x192'  # specify your model path here
    main(image_path, model_path)
