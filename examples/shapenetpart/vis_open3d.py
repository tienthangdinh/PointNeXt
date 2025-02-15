import open3d as o3d
import numpy as np

def load_obj(filename):
    """ Load OBJ file and return points & colors """
    points = []
    colors = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.split()
            if parts[0] == 'v':  # Read vertex
                x, y, z = map(float, parts[1:4])  # Position
                r, g, b = map(float, parts[4:7])  # Color
                points.append([x, y, z])
                colors.append([r / 255, g / 255, b / 255])  # Normalize colors
    return np.array(points), np.array(colors)

def visualize_point_cloud(filename):
    """ Visualize .obj point cloud using Open3D """
    points, colors = load_obj(filename)

    # Create Open3D Point Cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Display
    o3d.visualization.draw_geometries([pcd])

# Run visualization
visualize_point_cloud("/home/thang/Desktop/PointNeXt/log/shapenetpart/visual/lamp_0219_0007_pred.obj")
