import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_obj_matplotlib(filename):
    """ Load .obj file and return points and colors """
    points = []
    colors = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.split()
            if parts[0] == 'v':  # Read vertex
                x, y, z = map(float, parts[1:4])
                r, g, b = map(float, parts[4:7])
                points.append([x, y, z])
                colors.append([r / 255, g / 255, b / 255])  # Normalize

    return np.array(points), np.array(colors)

def visualize_matplotlib(filename):
    """ 3D Scatter Plot using Matplotlib """
    points, colors = load_obj_matplotlib(filename)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, marker='o', s=2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Point Cloud Visualization")

    plt.show()

# Run visualization
visualize_matplotlib("/home/thang/Desktop/PointNeXt/log/shapenetpart/visual/lamp_0219_0007_pred.obj")
