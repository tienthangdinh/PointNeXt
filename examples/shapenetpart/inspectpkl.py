import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pprint

# Path to the pkl file
pkl_path = "data/ShapeNetPart/shapenetcore_partanno_segmentation_benchmark_v0_normal/processed/test_2048_fps.pkl"

# Load the .pkl file
with open(pkl_path, "rb") as f:
    data, cls_labels = pickle.load(f)  # Unpack stored data


# Print basic type info
print(f"Type of loaded data: {type(data)}")

# Print the first few points of the first sample
print("\nFirst few points of the first object:")
print(data[0][:5])  # Print first 5 points

# Print unique class labels in dataset
print("\nUnique object categories in test set:", np.unique(cls_labels))

# Display general information
print(f"Total samples: {len(data)}")
print(f"Shape of first sample: {data[0].shape}")  # Expecting (2048, 7) -> XYZ + normal + label
print(f"Shape of first class label: {cls_labels[0].shape}")  # Expecting (1,) -> Single class label per sample

# Pick a random sample to inspect
sample_idx = np.random.randint(len(data))
point_cloud = data[sample_idx]  # (2048, 7)
class_label = cls_labels[sample_idx]  # (1,)

# Extract XYZ positions and part labels
xyz = point_cloud[:, :3]  # First 3 columns = X, Y, Z coordinates
normals = point_cloud[:, 3:6]  # Normal vectors (optional)
part_labels = point_cloud[:, 6].astype(int)  # Last column = segmentation labels

print(f"\nSample {sample_idx} Details:")
print(f"- Class Label: {class_label}")
print(f"- Point Cloud Shape: {xyz.shape} (Expecting 2048 points)")
print(f"- Unique Part Labels: {np.unique(part_labels)}")

# --- Visualization ---
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=part_labels, cmap="jet", marker="o")
plt.colorbar(scatter, ax=ax, label="Segmentation Labels")
ax.set_title(f"Point Cloud Sample {sample_idx} - Class {class_label[0]}")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
