import cv2
import numpy as np
import pandas as pd
import os

# Image paths
image1_path = "images/multiview/front.png"
image2_path = "images/multiview/angle_45.jpg"

# Load images
img1 = cv2.imread(image1_path)
img2 = cv2.imread(image2_path)

if img1 is None or img2 is None:
    raise FileNotFoundError("Input image not found!")

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

print("Image 1 size:", w1, h1)
print("Image 2 size:", w2, h2)

# Camera intrinsic matrices
f1 = max(w1, h1)
f2 = max(w2, h2)

K1 = np.array([
    [f1, 0, w1 / 2],
    [0, f1, h1 / 2],
    [0, 0, 1]
], dtype=np.float64)

K2 = np.array([
    [f2, 0, w2 / 2],
    [0, f2, h2 / 2],
    [0, 0, 1]
], dtype=np.float64)

# Load landmarks
data1 = pd.read_csv(
    "results/multiview/front_landmarks.csv"
)

data2 = pd.read_csv(
    "results/multiview/angle_45_landmarks.csv"
)

# Match landmarks by ID
matched = pd.merge(
    data1,
    data2,
    on="id",
    suffixes=("_1", "_2")
)

points1 = matched[
    ["x_pixel_1", "y_pixel_1"]
].to_numpy(dtype=np.float64)

points2 = matched[
    ["x_pixel_2", "y_pixel_2"]
].to_numpy(dtype=np.float64)

print("Matched landmarks:", len(points1))

# Normalize points using each camera matrix
p1 = cv2.undistortPoints(
    points1.reshape(-1, 1, 2), K1, None
).reshape(-1, 2)

p2 = cv2.undistortPoints(
    points2.reshape(-1, 1, 2), K2, None
).reshape(-1, 2)

# Estimate Essential Matrix
E, mask = cv2.findEssentialMat(
    p1,
    p2,
    focal=1.0,
    pp=(0.0, 0.0),
    method=cv2.RANSAC,
    prob=0.999,
    threshold=0.001
)

if E is None:
    raise RuntimeError("Essential matrix estimation failed.")

# Recover relative pose
inliers, R, t, pose_mask = cv2.recoverPose(
    E,
    p1,
    p2,
    focal=1.0,
    pp=(0.0, 0.0),
    mask=mask
)

print("\n===== CAMERA GEOMETRY =====")
print("Rotation matrix:\n", R)
print("Translation direction:\n", t)
print("Inlier points:", inliers)

# Normalized projection matrices
P1 = np.hstack((
    np.eye(3),
    np.zeros((3, 1))
))

P2 = np.hstack((R, t))

# Triangulation
X_h = cv2.triangulatePoints(
    P1,
    P2,
    p1.T,
    p2.T
)

X = (X_h[:3] / X_h[3]).T

# Keep inlier points
valid = pose_mask.ravel() > 0

X = X[valid]
ids = matched["id"].to_numpy()[valid]

# Save results
output = pd.DataFrame({
    "id": ids,
    "X": X[:, 0],
    "Y": X[:, 1],
    "Z": X[:, 2]
})

os.makedirs("results/triangulation", exist_ok=True)

output.to_csv(
    "results/triangulation/front_45_points.csv",
    index=False
)

print("\nTriangulated points:", len(X))
print("Saved successfully!")

