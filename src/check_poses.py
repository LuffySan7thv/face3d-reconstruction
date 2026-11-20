import cv2
import numpy as np
import pandas as pd

# Load estimated camera poses
poses = pd.read_csv("results/camera_poses.csv")

for _, row in poses.iterrows():

    image_name = row["image"]

    # Rotation vector
    rvec = np.array([
        [row["rx"]],
        [row["ry"]],
        [row["rz"]]
    ], dtype=np.float64)

    # Translation vector
    tvec = np.array([
        [row["tx"]],
        [row["ty"]],
        [row["tz"]]
    ], dtype=np.float64)

    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Extract Euler angles
    angles = cv2.RQDecomp3x3(R)[0]

    # Check rotation matrix properties
    orthogonal_error = np.linalg.norm(
        R.T @ R - np.eye(3)
    )

    determinant = np.linalg.det(R)

    print("\nImage:", image_name)
    print("Rotation matrix:\n", R)

    print("Euler angles:", angles)
    print("Translation:", tvec.ravel())

    print("Orthogonality error:", orthogonal_error)
    print("Determinant:", determinant)

    print("Reprojection error:",
          row["reprojection_error"])

