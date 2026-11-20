import cv2
import numpy as np
import csv
from pathlib import Path

# 1. Directories
image_dir = Path("images/multiview")
data_dir = Path("results/multiview")

# 2. Landmark indices
indices = [1, 10, 33, 61, 152, 263, 291]

# 3. Approximate 3D facial reference model
# Coordinates are in arbitrary units
model_points = np.array([
    [0, 0, 0], # Nose tip
    [0, 75, -15], # Forehead
    [-35, 35, -25], # Left eye corner
    [-25, -30, -20], # Left mouth corner
    [0, -70, -5], # Chin
    [35, 35, -25], # Right eye corner
    [25, -30, -20] # Right mouth corner
], dtype=np.float64)

# 4. Find images
image_files = sorted(
    list(image_dir.glob("*.png")) +
    list(image_dir.glob("*.jpg")) +
    list(image_dir.glob("*.jpeg"))
)

results = []

for image_path in image_files:

    csv_path = data_dir / (
        image_path.stem + "_landmarks.csv"
    )

    image = cv2.imread(str(image_path))

    if image is None or not csv_path.exists():
        print("Missing image or CSV:", image_path.name)
        continue

    h, w = image.shape[:2]

    data = np.loadtxt(
        csv_path,
        delimiter=",",
        skiprows=1
    )

    # 5. Select corresponding 2D landmarks
    image_points = data[indices, 1:3].astype(
        np.float64
    )

    # 6. Approximate camera matrix
    focal_length = max(w, h)

    camera_matrix = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    distortion = np.zeros((4, 1))

    # 7. Solve camera pose
    success, rvec, tvec = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        print("Pose estimation failed:", image_path.name)
        continue

    # 8. Rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # 9. Reprojection error
    projected, _ = cv2.projectPoints(
        model_points,
        rvec,
        tvec,
        camera_matrix,
        distortion
    )

    error = np.mean(
        np.linalg.norm(
            projected.reshape(-1, 2) - image_points,
            axis=1
        )
    )

    results.append([
        image_path.stem,
        *rvec.flatten(),
        *tvec.flatten(),
        error
    ])

    print("\nImage:", image_path.name)
    print("Rotation vector:", rvec.flatten())
    print("Translation:", tvec.flatten())
    print("Mean reprojection error:", error)

# 10. Save results
with open(
    "results/camera_poses.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "image",
        "rx", "ry", "rz",
        "tx", "ty", "tz",
        "reprojection_error"
    ])

    writer.writerows(results)

print("\nCamera pose estimation completed!")
