import pandas as pd
import numpy as np

# Load camera pose results
poses = pd.read_csv("results/camera_poses.csv")

print("\n===== CAMERA POSE REPORT =====\n")

for _, row in poses.iterrows():

    error = row["reprojection_error"]

    print("Image:", row["image"])
    print("Rotation vector:",
          row["rx"], row["ry"], row["rz"])

    print("Translation:",
          row["tx"], row["ty"], row["tz"])

    print("Reprojection error:", error)
    print("-----------------------------")

# Statistical analysis
errors = poses["reprojection_error"].to_numpy()

print("\n===== ERROR ANALYSIS =====")
print("Mean error:", np.mean(errors))
print("Maximum error:", np.max(errors))
print("Minimum error:", np.min(errors))
print("Standard deviation:", np.std(errors))

