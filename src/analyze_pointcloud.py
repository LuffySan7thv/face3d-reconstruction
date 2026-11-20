import numpy as np
import pandas as pd

data = pd.read_csv(
    "results/triangulation/front_45_points.csv"
)

points = data[["X", "Y", "Z"]].to_numpy()

# Calculate statistical properties
center = np.mean(points, axis=0)
std = np.std(points, axis=0)

print("Number of points:", len(points))
print("Center:", center)
print("Standard deviation:", std)

# Calculate distance from center
distances = np.linalg.norm(points - center, axis=1)

print("Minimum distance:", np.min(distances))
print("Maximum distance:", np.max(distances))
print("Mean distance:", np.mean(distances))

# Identify statistical outliers
threshold = np.mean(distances) + 2 * np.std(distances)

valid = distances < threshold
filtered_points = points[valid]

print("\nValid points:", len(filtered_points))
print("Removed points:", len(points) - len(filtered_points))

# Save filtered point cloud
output = pd.DataFrame(
    filtered_points,
    columns=["X", "Y", "Z"]
)

output.to_csv(
    "results/triangulation/filtered_points.csv",
    index=False
)

print("Filtered point cloud saved.")

