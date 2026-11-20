import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load triangulated points
data = pd.read_csv(
    "results/triangulation/front_45_points.csv"
)

X = data["X"].to_numpy()
Y = data["Y"].to_numpy()
Z = data["Z"].to_numpy()

# 3D visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    X, Y, Z,
    c=Z,
    cmap="viridis",
    s=12
)

ax.set_title("3D Triangulated Face Points")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.tight_layout()
plt.show()

