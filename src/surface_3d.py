import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# 1. Load facial landmarks
points = np.loadtxt(
    "results/face_landmarks.csv",
    delimiter=",",
    skiprows=1
)

x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

# 2. Create a regular computational grid
xi = np.linspace(x.min(), x.max(), 150)
yi = np.linspace(y.min(), y.max(), 150)

X, Y = np.meshgrid(xi, yi)

# 3. Cubic interpolation
Z = griddata(
    (x, y),
    z,
    (X, Y),
    method="cubic"
)

# 4. Plot the interpolated surface
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")

surface = ax.plot_surface(
    X, Y, Z,
    cmap="viridis",
    edgecolor="none",
    alpha=0.9
)

# Original landmarks
ax.scatter(x, y, z, color="red", s=5)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Relative Depth")

ax.set_title("Cubic Interpolation of Facial Surface")
ax.invert_yaxis()

fig.colorbar(surface, ax=ax, shrink=0.6)

plt.show()

