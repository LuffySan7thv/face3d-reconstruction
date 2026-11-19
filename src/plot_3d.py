import numpy as np
import matplotlib.pyplot as plt

# 1. Load numerical data
points = np.loadtxt(
    "results/face_landmarks.csv",
    delimiter=",",
    skiprows=1
)

# 2. Extract coordinates
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

# 3. Create 3D figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# 4. Plot facial landmarks
ax.scatter(x, y, z, s=5)

# 5. Set axis labels
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_title("3D Facial Landmark Coordinates")

# Image coordinate correction
ax.invert_yaxis()

plt.show()

