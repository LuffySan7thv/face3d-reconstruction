import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# =====================================
# 1. Load triangulated 3D points
# =====================================

data = pd.read_csv(
    "results/triangulation/filtered_points.csv"
)

points = data[["X", "Y", "Z"]].to_numpy()

x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

# =====================================
# 2. Create regular computational grid
# =====================================

n = 50

x_grid = np.linspace(x.min(), x.max(), n)
y_grid = np.linspace(y.min(), y.max(), n)

X, Y = np.meshgrid(x_grid, y_grid)

# Initial scattered-data interpolation
Z = griddata(
    (x, y),
    z,
    (X, Y),
    method="linear"
)

# Fill undefined areas
Z_nearest = griddata(
    (x, y),
    z,
    (X, Y),
    method="nearest"
)

Z = np.where(np.isnan(Z), Z_nearest, Z)

# =====================================
# 3. Natural cubic spline algorithm
# =====================================

def natural_spline(x_nodes, values, x_query):

    n = len(x_nodes)

    h = np.diff(x_nodes)

    A = np.zeros((n, n))
    b = np.zeros(n)

    A[0, 0] = 1
    A[-1, -1] = 1

    for i in range(1, n - 1):

        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]

        b[i] = 6 * (
            (values[i + 1] - values[i]) / h[i]
            -
            (values[i] - values[i - 1]) / h[i - 1]
        )

    # Solve spline coefficient system
    M = np.linalg.solve(A, b)

    result = np.zeros_like(x_query)

    for j, q in enumerate(x_query):

        i = np.searchsorted(x_nodes, q) - 1
        i = np.clip(i, 0, n - 2)

        hi = h[i]

        a = (x_nodes[i + 1] - q) / hi
        c = (q - x_nodes[i]) / hi

        result[j] = (
            a * values[i]
            + c * values[i + 1]
            + ((a**3 - a) * M[i]
            + (c**3 - c) * M[i + 1]) * hi**2 / 6
        )

    return result


# =====================================
# 4. Apply spline along X direction
# =====================================

Z_x = np.zeros_like(Z)

for j in range(n):

    Z_x[j, :] = natural_spline(
        x_grid,
        Z[j, :],
        x_grid
    )

# =====================================
# 5. Apply spline along Y direction
# =====================================

Z_final = np.zeros_like(Z_x)

for i in range(n):

    Z_final[:, i] = natural_spline(
        y_grid,
        Z_x[:, i],
        y_grid
    )


# Save numerical surface data
np.savez(
    "results/spline_surface.npz",
    X=X,
    Y=Y,
    Z=Z_final
)

print("Spline surface data saved.")

# Visualize reconstructed surface

fig = plt.figure(figsize=(11, 8))

ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(
    X,
    Y,
    Z_final,
    cmap="viridis",
    edgecolor="none"
)

ax.scatter(
    x, y, z,
    color="red",
    s=8
)

ax.set_title("Cubic Spline Surface Reconstruction")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.tight_layout()
plt.show()

print("Surface reconstruction completed.")
print("Grid dimensions:", Z_final.shape)
print("Number of surface samples:", Z_final.size)




