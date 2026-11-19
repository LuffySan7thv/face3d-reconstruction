import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


# ---------------------------------
# 1. Load facial landmark data
# ---------------------------------

points = np.loadtxt(
    "results/face_landmarks.csv",
    delimiter=",",
    skiprows=1
)

x = points[:, 0]
y = points[:, 1]
z = points[:, 2]


# ---------------------------------
# 2. Create a regular initial grid
# ---------------------------------

grid_size = 40

xi = np.linspace(x.min(), x.max(), grid_size)
yi = np.linspace(y.min(), y.max(), grid_size)

X, Y = np.meshgrid(xi, yi)

Z = griddata(
    (x, y),
    z,
    (X, Y),
    method="linear"
)

# Fill points outside the convex hull
Z_nearest = griddata(
    (x, y),
    z,
    (X, Y),
    method="nearest"
)

Z = np.where(np.isnan(Z), Z_nearest, Z)


# ---------------------------------
# 3. Natural cubic spline system
# ---------------------------------

def spline_second_derivatives(values):

    n = len(values)

    A = np.zeros((n, n))
    b = np.zeros(n)

    A[0, 0] = 1
    A[-1, -1] = 1

    for i in range(1, n - 1):

        A[i, i - 1] = 1
        A[i, i] = 4
        A[i, i + 1] = 1

        b[i] = 6 * (
            values[i + 1]
            - 2 * values[i]
            + values[i - 1]
        )

    return np.linalg.solve(A, b)


# ---------------------------------
# 4. Evaluate cubic spline
# ---------------------------------

def spline_evaluate(values, M, t):

    n = len(values)

    i = min(int(t), n - 2)

    a = i + 1 - t
    b = t - i

    return (
        M[i] * a**3 / 6
        + M[i + 1] * b**3 / 6
        + (values[i] - M[i] / 6) * a
        + (values[i + 1] - M[i + 1] / 6) * b
    )


# ---------------------------------
# 5. Spline interpolation along X
# ---------------------------------

dense_size = 100

tx = np.linspace(0, grid_size - 1, dense_size)
ty = np.linspace(0, grid_size - 1, dense_size)

Zx = np.zeros((grid_size, dense_size))

for j in range(grid_size):

    row = Z[j, :]

    M = spline_second_derivatives(row)

    for k, t in enumerate(tx):

        Zx[j, k] = spline_evaluate(row, M, t)


# ---------------------------------
# 6. Spline interpolation along Y
# ---------------------------------

Z_final = np.zeros((dense_size, dense_size))

for j in range(dense_size):

    column = Zx[:, j]

    M = spline_second_derivatives(column)

    for k, t in enumerate(ty):

        Z_final[k, j] = spline_evaluate(
            column, M, t
        )


# ---------------------------------
# 7. Create final coordinate grid
# ---------------------------------

x_final = np.linspace(xi.min(), xi.max(), dense_size)
y_final = np.linspace(yi.min(), yi.max(), dense_size)

X_final, Y_final = np.meshgrid(x_final, y_final)


# ---------------------------------
# 8. Plot 3D spline surface
# ---------------------------------

fig = plt.figure(figsize=(12, 9))

ax = fig.add_subplot(111, projection="3d")

surface = ax.plot_surface(
    X_final,
    Y_final,
    Z_final,
    cmap="viridis",
    edgecolor="none"
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Relative Depth")

ax.set_title("2D Tensor Product Cubic Spline")

ax.invert_yaxis()

fig.colorbar(surface, ax=ax, shrink=0.6)

plt.show()

print("2D spline surface completed!")
print("Initial grid:", Z.shape)
print("Final grid:", Z_final.shape)

