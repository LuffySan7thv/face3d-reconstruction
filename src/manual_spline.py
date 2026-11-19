import numpy as np
import matplotlib.pyplot as plt

# 1. Load facial landmarks
points = np.loadtxt(
    "results/face_landmarks.csv",
    delimiter=",",
    skiprows=1
)

# 2. Select ordered points on the face boundary
face_oval = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]

P = points[face_oval]

# 3. Parameter values
t = np.arange(len(P), dtype=float)

# 4. Build the cubic spline system
def calculate_second_derivatives(values):

    n = len(values)

    A = np.zeros((n, n))
    b = np.zeros(n)

    # Natural boundary conditions
    A[0, 0] = 1
    A[-1, -1] = 1

    # Interior equations
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


# 5. Calculate spline coefficients for X, Y, Z
M = np.zeros_like(P)

for j in range(3):
    M[:, j] = calculate_second_derivatives(P[:, j])


# 6. Evaluate cubic spline
def evaluate_spline(values, second_derivatives, t_dense):

    result = []

    for u in t_dense:

        i = min(int(u), len(values) - 2)
        h = 1.0

        a = i + 1 - u
        b = u - i

        value = (
            second_derivatives[i] * a**3 / 6
            + second_derivatives[i + 1] * b**3 / 6
            + (values[i] - second_derivatives[i] / 6) * a
            + (values[i + 1] - second_derivatives[i + 1] / 6) * b
        )

        result.append(value)

    return np.array(result)


# 7. Generate dense parameter values
t_dense = np.linspace(0, len(P) - 1, 500)

curve = np.zeros((len(t_dense), 3))

for j in range(3):
    curve[:, j] = evaluate_spline(
        P[:, j],
        M[:, j],
        t_dense
    )


# 8. Plot the original points and spline curve
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    P[:, 0], P[:, 1], P[:, 2],
    color="red",
    label="Original landmarks"
)

ax.plot(
    curve[:, 0],
    curve[:, 1],
    curve[:, 2],
    linewidth=2,
    label="Manual cubic spline"
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_title("Natural Cubic Spline - Face Boundary")
ax.invert_yaxis()
ax.legend()

plt.show()

print("Spline calculation completed!")
print("Original points:", len(P))
print("Interpolated points:", len(curve))

