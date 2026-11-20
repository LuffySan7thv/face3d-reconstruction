import numpy as np
import open3d as o3d

# =================================
# 1. Load spline surface
# =================================

data = np.load("results/spline_surface.npz")

X = data["X"]
Y = data["Y"]
Z = data["Z"]

rows, cols = X.shape

# =================================
# 2. Generate 3D vertices
# =================================

vertices = np.column_stack((
    X.ravel(),
    Y.ravel(),
    Z.ravel()
))

# =================================
# 3. Generate triangular faces
# =================================

triangles = []

for i in range(rows - 1):
    for j in range(cols - 1):

        p1 = i * cols + j
        p2 = p1 + 1
        p3 = p1 + cols
        p4 = p3 + 1

        triangles.append([p1, p2, p3])
        triangles.append([p2, p4, p3])

triangles = np.array(
    triangles,
    dtype=np.int32
)

# =================================
# 4. Construct triangular mesh
# =================================

mesh = o3d.geometry.TriangleMesh()

mesh.vertices = o3d.utility.Vector3dVector(
    vertices
)

mesh.triangles = o3d.utility.Vector3iVector(
    triangles
)

# =================================
# 5. Calculate surface normals
# =================================

mesh.compute_vertex_normals()

# =================================
# 6. Save 3D model
# =================================

mesh_path = "results/face_mesh.ply"

o3d.io.write_triangle_mesh(
    mesh_path,
    mesh
)

print("Vertices:", len(vertices))
print("Triangles:", len(triangles))
print("Mesh saved:", mesh_path)

# =================================
# 7. Display 3D mesh
# =================================

o3d.visualization.draw_geometries(
    [mesh],
    window_name="3D Face Surface"
)

