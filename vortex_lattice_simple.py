import numpy as np
import trimesh


def vortex_lattice(mesh, velocity, density, alpha, beta):
    # Convert the mesh to a surface mesh
    surface_mesh = mesh.split(only_watertight=True)[0]

    # Compute the surface normal vectors for each face of the mesh
    face_normals = surface_mesh.face_normals

    # Compute the bound vortex strength for each face of the mesh
    bound_vortex_strength = np.cross(face_normals, velocity)

    # Compute the induced velocity at each vertex of the mesh
    induced_velocity = np.zeros_like(surface_mesh.vertices)
    for i, face in enumerate(surface_mesh.faces):
        # Compute the vortex position as the centroid of the face
        vortex_position = np.mean(surface_mesh.vertices[face], axis=0)
        # Compute the vortex-vertex distance vector
        r = surface_mesh.vertices[face] - vortex_position[np.newaxis, :]
        # Compute the induced velocity contribution from this vortex
        induced_velocity[face] += (
            bound_vortex_strength[i]
            / (4 * np.pi * density * np.linalg.norm(r, axis=1)[:, np.newaxis])
            * np.cross(r, face_normals[i])
        )

    # Compute the total velocity at each vertex of the mesh
    total_velocity = induced_velocity + velocity[np.newaxis, :]

    # Compute the lift and drag coefficients for the mesh
    cl = np.sum(
        surface_mesh.area
        * np.sin(alpha - np.arctan2(total_velocity[:, 1], total_velocity[:, 0]))
    ) / (0.5 * density * np.linalg.norm(velocity) ** 2 * np.sum(surface_mesh.area))
    cd = np.sum(
        surface_mesh.area
        * np.cos(alpha - np.arctan2(total_velocity[:, 1], total_velocity[:, 0]))
    ) / (0.5 * density * np.linalg.norm(velocity) ** 2 * np.sum(surface_mesh.area))

    # Compute the lift and drag forces on the mesh
    lift = (
        0.5 * density * np.linalg.norm(velocity) ** 2 * np.sum(surface_mesh.area) * cl
    )
    drag = (
        0.5 * density * np.linalg.norm(velocity) ** 2 * np.sum(surface_mesh.area) * cd
    )

    cm = np.sum(
        surface_mesh.area
        * (surface_mesh.centroid[0] - mesh.centroid[0])
        * np.sin(beta - np.arctan2(total_velocity[:, 2], total_velocity[:, 0]))
    ) / (0.5 * density * np.linalg.norm(velocity) ** 2 * np.sum(surface_mesh.area))

    # Compute the pitching moment about the reference point for the mesh
    pitching_moment = (
        0.5 * density * np.linalg.norm(velocity) ** 2 * np.sum(surface_mesh.area) * cm
    )

    # Compute the yawing moment coefficient for the mesh
    cy = np.sum(
        surface_mesh.area
        * (surface_mesh.centroid[1] - mesh.centroid[1])
        * np.sin(beta - np.arctan2(total_velocity[:, 2], total_velocity[:, 1]))
    ) / (0.5 * density * np.linalg.norm(velocity) ** 2 * np.sum(surface_mesh.area))

    # Compute the yawing moment about the reference point for the mesh
    yawing_moment = (
        0.5 * density * np.linalg.norm(velocity) ** 2 * np.sum(surface_mesh.area) * cy
    )

    # Return the lift, drag, and moment coefficients for the mesh
    return lift, drag, pitching_moment, yawing_moment


# Load the mesh using the trimesh library
mesh = trimesh.creation.uv_sphere()

# Set the flow parameters
velocity = np.array([10.0, 0.0, 0.0]) # flow velocity in the x direction
density = 1.225 # air density
viscosity = 1.789e-5 # air viscosity
alpha = 0.0 # angle of attack in degrees
beta = 0.0 # sideslip angle in degrees

# Convert the angles of attack and sideslip to radians
alpha = np.deg2rad(alpha)
beta = np.deg2rad(beta)

# Compute the aerodynamic forces and moments on the mesh
lift, drag, pitching_moment, yawing_moment = vortex_lattice(mesh, velocity, density, alpha, beta)

# Print the results
print("Lift: ", lift)
print("Drag: ", drag)
print("Pitching moment: ", pitching_moment)
print("Yawing moment: ", yawing_moment)

