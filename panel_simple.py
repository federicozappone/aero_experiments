import numpy as np


def body_to_wind_dcm(alpha, beta):
    x0 = np.cos(alpha)
    x1 = np.cos(beta)
    x2 = np.sin(beta)
    x3 = np.sin(alpha)
    return (np.array([[x0*x1, x2, x1*x3], [-x0*x2, x1, -x2*x3], [-x3, 0, x0]]))

# Define a function to compute the aerodynamic forces and moments
def compute_aero_forces_and_moments(mesh, velocity, altitude, alpha, beta):
    # Compute the atmospheric properties at the given altitude
    T, P, rho, a = compute_atmospheric_properties(altitude)

    # Compute the dynamic viscosity of the atmosphere
    mu = 1.458e-6 * T**1.5 / (T + 110.4)

    # Compute the Mach number
    M = np.linalg.norm(velocity) / a

    # Compute the reference area and the centroid of the mesh
    centroid = mesh.center_mass

    # Compute the direction of the freestream velocity relative to the mesh
    freestream_direction = body_to_wind_dcm(alpha, beta) @ velocity

    # Compute the aerodynamic forces and moments
    F_aero = np.zeros(3)
    M_aero = np.zeros(3)

    for face in mesh.faces:
        # Compute the normal vector of the face
        normal = mesh.face_normals[face]

        # Compute the area of the face
        face_area = mesh.area_faces[face]

        # Compute the pressure force on the face
        F_p = 0.5 * rho * np.dot(freestream_direction, normal) * face_area * normal

        # Compute the viscous force on the face
        F_v = 0.5 * mu * face_area * np.dot(velocity, normal) * np.array(
            [normal[0], -normal[1], normal[2]]) / (np.linalg.norm(normal) * np.linalg.norm(velocity) * velocity)

        # Compute the shock wave correction factor
        if M >= 1:
            f = 1 + (gamma - 1) / 2 * M**2
        else:
            f = 1

        # Compute the aerodynamic force on the face
        F_aero_face = f * (F_p + F_v)

        # Accumulate the aerodynamic force and moment
        F_aero += F_aero_face
        M_aero += np.cross(mesh.vertices[face[0]] - centroid, F_aero_face)


    return F_aero, M_aero



