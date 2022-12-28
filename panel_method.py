import trimesh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri


# Load the mesh using the trimesh library
mesh = trimesh.creation.uv_sphere()

# Extract the vertices and faces of the mesh
vertices = mesh.vertices
faces = mesh.faces

# Compute the normal vector for each face
face_normals = np.array(mesh.face_normals)

# Normalize the normal vectors
face_normals /= np.linalg.norm(face_normals, axis=1)[:, np.newaxis]

# Compute the panel midpoints and lengths
panel_midpoints = (
    vertices[faces[:, 0]] + vertices[faces[:, 1]] + vertices[faces[:, 2]]
) / 3
panel_lengths = np.linalg.norm(vertices[faces[:, 1]] - vertices[faces[:, 0]])

# Set the flow conditions (density, velocity, viscosity, and angle of attack)
rho = 1.225  # density of air [kg/m^3]
V = 1.0  # velocity of the flow [m/s]
mu = 1.8e-5  # dynamic viscosity of air [Pa*s]
a = 343.0 # speed of sound [m/s]
alpha = 0.0  # angle of attack [rad]

# Compute the freestream velocity vector
V_inf = V * np.array([np.cos(alpha), np.sin(alpha), 0.0])

# Compute the Reynolds number
Re = rho * V * np.mean(panel_lengths) / mu

# Compute the viscous correction factor
if Re < 1e5:
    # Laminar flow
    f = 1.328 / np.sqrt(Re)
else:
    # Turbulent flow
    f = 0.455 / (np.log10(Re) ** 2.58)

# Compute the viscous velocity at each panel midpoint
viscous_velocity = np.zeros_like(panel_midpoints)
for i, panel_midpoint in enumerate(panel_midpoints):
    viscous_velocity[i] = np.sum(
        f
        * V_inf
        * np.exp(-1j * np.sum(panel_midpoint - panel_midpoints * face_normals, axis=1))
        / (4 * np.pi * np.linalg.norm(panel_midpoint - panel_midpoints, axis=1))
    )

# Compute the total velocity at each panel midpoint
total_velocity = V_inf + viscous_velocity

# Compute the Mach number at each panel midpoint
Mach = np.linalg.norm(total_velocity, axis=1) / a

# Compute the pressure coefficient at each panel
if np.any(Mach > 1.0):
    # Compressible flow with shock waves
    cp = 1 - (2 * Mach[0] ** 2 / (Mach**2 + 1)) * (Mach / Mach[0]) ** 2
else:
    # Incompressible flow
    cp = 1 - (total_velocity / V) ** 2

# print(np.conj(total_velocity))
print(cp)

# Compute the lift and drag forces on each panel
L = rho * V**2 * panel_lengths * np.real(cp * np.conj(total_velocity))
D = rho * V**2 * panel_lengths * np.imag(cp * np.conj(total_velocity))

# Sum the lift and drag forces on all panels to get the total lift and drag forces
L_tot = np.sum(L)
D_tot = np.sum(D)

# Compute the lift and drag coefficients
CL = L_tot / (0.5 * rho * V**2 * np.sum(panel_lengths))
CD = D_tot / (0.5 * rho * V**2 * np.sum(panel_lengths))

# Compute the moments about the center of pressure
moments = np.cross(panel_midpoints, L)

# Compute the center of pressure
CP = np.sum(moments, axis=0) / L_tot

# Compute the moment coefficients
Cmx = moments[:,0].sum() / (0.5 * rho * V**2 * np.sum(panel_lengths))
Cmy = moments[:,1].sum() / (0.5 * rho * V**2 * np.sum(panel_lengths))
Cmz = moments[:,2].sum() / (0.5 * rho * V**2 * np.sum(panel_lengths))

# Print the results
print("Lift force: {:.2f} N".format(L_tot))
print("Drag force: {:.2f} N".format(D_tot))
print("Lift coefficient: {:.2f}".format(CL))
print("Drag coefficient: {:.2f}".format(CD))
print("Center of pressure: ({:.2f}, {:.2f}, {:.2f})".format(CP[0], CP[1], CP[2]))
print("Moment coefficients: Cmx = {:.2f}, Cmy = {:.2f}, Cmz = {:.2f}".format(Cmx, Cmy, Cmz))

# Compute the pressure on each panel
pressure = rho * V**2 * cp

# Create a Triangulation object
triang = tri.Triangulation(panel_midpoints[:,0], panel_midpoints[:,1])

# Create a Figure object and set the axes limits
fig, ax = plt.subplots(figsize=(10,10))
ax.set_xlim(panel_midpoints[:,0].min(), panel_midpoints[:,0].max())
ax.set_ylim(panel_midpoints[:,1].min(), panel_midpoints[:,1].max())

# Display the pressure on the mesh
ax.tricontourf(triang, pressure, cmap='RdYlBu')

# Display the mesh
ax.triplot(triang, color='k')

# Show the plot
plt.show()
