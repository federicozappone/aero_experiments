import numpy as np
import matplotlib.pyplot as plt


def collision_step(f, u, v, rho, nu, tau, omega):
    # Compute equilibrium distribution function
    feq = w * rho * (1 + 3 * (c[:, 0] * u + c[:, 1] * v) +
                     4.5 * ((c[:, 0] * u) ** 2 + (c[:, 1] * v) ** 2 -
                            (u ** 2 + v ** 2)))
    # Add viscosity term
    feq += w * nu * (3 * (c[:, 0] ** 2 + c[:, 1] ** 2) - 2) * (c[:, 0] * u + c[:, 1] * v)
    # Relax distribution function towards equilibrium
    f = (1 - omega) * f + omega * feq
    return f

# Physical constants
nu = 0.1  # Viscosity
rho = 1.0  # Density

# Simulation parameters
nx = 200  # Number of lattice points in x direction
ny = 100  # Number of lattice points in y direction
tau = 1.0  # Relaxation time
omega = 1.0 / tau  # Relaxation frequency
max_iter = 10000  # Maximum number of iterations

# Lattice velocities
c = np.array([[0, 1], [1, 0], [0, -1], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]])

# Lattice weights
w = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])

# Initialize distribution functions and flow field
f = np.zeros((nx, ny, len(w)))
u = np.zeros((nx, ny))
v = np.zeros((nx, ny))

# Define geometry of the simulation domain
# cylinder_center is a tuple containing the x and y coordinates of the cylinder center
cylinder_center = (nx // 2, ny // 2)
# cylinder_radius is the radius of the cylinder
cylinder_radius = ny // 4

# Initialize distribution functions at equilibrium
for i in range(nx):
    for j in range(ny):
        # Compute equilibrium distribution function for each velocity
        feq = w * rho * (1 + 3 * (c[:, 0] * u[i, j] + c[:, 1] * v[i, j]) +
                         4.5 * ((c[:, 0] * u[i, j]) ** 2 + (c[:, 1] * v[i, j]) ** 2 -
                                (u[i, j] ** 2 + v[i, j] ** 2)))
        # Set initial distribution function to equilibrium
        f[i, j, :] = feq

# Iterate until flow reaches steady state or maximum number of iterations is reached
for t in range(max_iter):
    # Collision step
    for i in range(nx):
        for j in range(ny):
            # Compute density and velocity at lattice point (i, j)
            rho = np.sum(f[i, j, :])
            u[i, j] = (np.sum(f[i, j, :] * c[:, 0]) / rho)
            v[i, j] = (np.sum(f[i, j, :] * c[:, 1]) / rho)
            # Relax distribution function towards equilibrium with viscosity and compressibility
            f[i, j, :] = collision_step(f[i, j, :], u[i, j], v[i, j], rho, nu, tau, omega)

    # Streaming step
    for i in range(nx):
        for j in range(ny):
            # Compute flow field at lattice point (i, j)
            u[i, j] = (np.sum(f[i, j, :] * c[:, 0]) / rho)
            v[i, j] = (np.sum(f[i, j, :] * c[:, 1]) / rho)
            
            # Check if lattice point is inside the cylinder
            if (i - cylinder_center[0]) ** 2 + (j - cylinder_center[1]) ** 2 < cylinder_radius ** 2:
                # Set velocity to zero inside the cylinder
                u[i, j] = 0
                v[i, j] = 0
                # Set distribution functions to equilibrium with zero velocity
                f[i, j, :] = w * rho
                
            # Iterate over all lattice velocities
            for k in range(len(w)):
                # Compute new lattice indices after streaming
                i_new = i + c[k, 0]
                j_new = j + c[k, 1]
                
                # Implement periodic boundary conditions
                if i_new < 0:
                    i_new += nx
                elif i_new >= nx:
                    i_new -= nx
                if j_new < 0:
                    j_new += ny
                elif j_new >= ny:
                    j_new -= ny
                    
                # Stream distribution function from lattice point (i, j) to (i_new, j_new)
                f[i_new, j_new, k] = f[i, j, k]


# Visualize flow field and any relevant quantities

# Compute pressure field
pressure = rho * (1 - (u ** 2 + v ** 2) / 2)

# Set up figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot velocity field
ax1.quiver(u, v)
ax1.set_title('Velocity Field')

# Plot pressure field
p = ax2.pcolor(pressure)
fig.colorbar(p)
ax2.set_title('Pressure Field')

plt.show()
