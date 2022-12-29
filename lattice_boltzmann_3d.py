import numpy as np
import matplotlib.pyplot as plt


# Physical constants
nu = 0.1  # Viscosity
rho = 1.0  # Density
cs = 1.0   # Speed of sound
kappa = 1.0  # Thermal conductivity
T0 = 1.0  # Reference temperature

# Simulation parameters
nx = 200  # Number of lattice points in x direction
ny = 100  # Number of lattice points in y direction
nz = 50   # Number of lattice points in z direction
tau = 1.0  # Relaxation time
omega = 1.0 / tau  # Relaxation frequency
max_iter = 10000  # Maximum number of iterations

# Lattice velocities (D3Q19 lattice)
c = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
              [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
              [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1]])

# Lattice weights (D3Q19 lattice)
w = np.array([1 / 3, 1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18,
              1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36,
              1 / 36, 1 / 36, 1 / 36, 1 / 36])

# Initialize distribution functions and flow field
f = np.zeros((nx, ny, nz, len(w)))
u = np.zeros((nx, ny, nz))
v = np.zeros((nx, ny, nz))
w = np.zeros((nx, ny, nz))

# Initialize distribution functions at equilibrium
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            # Compute equilibrium distribution function for each velocity
            ue = u[i, j, k]
            ve = v[i, j, k]
            we = w[i, j, k]
            pe = cs ** 2 * (rho - 1)
            Te = T[i, j, k] - T0  # Temperature deviation from reference temperature
            feq = w * rho * (1 + 3 * (c[:, 0] * ue + c[:, 1] * ve + c[:, 2] * we) +
                             4.5 * ((c[:, 0] * ue) ** 2 + (c[:, 1] * ve) ** 2 + (c[:, 2] * we) ** 2 -
                                    (ue ** 2 + ve ** 2 + we ** 2)) +
                             1.5 * pe + 3 * Te)
            # Set initial distribution function to equilibrium
            f[i, j, k, :] = feq

# Iterate until flow reaches steady state or maximum number of iterations is reached
for t in range(max_iter):
    # Collision step
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Compute equilibrium distribution function
                ue = u[i, j, k]
                ve = v[i, j, k]
                we = w[i, j, k]
                pe = cs ** 2 * (rho[i, j, k] - 1)
                Te = T[i, j, k] - T0  # Temperature deviation from reference temperature
                feq = w * rho[i, j, k] * (1 + 3 * (c[:, 0] * ue + c[:, 1] * ve + c[:, 2] * we) +
                                         4.5 * ((c[:, 0] * ue) ** 2 + (c[:, 1] * ve) ** 2 +
                                                (c[:, 2] * we) ** 2 - (ue ** 2 + ve ** 2 + we ** 2)) +
                                         1.5 * pe + 3 * Te)
                # Relax distribution function towards equilibrium
                f[i, j, k, :] = (1 - omega) * f[i, j, k, :] + omega * feq
                # Compute viscous stress tensor
                sigma = nu * (np.sum(c[:, 0] ** 2 * f[i, j, k, :] - feq) +
                              np.sum(c[:, 1] ** 2 * f[i, j, k, :] - feq) +
                              np.sum(c[:, 2] ** 2 * f[i, j, k, :] - feq))
                # Modify distribution function to include viscous forces
                f[i, j, k, :] += w * sigma / (2 * cs ** 2)
                # Compute heat exchange between fluid and environment
                q = kappa * (T[i, j, k] - T_env)
                # Modify distribution function to include heat exchange
                f[i, j, k, :] += w * q / (2 * cs ** 2)
    
    # Streaming step
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Compute flow field at lattice point (i, j, k)
                u[i, j, k] = (np.sum(f[i, j, k, :] * c[:, 0]) / rho[i, j, k])
                v[i, j, k] = (np.sum(f[i, j, k, :] * c[:, 1]) / rho[i, j, k])
                w[i, j, k] = (np.sum(f[i, j, k, :] * c[:, 2]) / rho[i, j, k])
                
                # Iterate over all lattice velocities
                for l in range(len(w)):
                    # Compute new lattice indices and handle periodic boundary conditions
                    ni = i + c[l, 0]
                    nj = j + c[l, 1]
                    nk = k + c[l, 2]
                    if ni < 0:
                        ni += nx
                    if ni >= nx:
                        ni -= nx
                    if nj < 0:
                        nj += ny
                    if nj >= ny:
                        nj -= ny
                    if nk < 0:
                        nk += nz
                    if nk >= nz:
                        nk -= nz

                    # Stream distribution function from lattice point
                    f[ni, nj, nk, l] = f[i, j, k, l]

    # Compute macroscopic flow and temperature fields
    rho = np.sum(f, axis=3)
    u = (np.sum(f * c[:, 0], axis=3) / rho).T
    v = (np.sum(f * c[:, 1], axis=3) / rho).T
    w = (np.sum(f * c[:, 2], axis=3) / rho).T
    T = (np.sum(f * e, axis=3) / rho).T + T0

    # Update plot
    ax1.clear()
    ax1.quiver(X, Y, u, v)
    ax1.set_title("Velocity field")
    ax2.clear()
    ax2.contourf(X, Y, T)
    ax2.set_title("Temperature field")
    plt.pause(0.01)

# Show plot
plt.show()