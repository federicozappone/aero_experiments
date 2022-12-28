import numpy as np
import matplotlib.pyplot as plt


def navier_stokes_solver(u, v, w, p, rho, nu, dx, dy, dz, dt, nt):
    """
    Solve the Navier-Stokes equations in 3D using the finite difference method.

    Parameters
    ----------
    u, v, w : numpy arrays
        Initial velocity field in the x, y, and z directions, respectively.
    p : numpy array
        Initial pressure field.
    rho : float
        Density of the fluid.
    nu : float
        Kinematic viscosity of the fluid.
    dx, dy, dz : floats
        Spacing between grid points in the x, y, and z directions, respectively.
    dt : float
        Time step size.
    nt : int
        Number of time steps to take.

    Returns
    -------
    u, v, w : numpy arrays
        Velocity field at the final time step in the
    x, y, and z directions, respectively.
    p : numpy array
    Pressure field at the final time step.
    """
    # Compute the number of grid points in each direction
    nx, ny, nz = u.shape

    # Set the initial values for the pressure correction and the residual
    p_corr = np.zeros_like(p)
    res = np.zeros_like(p)

    # Set the boundary conditions for the velocity field
    u[:, :, 0] = 0
    u[:, :, -1] = 0
    u[:, 0, :] = 0
    u[:, -1, :] = 0
    u[0, :, :] = 0
    u[-1, :, :] = 0
    v[:, :, 0] = 0
    v[:, :, -1] = 0
    v[:, 0, :] = 0
    v[:, -1, :] = 0
    v[0, :, :] = 0
    v[-1, :, :] = 0
    w[:, :, 0] = 0
    w[:, :, -1] = 0
    w[:, 0, :] = 0
    w[:, -1, :] = 0
    w[0, :, :] = 0
    w[-1, :, :] = 0

    # Set the boundary conditions for the pressure field
    p[:, :, 0] = 0
    p[:, :, -1] = 0
    p[:, 0, :] = 0
    p[:, -1, :] = 0
    p[0, :, :] = 0
    p[-1, :, :] = 0

    for _ in range(nt):
        # Update the velocity field using the Euler method
        u_star = u + dt * (
            -u * np.gradient(u, dx, axis=0)
            - v * np.gradient(u, dy, axis=1)
            - w * np.gradient(u, dz, axis=2)
            + (1 / rho) * np.gradient(p, dx, axis=0)
            - nu
            * (
                np.gradient(u, dx, axis=0, edge_order=2)
                + np.gradient(u, dy, axis=1, edge_order=2)
                + np.gradient(u, dz, axis=2, edge_order=2)
            )
        )
        v_star = v + dt * (
            -u * np.gradient(v, dx, axis=0)
            - v * np.gradient(v, dy, axis=1)
            - w * np.gradient(v, dz, axis=2)
            + (1 / rho) * np.gradient(p, dy, axis=1)
            - nu
            * (
                np.gradient(v, dx, axis=0, edge_order=2)
                + np.gradient(v, dy, axis=1, edge_order=2)
                + np.gradient(v, dz, axis=2, edge_order=2)
            )
        )
        w_star = w + dt * (
            -u * np.gradient(w, dx, axis=0)
            - v * np.gradient(w, dy, axis=1)
            - w * np.gradient(w, dz, axis=2)
            + (1 / rho) * np.gradient(p, dz, axis=2)
            - nu
            * (
                np.gradient(w, dx, axis=0, edge_order=2)
                + np.gradient(w, dy, axis=1, edge_order=2)
                + np.gradient(w, dz, axis=2, edge_order=2)
            )
        )

        # Compute the residual
        res = (rho / dt) * (
            np.gradient(u_star, dx, axis=0)
            + np.gradient(v_star, dy, axis=1)
            + np.gradient(w_star, dz, axis=2)
        )

        # Update the pressure correction using the SIMPLE algorithm
        p_corr = (
            p_corr
            + (rho / dt)
            * (
                np.gradient(p_corr, dx, axis=0)
                + np.gradient(p_corr, dy, axis=1)
                + np.gradient(p_corr, dz, axis=2)
            )
            - res
        )

        # Update the velocity field using the pressure correction
        u = u_star - (dt / rho) * np.gradient(p_corr, dx, axis=0)
        v = v_star - (dt / rho) * np.gradient(p_corr, dy, axis=1)
        w = w_star - (dt / rho) * np.gradient(p_corr, dz, axis=2)

        # Update the pressure field using the pressure correction
        p = p + p_corr

        # Create the x, y, and z grids
        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        z = np.linspace(0, Lz, nz)

        # Plot the velocity field in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        ax.quiver(X, Y, Z, u, v, w, length=0.1, normalize=True)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.title("Velocity field")

        plt.show()

    return u, v, w, p


# Set the size of the domain
Lx = 1.0
Ly = 1.0
Lz = 1.0

# Set the number of grid points in each direction
nx = 10
ny = 10
nz = 10

# Set the grid spacing in each direction
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dz = Lz / (nz - 1)

# Set the time step size
dt = 0.001

# Set the number of time steps to take
nt = 10

# Set the fluid properties
rho = 1.0
nu = 0.1

# Set the initial velocity field
u = np.ones((nx, ny, nz)) * 0.01
v = np.zeros((nx, ny, nz))
w = np.zeros((nx, ny, nz))

# Set the initial pressure field
p = np.ones((nx, ny, nz)) * 0.0

# Solve the Navier-Stokes equations
u, v, w, p = navier_stokes_solver(u, v, w, p, rho, nu, dx, dy, dz, dt, nt)

# Print the final velocity and pressure fields
print("Final velocity field:")
print("u =", u)
print("v =", v)
print("w =", w)
print("Final pressure field:")
print("p =", p)

