import numpy as np
import matplotlib.pyplot as plt


def plot(xlabel, x, ylabel, y, title):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()


def body_to_wind_dcm(alpha, beta):
    x0 = np.cos(alpha)
    x1 = np.cos(beta)
    x2 = np.sin(beta)
    x3 = np.sin(alpha)
    return np.array([[x0 * x1, x2, x1 * x3], [-x0 * x2, x1, -x2 * x3], [-x3, 0, x0]])


def quat2rotm(q):
    # Extract scalar and vector parts of quaternion
    s = q[0]
    v = q[1:4]

    # Compute rotational matrix
    R = np.zeros((3, 3))
    R[0, 0] = 1 - 2 * (v[1] ** 2 + v[2] ** 2)
    R[0, 1] = 2 * (v[0] * v[1] - s * v[2])
    R[0, 2] = 2 * (v[0] * v[2] + s * v[1])
    R[1, 0] = 2 * (v[0] * v[1] + s * v[2])
    R[1, 1] = 1 - 2 * (v[0] ** 2 + v[2] ** 2)
    R[1, 2] = 2 * (v[1] * v[2] - s * v[0])
    R[2, 0] = 2 * (v[0] * v[2] - s * v[1])
    R[2, 1] = 2 * (v[1] * v[2] + s * v[0])
    R[2, 2] = 1 - 2 * (v[0] ** 2 + v[1] ** 2)

    return R


def rotm2euler(R):
    # Compute Euler angles
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arcsin(-R[2, 0])
    yaw = np.arctan2(R[1, 0], R[0, 0])

    return np.array([roll, pitch, yaw])


def quatmultiply(q1, q2):
    # Extract scalar and vector parts of quaternions
    s1 = q1[0]
    v1 = q1[1:4]
    s2 = q2[0]
    v2 = q2[1:4]

    # Compute quaternion product
    s = s1 * s2 - np.dot(v1, v2)
    v = s1 * v2 + s2 * v1 + np.cross(v1, v2)

    return np.concatenate(([s], v))


def calc_J2(r, J2, mu, R):
    z2 = r[2] ** 2
    r_mag = np.linalg.norm(r)
    r2 = r_mag**2
    tx = r[0] / r_mag * (5 * z2 / r2 - 1)
    ty = r[1] / r_mag * (5 * z2 / r2 - 1)
    tz = r[2] / r_mag * (5 * z2 / r2 - 3)
    return 1.5 * J2 * mu * R**2 / r2**2 * np.array([tx, ty, tz])


# Constants
MU_MARS = 4.282828e13  # Mars gravitational parameter [m^3/s^2]
R_MARS = 3.3962e6  # Mars equatorial radius [m]
W_MARS = 7.088e-5  # Mars angular velocity [rad/s]
J2_MARS = 1.964e-3  # Mars J2 coefficient

m = 3.0e3  # Mass of spacecraft [kg]
I = np.array(
    [[4800, 0.0, 0.0], [0.0, 3800, 0.0], [0.0, 0.0, 2900]]
)  # Moment of inertia tensor [kg*m^2]
Iinv = np.linalg.inv(I)  # Inverse of moment of inertia tensor [kg*m^2]

A = 15.904e3  # Reference area of the vehicle [m^2]
d = 4.5e3  # Reference diameter of the vehicle [m]

gamma = np.deg2rad(-15.5)  # Flight path angle [rad]

# Initial state
altitude = 125e3  # Entry interface [m]

r0 = np.array([0.0, altitude + R_MARS, 0.0])  # Initial position [m]
v0 = 5.6e3 * np.array([np.cos(gamma), np.sin(gamma), 0.0])  # Initial velocity [m/s]
q0 = np.array([1.0, 0.0, 0.0, 0.0])  # Initial orientation
w0 = np.array([0.0, 0.0, 0.0])  # Initial angular rate [rad/s]

# Initial state vector
x0 = np.concatenate([r0, v0, q0, w0])

# Simulation time
T = 60.0e3  # End time [s]
dt = 1.0  # Time step [s]

# Initialize time steps
time_range = np.arange(0, T, dt)


# Dynamical model
def odes(t, X):
    # Extract state
    r = X[0:3]
    v = X[3:6]
    q = X[6:10]
    w = X[10:13]

    # Compute rotational matrix from quaternion
    R = quat2rotm(q)

    # Compute attitude
    roll, pitch, yaw = rotm2euler(R)

    # Compute bank angle
    phi = np.arcsin(np.sin(roll) * np.cos(pitch))

    # Compute the magnitude of the velocity vector
    v_mag = np.linalg.norm(v)

    # Compute the magnitude of the position vector
    r_mag = np.linalg.norm(r)
    altitude = r_mag - R_MARS

    # Simple atmospheric model for testing
    rho = 0.015 * np.exp(-altitude / 12e3)

    # Vector normal to the velocity vector
    normal = np.array([v[1], v[0], 0.0])

    beta = 146  # Ballistic coefficient
    ld = 0.20  # L/D ratio
    a_aero = 0.5 * rho * v_mag * (ld * np.cos(phi) * normal / beta - v / beta)

    # Compute acceleration due to gravity and oblateness
    a_grav = -(MU_MARS * r) / r_mag**3
    a_j2 = calc_J2(r, J2_MARS, MU_MARS, R_MARS)

    a_total = a_grav + a_j2 + a_aero
    M_total = np.zeros(3)

    # Compute kinematic equations
    rdot = v
    vdot = a_total
    qdot = 0.5 * quatmultiply(q, np.concatenate(([0], w)))
    wdot = Iinv @ (M_total - np.cross(w, I @ w))

    return np.concatenate([rdot, vdot, qdot, wdot])


def run_simulation(times=1):
    result = []

    for i in range(times):
        # Initialize an empty list to store the state
        state = []

        # Loop through the time range and calculate the state at each time step
        for t in time_range:
            if t == 0:  # Set the initial state
                state.append(x0)
            else:  # Use RK4 method to calculate the next state
                current_state = state[-1]
                k1 = odes(t, current_state)
                k2 = odes(t + 0.5 * dt, current_state + 0.5 * dt * k1)
                k3 = odes(t + 0.5 * dt, current_state + 0.5 * dt * k2)
                k4 = odes(t + dt, current_state + dt * k3)
                next_state = current_state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                state.append(next_state)

                # Check if we reached the ground
                if np.linalg.norm(next_state[0:3]) - R_MARS <= 0.0:
                    break

        state = np.array(state)

        # Print the final state
        print(f"Simulation {i+1} final state:", state[-1])

        result.append(state)

    return result


def main():
    result = run_simulation(100)

    h = []
    V = []

    landing_latitudes = []
    landing_longitudes = []

    for s in result[-1]:
        h.append((np.linalg.norm(s[0:3]) - R_MARS) / 1e3)
        V.append(np.linalg.norm(s[3:6]))

    plot("V (m/s)", V, "h (km)", h, "Mars Entry")


    for res in result:
        landing_latitudes.append(np.degrees(np.arcsin(res[-1][2] / R_MARS)))
        landing_longitudes.append(np.degrees(np.arctan2(res[-1][1], res[-1][0])))


    plt.figure()
    plt.imshow(plt.imread("mars_topo.webp"), extent=[-180, 180, -90, 90])
    plt.plot(landing_longitudes, landing_latitudes, marker="o", markersize=5)

    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(range(-180, 200, 20))
    plt.yticks(range(-90, 100, 10))
    plt.xlabel(r"Longitude (degrees $^\circ$)")
    plt.ylabel(r"Latitude (degrees $^\circ$)")
    plt.tight_layout()

    plt.grid(linestyle="dotted")
    plt.show()


if __name__ == "__main__":
    main()

