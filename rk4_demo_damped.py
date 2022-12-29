# This is just my code for a simple RK4 integrator that I keep as reference for other things
import numpy as np
import matplotlib.pyplot as plt


# Define the function for the differential equation
def damped_harmonic_oscillator(t, y, b, k, m):
    x, v = y
    dxdt = v
    dvdt = -b * v - k * x / m
    return np.array([dxdt, dvdt])

# Set the parameters
b = 0.1  # Damping coefficient
k = 1.0  # Spring constant
m = 1.0  # Mass of the object

# Set the initial state
x0 = 1.0  # Initial position
v0 = 0.0  # Initial velocity
y0 = np.array([x0, v0])

# Set the time range for the simulation
t_start = 0.0  # Start time
t_end = 100.0  # End time
dt = 0.01  # Time step

# Solve the differential equation using the Euler method
time_range = np.arange(t_start, t_end, dt)
y = y0
x_values = []  # List to store the position values
v_values = []  # List to store the velocity values


# Initialize an empty list to store the state at each time step
state = []

# Loop through the time range and calculate the state at each time step
for t in time_range:
    if t == 0: # Set the initial state
        state.append(y0)
        x, v = y0
        x_values.append(x)
        v_values.append(v)
    else: # Use RK4 method to calculate the next state
        current_state = state[-1]
        k1 = damped_harmonic_oscillator(t, current_state, b, k, m)
        k2 = damped_harmonic_oscillator(t + 0.5 * dt, current_state + 0.5 * dt * k1, b, k, m)
        k3 = damped_harmonic_oscillator(t + 0.5 * dt, current_state + 0.5 * dt * k2, b, k, m)
        k4 = damped_harmonic_oscillator(t + dt, current_state + dt * k3, b, k, m)
        next_state = current_state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        state.append(next_state)

        x, v = next_state
        x_values.append(x)
        v_values.append(v)


# Plot the position and velocity as a function of time
plt.plot(time_range, x_values, label="Position")
plt.plot(time_range, v_values, label="Velocity")
plt.xlabel("Time (s)")
plt.legend()
plt.show()
