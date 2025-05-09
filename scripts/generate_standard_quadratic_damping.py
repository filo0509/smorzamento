import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

def simulate_disco_forato(
    mass=0.2016,           # Mass in kg
    k=14.92,               # Spring constant in N/m
    b=0.05,                # Quadratic damping coefficient in kg/m
    x0=0.1,                # Initial position in m
    v0=0,                  # Initial velocity in m/s
    t_max=20.0,            # Maximum simulation time in seconds
    dt=0.01,               # Time step in seconds
    noise_level=0.001      # Measurement noise level
):
    """
    Simulate a spring-mass-damper system with quadratic damping.
    
    This simulates a mass attached to a spring and a perforated disk (disco forato)
    where damping is proportional to velocity squared.
    
    The equation of motion is:
    m * x''(t) + b * x'(t)^2 * sign(x'(t)) + k * x(t) = 0
    """
    # Natural frequency
    omega_n = np.sqrt(k / mass)
    print(f"Natural frequency: {omega_n:.4f} rad/s = {omega_n/(2*np.pi):.4f} Hz")
    
    # Define the ODE system
    def spring_mass_quadratic_damping(t, state):
        x, v = state
        # Quadratic damping: force proportional to v^2 with correct sign
        damping_force = -b * v * np.abs(v)
        # Spring force: F = -kx
        spring_force = -k * x
        # Sum forces: F = ma -> a = F/m
        a = (spring_force + damping_force) / mass
        return [v, a]
    
    # Time points for simulation
    t_span = (0, t_max)
    t_eval = np.arange(0, t_max, dt)
    
    # Initial state [position, velocity]
    initial_state = [x0, v0]
    
    # Solve ODE system
    solution = solve_ivp(
        spring_mass_quadratic_damping,
        t_span,
        initial_state,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8,
        atol=1e-8
    )
    
    # Extract results
    time = solution.t
    position = solution.y[0]
    velocity = solution.y[1]
    
    # Calculate acceleration
    acceleration = np.zeros_like(velocity)
    for i in range(len(time)):
        x, v = position[i], velocity[i]
        damping_force = -b * v * np.abs(v)
        spring_force = -k * x
        acceleration[i] = (spring_force + damping_force) / mass
    
    # Add realistic measurement noise
    position_noisy = position + np.random.normal(0, noise_level, position.shape)
    velocity_noisy = velocity + np.random.normal(0, noise_level * 10, velocity.shape)
    acceleration_noisy = acceleration + np.random.normal(0, noise_level * 100, acceleration.shape)
    
    # Create DataFrame
    data = pd.DataFrame({
        'time': time,
        'position': position_noisy,
        'velocity': velocity_noisy,
        'acceleration': acceleration_noisy
    })
    
    # Calculate theoretical damping parameter beta
    beta = (8 * b * omega_n) / (3 * np.pi * mass)
    A0 = np.abs(x0)
    
    # Plot to verify the simulation
    plt.figure(figsize=(12, 8))
    plt.plot(time, position_noisy)
    
    # Find peaks for verification
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(position_noisy, height=0.005, distance=5/dt)
    peak_times = time[peaks]
    peak_positions = position_noisy[peaks]
    
    # Theoretical envelope for quadratic damping
    envelope_times = np.linspace(0, t_max, 500)
    envelope = A0 / (1 + beta * A0 * envelope_times)
    
    # Plot
    plt.plot(time, position_noisy, 'b-', alpha=0.7, label='Position')
    plt.plot(peak_times, peak_positions, 'ro', label='Peaks')
    plt.plot(envelope_times, envelope, 'g-', label=f'Theoretical Envelope (β={beta:.4f})')
    plt.plot(envelope_times, -envelope, 'g-')
    
    plt.title('Simulated Disco Forato Oscillation with Quadratic Damping')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend()
    plt.grid(True)
    
    return data, beta, omega_n

def generate_data_files(
    mass=0.2016,
    spring_constants=[14.92],
    damping_coefficients=[0.05],
    initial_amplitudes=[0.1],
    noise_levels=[0.001]
):
    """Generate multiple data files with different parameters."""
    
    # Create data_simulated directory if it doesn't exist
    os.makedirs('data_simulated', exist_ok=True)
    
    file_count = 0
    for k in spring_constants:
        for b in damping_coefficients:
            for x0 in initial_amplitudes:
                for noise in noise_levels:
                    file_count += 1
                    
                    # Generate data
                    print(f"\nGenerating file {file_count} with parameters:")
                    print(f"Mass: {mass} kg")
                    print(f"Spring constant: {k} N/m")
                    print(f"Damping coefficient: {b} kg/m")
                    print(f"Initial amplitude: {x0} m")
                    
                    data, beta, omega_n = simulate_disco_forato(
                        mass=mass,
                        k=k,
                        b=b,
                        x0=x0,
                        noise_level=noise
                    )
                    
                    # Calculate expected half-life for naming
                    half_life = 1 / (beta * x0)
                    print(f"Theoretical half-life: {half_life:.4f} s")
                    
                    # Create filename
                    filename = f"molla_dura_discoforato_{k:.5f}_{mass:.4f}_{file_count}.csv"
                    filepath = os.path.join('data_simulated', filename)
                    
                    # Save data
                    data.to_csv(filepath, sep=';', index=False)
                    print(f"Saved to {filepath}")
                    
                    # Plot position vs time (diagnostic)
                    plt.figure(figsize=(10, 6))
                    plt.plot(data['time'], data['position'])
                    plt.title(f'Position vs Time (File {file_count})')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Position (m)')
                    plt.grid(True)
                    
                    # Calculate and print some useful information
                    T = 2 * np.pi / omega_n
                    print(f"Natural period: {T:.4f} s")
                    print(f"Expected frequency: {1/T:.4f} Hz")
                    print(f"Quadratic damping coefficient (b): {b:.4f} kg/m")
                    print(f"Damping parameter (β): {beta:.6f}")
                    print(f"Dataset size: {len(data)} points")
    
    print(f"\nGenerated {file_count} data files in the 'data_simulated' directory.")

if __name__ == "__main__":
    # Generate a standard dataset
    data, beta, omega_n = simulate_disco_forato()
    
    # Save to file
    filename = "molla_dura_discoforato_14.92000_0.2016_1.csv"
    filepath = os.path.join('data_simulated', filename)
    data.to_csv(filepath, sep=';', index=False)
    print(f"Saved to {filepath}")
    
    # Optional: Generate multiple files with different parameters
    response = input("Do you want to generate additional data files with different parameters? (y/n): ")
    if response.lower() == 'y':
        # Ensure data_simulated directory exists
        os.makedirs('data_simulated', exist_ok=True)
        # Ensure results directory exists
        os.makedirs('results/plots', exist_ok=True)
        
        # Generate additional files with varying parameters
        generate_data_files(
            spring_constants=[14.92, 10.5],  # Different spring constants
            damping_coefficients=[0.05, 0.1, 0.15],  # Different damping strengths
            initial_amplitudes=[0.1, 0.15],  # Different initial displacements
            noise_levels=[0.001, 0.002]  # Different noise levels
        )
    
    plt.show()