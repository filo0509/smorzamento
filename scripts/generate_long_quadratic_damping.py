import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

def simulate_disco_forato_long(
    mass=0.2016,           # Mass in kg
    k=14.92,               # Spring constant in N/m
    b=0.05,                # Quadratic damping coefficient in kg/m
    x0=0.1,                # Initial position in m
    v0=0,                  # Initial velocity in m/s
    t_max=100.0,           # Maximum simulation time in seconds
    dt=0.01,               # Time step in seconds
    noise_level=0.00052,   # Base noise level (5% higher than before)
    noise_bursts=True      # Add occasional noise bursts to simulate disturbances
):
    """
    Generate a long (100 second) simulation of a spring-mass system with
    a perforated disk (disco forato) exhibiting quadratic damping.
    
    The simulation includes realistic noise and occasional disturbances.
    """
    # Natural frequency
    omega_n = np.sqrt(k / mass)
    print(f"Natural frequency: {omega_n:.4f} rad/s = {omega_n/(2*np.pi):.4f} Hz")
    
    # Define the ODE system for quadratic damping
    def system_ode(t, state):
        x, v = state
        # Quadratic damping force: proportional to v^2 with sign matching v
        damping_force = -b * v * np.abs(v)
        # Spring force
        spring_force = -k * x
        # Total acceleration
        a = (spring_force + damping_force) / mass
        return [v, a]
    
    # Time points
    t_span = (0, t_max)
    t_eval = np.arange(0, t_max, dt)
    
    # Initial state [position, velocity]
    initial_state = [x0, v0]
    
    # Solve the ODE
    solution = solve_ivp(
        system_ode,
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
    acceleration = np.zeros_like(position)
    for i in range(len(time)):
        x, v = position[i], velocity[i]
        damping_force = -b * v * np.abs(v)
        spring_force = -k * x
        acceleration[i] = (spring_force + damping_force) / mass
    
    # Add time-dependent noise (increases slightly over time to simulate drift)
    time_factor = 1 + 0.5 * (time / t_max)
    position_noise = np.random.normal(0, noise_level * time_factor, position.shape)
    velocity_noise = np.random.normal(0, noise_level * 10 * time_factor, velocity.shape)
    acceleration_noise = np.random.normal(0, noise_level * 100 * time_factor, acceleration.shape)
    
    # Add occasional noise bursts/disturbances at random times
    if noise_bursts:
        # Generate 3-8 random disturbance points
        n_bursts = np.random.randint(3, 9)
        burst_times = np.random.choice(len(time), n_bursts, replace=False)
        burst_width = int(0.2 / dt)  # 0.2 second burst width
        
        for burst_idx in burst_times:
            # Only affect points if they're far enough from the end
            if burst_idx + burst_width < len(time):
                # Create a burst shape (gaussian-like)
                burst_shape = np.exp(-0.5 * np.linspace(-3, 3, burst_width)**2)
                
                # Apply the burst with random amplitude (stronger for acceleration)
                burst_start = max(0, burst_idx - burst_width//2)
                burst_end = min(len(time), burst_idx + burst_width//2)
                burst_range = burst_end - burst_start
                
                # Scale the burst shape to the available range
                scaled_burst = burst_shape[:burst_range] * np.random.uniform(2, 5) * noise_level
                
                position_noise[burst_start:burst_end] += scaled_burst * 3
                velocity_noise[burst_start:burst_end] += scaled_burst * 30
                acceleration_noise[burst_start:burst_end] += scaled_burst * 300
    
    # Apply noise to signals
    position_noisy = position + position_noise
    velocity_noisy = velocity + velocity_noise
    acceleration_noisy = acceleration + acceleration_noise
    
    # Calculate theoretical damping parameter beta
    beta = (8 * b * omega_n) / (3 * np.pi * mass)
    A0 = np.abs(x0)
    
    # Calculate half-life for quadratic damping
    half_life = 1 / (beta * A0)
    
    # Return data in a dictionary
    return {
        'time': time,
        'position': position_noisy,
        'velocity': velocity_noisy,
        'acceleration': acceleration_noisy,
        'omega_n': omega_n,
        'beta': beta,
        'half_life': half_life,
        'clean_position': position  # For comparison plots
    }

def generate_and_save_file(params, file_number):
    """Generate and save data for a single experiment configuration."""
    print(f"Generating file {file_number} with parameters:")
    for key, value in params.items():
        if key not in ['plot']:
            print(f"  {key}: {value}")
    
    # Run simulation
    result = simulate_disco_forato_long(
        mass=params['mass'],
        k=params['k'],
        b=params['b'],
        x0=params['x0'],
        t_max=params['t_max'],
        noise_level=params['noise_level']
    )
    
    # Create DataFrame
    data = pd.DataFrame({
        'time': result['time'],
        'position': result['position'],
        'velocity': result['velocity'],
        'acceleration': result['acceleration']
    })
    
    # Create filename with parameters encoded
    filename = f"molla_dura_discoforato_long_{params['k']:.5f}_{params['mass']:.4f}_{file_number}.csv"
    filepath = os.path.join('data_simulated', filename)
    
    # Save to CSV
    data.to_csv(filepath, sep=';', index=False)
    print(f"Saved to {filepath} ({len(data)} data points)")
    
    # Generate diagnostic plot if requested
    if params.get('plot', True):
        plt.figure(figsize=(15, 8))
        
        # Plot both noisy and clean signals
        plt.plot(result['time'], result['position'], 'b-', alpha=0.7, label='Noisy Position')
        plt.plot(result['time'], result['clean_position'], 'g-', alpha=0.5, label='Clean Signal')
        
        # Add envelope for theoretical decay
        if abs(params['x0']) > 0.01:
            envelope_times = np.linspace(0, params['t_max'], 1000)
            A0 = abs(params['x0'])
            envelope = A0 / (1 + result['beta'] * A0 * envelope_times)
            plt.plot(envelope_times, envelope, 'r--', linewidth=1.5, label='Theoretical Envelope')
            plt.plot(envelope_times, -envelope, 'r--', linewidth=1.5)
        
        plt.title(f'Disco Forato Simulation (quadratic damping, b={params["b"]} kg/m, 100s duration)')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Show close-up of first few seconds
        plt.figure(figsize=(12, 6))
        early_time = 15  # seconds to show in detail
        early_idx = int(early_time / params.get('dt', 0.01))
        plt.plot(result['time'][:early_idx], result['position'][:early_idx], 'b-', label='Position')
        
        plt.title(f'Disco Forato - First {early_time} seconds (close-up)')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.grid(True)
        plt.legend()
        
        # Show later portion to see noise and long-term behavior
        plt.figure(figsize=(12, 6))
        later_start = 60  # seconds
        later_idx = int(later_start / params.get('dt', 0.01))
        plt.plot(result['time'][later_idx:], result['position'][later_idx:], 'b-', label='Position')
        
        plt.title(f'Disco Forato - Later Portion (t > {later_start}s)')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    return filepath

def main():
    """Generate multiple quadratic damping data files with varying parameters."""
    # Ensure data_simulated directory exists
    os.makedirs('data_simulated', exist_ok=True)
    # Ensure plots directory exists
    os.makedirs('results/plots', exist_ok=True)
    
    # Common parameters for all simulations
    base_params = {
        't_max': 100.0,     # 100 seconds of data
        'noise_level': 0.00052,  # 5% more noise than before
        'dt': 0.01,         # Time step
        'plot': True        # Generate diagnostic plots
    }
    
    # Different experimental configurations
    experiments = [
        # Standard configuration with reference mass
        {
            'mass': 0.2016,     # kg
            'k': 14.92,         # N/m
            'b': 0.05,          # kg/m - medium damping
            'x0': 0.10,         # Initial displacement
            'description': 'Reference configuration'
        },
        
        # Light damping - shows more oscillations
        {
            'mass': 0.2016,
            'k': 14.92,
            'b': 0.025,         # Lighter damping
            'x0': 0.12,
            'description': 'Light damping'
        },
        
        # Heavy damping - shows fewer oscillations
        {
            'mass': 0.2016,
            'k': 14.92,
            'b': 0.12,          # Heavier damping
            'x0': 0.15,
            'description': 'Heavy damping'
        }
    ]
    
    # Generate all files
    generated_files = []
    for i, exp in enumerate(experiments, 1):
        # Combine base parameters with experiment-specific ones
        params = base_params.copy()
        params.update(exp)
        
        print(f"\nExperiment {i}/{len(experiments)}: {exp['description']}")
        filepath = generate_and_save_file(params, i)
        generated_files.append(filepath)
    
    print("\nGenerated files:")
    for filepath in generated_files:
        print(f"- {os.path.basename(filepath)}")
    
    print("\nThese files contain 100 seconds of simulated data with quadratic damping (F_d ∝ v²),")
    print("as would be observed with a perforated disk moving through air.")
    print("The noise level has been increased by 5% compared to previous simulations.")
    print("Files are saved in the data_simulated directory.")

if __name__ == "__main__":
    main()