import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

def simulate_quadratic_damping(
    mass,                  # Mass in kg
    k,                     # Spring constant in N/m
    b,                     # Quadratic damping coefficient in kg/m
    x0,                    # Initial position in m
    v0=0,                  # Initial velocity in m/s
    t_max=15.0,            # Maximum simulation time in seconds
    dt=0.01,               # Time step in seconds
    noise_level=0.0005     # Measurement noise level
):
    """
    Simulate oscillation with quadratic damping (proportional to v^2).
    Represents a mass connected to a spring and a perforated disk (disco forato).
    
    Equation of motion: m*x'' + b*x'*|x'| + k*x = 0
    """
    # Natural frequency
    omega_n = np.sqrt(k / mass)
    
    # Define the ODE system for quadratic damping
    def system_ode(t, state):
        x, v = state
        # Force proportional to v^2, with sign of v
        damping_force = -b * v * np.abs(v)
        # Spring force
        spring_force = -k * x
        # Newton's second law
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
    
    # Add realistic measurement noise
    position_noisy = position + np.random.normal(0, noise_level, position.shape)
    velocity_noisy = velocity + np.random.normal(0, noise_level*10, velocity.shape)
    acceleration_noisy = acceleration + np.random.normal(0, noise_level*100, acceleration.shape)
    
    # Calculate theoretical damping parameter beta
    beta = (8 * b * omega_n) / (3 * np.pi * mass)
    
    # Calculate half-life for quadratic damping
    half_life = 1 / (beta * abs(x0))
    
    return {
        'time': time,
        'position': position_noisy,
        'velocity': velocity_noisy,
        'acceleration': acceleration_noisy,
        'omega_n': omega_n,
        'beta': beta,
        'half_life': half_life
    }

def generate_and_save_file(params, file_number):
    """Generate and save data for a single experiment configuration."""
    # Simulation
    result = simulate_quadratic_damping(
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
    filename = f"molla_dura_discoforato_{params['k']:.5f}_{params['mass']:.4f}_{file_number}.csv"
    filepath = os.path.join('data_simulated', filename)
    
    # Save to CSV
    data.to_csv(filepath, sep=';', index=False)
    
    # Print information about the file
    print(f"Generated: {filename}")
    print(f"  Mass: {params['mass']} kg")
    print(f"  Spring constant: {params['k']} N/m")
    print(f"  Quadratic damping (b): {params['b']} kg/m")
    print(f"  Natural frequency: {result['omega_n']/(2*np.pi):.4f} Hz")
    print(f"  Damping parameter (β): {result['beta']:.6f}")
    print(f"  Half-life: {result['half_life']:.2f} seconds")
    print(f"  Data points: {len(data)}")
    
    # Optionally generate a diagnostic plot
    if params.get('plot', False):
        plt.figure(figsize=(10, 6))
        plt.plot(result['time'], result['position'], 'b-', label='Position')
        
        # Add theoretical envelope if initial amplitude is significant
        if abs(params['x0']) > 0.01:
            envelope_times = np.linspace(0, params['t_max'], 500)
            A0 = abs(params['x0'])
            envelope = A0 / (1 + result['beta'] * A0 * envelope_times)
            plt.plot(envelope_times, envelope, 'r--', label='Theoretical Envelope')
            plt.plot(envelope_times, -envelope, 'r--')
        
        plt.title(f'Disco Forato Simulation (b={params["b"]} kg/m)')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plot_filename = os.path.splitext(filename)[0] + '_plot.png'
        plt.savefig(os.path.join('results/plots', plot_filename))
        plt.close()
    
    return filepath

def main():
    """Generate multiple disco forato data files with varying parameters."""
    # Ensure data_simulated directory exists
    os.makedirs('data_simulated', exist_ok=True)
    
    # Base parameters
    base_params = {
        'mass': 0.2016,     # kg
        'k': 14.92,         # N/m
        't_max': 15.0,      # seconds
        'noise_level': 0.0005,
        'plot': True
    }
    
    # List of experimental configurations
    experiments = [
        # Light damping
        {'b': 0.03, 'x0': 0.10, 'description': 'Light damping'},
        
        # Medium damping
        {'b': 0.08, 'x0': 0.12, 'description': 'Medium damping'},
        
        # Heavy damping
        {'b': 0.15, 'x0': 0.15, 'description': 'Heavy damping'},
        
        # Different spring constant
        {'k': 10.5, 'b': 0.05, 'x0': 0.10, 'description': 'Different spring'},
        
        # Different mass
        {'mass': 0.15, 'b': 0.05, 'x0': 0.10, 'description': 'Different mass'}
    ]
    
    # Generate each file
    generated_files = []
    for i, exp in enumerate(experiments, 1):
        # Combine base parameters with experiment-specific ones
        params = base_params.copy()
        params.update(exp)
        
        print(f"\nGenerating dataset {i}/{len(experiments)}: {exp['description']}")
        filepath = generate_and_save_file(params, i)
        generated_files.append(filepath)
    
    print(f"\nGenerated {len(generated_files)} disco forato data files:")
    print("\nGenerated 5 quadratic damping data files:")
    for filepath in generated_files:
        print(f"- {os.path.basename(filepath)}")
    
    print("\nThese files simulate mass-spring systems with quadratic damping (F_d ∝ v²),")
    print("as would be observed with a perforated disk moving through air.")
    print(f"Files are saved in the data_simulated directory and plots in the results/plots directory.")

if __name__ == "__main__":
    main()