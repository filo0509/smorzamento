import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import os
import glob

def load_latest_data_file():
    """Find the most recently modified quadratic damping data file, or prompt for selection if not found."""
    data_dir = "data_simulated"
    # Search for files that might contain quadratic damping data
    data_files = glob.glob(os.path.join(data_dir, "*discoforato*.csv"))
    
    if not data_files:
        # If no forato files found, list all available files
        all_files = glob.glob(os.path.join(data_dir, "*.csv"))
        if not all_files:
            data_dir = "data_raw"
            all_files = glob.glob(os.path.join(data_dir, "*.csv"))
            
        print("No quadratic damping files found. Please select from available data files:")
        for i, file in enumerate(all_files):
            print(f"[{i}] {os.path.basename(file)}")
        
        selection = int(input("Enter file number: "))
        if 0 <= selection < len(all_files):
            return all_files[selection]
        else:
            raise ValueError("Invalid selection")
    
    # If forato files found, use the most recent one
    latest_file = max(data_files, key=os.path.getmtime)
    print(f"Using most recent file: {os.path.basename(latest_file)}")
    return latest_file

def extract_mass_from_filename(filename):
    """Try to extract mass value from filename, or prompt user for input if not found."""
    try:
        # Look for patterns like: disco_forato_X.XXXX_Y.YYYY where Y.YYYY might be the mass in kg
        parts = os.path.basename(filename).split('_')
        for part in parts:
            if part.replace('.', '').isdigit():
                value = float(part)
                if 0.05 < value < 1.0:  # Reasonable mass range in kg
                    return value
        
        # If not found, ask user
        print(f"Could not automatically extract mass from {filename}")
        return float(input("Enter mass value in kg: "))
    except Exception as e:
        print(f"Error extracting mass: {e}")
        return float(input("Enter mass value in kg: "))

def quadratic_damping_envelope(t, A0, beta):
    """
    Envelope function for oscillations with damping proportional to v^2
    
    For quadratic damping (F_d = -b*v^2*sign(v)), the amplitude follows:
    A(t) = A0 / (1 + beta*A0*t)
    
    where beta is related to the damping coefficient b by:
    beta = (8*b*omega)/(3*pi*m)
    
    Parameters:
    t: time from start of decay
    A0: initial amplitude
    beta: damping parameter
    
    Returns:
    Amplitude at time t
    """
    return A0 / (1 + beta * A0 * t)

def quadratic_damped_oscillation(t, A0, beta, omega, phi, offset):
    """
    Full model for oscillations with quadratic damping
    
    Parameters:
    t: time array
    A0: initial amplitude
    beta: quadratic damping parameter
    omega: angular frequency
    phi: phase
    offset: equilibrium position
    
    Returns:
    Position at time t
    """
    # Calculate amplitude envelope
    t_rel = t - t[0]  # time relative to start
    amplitude = quadratic_damping_envelope(t_rel, A0, beta)
    
    # Multiply by oscillation
    return amplitude * np.cos(omega * t_rel + phi) + offset

# Main analysis function
def analyze_disco_forato_data(filename=None, mass=None):
    """Analyze the disco forato oscillation data with quadratic damping model."""
    # Load the data
    if filename is None:
        filename = load_latest_data_file()
    
    print(f"Analyzing: {filename}")
    df = pd.read_csv(filename, sep=';')
    t = df['time'].values
    x = df['position'].values
    
    if 'velocity' in df.columns:
        v = df['velocity'].values
    else:
        # Calculate velocity using central differences if not provided
        dt = np.diff(t)
        v = np.zeros_like(t)
        v[1:-1] = (x[2:] - x[:-2]) / (t[2:] - t[:-2])
        v[0] = (x[1] - x[0]) / (t[1] - t[0])
        v[-1] = (x[-1] - x[-2]) / (t[-1] - t[-2])
    
    # Extract mass from filename or ask user
    if mass is None:
        mass = extract_mass_from_filename(filename)
    
    print(f"Using mass: {mass} kg")
    
    # Plot raw data
    plt.figure(figsize=(12, 6))
    plt.plot(t, x, 'b-', label='Position Data')
    plt.title('Raw Oscillation Data (Quadratic Damping)')
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig('results/plots/quadratic_damping_raw_data.png')
    plt.show()
    
    # Determine equilibrium position (mean of the signal)
    x_equilibrium = np.mean(x)
    x_centered = x - x_equilibrium
    
    # Find peaks and valleys for the centered data
    peaks, _ = find_peaks(x_centered, height=0.001, distance=10)
    valleys, _ = find_peaks(-x_centered, height=0.001, distance=10)
    
    # Extract peak and valley data
    peaks_t = t[peaks]
    peaks_x = x_centered[peaks]
    valleys_t = t[valleys]
    valleys_x = x_centered[valleys]
    
    # Combine all extrema for a better fit
    all_extrema_t = np.concatenate((peaks_t, valleys_t))
    all_extrema_amp = np.concatenate((np.abs(peaks_x), np.abs(valleys_x)))
    
    # Sort by time
    sort_idx = np.argsort(all_extrema_t)
    all_extrema_t = all_extrema_t[sort_idx]
    all_extrema_amp = all_extrema_amp[sort_idx]
    
    # Fit the decay envelope using quadratic damping model
    try:
        # Initial guess for parameters
        # For quadratic damping, amplitude follows A(t) = A0/(1 + beta*A0*t)
        A0_guess = np.max(all_extrema_amp)
        
        # Estimate beta from a few points
        # If A(t) = A0/(1 + beta*A0*t), then beta = (1/A0)*(1/t)*((A0/A) - 1)
        t_half_idx = len(all_extrema_amp) // 2
        if t_half_idx > 0:
            A_half = all_extrema_amp[t_half_idx]
            t_half = all_extrema_t[t_half_idx] - all_extrema_t[0]
            beta_guess = (1/A0_guess) * (1/t_half) * ((A0_guess/A_half) - 1)
        else:
            beta_guess = 0.1  # fallback guess
        
        # Perform the curve fit
        params, params_covariance = curve_fit(
            lambda t, A0, beta: quadratic_damping_envelope(t - all_extrema_t[0], A0, beta),
            all_extrema_t, all_extrema_amp, 
            p0=[A0_guess, beta_guess],
            bounds=([0, 0], [np.inf, np.inf]),
            maxfev=10000
        )
        
        A0, beta = params
        print(f"Quadratic Damping Model Parameters:")
        print(f"Initial amplitude (A₀): {A0:.6f}")
        print(f"Damping parameter (β): {beta:.6f}")
        
        # Calculate time for amplitude to decrease to half its initial value
        # For quadratic damping with A(t) = A0/(1 + beta*A0*t)
        # Solving for t when A = A0/2: t = 1/(beta*A0)
        half_life = 1 / (beta * A0)
        print(f"Half-life of oscillation: {half_life:.6f} seconds")
        print(f"This is the time it takes for the amplitude to decrease to 50% of its initial value.")
        
        # Calculate damping coefficient b from beta
        # Estimate omega (angular frequency) from peak-to-peak times
        if len(peaks_t) > 1:
            period = np.mean(np.diff(peaks_t))
            omega = 2 * np.pi / period
        else:
            # If we don't have multiple peaks, make an educated guess
            omega = 10.0  # typical angular frequency in rad/s
        
        # For quadratic damping, β = (8*b*ω)/(3*π*m)
        # Solving for b: b = (3*π*m*β)/(8*ω)
        b = (3 * np.pi * mass * beta) / (8 * omega)
        print(f"Estimated angular frequency (ω): {omega:.6f} rad/s")
        print(f"Quadratic damping coefficient (b): {b:.6f} kg/m")
        
        # Calculate goodness of fit metrics
        predicted = quadratic_damping_envelope(all_extrema_t - all_extrema_t[0], A0, beta)
        residuals = all_extrema_amp - predicted
        
        # Degrees of freedom
        n_params = 2  # A0 and beta
        n_data = len(all_extrema_amp)
        dof = n_data - n_params
        
        # Chi-squared using standard deviation of residuals as uncertainty
        sigma = np.std(residuals)
        chi_squared = np.sum((residuals / sigma)**2)
        reduced_chi_squared = chi_squared / dof
        
        # R-squared
        ss_total = np.sum((all_extrema_amp - np.mean(all_extrema_amp))**2)
        ss_residual = np.sum(residuals**2)
        r_squared = 1 - (ss_residual / ss_total)
        
        print("\nGoodness of Fit Metrics:")
        print(f"Chi-squared: {chi_squared:.4f}")
        print(f"Reduced chi-squared: {reduced_chi_squared:.4f}")
        print(f"R-squared: {r_squared:.6f}")
        
        # Interpret reduced chi-squared
        if 0.8 <= reduced_chi_squared <= 1.2:
            interpretation = "excellent fit (χ²/dof ≈ 1)"
        elif 0.5 <= reduced_chi_squared <= 1.5:
            interpretation = "good fit (0.5 < χ²/dof < 1.5)"
        elif reduced_chi_squared < 0.5:
            interpretation = "potentially overfitting (χ²/dof < 0.5)"
        else:
            interpretation = "poor fit (χ²/dof > 1.5) - model may not fully describe the data"
        
        print(f"Interpretation: {interpretation}")
        
        # Plot the data with the fitted envelope
        plt.figure(figsize=(12, 8))
        plt.plot(t, x_centered, 'b-', alpha=0.5, label='Centered Position Data')
        plt.plot(peaks_t, peaks_x, 'ro', markersize=4, label='Peaks')
        plt.plot(valleys_t, valleys_x, 'go', markersize=4, label='Valleys')
        
        # Generate the decay envelope
        t_fit = np.linspace(all_extrema_t[0], all_extrema_t[-1], 1000)
        upper_envelope = quadratic_damping_envelope(t_fit - all_extrema_t[0], A0, beta)
        lower_envelope = -upper_envelope
        
        plt.plot(t_fit, upper_envelope, 'r-', linewidth=2, label=f'Quadratic Damping Envelope')
        plt.plot(t_fit, lower_envelope, 'r-', linewidth=2)
        
        # Mark the half-life on the plot
        t_at_half = all_extrema_t[0] + half_life
        amp_at_half = A0 / 2
        plt.axvline(x=t_at_half, color='k', linestyle='--', label=f'Half-life = {half_life:.4f}s')
        plt.axhline(y=amp_at_half, color='m', linestyle=':', label='A₀/2 threshold')
        
        plt.title('Damped Oscillation Analysis (Quadratic Damping)')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (centered)')
        plt.grid(True)
        plt.legend()
        plt.savefig('results/plots/quadratic_damping_analysis.png')
        plt.show()
        
        # Plot residuals
        plt.figure(figsize=(12, 4))
        plt.scatter(all_extrema_t, residuals, color='purple', s=20, alpha=0.7)
        plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        plt.title('Residuals of Quadratic Damping Fit')
        plt.xlabel('Time (s)')
        plt.ylabel('Residual')
        plt.grid(True, alpha=0.3)
        plt.savefig('results/plots/quadratic_damping_residuals.png')
        plt.show()
        
        # Plot observed vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(predicted, all_extrema_amp, alpha=0.7)
        
        # Perfect fit line
        min_val = min(min(predicted), min(all_extrema_amp))
        max_val = max(max(predicted), max(all_extrema_amp))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Fit')
        
        plt.title('Observed vs Predicted Amplitudes')
        plt.xlabel('Predicted Amplitude')
        plt.ylabel('Observed Amplitude')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/plots/quadratic_damping_observed_vs_predicted.png')
        plt.show()
        
        # Now fit the full oscillation model with quadratic damping
        try:
            # Estimate initial parameters
            if len(peaks_t) > 1:
                T = np.mean(np.diff(peaks_t))
                omega_guess = 2 * np.pi / T
            else:
                omega_guess = 10.0  # fallback
            
            initial_params = [
                A0,             # initial amplitude (from envelope fit)
                beta,           # damping parameter (from envelope fit)
                omega_guess,    # angular frequency
                0,              # phase
                x_equilibrium   # offset
            ]
            
            # Use only a portion of the data for fitting to improve performance
            fit_length = min(2000, len(t))
            
            # Perform the curve fit
            params_full, _ = curve_fit(
                quadratic_damped_oscillation,
                t[:fit_length],
                x[:fit_length],
                p0=initial_params,
                bounds=([0, 0, 0, -np.pi, -np.inf], 
                        [np.inf, np.inf, np.inf, np.pi, np.inf]),
                maxfev=10000
            )
            
            A0_full, beta_full, omega_full, phi_full, offset_full = params_full
            
            print("\nFull Model Parameters:")
            print(f"Amplitude (A₀): {A0_full:.6f}")
            print(f"Damping parameter (β): {beta_full:.6f}")
            print(f"Angular frequency (ω): {omega_full:.6f} rad/s")
            print(f"Frequency (f): {omega_full/(2*np.pi):.6f} Hz")
            print(f"Period (T): {2*np.pi/omega_full:.6f} seconds")
            print(f"Phase (φ): {phi_full:.6f} radians")
            print(f"Offset: {offset_full:.6f}")
            
            # Calculate spring constant
            k = mass * omega_full**2
            print(f"Spring constant (k): {k:.6f} N/m")
            
            # Calculate quadratic damping coefficient from full fit
            b_full = (3 * np.pi * mass * beta_full) / (8 * omega_full)
            print(f"Quadratic damping coefficient (b): {b_full:.6f} kg/m")
            
            # Plot the full fitted model
            plt.figure(figsize=(12, 8))
            plt.plot(t, x, 'b-', alpha=0.5, label='Position Data')
            
            t_fit = np.linspace(t[0], t[fit_length-1], 1000)
            x_fit = quadratic_damped_oscillation(t_fit, A0_full, beta_full, omega_full, phi_full, offset_full)
            plt.plot(t_fit, x_fit, 'r-', linewidth=2, label='Fitted Model (Quadratic Damping)')
            
            plt.title('Disco Forato - Quadratic Damping Model Fit')
            plt.title('Damped Oscillation - Full Model Fit')
            plt.xlabel('Time (s)')
            plt.ylabel('Position')
            plt.legend()
            plt.grid(True)
            plt.savefig('results/plots/quadratic_damping_full_model_fit.png')
            plt.show()
            
            # Compare with linear damping
            print("\nComparison with Linear Damping Model:")
            print("In quadratic damping (F_d = -b*v^2*sign(v)), amplitude follows A(t) = A0/(1 + beta*A0*t)")
            print("In linear damping (F_d = -c*v), amplitude follows A(t) = A0*exp(-t/tau)")
            print("The quadratic model better captures air resistance at higher velocities.")
            
            # Analyze the velocity-squared relationship
            plt.figure(figsize=(10, 6))
            # Calculate local energy dissipation rate (proportional to force × velocity)
            # For quadratic damping, we expect energy loss proportional to v^3
            v_abs = np.abs(v)
            v_squared = v_abs**2
            v_cubed = v_abs**3
            
            # Plot velocity vs acceleration (should show quadratic relationship)
            if 'acceleration' in df.columns:
                a = df['acceleration'].values
                a_abs = np.abs(a)
                
                # Filter out noise and keep only data points where velocity is significant
                mask = v_abs > np.percentile(v_abs, 50)
                
                # Calculate acceleration magnitude due to damping
                # Total a = -kx/m - b*v^2*sign(v)/m
                a_spring = -k * x_centered / mass  # spring acceleration
                a_damping = a - a_spring          # damping acceleration
                
                plt.scatter(v_abs[mask], np.abs(a_damping[mask]), s=2, alpha=0.4, label='|a_damping| vs |v|')
                
                # Fit a quadratic relationship: |a_damping| = B*v^2
                try:
                    from sklearn.linear_model import LinearRegression
                    model = LinearRegression(fit_intercept=False)
                    X = v_squared[mask].reshape(-1, 1)
                    y = np.abs(a_damping[mask])
                    model.fit(X, y)
                    b_from_accel = model.coef_[0] * mass
                    
                    v_fit = np.linspace(0, np.max(v_abs[mask]), 100)
                    a_fit = model.coef_[0] * v_fit**2
                    
                    plt.plot(v_fit, a_fit, 'r-', linewidth=2, 
                             label=f'Fitted a = ({model.coef_[0]:.4f})*v^2')
                    
                    print(f"Damping coefficient from a-v relationship: {b_from_accel:.6f} kg/m")
                    
                    # Calculate R^2 for this fit
                    y_pred = model.predict(X)
                    ss_total = np.sum((y - np.mean(y))**2)
                    ss_residual = np.sum((y - y_pred)**2)
                    r_squared_av = 1 - (ss_residual / ss_total)
                    print(f"R-squared for quadratic damping (a vs v^2): {r_squared_av:.6f}")
                    
                    plt.title(f'Damping Acceleration vs Velocity (R² = {r_squared_av:.4f})')
                    plt.xlabel('Velocity Magnitude |v|')
                    plt.ylabel('Damping Acceleration Magnitude |a_damping|')
                    plt.grid(True)
                    plt.legend()
                    plt.savefig('results/plots/quadratic_damping_a_vs_v_squared.png')
                    plt.show()
                    
                except Exception as e:
                    print(f"Could not fit a-v relationship: {e}")
            
            # Create a direct comparison of linear vs quadratic damping models
            try:
                # Fit with linear damping model for comparison
                def linear_damping_envelope(t, A0, tau):
                    return A0 * np.exp(-t/tau)
                
                params_linear, _ = curve_fit(
                    lambda t, A0, tau: linear_damping_envelope(t - all_extrema_t[0], A0, tau),
                    all_extrema_t, all_extrema_amp, 
                    p0=[A0, 1.0],
                    bounds=([0, 0], [np.inf, np.inf]),
                    maxfev=10000
                )
                
                A0_linear, tau = params_linear
                
                # Calculate R-squared for linear model
                predicted_linear = linear_damping_envelope(all_extrema_t - all_extrema_t[0], A0_linear, tau)
                residuals_linear = all_extrema_amp - predicted_linear
                ss_residual_linear = np.sum(residuals_linear**2)
                r_squared_linear = 1 - (ss_residual_linear / ss_total)
                
                print("\nModel Comparison:")
                print(f"R-squared (Quadratic Damping): {r_squared:.6f}")
                print(f"R-squared (Linear Damping): {r_squared_linear:.6f}")
                
                # Plot comparison
                plt.figure(figsize=(12, 8))
                plt.scatter(all_extrema_t, all_extrema_amp, c='k', s=20, alpha=0.7, label='Measured Amplitudes')
                
                # Generate model predictions
                t_comp = np.linspace(all_extrema_t[0], all_extrema_t[-1], 1000)
                amp_quadratic = quadratic_damping_envelope(t_comp - all_extrema_t[0], A0, beta)
                amp_linear = linear_damping_envelope(t_comp - all_extrema_t[0], A0_linear, tau)
                
                plt.plot(t_comp, amp_quadratic, 'r-', linewidth=2, 
                         label=f'Quadratic Damping: A(t) = A₀/(1 + βA₀t), R² = {r_squared:.4f}')
                plt.plot(t_comp, amp_linear, 'b-', linewidth=2, 
                         label=f'Linear Damping: A(t) = A₀e^(-t/τ), R² = {r_squared_linear:.4f}')
                
                plt.title('Damping Model Comparison')
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude')
                plt.grid(True)
                plt.legend()
                plt.savefig('results/plots/quadratic_damping_model_comparison.png')
                plt.show()
                
                # Create residual comparison
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.scatter(all_extrema_t, residuals, c='r', s=20, alpha=0.7)
                plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
                plt.title('Quadratic Damping Residuals')
                plt.xlabel('Time (s)')
                plt.ylabel('Residual')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(1, 2, 2)
                plt.scatter(all_extrema_t, residuals_linear, c='b', s=20, alpha=0.7)
                plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
                plt.title('Linear Damping Residuals')
                plt.xlabel('Time (s)')
                plt.ylabel('Residual')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig('results/plots/quadratic_damping_residuals_comparison.png')
                plt.show()
                
            except Exception as e:
                print(f"Could not perform model comparison: {e}")
        
        except Exception as e:
            print(f"Could not fit full oscillation model: {e}")
    
    except Exception as e:
        print(f"Error fitting damping envelope: {e}")

# Execute if run directly
if __name__ == "__main__":
    try:
        analyze_disco_forato_data()
    except Exception as e:
        print(f"Analysis failed: {e}")