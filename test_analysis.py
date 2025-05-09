import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff, fs, order=4):
    """Apply a low-pass Butterworth filter to the data."""
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    # Get filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Apply filter
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def process_data(t, x, fs=100):
    """Clean and process raw data for analysis."""
    # Apply smoothing to reduce noise
    # First, apply a low-pass filter to remove high-frequency noise
    x_filtered = butter_lowpass_filter(x, cutoff=5.0, fs=fs, order=3)
    
    # Then apply Savitzky-Golay filter for additional smoothing while preserving peaks
    x_smoothed = savgol_filter(x_filtered, window_length=21, polyorder=3)
    
    return x_smoothed

def quadratic_damping_envelope(t, A0, beta):
    """
    Envelope for quadratic damping: A(t) = A0 / (1 + beta*A0*t)
    """
    return A0 / (1 + beta * A0 * t)

def linear_damping_envelope(t, A0, tau):
    """
    Envelope for linear damping: A(t) = A0 * exp(-t/tau)
    """
    return A0 * np.exp(-t/tau)

def test_analysis(filename=None, mass=None, save_plots=True, show_plots=False):
    """
    Test function to analyze oscillation data without requiring user interaction.
    
    Parameters:
    filename: path to data file (if None, a default will be used)
    mass: mass in kg (if None, will be extracted from filename or default used)
    save_plots: whether to save plots to results directory
    show_plots: whether to display plots interactively
    """
    # Set default file if none provided
    if filename is None:
        filename = "data_simulated/molla_dura_discoforato_long_14.92000_0.2016_2.csv"
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found.")
        return
    
    print(f"Analyzing: {filename}")
    
    # Load data
    df = pd.read_csv(filename, sep=';')
    t = df['time'].values
    x = df['position'].values
    
    # Extract or set mass
    if mass is None:
        # Try to extract from filename
        parts = os.path.basename(filename).split('_')
        mass = 0.2016  # Default
        for part in parts:
            if part.replace('.', '').isdigit():
                value = float(part)
                if 0.05 < value < 0.5:  # Reasonable mass range in kg
                    mass = value
                    break
    
    print(f"Using mass: {mass} kg")
    print(f"Data range: {t[0]:.2f}s to {t[-1]:.2f}s ({len(t)} points)")
    
    # Process and smooth data
    fs = 1.0 / np.mean(np.diff(t))  # Sampling frequency
    x_smoothed = process_data(t, x, fs)
    
    # Create results directory if it doesn't exist
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/analysis', exist_ok=True)
    
    # Plot raw and processed data
    plt.figure(figsize=(12, 6))
    plt.plot(t, x, 'b-', alpha=0.3, label='Raw Position Data')
    plt.plot(t, x_smoothed, 'r-', label='Processed Position Data')
    plt.title('Raw vs. Processed Position Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.grid(True, alpha=0.3)
    plt.legend()
    if save_plots:
        plt.savefig('results/plots/test_preprocessing.png')
    if show_plots:
        plt.show()
    plt.close()
    
    # Determine equilibrium position
    x_equilibrium = np.mean(x_smoothed)
    x_centered = x_smoothed - x_equilibrium
    
    # Find peaks and valleys
    min_height = np.max(np.abs(x_centered)) * 0.05  # 5% of max amplitude
    min_distance = fs * 0.3  # Minimum 0.3 seconds between peaks
    
    peaks, _ = find_peaks(x_centered, height=min_height, distance=min_distance)
    valleys, _ = find_peaks(-x_centered, height=min_height, distance=min_distance)
    
    # Extract extrema data
    peaks_t = t[peaks]
    peaks_x = x_centered[peaks]
    valleys_t = t[valleys]
    valleys_x = x_centered[valleys]
    
    # Combine and sort extrema
    all_extrema_t = np.concatenate((peaks_t, valleys_t))
    all_extrema_amp = np.concatenate((np.abs(peaks_x), np.abs(valleys_x)))
    
    sort_idx = np.argsort(all_extrema_t)
    all_extrema_t = all_extrema_t[sort_idx]
    all_extrema_amp = all_extrema_amp[sort_idx]
    
    # Define minimum amplitude threshold - extrema below this are likely noise
    amp_threshold = np.max(all_extrema_amp) * 0.03  # 3% of max amplitude
    valid_idx = all_extrema_amp > amp_threshold
    
    # Filter out low-amplitude extrema (likely noise)
    all_extrema_t = all_extrema_t[valid_idx]
    all_extrema_amp = all_extrema_amp[valid_idx]
    
    # Print extrema statistics
    print(f"Found {len(peaks_t)} peaks and {len(valleys_t)} valleys")
    print(f"After filtering: {len(all_extrema_t)} valid extrema")
    
    if len(all_extrema_t) < 5:
        print("Warning: Too few extrema found for reliable analysis.")
        return
    
    # Fit quadratic damping envelope
    try:
        # Initial parameter estimates
        A0_guess = np.max(all_extrema_amp)
        
        # For quadratic damping model
        # A(t) = A0/(1 + beta*A0*t)
        # At t = t_half, A = A0/2, solving:
        # beta = 1/(A0*t_half)
        
        # Estimate t_half from data
        half_amp = A0_guess / 2
        t_half_idx = np.argmin(np.abs(all_extrema_amp - half_amp))
        if t_half_idx > 0:
            t_half = all_extrema_t[t_half_idx] - all_extrema_t[0]
            beta_guess = 1 / (A0_guess * t_half)
        else:
            beta_guess = 0.1  # fallback
        
        # Time relative to first extrema
        t_rel = all_extrema_t - all_extrema_t[0]
        
        # Fit quadratic damping model
        params_quad, pcov_quad = curve_fit(
            quadratic_damping_envelope, 
            t_rel, all_extrema_amp,
            p0=[A0_guess, beta_guess],
            bounds=([0, 0], [np.inf, np.inf]),
            maxfev=20000
        )
        
        A0_quad, beta_quad = params_quad
        
        # Calculate standard errors
        perr_quad = np.sqrt(np.diag(pcov_quad))
        
        print("\nQuadratic Damping Model Results:")
        print(f"Initial amplitude (A₀): {A0_quad:.6f} ± {perr_quad[0]:.6f}")
        print(f"Damping parameter (β): {beta_quad:.6f} ± {perr_quad[1]:.6f}")
        
        # Calculate half-life
        half_life = 1 / (beta_quad * A0_quad)
        print(f"Half-life: {half_life:.4f} seconds")
        
        # Fit linear damping model for comparison
        params_lin, pcov_lin = curve_fit(
            linear_damping_envelope,
            t_rel, all_extrema_amp,
            p0=[A0_guess, half_life],
            bounds=([0, 0], [np.inf, np.inf]),
            maxfev=20000
        )
        
        A0_lin, tau = params_lin
        perr_lin = np.sqrt(np.diag(pcov_lin))
        
        print("\nLinear Damping Model (for comparison):")
        print(f"Initial amplitude (A₀): {A0_lin:.6f} ± {perr_lin[0]:.6f}")
        print(f"Time constant (τ): {tau:.6f} ± {perr_lin[1]:.6f}")
        
        # Calculate goodness of fit metrics
        # Quadratic model predictions
        pred_quad = quadratic_damping_envelope(t_rel, A0_quad, beta_quad)
        residuals_quad = all_extrema_amp - pred_quad
        ss_residual_quad = np.sum(residuals_quad**2)
        
        # Linear model predictions
        pred_lin = linear_damping_envelope(t_rel, A0_lin, tau)
        residuals_lin = all_extrema_amp - pred_lin
        ss_residual_lin = np.sum(residuals_lin**2)
        
        # Common statistics
        ss_total = np.sum((all_extrema_amp - np.mean(all_extrema_amp))**2)
        r_squared_quad = 1 - (ss_residual_quad / ss_total)
        r_squared_lin = 1 - (ss_residual_lin / ss_total)
        
        # Calculate AIC (Akaike Information Criterion) for model comparison
        n = len(all_extrema_amp)
        k_quad = len(params_quad)
        k_lin = len(params_lin)
        
        # AIC = 2k + n*ln(RSS/n)
        aic_quad = 2*k_quad + n*np.log(ss_residual_quad/n)
        aic_lin = 2*k_lin + n*np.log(ss_residual_lin/n)
        
        print("\nModel Comparison:")
        print(f"R² (Quadratic Damping): {r_squared_quad:.6f}")
        print(f"R² (Linear Damping): {r_squared_lin:.6f}")
        print(f"AIC (Quadratic Damping): {aic_quad:.2f}")
        print(f"AIC (Linear Damping): {aic_lin:.2f}")
        
        # Physical parameters
        # Estimate angular frequency from peak-to-peak times
        if len(peaks_t) > 1:
            periods = np.diff(peaks_t)
            omega = 2 * np.pi / np.mean(periods)
        else:
            # Rough estimate from data
            omega = 2 * np.pi  # Default to 1 Hz
        
        # For quadratic damping, β = (8*b*ω)/(3*π*m)
        # Therefore b = (3*π*m*β)/(8*ω)
        b = (3 * np.pi * mass * beta_quad) / (8 * omega)
        
        # Spring constant k = m*ω²
        k = mass * omega**2
        
        print("\nPhysical Parameters:")
        print(f"Estimated angular frequency (ω): {omega:.4f} rad/s")
        print(f"Estimated frequency (f): {omega/(2*np.pi):.4f} Hz")
        print(f"Estimated period (T): {2*np.pi/omega:.4f} s")
        print(f"Spring constant (k): {k:.4f} N/m")
        print(f"Quadratic damping coefficient (b): {b:.6f} kg/m")
        
        # Plot results
        plt.figure(figsize=(12, 8))
        
        # Plot data points
        plt.scatter(all_extrema_t, all_extrema_amp, c='k', s=30, alpha=0.7, label='Amplitude Peaks')
        
        # Plot model fits
        t_model = np.linspace(all_extrema_t[0], all_extrema_t[-1], 1000)
        t_model_rel = t_model - all_extrema_t[0]
        
        amp_quad = quadratic_damping_envelope(t_model_rel, A0_quad, beta_quad)
        amp_lin = linear_damping_envelope(t_model_rel, A0_lin, tau)
        
        plt.plot(t_model, amp_quad, 'r-', linewidth=2, 
                 label=f'Quadratic Damping Model (R² = {r_squared_quad:.4f})')
        plt.plot(t_model, amp_lin, 'b--', linewidth=2, 
                 label=f'Linear Damping Model (R² = {r_squared_lin:.4f})')
        
        # Mark half-life
        half_amp = A0_quad / 2
        t_half_mark = all_extrema_t[0] + half_life
        plt.axvline(x=t_half_mark, color='g', linestyle=':', label=f'Half-life = {half_life:.2f}s')
        plt.axhline(y=half_amp, color='g', linestyle=':')
        
        plt.title('Quadratic Damping Amplitude Decay Analysis')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        plt.legend()
        if save_plots:
            plt.savefig('results/plots/test_amplitude_decay.png')
        if show_plots:
            plt.show()
        plt.close()
        
        # Plot the full oscillation with envelope
        plt.figure(figsize=(15, 8))
        plt.plot(t, x_centered, 'b-', alpha=0.5, label='Position (centered)')
        
        # Plot quadratic damping envelope
        t_env = np.linspace(all_extrema_t[0], t[-1], 1000)
        t_env_rel = t_env - all_extrema_t[0]
        env_quad = quadratic_damping_envelope(t_env_rel, A0_quad, beta_quad)
        
        plt.plot(t_env, env_quad, 'r-', linewidth=2, label='Quadratic Damping Envelope')
        plt.plot(t_env, -env_quad, 'r-', linewidth=2)
        
        # Mark peaks and valleys
        plt.plot(peaks_t, peaks_x, 'ro', markersize=3)
        plt.plot(valleys_t, valleys_x, 'go', markersize=3)
        
        plt.title('Oscillation with Quadratic Damping Envelope')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (centered)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        if save_plots:
            plt.savefig('results/plots/test_full_oscillation.png')
        if show_plots:
            plt.show()
        plt.close()
        
        # Plot residuals
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(all_extrema_t, residuals_quad, c='r', s=20, alpha=0.7)
        plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        plt.title('Quadratic Damping Residuals')
        plt.xlabel('Time (s)')
        plt.ylabel('Residual')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.scatter(all_extrema_t, residuals_lin, c='b', s=20, alpha=0.7)
        plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        plt.title('Linear Damping Residuals')
        plt.xlabel('Time (s)')
        plt.ylabel('Residual')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('results/plots/test_residuals.png')
        if show_plots:
            plt.show()
        plt.close()
        
        # Save results to a file
        results_file = 'results/analysis/test_results.txt'
        with open(results_file, 'w') as f:
            f.write(f"Analysis of {filename}\n")
            f.write(f"Mass: {mass} kg\n\n")
            
            f.write("Quadratic Damping Model Results:\n")
            f.write(f"Initial amplitude (A₀): {A0_quad:.6f} ± {perr_quad[0]:.6f}\n")
            f.write(f"Damping parameter (β): {beta_quad:.6f} ± {perr_quad[1]:.6f}\n")
            f.write(f"Half-life: {half_life:.4f} seconds\n\n")
            
            f.write("Linear Damping Model Results:\n")
            f.write(f"Initial amplitude (A₀): {A0_lin:.6f} ± {perr_lin[0]:.6f}\n")
            f.write(f"Time constant (τ): {tau:.6f} ± {perr_lin[1]:.6f}\n\n")
            
            f.write("Model Comparison:\n")
            f.write(f"R² (Quadratic Damping): {r_squared_quad:.6f}\n")
            f.write(f"R² (Linear Damping): {r_squared_lin:.6f}\n")
            f.write(f"AIC (Quadratic Damping): {aic_quad:.2f}\n")
            f.write(f"AIC (Linear Damping): {aic_lin:.2f}\n\n")
            
            f.write("Physical Parameters:\n")
            f.write(f"Angular frequency (ω): {omega:.4f} rad/s\n")
            f.write(f"Frequency (f): {omega/(2*np.pi):.4f} Hz\n")
            f.write(f"Period (T): {2*np.pi/omega:.4f} s\n")
            f.write(f"Spring constant (k): {k:.4f} N/m\n")
            f.write(f"Quadratic damping coefficient (b): {b:.6f} kg/m\n")
        
        print(f"\nResults saved to {results_file}")
        print(f"Plots saved to results/plots/ directory")
        
        return {
            "A0_quad": A0_quad,
            "beta_quad": beta_quad,
            "half_life": half_life,
            "R2_quad": r_squared_quad,
            "R2_lin": r_squared_lin,
            "omega": omega,
            "k": k,
            "b": b
        }
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return None

if __name__ == "__main__":
    print("Running non-interactive test analysis...")
    
    # Test with a standard file
    test_analysis(filename="data_simulated/molla_dura_discoforato_long_14.92000_0.2016_2.csv", 
                  mass=0.2016, 
                  save_plots=True, 
                  show_plots=False)
    
    # You can uncomment this to test with a different file
    # test_analysis(filename="data_simulated/molla_dura_discoforato_14.92000_0.2016_1.csv",
    #               mass=0.2016, 
    #               save_plots=True, 
    #               show_plots=False)
    
    print("\nTest analysis complete.")