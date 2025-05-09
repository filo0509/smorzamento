import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
import os
import glob
from scipy.signal import butter, filtfilt

def load_disco_forato_data():
    """Find and load a quadratic damping data file."""
    data_dir = "data_simulated"
    # Search for long quadratic damping files
    data_files = glob.glob(os.path.join(data_dir, "*discoforato*long*.csv"))
    
    if not data_files:
        # Search for any quadratic damping files
        data_files = glob.glob(os.path.join(data_dir, "*discoforato*.csv"))
    
    if not data_files:
        # If no quadratic damping files found, list all available files
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
    
    # If multiple files, let user choose
    if len(data_files) > 1:
        print("Multiple disco forato files found. Please select one:")
        for i, file in enumerate(data_files):
            print(f"[{i}] {os.path.basename(file)}")
        
        selection = int(input("Enter file number (or -1 for most recent): "))
        if selection == -1:
            return max(data_files, key=os.path.getmtime)
        elif 0 <= selection < len(data_files):
            return data_files[selection]
        else:
            raise ValueError("Invalid selection")
    
    # If only one file, use it
    return data_files[0]

def extract_mass_from_filename(filename):
    """Extract mass value from filename or prompt user."""
    try:
        # Extract mass from filename format like "*_X.XXXXX_Y.YYYY_*.csv"
        parts = os.path.basename(filename).split('_')
        for part in parts:
            if part.replace('.', '').isdigit():
                value = float(part)
                if 0.05 < value < 0.5:  # Reasonable mass range in kg
                    return value
        
        # If not found, ask user
        print(f"Could not extract mass from {filename}")
        return float(input("Enter mass value in kg: "))
    except Exception as e:
        print(f"Error extracting mass: {e}")
        return float(input("Enter mass value in kg: "))

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
    
    # Calculate velocity using central differences on filtered data
    dt = np.diff(t)
    v = np.zeros_like(t)
    v[1:-1] = (x_smoothed[2:] - x_smoothed[:-2]) / (t[2:] - t[:-2])
    v[0] = (x_smoothed[1] - x_smoothed[0]) / (t[1] - t[0])
    v[-1] = (x_smoothed[-1] - x_smoothed[-2]) / (t[-1] - t[-2])
    
    # Smooth velocity
    v_smoothed = savgol_filter(v, window_length=21, polyorder=3)
    
    return x_smoothed, v_smoothed

def quadratic_damping_envelope(t, A0, beta):
    """
    Envelope for quadratic damping: A(t) = A0 / (1 + beta*A0*t)
    
    Parameters:
    t: time from start of decay
    A0: initial amplitude
    beta: damping parameter (related to physical damping coefficient)
    """
    return A0 / (1 + beta * A0 * t)

def linear_damping_envelope(t, A0, tau):
    """
    Envelope for linear damping: A(t) = A0 * exp(-t/tau)
    
    Parameters:
    t: time from start of decay
    A0: initial amplitude
    tau: time constant
    """
    return A0 * np.exp(-t/tau)

def analyze_oscillation_data(filename=None, mass=None, max_time=None, plot_interim=True):
    """
    Analyze oscillation data with quadratic damping.
    
    Parameters:
    filename: path to data file
    mass: mass in kg (if None, extracted from filename)
    max_time: maximum time to include in analysis (if None, use all data)
    plot_interim: whether to plot intermediate steps
    """
    # Load data
    if filename is None:
        filename = load_disco_forato_data()
    
    print(f"Analyzing: {filename}")
    df = pd.read_csv(filename, sep=';')
    t = df['time'].values
    x = df['position'].values
    
    # Limit time range if specified
    if max_time is not None and max_time < t[-1]:
        idx = np.searchsorted(t, max_time)
        t = t[:idx]
        x = x[:idx]
    
    # Get mass
    if mass is None:
        mass = extract_mass_from_filename(filename)
    
    print(f"Using mass: {mass} kg")
    print(f"Data range: {t[0]:.2f}s to {t[-1]:.2f}s ({len(t)} points)")
    
    # Process and smooth data
    fs = 1.0 / np.mean(np.diff(t))  # Sampling frequency
    x_smoothed, v_smoothed = process_data(t, x, fs)
    
    # Plot raw and processed data
    if plot_interim:
        plt.figure(figsize=(12, 6))
        plt.plot(t, x, 'b-', alpha=0.3, label='Raw Position Data')
        plt.plot(t, x_smoothed, 'r-', label='Processed Position Data')
        plt.title('Raw vs. Processed Position Data')
        plt.xlabel('Time (s)')
        plt.ylabel('Position')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig('results/plots/quadratic_damping_long_preprocessing.png')
        plt.show()
    
    # Determine equilibrium position
    x_equilibrium = np.mean(x_smoothed)
    x_centered = x_smoothed - x_equilibrium
    
    # Find peaks and valleys
    # Adjust minimum height based on data amplitude decay
    min_height = np.max(np.abs(x_centered)) * 0.05  # 5% of max amplitude
    min_distance = fs * 0.3  # Minimum 0.3 seconds between peaks
    
    peaks, peak_props = find_peaks(x_centered, height=min_height, distance=min_distance)
    valleys, valley_props = find_peaks(-x_centered, height=min_height, distance=min_distance)
    
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
        
        plt.title('Oscillation Amplitude Decay Analysis')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig('results/plots/quadratic_damping_long_amplitude_decay.png')
        plt.show()
        
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
        plt.savefig('results/plots/quadratic_damping_long_full_oscillation.png')
        plt.show()
        
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
        plt.savefig('results/plots/quadratic_damping_long_residuals.png')
        plt.show()
        
        # If available, analyze velocity-squared relationship
        if 'velocity' in df.columns:
            v_raw = df['velocity'].values
            if max_time is not None:
                v_raw = v_raw[:len(t)]
            
            # Process velocity data
            v_smoothed = savgol_filter(v_raw, window_length=21, polyorder=3)
            
            # If acceleration is available
            if 'acceleration' in df.columns:
                a_raw = df['acceleration'].values
                if max_time is not None:
                    a_raw = a_raw[:len(t)]
                
                # Process acceleration data
                a_smoothed = savgol_filter(a_raw, window_length=21, polyorder=3)
                
                # Plot |v|² vs |a| (should be linear for quadratic damping)
                plt.figure(figsize=(10, 6))
                
                # Calculate v² and prepare for plotting
                v_squared = v_smoothed**2
                
                # Filter out very small values to reduce noise
                threshold = np.percentile(np.abs(v_smoothed), 70)
                mask = np.abs(v_smoothed) > threshold
                
                plt.scatter(v_squared[mask], np.abs(a_smoothed[mask]), s=2, alpha=0.3)
                plt.title('|a| vs v² Relationship (Quadratic Damping Test)')
                plt.xlabel('Velocity Squared (v²)')
                plt.ylabel('Acceleration Magnitude |a|')
                plt.grid(True, alpha=0.3)
                plt.savefig('results/plots/quadratic_damping_long_v_squared_test.png')
                plt.show()
    
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    try:
        # Default analysis with all data
        analyze_oscillation_data()
        
        # Optionally, you can run with limited time range for faster analysis
        # analyze_oscillation_data(max_time=30)  # Analyze only first 30 seconds
    except Exception as e:
        print(f"Analysis failed: {e}")