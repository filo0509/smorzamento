import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from scipy.signal import butter, filtfilt
import glob
import matplotlib
matplotlib.style.use('seaborn-v0_8-whitegrid')

# Set consistent plot styling
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12

# Plot colors
COLOR_POSITION = '#1f77b4'  # Blue
COLOR_VELOCITY = '#ff7f0e'  # Orange
COLOR_ACCEL = '#2ca02c'     # Green
COLOR_QUAD_FIT = '#d62728'  # Red
COLOR_LIN_FIT = '#9467bd'   # Purple
COLOR_PEAKS = '#8c564b'     # Brown

def ensure_dirs_exist():
    """Ensure all required directories exist."""
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/analysis', exist_ok=True)

def butter_lowpass_filter(data, cutoff, fs, order=4):
    """Apply a low-pass Butterworth filter to the data."""
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def process_data(t, x, fs=100):
    """Clean and process raw data for analysis."""
    # Apply smoothing to reduce noise
    x_filtered = butter_lowpass_filter(x, cutoff=5.0, fs=fs, order=3)
    x_smoothed = savgol_filter(x_filtered, window_length=21, polyorder=3)
    return x_smoothed

def quadratic_damping_envelope(t, A0, beta):
    """Envelope for quadratic damping: A(t) = A0 / (1 + beta*A0*t)"""
    return A0 / (1 + beta * A0 * t)

def linear_damping_envelope(t, A0, tau):
    """Envelope for linear damping: A(t) = A0 * exp(-t/tau)"""
    return A0 * np.exp(-t/tau)

def extract_mass_from_filename(filename):
    """Try to extract mass value from filename."""
    try:
        parts = os.path.basename(filename).split('_')
        for part in parts:
            if part.replace('.', '').isdigit():
                value = float(part)
                if 0.05 < value < 0.5:  # Reasonable mass range in kg
                    return value
        return 0.2016  # Default
    except:
        return 0.2016  # Default

def plot_raw_data(filename, output_prefix, show_plots=False):
    """Generate standardized raw data plot."""
    try:
        df = pd.read_csv(filename, sep=';')
        t = df['time'].values
        x = df['position'].values
        
        # Calculate velocity if not available
        if 'velocity' in df.columns:
            v = df['velocity'].values
        else:
            dt = np.diff(t)
            v = np.zeros_like(t)
            v[1:] = np.diff(x) / dt
        
        # Calculate acceleration if not available
        if 'acceleration' in df.columns:
            a = df['acceleration'].values
        else:
            dt = np.diff(t)
            a = np.zeros_like(t)
            a[1:-1] = (v[2:] - v[:-2]) / (t[2:] - t[:-2])
        
        # Create a multi-panel figure
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Position plot
        axes[0].plot(t, x, color=COLOR_POSITION, label='Position')
        axes[0].set_ylabel('Position (m)')
        axes[0].set_title('Position vs Time')
        axes[0].legend(loc='upper right')
        
        # Velocity plot
        axes[1].plot(t, v, color=COLOR_VELOCITY, label='Velocity')
        axes[1].set_ylabel('Velocity (m/s)')
        axes[1].set_title('Velocity vs Time')
        axes[1].legend(loc='upper right')
        
        # Acceleration plot
        axes[2].plot(t, a, color=COLOR_ACCEL, label='Acceleration')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Acceleration (m/s²)')
        axes[2].set_title('Acceleration vs Time')
        axes[2].legend(loc='upper right')
        
        plt.suptitle(f'Raw Data Analysis: {os.path.basename(filename)}', fontsize=18)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save figure
        figname = f'results/plots/{output_prefix}_raw_data.png'
        plt.savefig(figname, dpi=150)
        print(f"Saved {figname}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
            
        return True
    except Exception as e:
        print(f"Error in plot_raw_data: {e}")
        return False

def analyze_damping(filename, output_prefix, is_quadratic=True, show_plots=False):
    """Generate standardized damping analysis plots."""
    try:
        df = pd.read_csv(filename, sep=';')
        t = df['time'].values
        x = df['position'].values
        
        # Get mass
        mass = extract_mass_from_filename(filename)
        
        # Process data
        fs = 1.0 / np.mean(np.diff(t))
        x_smoothed = process_data(t, x, fs)
        
        # Center data
        x_equilibrium = np.mean(x_smoothed)
        x_centered = x_smoothed - x_equilibrium
        
        # Find peaks and valleys
        min_height = np.max(np.abs(x_centered)) * 0.05
        min_distance = fs * 0.3
        
        peaks, _ = find_peaks(x_centered, height=min_height, distance=min_distance)
        valleys, _ = find_peaks(-x_centered, height=min_height, distance=min_distance)
        
        # Extract peaks and valleys
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
        
        # Filter out noise
        amp_threshold = np.max(all_extrema_amp) * 0.03
        valid_idx = all_extrema_amp > amp_threshold
        all_extrema_t = all_extrema_t[valid_idx]
        all_extrema_amp = all_extrema_amp[valid_idx]
        
        if len(all_extrema_t) < 5:
            print(f"Warning: Too few extrema found in {filename}")
            return False
        
        # Fit damping models
        t_rel = all_extrema_t - all_extrema_t[0]
        A0_guess = np.max(all_extrema_amp)
        
        # For quadratic damping model
        half_amp = A0_guess / 2
        t_half_idx = np.argmin(np.abs(all_extrema_amp - half_amp))
        if t_half_idx > 0:
            t_half = all_extrema_t[t_half_idx] - all_extrema_t[0]
            beta_guess = 1 / (A0_guess * t_half)
        else:
            beta_guess = 0.1
            
        # Fit quadratic damping model
        params_quad, _ = curve_fit(
            quadratic_damping_envelope, 
            t_rel, all_extrema_amp,
            p0=[A0_guess, beta_guess],
            bounds=([0, 0], [np.inf, np.inf]),
            maxfev=20000
        )
        
        A0_quad, beta_quad = params_quad
        half_life = 1 / (beta_quad * A0_quad)
        
        # Fit linear damping model
        params_lin, _ = curve_fit(
            linear_damping_envelope,
            t_rel, all_extrema_amp,
            p0=[A0_guess, half_life],
            bounds=([0, 0], [np.inf, np.inf]),
            maxfev=20000
        )
        
        A0_lin, tau = params_lin
        
        # Calculate goodness of fit
        pred_quad = quadratic_damping_envelope(t_rel, A0_quad, beta_quad)
        pred_lin = linear_damping_envelope(t_rel, A0_lin, tau)
        
        residuals_quad = all_extrema_amp - pred_quad
        residuals_lin = all_extrema_amp - pred_lin
        
        ss_total = np.sum((all_extrema_amp - np.mean(all_extrema_amp))**2)
        ss_residual_quad = np.sum(residuals_quad**2)
        ss_residual_lin = np.sum(residuals_lin**2)
        
        r_squared_quad = 1 - (ss_residual_quad / ss_total)
        r_squared_lin = 1 - (ss_residual_lin / ss_total)
        
        # Plot 1: Amplitude decay with both models
        plt.figure(figsize=(12, 8))
        
        # Plot data points
        plt.scatter(all_extrema_t, all_extrema_amp, color=COLOR_PEAKS, s=30, alpha=0.7, label='Amplitude Peaks')
        
        # Plot model fits
        t_model = np.linspace(all_extrema_t[0], all_extrema_t[-1], 1000)
        t_model_rel = t_model - all_extrema_t[0]
        
        amp_quad = quadratic_damping_envelope(t_model_rel, A0_quad, beta_quad)
        amp_lin = linear_damping_envelope(t_model_rel, A0_lin, tau)
        
        plt.plot(t_model, amp_quad, color=COLOR_QUAD_FIT, linewidth=2.5, 
                 label=f'Quadratic Damping: A₀/(1+βA₀t), R² = {r_squared_quad:.4f}')
        plt.plot(t_model, amp_lin, color=COLOR_LIN_FIT, linewidth=2.5, linestyle='--',
                 label=f'Linear Damping: A₀e^(-t/τ), R² = {r_squared_lin:.4f}')
        
        # Mark half-life if quadratic model
        if is_quadratic:
            half_amp = A0_quad / 2
            t_half_mark = all_extrema_t[0] + half_life
            plt.axvline(x=t_half_mark, color='green', linestyle=':', label=f'Half-life = {half_life:.2f}s')
            plt.axhline(y=half_amp, color='green', linestyle=':')
        
        plt.title('Amplitude Decay Analysis')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        
        # Save figure
        figname = f'results/plots/{output_prefix}_amplitude_decay.png'
        plt.savefig(figname, dpi=150)
        print(f"Saved {figname}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # Plot 2: Full oscillation with envelope
        plt.figure(figsize=(12, 8))
        
        plt.plot(t, x_centered, color=COLOR_POSITION, alpha=0.5, label='Position (centered)')
        
        # Plot dominant envelope (quadratic or linear)
        env_model = amp_quad if is_quadratic else amp_lin
        env_label = 'Quadratic Damping Envelope' if is_quadratic else 'Linear Damping Envelope'
        
        plt.plot(t_model, env_model, color=COLOR_QUAD_FIT if is_quadratic else COLOR_LIN_FIT, 
                 linewidth=2.5, label=env_label)
        plt.plot(t_model, -env_model, color=COLOR_QUAD_FIT if is_quadratic else COLOR_LIN_FIT, 
                 linewidth=2.5)
        
        # Plot peaks and valleys
        plt.scatter(peaks_t, peaks_x, color='red', s=20, alpha=0.7)
        plt.scatter(valleys_t, valleys_x, color='green', s=20, alpha=0.7)
        
        model_type = "Quadratic Damping" if is_quadratic else "Linear Damping"
        plt.title(f'Oscillation with {model_type} Envelope')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (centered)')
        plt.legend()
        
        # Save figure
        figname = f'results/plots/{output_prefix}_full_oscillation.png'
        plt.savefig(figname, dpi=150)
        print(f"Saved {figname}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # Plot 3: Residuals comparison
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(all_extrema_t, residuals_quad, color=COLOR_QUAD_FIT, s=30, alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.title('Quadratic Damping Residuals')
        plt.xlabel('Time (s)')
        plt.ylabel('Residual')
        
        plt.subplot(1, 2, 2)
        plt.scatter(all_extrema_t, residuals_lin, color=COLOR_LIN_FIT, s=30, alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.title('Linear Damping Residuals')
        plt.xlabel('Time (s)')
        plt.ylabel('Residual')
        
        plt.tight_layout()
        
        # Save figure
        figname = f'results/plots/{output_prefix}_residuals.png'
        plt.savefig(figname, dpi=150)
        print(f"Saved {figname}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # Save analysis results to file
        results_file = f'results/analysis/{output_prefix}_analysis.txt'
        with open(results_file, 'w') as f:
            f.write(f"Analysis of {os.path.basename(filename)}\n")
            f.write(f"Mass: {mass} kg\n\n")
            
            f.write("Quadratic Damping Model Results:\n")
            f.write(f"Initial amplitude (A₀): {A0_quad:.6f}\n")
            f.write(f"Damping parameter (β): {beta_quad:.6f}\n")
            f.write(f"Half-life: {half_life:.4f} seconds\n\n")
            
            f.write("Linear Damping Model Results:\n")
            f.write(f"Initial amplitude (A₀): {A0_lin:.6f}\n")
            f.write(f"Time constant (τ): {tau:.6f} seconds\n\n")
            
            f.write("Model Comparison:\n")
            f.write(f"R² (Quadratic Damping): {r_squared_quad:.6f}\n")
            f.write(f"R² (Linear Damping): {r_squared_lin:.6f}\n\n")
            
            # Calculate physical parameters
            if len(peaks_t) > 1:
                periods = np.diff(peaks_t)
                omega = 2 * np.pi / np.mean(periods)
                f = omega / (2 * np.pi)
                period = 2 * np.pi / omega
                k = mass * omega**2
                
                if is_quadratic:
                    b = (3 * np.pi * mass * beta_quad) / (8 * omega)
                    f.write(f"Physical Parameters (Quadratic Damping):\n")
                    f.write(f"Angular frequency (ω): {omega:.4f} rad/s\n")
                    f.write(f"Frequency (f): {f:.4f} Hz\n")
                    f.write(f"Period (T): {period:.4f} s\n")
                    f.write(f"Spring constant (k): {k:.4f} N/m\n")
                    f.write(f"Quadratic damping coefficient (b): {b:.6f} kg/m\n")
                else:
                    gamma = mass / tau
                    f.write(f"Physical Parameters (Linear Damping):\n")
                    f.write(f"Angular frequency (ω): {omega:.4f} rad/s\n")
                    f.write(f"Frequency (f): {f:.4f} Hz\n")
                    f.write(f"Period (T): {period:.4f} s\n")
                    f.write(f"Spring constant (k): {k:.4f} N/m\n")
                    f.write(f"Linear damping coefficient (γ): {gamma:.6f} kg/s\n")
        
        print(f"Saved analysis to {results_file}")
        return True
        
    except Exception as e:
        print(f"Error in analyze_damping: {e}")
        return False

def process_all_files():
    """Process all data files and generate standardized plots."""
    ensure_dirs_exist()
    
    # Process quadratic damping files (simulated)
    quadratic_files = glob.glob('data_simulated/*discoforato*.csv')
    print(f"Found {len(quadratic_files)} quadratic damping files")
    
    for i, file in enumerate(quadratic_files, 1):
        basename = os.path.splitext(os.path.basename(file))[0]
        print(f"\nProcessing file {i}/{len(quadratic_files)}: {basename}")
        
        # For quadratic damping files
        prefix = f"quadratic_damping_{i:02d}"
        plot_raw_data(file, prefix)
        analyze_damping(file, prefix, is_quadratic=True)
    
    # Process linear damping files (raw data)
    linear_files = glob.glob('data_raw/*discopieno*.csv')
    print(f"\nFound {len(linear_files)} linear damping files")
    
    for i, file in enumerate(linear_files, 1):
        basename = os.path.splitext(os.path.basename(file))[0]
        print(f"\nProcessing file {i}/{len(linear_files)}: {basename}")
        
        # For linear damping files
        prefix = f"linear_damping_{i:02d}"
        plot_raw_data(file, prefix)
        analyze_damping(file, prefix, is_quadratic=False)

def cleanup_old_plots():
    """Remove inconsistently named plot files."""
    old_patterns = [
        'disco_forato_*.png',
        'exponential_*.png',
        'test_*.png',
        'raw_data.png',
        'observed_vs_predicted.png'
    ]
    
    total_removed = 0
    for pattern in old_patterns:
        old_files = glob.glob(f'results/plots/{pattern}')
        for file in old_files:
            try:
                os.remove(file)
                print(f"Removed old file: {file}")
                total_removed += 1
            except Exception as e:
                print(f"Error removing {file}: {e}")
    
    print(f"Removed {total_removed} old plot files")

if __name__ == "__main__":
    print("Generating standardized plots for all data files...")
    
    # Automatically clean up old plots without asking
    print("Cleaning up old plot files...")
    cleanup_old_plots()
    
    # Process all files
    process_all_files()
    
    print("\nAll plots have been standardized and saved to the results/plots directory.")
    print("Analysis results are available in the results/analysis directory.")