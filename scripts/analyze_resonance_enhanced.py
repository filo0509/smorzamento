import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import re
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, butter, filtfilt, savgol_filter
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

# Set up matplotlib aesthetic parameters
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

def read_battimenti_file(filepath):
    """Read and parse a resonance experiment file."""
    try:
        # Try different parsing options since files have varied formats
        try:
            df = pd.read_csv(filepath, sep=';', decimal=',', encoding='utf-8-sig')
        except:
            try:
                df = pd.read_csv(filepath, sep=';', decimal='.', encoding='utf-8-sig')
            except:
                print(f"Failed to read file: {filepath}")
                return None
        
        # Extract frequency from filename
        freq_match = re.search(r'frequenza_(\d+\.\d+)hz', filepath.lower())
        if not freq_match:
            freq_match = re.search(r'frequenza_(\d+)hz', filepath.lower())
        frequency = float(freq_match.group(1)) if freq_match else None
        
        # Extract mass from filename
        mass_match = re.search(r'(\d+\.\d+)kg', filepath.lower())
        mass = float(mass_match.group(1)) if mass_match else None
        
        # Standardize column names
        if df.columns[0].startswith('Time'):
            time_col = df.columns[0]
            pos_col = [col for col in df.columns if 'Position' in col][0]
            vel_col = [col for col in df.columns if 'Velocity' in col][0] if any('Velocity' in col for col in df.columns) else None
            acc_col = [col for col in df.columns if 'Acceleration' in col][0] if any('Acceleration' in col for col in df.columns) else None
        else:
            # Try positional assignment
            time_col = df.columns[0]
            pos_col = df.columns[1] if len(df.columns) > 1 else None
            vel_col = df.columns[2] if len(df.columns) > 2 else None
            acc_col = df.columns[3] if len(df.columns) > 3 else None
        
        # Extract data
        time = df[time_col].values
        position = df[pos_col].values if pos_col else None
        velocity = df[vel_col].values if vel_col else None
        acceleration = df[acc_col].values if acc_col else None
        
        # Clean and preprocess data
        position = pd.Series(position).interpolate().ffill().bfill().values
        if velocity is not None:
            velocity = pd.Series(velocity).interpolate().ffill().bfill().values
        if acceleration is not None:
            acceleration = pd.Series(acceleration).interpolate().ffill().bfill().values
            
        return {
            'time': time,
            'position': position,
            'velocity': velocity,
            'acceleration': acceleration,
            'frequency': frequency,
            'mass': mass,
            'filename': os.path.basename(filepath)
        }
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return None

def smooth_data(data, method='savgol'):
    """Apply smoothing to position data."""
    if method == 'savgol':
        # Savitzky-Golay filter - better preserves peak heights
        window_length = min(51, len(data) // 5)
        if window_length % 2 == 0:  # Window length must be odd
            window_length += 1
        return savgol_filter(data, window_length, 3)
    elif method == 'lowpass':
        # Butterworth low-pass filter - more aggressive smoothing
        b, a = butter(3, 0.1, 'low')
        return filtfilt(b, a, data)
    else:
        return data  # No smoothing

def calculate_steady_state_amplitude(time, position, drive_freq):
    """
    Calculate the amplitude of steady-state oscillation using FFT analysis
    and peak detection with additional validation.
    """
    # Skip initial transient (first 30% of the data)
    start_idx = int(len(time) * 0.3)
    
    # Apply smoothing
    position_smooth = smooth_data(position[start_idx:])
    time_steady = time[start_idx:]
    
    # Find peaks and valleys
    peaks, _ = find_peaks(position_smooth, height=0, distance=5)
    valleys, _ = find_peaks(-position_smooth, height=0, distance=5)
    
    # Different approaches to calculate amplitude
    amplitudes = []
    
    # 1. Using peaks and valleys
    if len(peaks) >= 3 and len(valleys) >= 3:
        avg_peak = np.mean(position_smooth[peaks])
        avg_valley = np.mean(position_smooth[valleys])
        peak_valley_amp = (avg_peak - avg_valley) / 2
        amplitudes.append(peak_valley_amp)
    
    # 2. Using max-min
    max_min_amp = (np.max(position_smooth) - np.min(position_smooth)) / 2
    amplitudes.append(max_min_amp)
    
    # 3. Using FFT
    if len(time_steady) > 20:  # Need enough points for FFT
        n = len(time_steady)
        dt = np.mean(np.diff(time_steady))
        yf = np.abs(np.fft.rfft(position_smooth)) * 2.0 / n
        xf = np.fft.rfftfreq(n, dt)
        
        # Find frequency closest to drive frequency
        idx = np.argmin(np.abs(xf - drive_freq))
        fft_amp = yf[idx]
        amplitudes.append(fft_amp)
    
    # Combine results with median to be robust against outliers
    amplitude = np.median(amplitudes)
    
    # Calculate RMS as another metric
    rms_amplitude = np.sqrt(np.mean(position_smooth**2))
    
    return {
        'amplitude': amplitude,
        'rms_amplitude': rms_amplitude,
        'peak_data': (peaks, position_smooth[peaks]) if len(peaks) > 0 else ([], []),
        'valley_data': (valleys, position_smooth[valleys]) if len(valleys) > 0 else ([], [])
    }

def lorentzian(f, f0, gamma, A):
    """
    Lorentzian function for resonance curve.
    
    Parameters:
    f: frequency
    f0: resonance frequency
    gamma: full width at half maximum (FWHM)
    A: amplitude parameter
    """
    return A / ((f - f0)**2 + (gamma/2)**2)

def damped_harmonic_oscillator(t, A, omega, phi, tau, offset):
    """
    Function for damped harmonic oscillation.
    
    Parameters:
    t: time
    A: initial amplitude
    omega: angular frequency
    phi: phase
    tau: characteristic damping time
    offset: equilibrium position
    """
    return A * np.exp(-t/tau) * np.cos(omega*t + phi) + offset

def plot_time_series(data, results_dir, amplitude_info=None):
    """Plot time series data with amplitude analysis."""
    time = data['time']
    position = data['position']
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)
    
    # Main position plot
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time, position, linewidth=1, alpha=0.6, color='#1f77b4', label='Raw Data')
    
    # Add smoothed data if amplitude info is provided
    if amplitude_info:
        start_idx = int(len(time) * 0.3)
        t_steady = time[start_idx:]
        p_steady = smooth_data(position[start_idx:])
        ax1.plot(t_steady, p_steady, linewidth=1.5, color='#ff7f0e', label='Smoothed Data')
        
        # Plot detected peaks and valleys
        peaks, peak_pos = amplitude_info['peak_data']
        valleys, valley_pos = amplitude_info['valley_data']
        
        if len(peaks) > 0:
            ax1.scatter(t_steady[peaks], peak_pos, color='red', s=30, alpha=0.7, label='Peaks')
        if len(valleys) > 0:
            ax1.scatter(t_steady[valleys], valley_pos, color='green', s=30, alpha=0.7, label='Valleys')
            
        # Add horizontal lines for amplitude
        amp = amplitude_info['amplitude']
        mean_level = np.mean(p_steady)
        ax1.axhline(mean_level + amp, color='purple', linestyle='--', alpha=0.7, 
                   label=f'Amplitude: {amp:.5f} m')
        ax1.axhline(mean_level - amp, color='purple', linestyle='--', alpha=0.7)
    
    ax1.set_title(f"Oscillation at {data['frequency']:.3f} Hz", fontsize=16)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Position (m)")
    ax1.legend(loc='upper right')
    
    # Velocity subplot if available
    if data['velocity'] is not None:
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.plot(time, data['velocity'], linewidth=1, color='#2ca02c')
        ax2.set_ylabel("Velocity (m/s)")
        ax2.set_title("Velocity", fontsize=12)
    
    # Acceleration subplot if available
    if data['acceleration'] is not None:
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.plot(time, data['acceleration'], linewidth=1, color='#d62728')
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Acceleration (m/s²)")
        ax3.set_title("Acceleration", fontsize=12)
    
    filename = f"timeseries_{data['frequency']:.3f}hz.png"
    plt.savefig(os.path.join(results_dir, filename))
    plt.close()
    
    return filename

def plot_resonance_curve(frequencies, amplitudes, fit_params, results_dir, damping_info=None):
    """Create an elaborate resonance curve plot with Lorentzian fit."""
    f0, gamma, A = fit_params
    
    # Create high-resolution x values for the fit curve
    freq_fit = np.linspace(min(frequencies) * 0.8, max(frequencies) * 1.2, 1000)
    amp_fit = lorentzian(freq_fit, f0, gamma, A)
    
    # Create the main figure
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[4, 1])
    
    # Main plot
    ax_main = fig.add_subplot(gs[0, 0])
    
    # Plot measured data
    ax_main.scatter(frequencies, amplitudes, s=100, color='#1f77b4', 
                   edgecolors='black', linewidth=1.5, label='Measured Amplitudes', zorder=5)
    
    # Plot fitted curve
    ax_main.plot(freq_fit, amp_fit, '-', color='#ff7f0e', linewidth=2.5, label='Lorentzian Fit')
    
    # Add half-maximum markers
    half_max_amp = lorentzian(f0, f0, gamma, A) / 2
    ax_main.axhline(y=half_max_amp, color='green', linestyle=':', alpha=0.7, 
                   label='Half Maximum')
    
    # Find the frequency points where amplitude equals half maximum
    half_max_freqs = []
    for i in range(len(freq_fit)-1):
        if (amp_fit[i] < half_max_amp and amp_fit[i+1] >= half_max_amp) or \
           (amp_fit[i] >= half_max_amp and amp_fit[i+1] < half_max_amp):
            half_max_freqs.append(freq_fit[i])
    
    # Mark the resonance frequency
    ax_main.axvline(x=f0, color='red', linestyle='--', alpha=0.7, 
                   label=f'Resonance: {f0:.4f} Hz')
    
    # Mark FWHM if we found the half-max points
    if len(half_max_freqs) >= 2:
        fwhm_left = min(half_max_freqs)
        fwhm_right = max(half_max_freqs)
        ax_main.plot([fwhm_left, fwhm_right], [half_max_amp, half_max_amp], 'g-', linewidth=2.5,
                    label=f'FWHM: {gamma:.4f} Hz')
        
        # Add shaded area for FWHM
        ax_main.fill_between([fwhm_left, fwhm_right], [0, 0], [half_max_amp, half_max_amp],
                           alpha=0.2, color='green')
    
    # Label the points with frequencies
    for freq, amp in zip(frequencies, amplitudes):
        ax_main.annotate(f"{freq:.3f} Hz", 
                        xy=(freq, amp),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontsize=10)
    
    ax_main.set_title('Resonance Curve with Lorentzian Fit', fontsize=18)
    ax_main.set_xlabel('Frequency (Hz)', fontsize=14)
    ax_main.set_ylabel('Amplitude (m)', fontsize=14)
    ax_main.legend(loc='upper right', fontsize=12)
    
    # Add quality factor annotation
    Q = f0 / gamma
    bbox_props = dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.8)
    ax_main.text(0.05, 0.95, f"Q = {Q:.2f}", transform=ax_main.transAxes, 
                fontsize=14, verticalalignment='top', bbox=bbox_props)
    
    # Add damping ratio if available
    if damping_info and 'damping_ratio' in damping_info:
        zeta = damping_info['damping_ratio']
        ax_main.text(0.05, 0.85, f"ζ = {zeta:.5f}", transform=ax_main.transAxes,
                    fontsize=14, verticalalignment='top', bbox=bbox_props)
    
    # Residuals plot
    ax_resid = fig.add_subplot(gs[1, 0], sharex=ax_main)
    
    # Calculate residuals (observed - predicted)
    predicted_amps = [lorentzian(f, f0, gamma, A) for f in frequencies]
    residuals = np.array(amplitudes) - np.array(predicted_amps)
    
    # Plot residuals
    ax_resid.scatter(frequencies, residuals, color='purple', s=70, alpha=0.7)
    ax_resid.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add shaded area for +/- 1 stdev
    stdev = np.std(residuals)
    ax_resid.axhspan(-stdev, stdev, alpha=0.2, color='gray', label=f'±σ: {stdev:.5f}')
    
    ax_resid.set_title('Residuals', fontsize=14)
    ax_resid.set_xlabel('Frequency (Hz)', fontsize=12)
    ax_resid.set_ylabel('Residual (Obs-Pred)', fontsize=12)
    ax_resid.legend(loc='upper right')
    
    # Right-side amplitude distribution plot
    ax_ampdist = fig.add_subplot(gs[0, 1], sharey=ax_main)
    
    # Create histogram instead of KDE plot
    counts, bins = np.histogram(amplitudes, bins=10)
    ax_ampdist.barh(height=0.8 * (bins[1:] - bins[:-1]), width=counts, 
                   left=0, y=0.5 * (bins[1:] + bins[:-1]), 
                   color='#1f77b4', alpha=0.5)
    
    # Add individual points
    for amp in amplitudes:
        ax_ampdist.axhline(y=amp, color='#1f77b4', alpha=0.3, linewidth=1)
    
    ax_ampdist.set_title('Amplitude Distribution', fontsize=14)
    ax_ampdist.set_xlabel('Count', fontsize=12)
    ax_ampdist.set_yticklabels([])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(results_dir, "resonance_curve_enhanced.png")
    plt.savefig(output_file)
    
    return "resonance_curve_enhanced.png"

def analyze_resonance_detailed():
    """Perform enhanced resonance analysis on the battimenti files."""
    # Setup directories
    data_dir = "data_raw"
    results_dir = os.path.join("results", "plots", "resonance_enhanced")
    analysis_dir = os.path.join("results", "analysis")
    
    # Create directories if they don't exist
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Find all battimenti files
    battimenti_files = glob.glob(os.path.join(data_dir, "*battimenti*.csv"))
    
    if not battimenti_files:
        print("No resonance data files found!")
        return
    
    print(f"Found {len(battimenti_files)} resonance data files.")
    
    # Process each file and extract amplitude vs frequency data
    results = []
    for file_path in battimenti_files:
        filename = os.path.basename(file_path)
        print(f"Processing {filename}...")
        
        # Read and parse the file
        data = read_battimenti_file(file_path)
        if data is None or data['frequency'] is None:
            print(f"  Skipping due to parsing issues.")
            continue
        
        # Calculate amplitude
        amplitude_info = calculate_steady_state_amplitude(
            data['time'], data['position'], data['frequency'])
        
        # Add to results list
        results.append({
            'frequency': data['frequency'],
            'amplitude': amplitude_info['amplitude'],
            'rms_amplitude': amplitude_info['rms_amplitude'],
            'mass': data['mass'],
            'filename': data['filename']
        })
        
        # Generate time series plot
        plot_file = plot_time_series(data, results_dir, amplitude_info)
        print(f"  Generated {plot_file}")
    
    if not results:
        print("No valid data was extracted from the files.")
        return
    
    # Sort results by frequency
    results.sort(key=lambda x: x['frequency'])
    
    # Extract arrays for curve fitting
    frequencies = np.array([r['frequency'] for r in results])
    amplitudes = np.array([r['amplitude'] for r in results])
    
    # Fit Lorentzian curve
    try:
        # Initial guesses
        f0_guess = frequencies[np.argmax(amplitudes)]  # Resonance frequency
        gamma_guess = 0.2  # Width parameter (FWHM)
        A_guess = np.max(amplitudes) * (gamma_guess/2)**2  # Amplitude parameter
        
        # Perform curve fit
        params, params_covariance = curve_fit(
            lorentzian, 
            frequencies, 
            amplitudes, 
            p0=[f0_guess, gamma_guess, A_guess],
            bounds=([0.5, 0.01, 0], [3.0, 2.0, 100])
        )
        
        f0, gamma, A = params
        f0_err, gamma_err, A_err = np.sqrt(np.diag(params_covariance))
        
        # Calculate quality factor Q = f0/Δf
        Q = f0 / gamma
        
        # Calculate physical parameters
        mass = results[0]['mass']  # Assuming all files have the same mass
        
        # Calculate damping parameters
        omega0 = 2 * np.pi * f0  # Angular resonance frequency
        k = mass * omega0**2     # Spring constant
        b = mass * gamma * np.pi  # Damping coefficient for FWHM definition
        zeta = b / (2 * np.sqrt(k * mass))  # Damping ratio
        
        damping_info = {
            'omega0': omega0,
            'k': k,
            'b': b,
            'damping_ratio': zeta
        }
        
        print("\nResonance Analysis Results:")
        print(f"Resonance Frequency (f₀): {f0:.4f} ± {f0_err:.4f} Hz")
        print(f"Width Parameter (FWHM γ): {gamma:.4f} ± {gamma_err:.4f} Hz")
        print(f"Quality Factor (Q = f₀/γ): {Q:.2f}")
        
        if mass:
            print(f"Mass: {mass:.4f} kg")
            print(f"Spring Constant (k): {k:.4f} N/m")
            print(f"Damping Coefficient (b): {b:.4f} kg/s")
            print(f"Damping Ratio (ζ): {zeta:.6f}")
        
        # Generate enhanced resonance curve plot
        plot_file = plot_resonance_curve(frequencies, amplitudes, params, results_dir, damping_info)
        print(f"Generated enhanced resonance curve: {plot_file}")
        
        # Create a detailed report
        report_path = os.path.join(analysis_dir, "resonance_analysis_enhanced.txt")
        
        with open(report_path, 'w') as f:
            f.write("Enhanced Resonance Analysis Results\n")
            f.write("=================================\n\n")
            
            f.write("Resonance Parameters:\n")
            f.write("-----------------\n")
            f.write(f"Resonance Frequency (f₀): {f0:.6f} ± {f0_err:.6f} Hz\n")
            f.write(f"FWHM (γ): {gamma:.6f} ± {gamma_err:.6f} Hz\n")
            f.write(f"Amplitude Parameter (A): {A:.8f} ± {A_err:.8f}\n")
            f.write(f"Quality Factor (Q = f₀/γ): {Q:.4f}\n\n")
            
            f.write("Physical Parameters:\n")
            f.write("------------------\n")
            if mass:
                f.write(f"Mass (m): {mass:.6f} kg\n")
                f.write(f"Angular Frequency (ω₀): {omega0:.6f} rad/s\n")
                f.write(f"Spring Constant (k): {k:.6f} N/m\n")
                f.write(f"Damping Coefficient (b): {b:.6f} kg/s\n")
                f.write(f"Damping Ratio (ζ): {zeta:.6f}\n\n")
                
                f.write("Additional Derived Parameters:\n")
                f.write("----------------------------\n")
                f.write(f"Natural Period (T₀ = 2π/ω₀): {2*np.pi/omega0:.6f} s\n")
                f.write(f"Critical Damping Coefficient (bcrit = 2√km): {2*np.sqrt(k*mass):.6f} kg/s\n")
                f.write(f"Energy Dissipation Rate (P = b·ω₀²·A²/2): {b*omega0**2*np.max(amplitudes)**2/2:.6f} W\n\n")
            
            f.write("Lorentzian Fit Summary:\n")
            f.write("--------------------\n")
            f.write(f"Fit Function: A / ((f - f₀)² + (γ/2)²)\n")
            
            # Calculate goodness of fit metrics
            predicted = [lorentzian(f, f0, gamma, A) for f in frequencies]
            residuals = amplitudes - predicted
            SS_res = np.sum(residuals**2)
            SS_tot = np.sum((amplitudes - np.mean(amplitudes))**2)
            r_squared = 1 - (SS_res / SS_tot)
            rmse = np.sqrt(np.mean(residuals**2))
            
            f.write(f"R-squared: {r_squared:.6f}\n")
            f.write(f"RMSE: {rmse:.8f}\n\n")
            
            f.write("Detailed Measurements:\n")
            f.write("--------------------\n")
            f.write("Freq (Hz) | Amplitude (m) | RMS Amplitude (m) | Filename\n")
            f.write("-" * 75 + "\n")
            for result in results:
                f.write(f"{result['frequency']:.4f} | {result['amplitude']:.8f} | {result['rms_amplitude']:.8f} | {result['filename']}\n")
            
            f.write("\n\nAnalysis completed on: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
            
        print(f"\nEnhanced analysis results saved to {report_path}")
        return report_path
            
    except Exception as e:
        print(f"Error during curve fitting: {e}")
        
        # Even if fit fails, plot the raw data
        plt.figure(figsize=(10, 6))
        plt.scatter(frequencies, amplitudes, s=80, color='blue', edgecolors='black')
        plt.title('Resonance Data (Fitting Failed)', fontsize=16)
        plt.xlabel('Frequency (Hz)', fontsize=14)
        plt.ylabel('Amplitude (m)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Save plot
        output_file = os.path.join(results_dir, "resonance_data_raw.png")
        plt.savefig(output_file)
        plt.close()
        
        print(f"Raw data plot saved to {output_file}")
        return None

if __name__ == "__main__":
    analyze_resonance_detailed()