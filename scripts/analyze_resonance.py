import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import re
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

def read_battimenti_file(filepath):
    """Read data from a battimenti file and handle various formats."""
    # Read CSV file and handle different delimiters and decimal separators
    try:
        # For files with comma as decimal separator
        df = pd.read_csv(filepath, sep=';', decimal=',', encoding='utf-8-sig')
    except:
        try:
            # Try with dot as decimal separator
            df = pd.read_csv(filepath, sep=';', decimal='.', encoding='utf-8-sig')
        except:
            print(f"Failed to read file: {filepath}")
            return None
    
    # Extract frequency from filename
    freq_match = re.search(r'frequenza_(\d+\.\d+)hz', filepath.lower())
    if not freq_match:
        freq_match = re.search(r'frequenza_(\d+)hz', filepath.lower())
    
    frequency = float(freq_match.group(1)) if freq_match else None
    
    # Standardize column names
    if df.columns[0].startswith('Time'):
        time_col = df.columns[0]
        pos_col = [col for col in df.columns if 'Position' in col][0]
        vel_col = [col for col in df.columns if 'Velocity' in col][0] if any('Velocity' in col for col in df.columns) else None
        acc_col = [col for col in df.columns if 'Acceleration' in col][0] if any('Acceleration' in col for col in df.columns) else None
    else:
        # If columns aren't named as expected, try positional assignment
        time_col = df.columns[0]
        pos_col = df.columns[1] if len(df.columns) > 1 else None
        vel_col = df.columns[2] if len(df.columns) > 2 else None
        acc_col = df.columns[3] if len(df.columns) > 3 else None
    
    # Extract data
    time = df[time_col].values
    position = df[pos_col].values if pos_col else None
    velocity = df[vel_col].values if vel_col else None
    acceleration = df[acc_col].values if acc_col else None
    
    # Replace any NaN values with interpolated values or zeros
    if position is not None:
        position = pd.Series(position).interpolate().fillna(method='bfill').fillna(0).values
    if velocity is not None:
        velocity = pd.Series(velocity).interpolate().fillna(method='bfill').fillna(0).values
    if acceleration is not None:
        acceleration = pd.Series(acceleration).interpolate().fillna(method='bfill').fillna(0).values
    
    return {
        'time': time,
        'position': position,
        'velocity': velocity,
        'acceleration': acceleration,
        'frequency': frequency,
        'filename': os.path.basename(filepath)
    }

def find_amplitude(time, position):
    """Find the amplitude of oscillation after steady state is reached."""
    # Skip initial transient (first 20% of the data)
    start_idx = int(len(time) * 0.2)
    
    # Find peaks and valleys
    peaks, _ = find_peaks(position[start_idx:], height=0, distance=5)
    valleys, _ = find_peaks(-position[start_idx:], height=0, distance=5)
    
    # Adjust peak indices to original array
    peaks = peaks + start_idx
    valleys = valleys + start_idx
    
    if len(peaks) < 3 or len(valleys) < 3:
        print("Warning: Not enough peaks/valleys found for reliable amplitude estimation")
        # Use simple max-min as fallback
        amplitude = (np.max(position[start_idx:]) - np.min(position[start_idx:])) / 2
        return amplitude
    
    # Calculate average peak and valley values
    avg_peak = np.mean(position[peaks])
    avg_valley = np.mean(position[valleys])
    
    # Calculate amplitude
    amplitude = (avg_peak - avg_valley) / 2
    
    return amplitude

def lorentzian(f, f0, gamma, A):
    """Lorentzian function for resonance curve."""
    return A / ((f - f0)**2 + (gamma/2)**2)

def analyze_resonance():
    """Analyze resonance data and plot Lorentzian curve."""
    # Find all battimenti files
    data_dir = "data_raw"
    battimenti_files = glob.glob(os.path.join(data_dir, "*battimenti*.csv"))
    
    if not battimenti_files:
        print("No resonance data files found!")
        return
    
    print(f"Found {len(battimenti_files)} resonance data files.")
    
    # Process each file and extract amplitude and frequency
    results = []
    for file_path in battimenti_files:
        print(f"Processing {os.path.basename(file_path)}...")
        data = read_battimenti_file(file_path)
        
        if data is None or data['frequency'] is None:
            print(f"Skipping {file_path} due to parsing issues.")
            continue
        
        amplitude = find_amplitude(data['time'], data['position'])
        results.append({
            'frequency': data['frequency'],
            'amplitude': amplitude,
            'filename': data['filename']
        })
        
        # Plot time series for this file
        plt.figure(figsize=(10, 6))
        plt.plot(data['time'], data['position'], linewidth=1)
        plt.title(f"Oscillation at {data['frequency']} Hz")
        plt.xlabel("Time (s)")
        plt.ylabel("Position (m)")
        plt.grid(True, alpha=0.3)
        save_path = os.path.join("results", "plots", "resonance", f"timeseries_{data['frequency']:.3f}hz.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    
    # Sort results by frequency
    results.sort(key=lambda x: x['frequency'])
    
    # Extract frequency and amplitude arrays
    frequencies = np.array([r['frequency'] for r in results])
    amplitudes = np.array([r['amplitude'] for r in results])
    
    # Fit Lorentzian curve
    try:
        # Initial guesses
        f0_guess = frequencies[np.argmax(amplitudes)]  # Resonance frequency
        gamma_guess = 0.5  # Width parameter
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
        
        # Quality factor Q = f0/Δf where Δf is FWHM
        Q = f0 / gamma
        
        print("\nResonance Analysis Results:")
        print(f"Resonance Frequency (f₀): {f0:.4f} ± {f0_err:.4f} Hz")
        print(f"Width Parameter (γ): {gamma:.4f} ± {gamma_err:.4f} Hz")
        print(f"Quality Factor (Q): {Q:.2f}")
        
        # Generate points for the fitted curve
        freq_fit = np.linspace(min(frequencies) * 0.8, max(frequencies) * 1.2, 1000)
        amp_fit = lorentzian(freq_fit, f0, gamma, A)
        
        # Plot resonance curve
        plt.figure(figsize=(12, 8))
        
        # Plot individual data points
        plt.scatter(frequencies, amplitudes, s=80, color='#1f77b4', edgecolors='black', linewidth=1.5, 
                   label='Measured Amplitudes', zorder=10)
        
        # Plot fitted Lorentzian
        plt.plot(freq_fit, amp_fit, '-', color='#ff7f0e', linewidth=2.5, 
                label=f'Lorentzian Fit\nf₀ = {f0:.4f} Hz\nQ = {Q:.2f}')
        
        # Mark the resonance frequency and half-max points
        plt.axvline(x=f0, color='red', linestyle='--', alpha=0.7, 
                   label=f'Resonance: {f0:.4f} Hz')
        
        # Mark the half-width points f0±γ/2
        half_max = lorentzian(f0, f0, gamma, A) / 2
        plt.axhline(y=half_max, color='green', linestyle=':', alpha=0.7, 
                   label='Half Maximum')
        
        # Format the plot
        plt.title('Resonance Curve with Lorentzian Fit', fontsize=16)
        plt.xlabel('Frequency (Hz)', fontsize=14)
        plt.ylabel('Amplitude (m)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Add annotations for each data point
        for freq, amp, result in zip(frequencies, amplitudes, results):
            plt.annotate(f"{freq:.3f} Hz", 
                        xy=(freq, amp),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontsize=9)
        
        # Save and show the plot
        save_path = os.path.join("results", "plots", "resonance", "resonance_curve.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        
        # Create a more detailed report file
        report_path = os.path.join("results", "analysis", "resonance_analysis.txt")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("Resonance Analysis Results\n")
            f.write("==========================\n\n")
            f.write(f"Resonance Frequency (f₀): {f0:.6f} ± {f0_err:.6f} Hz\n")
            f.write(f"Width Parameter (γ): {gamma:.6f} ± {gamma_err:.6f} Hz\n")
            f.write(f"Amplitude Parameter (A): {A:.6f} ± {A_err:.6f}\n")
            f.write(f"Quality Factor (Q = f₀/γ): {Q:.4f}\n\n")
            
            f.write("Damped Harmonic Oscillator Parameters:\n")
            # Extract mass from filename
            mass_match = re.search(r'(\d+\.\d+)kg', results[0]['filename'])
            mass = float(mass_match.group(1)) if mass_match else None
            
            if mass:
                f.write(f"Mass (m): {mass:.6f} kg\n")
                
                # Calculate spring constant: k = m * (2πf₀)²
                omega0 = 2 * np.pi * f0
                k = mass * omega0**2
                f.write(f"Angular Frequency (ω₀): {omega0:.6f} rad/s\n")
                f.write(f"Spring Constant (k): {k:.6f} N/m\n")
                
                # Calculate damping coefficient: b = m * γ
                b = mass * gamma * 2 * np.pi
                f.write(f"Damping Coefficient (b): {b:.6f} kg/s\n")
                
                # Damping ratio: ζ = b/(2√(km))
                zeta = b/(2*np.sqrt(k*mass))
                f.write(f"Damping Ratio (ζ): {zeta:.6f}\n")
            
            f.write("\nDetailed Measurements:\n")
            f.write("Frequency (Hz) | Amplitude (m) | Filename\n")
            f.write("-" * 60 + "\n")
            for result in results:
                f.write(f"{result['frequency']:.4f} | {result['amplitude']:.6f} | {result['filename']}\n")
                
        print(f"\nDetailed results saved to {report_path}")
        print(f"Resonance curve plot saved to {save_path}")
        
    except Exception as e:
        print(f"Error fitting Lorentzian curve: {e}")
        
        # Even if fit fails, plot the raw data
        plt.figure(figsize=(10, 6))
        plt.scatter(frequencies, amplitudes, s=80, color='blue', edgecolors='black')
        plt.title('Resonance Data (Fitting Failed)', fontsize=16)
        plt.xlabel('Frequency (Hz)', fontsize=14)
        plt.ylabel('Amplitude (m)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add annotations for each data point
        for freq, amp, result in zip(frequencies, amplitudes, results):
            plt.annotate(f"{freq:.3f} Hz", 
                        xy=(freq, amp),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontsize=9)
        
        save_path = os.path.join("results", "plots", "resonance", "resonance_data_raw.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

if __name__ == "__main__":
    analyze_resonance()