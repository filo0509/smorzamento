import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# Load the data
filename = 'data_raw/molla_durissima_discopieno_14.91945_0.2016_1.csv'
df = pd.read_csv(filename, sep=';')
t = df['time'].values
x = df['position'].values

# Plot raw data
plt.figure(figsize=(12, 6))
plt.plot(t, x, 'b-', label='Position Data')
plt.title('Raw Oscillation Data')
plt.xlabel('Time (s)')
plt.ylabel('Position')
plt.grid(True)
plt.legend(loc='upper right')
plt.savefig('results/plots/exponential_damping_raw_data.png')
plt.show()

# Determine equilibrium position (mean of the signal)
x_equilibrium = np.mean(x)
x_centered = x - x_equilibrium

# Find peaks and valleys for the centered data
peaks, _ = find_peaks(x_centered, height=0.001, distance=10)  # Adjust these parameters as needed
valleys, _ = find_peaks(-x_centered, height=0.001, distance=10)

# Extract peak and valley data
peaks_t = t[peaks]
peaks_x = x_centered[peaks]
valleys_t = t[valleys]
valleys_x = x_centered[valleys]

# Function to fit exponential decay of amplitude
def exponential_decay(t, A0, tau):
    return A0 * np.exp(-t/tau)

# Convert to absolute amplitude values for fitting
peaks_amplitude = np.abs(peaks_x)
valleys_amplitude = np.abs(valleys_x)

# Combine all extrema for a better fit
all_extrema_t = np.concatenate((peaks_t, valleys_t))
all_extrema_amp = np.concatenate((peaks_amplitude, valleys_amplitude))

# Sort by time
sort_idx = np.argsort(all_extrema_t)
all_extrema_t = all_extrema_t[sort_idx]
all_extrema_amp = all_extrema_amp[sort_idx]

# Fit the decay envelope
try:
    # Initial guess: [initial amplitude, time constant]
    initial_guess = [np.max(all_extrema_amp), (all_extrema_t[-1] - all_extrema_t[0])/5]
    params, params_covariance = curve_fit(exponential_decay, all_extrema_t - all_extrema_t[0], all_extrema_amp, p0=initial_guess, maxfev=10000)

    A0, tau = params
    print(f"Initial amplitude (A₀): {A0:.6f}")
    print(f"Characteristic time (τ): {tau:.6f} seconds")
    print(f"This is the time it takes for the amplitude to decrease to 1/e (piò o meno 36.8%) of its initial value.")

    # Calculate damping coefficient
    mass = 0.2016  # Given in the file name
    gamma = 2 * mass / tau
    print(f"Damping coefficient (γ): {gamma:.6f} kg/s")

    # Chi-Squared Test for goodness of fit
    # Calculate predicted values using our fitted model
    predicted = exponential_decay(all_extrema_t - all_extrema_t[0], A0, tau)

    # Calculate chi-squared statistic
    # χ² = Σ((observed - expected)²/expected)
    # For amplitude data without known uncertainties, we can estimate uncertainty as sqrt(data)
    # or use a constant uncertainty estimate
    residuals = all_extrema_amp - predicted
    n_params = 2  # A0 and tau
    n_data = len(all_extrema_amp)
    dof = n_data - n_params  # degrees of freedom

    # We'll use two approaches for chi-squared:

    # 1. Using data value as uncertainty estimate (common in particle counting)
    chi_squared_1 = np.sum(residuals**2 / predicted)
    reduced_chi_squared_1 = chi_squared_1 / dof

    # 2. Using standard deviation of residuals as constant uncertainty
    sigma = np.std(residuals)
    chi_squared_2 = np.sum((residuals / sigma)**2)
    reduced_chi_squared_2 = chi_squared_2 / dof

    print("Chi-Squared Test Results:")
    print(f"Method 1 - Using data as uncertainty:")
    print(f"Chi-squared: {chi_squared_1:.4f}")
    print(f"Reduced chi-squared: {reduced_chi_squared_1:.4f}")
    print(f"Degrees of freedom: {dof}")

    print("Method 2 - Using constant uncertainty (std of residuals):")
    print(f"Chi-squared: {chi_squared_2:.4f}")
    print(f"Reduced chi-squared: {reduced_chi_squared_2:.4f}")

    # Interpret reduced chi-squared
    if 0.8 <= reduced_chi_squared_2 <= 1.2:
        interpretation = "excellent fit (χ²/dof ≈ 1)"
    elif 0.5 <= reduced_chi_squared_2 <= 1.5:
        interpretation = "good fit (0.5 < χ²/dof < 1.5)"
    elif reduced_chi_squared_2 < 0.5:
        interpretation = "potentially overfitting (χ²/dof < 0.5)"
    else:
        interpretation = "poor fit (chi^2/dof > 1.5) - model may not fully describe the data"

    print(f"Interpretation: {interpretation}")

    # Add R-squared value (coefficient of determination)
    ss_total = np.sum((all_extrema_amp - np.mean(all_extrema_amp))**2)
    ss_residual = np.sum(residuals**2)
    r_squared = 1 - (ss_residual / ss_total)
    print(f"R-squared: {r_squared:.6f}")

except Exception as e:
    print(f"Error fitting exponential decay: {e}")
    tau = None
    A0 = None

# Plot with the fitted envelope
plt.figure(figsize=(12, 8))
plt.plot(t, x_centered, 'b-', alpha=0.5, label='Centered Position Data')
plt.plot(peaks_t, peaks_x, 'ro', markersize=4, label='Peaks')
plt.plot(valleys_t, valleys_x, 'go', markersize=4, label='Valleys')

if tau is not None:
    # Generate the decay envelope
    t_fit = np.linspace(all_extrema_t[0], all_extrema_t[-1], 1000)
    upper_envelope = exponential_decay(t_fit - all_extrema_t[0], A0, tau)
    lower_envelope = -upper_envelope

    plt.plot(t_fit, upper_envelope, 'r-', linewidth=2, label=f'Exponential Envelope (τ = {tau:.4f}s)')
    plt.plot(t_fit, lower_envelope, 'r-', linewidth=2, label='Lower Envelope')

    # Mark the characteristic time (tau) on the plot
    t_at_tau = all_extrema_t[0] + tau
    amp_at_tau = A0 / np.e
    plt.axvline(x=t_at_tau, color='k', linestyle='--', label=f'τ = {tau:.4f}s')
    plt.axhline(y=amp_at_tau, color='m', linestyle=':', label='A₀/e threshold')
    plt.plot([all_extrema_t[0], t_at_tau], [A0, amp_at_tau], 'k-', linewidth=1.5, label='Decay to A₀/e')
    plt.annotate(f'τ = {tau:.4f}s', xy=(t_at_tau, amp_at_tau), xytext=(t_at_tau + 0.1, amp_at_tau * 1.1),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)

    # Plot residuals as small markers to visualize goodness of fit
    predicted = exponential_decay(all_extrema_t - all_extrema_t[0], A0, tau)
    residuals = all_extrema_amp - predicted
    plt.figure(figsize=(12, 4))
    plt.scatter(all_extrema_t, residuals, color='purple', s=20, alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.title('Residuals of Exponential Decay Fit')
    plt.xlabel('Time (s)')
    plt.ylabel('Residual (Observed - Predicted)')
    plt.grid(True, alpha=0.3)
    plt.savefig('results/plots/exponential_decay_residuals.png')
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
    plt.savefig('results/plots/exponential_observed_vs_predicted.png')
    plt.show()

    # Add annotations
    plt.annotate(f'A₀ = {A0:.4f}',
                xy=(all_extrema_t[0], A0),
                xytext=(all_extrema_t[0]+0.1, A0*1.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

    plt.annotate(f'A₀/e = {amp_at_tau:.4f}',
                xy=(t_at_tau, amp_at_tau),
                xytext=(t_at_tau+0.1, amp_at_tau*1.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

plt.figure(figsize=(12, 8))
plt.plot(t, x_centered, 'b-', alpha=0.5, label='Centered Position Data')
plt.plot(peaks_t, peaks_x, 'ro', markersize=4, label='Peaks')
plt.plot(valleys_t, valleys_x, 'go', markersize=4, label='Valleys')

if tau is not None:
    t_fit = np.linspace(all_extrema_t[0], all_extrema_t[-1], 1000)
    upper_envelope = exponential_decay(t_fit - all_extrema_t[0], A0, tau)
    lower_envelope = -upper_envelope
    plt.plot(t_fit, upper_envelope, 'r-', linewidth=2, label=f'Exponential Envelope (τ = {tau:.4f}s)')
    plt.plot(t_fit, lower_envelope, 'r-', linewidth=2)

plt.title('Damped Oscillation Analysis')
plt.xlabel('Time (s)')
plt.ylabel('Position (centered)')
plt.grid(True)
plt.legend()
plt.savefig('results/plots/exponential_damping_analysis.png')
plt.show()

# già che ci siamo...
def damped_oscillation(t, A, tau, omega, phi, offset):
    return A * np.exp(-(t-t[0])/tau) * np.cos(omega * (t-t[0]) + phi) + offset

try:
    # Estimate period from time between peaks
    if len(peaks_t) > 1:
        T = np.mean(np.diff(peaks_t))
    else:
        T = 0.7  # Fallback estimate

    omega_guess = 2 * np.pi / T

    # Initial parameters: [amplitude, tau, angular frequency, phase, offset]
    initial_params = [
        np.max(np.abs(x_centered)),  # amplitude
        tau if tau is not None else 10.0,  # damping time
        omega_guess,  # angular frequency
        0,  # phase
        x_equilibrium  # offset
    ]

    # Fit only a portion of the data to improve fit quality (first few oscillations)
    fit_length = min(2000, len(t))  # Adjust based on your data

    params_full, _ = curve_fit(
        damped_oscillation,
        t[:fit_length],
        x[:fit_length],
        p0=initial_params,
        maxfev=10000
    )

    A_full, tau_full, omega_full, phi_full, offset_full = params_full

    print("Full model parameters:")
    print(f"Amplitude (A): {A_full:.6f}")
    print(f"Damping time (τ): {tau_full:.6f} seconds")
    print(f"Angular frequency (ω): {omega_full:.6f} rad/s")
    print(f"Frequency (f): {omega_full/(2*np.pi):.6f} Hz")
    print(f"Period (T): {2*np.pi/omega_full:.6f} seconds")
    print(f"Phase (φ): {phi_full:.6f} radians")
    print(f"Offset: {offset_full:.6f}")

    # Calculate spring constant
    k = mass * omega_full**2
    print(f"Spring constant (k): {k:.6f} N/m")

    # Calculate critical damping coefficient
    gamma_critical = 2 * mass * omega_full
    print(f"Critical damping coefficient: {gamma_critical:.6f} kg/s")

    # Calculate damping ratio
    if tau_full > 0:
        gamma_actual = 2 * mass / tau_full
        damping_ratio = gamma_actual / gamma_critical
        print(f"Damping ratio (ζ): {damping_ratio:.6f}")
        print(f"System is {'under' if damping_ratio < 1 else 'over'}damped")

    # Plot the full fitted model
    plt.figure(figsize=(12, 8))
    plt.plot(t, x, 'b-', alpha=0.5, label='Position Data')

    t_fit = np.linspace(t[0], t[fit_length-1], 1000)
    x_fit = damped_oscillation(t_fit, A_full, tau_full, omega_full, phi_full, offset_full)
    plt.plot(t_fit, x_fit, 'r-', linewidth=2, label='Fitted Model')

    plt.title('Damped Oscillation - Full Model Fit')
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/exponential_damping_full_fit.png')
    plt.show()

except Exception as e:
    print(f"Could not fit full model: {e}")
