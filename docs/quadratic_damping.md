# Disco Forato (Perforated Disk) Analysis

This analysis script examines data from an oscillating system with a perforated disk that produces quadratic damping (proportional to v²).

## Physics Background

In this experiment, a mass is attached to a spring and a perforated disk that moves through air. The perforated disk creates a damping force that is proportional to the square of velocity:

F_d = -b·v²·sign(v)

This is different from linear damping (F_d = -c·v) seen in the "disco pieno" (full disk) experiment. Quadratic damping is typical for:
- Air resistance at higher velocities
- Turbulent fluid resistance
- Systems where energy dissipation scales with kinetic energy

## Mathematical Model

For a system with quadratic damping, the amplitude decay follows:

A(t) = A₀/(1 + β·A₀·t)

Where:
- A₀ is the initial amplitude
- β is the damping parameter
- t is the time from the start of decay

The damping parameter β is related to the physical damping coefficient b by:

β = (8·b·ω)/(3·π·m)

Where:
- b is the quadratic damping coefficient (kg/m)
- ω is the angular frequency (rad/s)
- m is the oscillating mass (kg)

## Key Metrics

The script calculates:
- Initial amplitude (A₀)
- Damping parameter (β)
- Half-life of oscillation (time for amplitude to reach A₀/2)
- Quadratic damping coefficient (b)
- Spring constant (k)
- Angular frequency (ω)
- Goodness of fit metrics (R², chi-squared)

## Comparison with Linear Damping

For comparison, the script also fits the data to a linear damping model:
A(t) = A₀·e^(-t/τ)

The script compares both models to determine which better describes the data.

## Usage

```python
python analyze_disco_forato.py
```

If no explicit "disco forato" files are found, the script will list available data files and prompt for selection.

## Output

The script generates several visualization plots:
- Raw oscillation data
- Amplitude decay with fitted quadratic damping envelope
- Residuals of the fit
- Full oscillation model fit
- Comparison between quadratic and linear damping models
- Velocity-squared vs. acceleration analysis (to verify the v² relationship)

All plots are saved as PNG files for further analysis and reporting.

## Requirements

- NumPy
- SciPy
- Pandas
- Matplotlib
- scikit-learn (optional, for additional regression analysis)