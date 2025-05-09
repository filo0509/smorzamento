# Smorzamento: Oscillation Damping Analysis

## Project Overview

This project provides tools for analyzing oscillatory systems with different damping mechanisms, focusing on:
- Linear damping (proportional to velocity)
- Quadratic damping (proportional to velocity squared)

The project includes data collection, simulation generation, and analytical tools to study the physics of damped oscillations.

[View Interactive Plot Gallery](results/gallery/index.html)

## Directory Structure

```
smorzamento/
├── data_raw/        # Original experimental data files
├── data_simulated/  # Generated simulation data files
├── scripts/         # Analysis and data generation scripts
├── results/         # Output from analysis
│   ├── plots/       # Generated visualizations
│   │   ├── raw_data/          # Data visualization plots
│   │   ├── amplitude_decay/   # Amplitude decay analysis
│   │   ├── full_oscillation/  # Full oscillation with envelopes
│   │   ├── residuals/         # Fit residual analysis
│   │   ├── model_comparison/  # Comparing different models
│   │   └── other/             # Additional visualizations
│   └── analysis/    # Analysis results and numerical data
└── docs/            # Documentation files
```

## Data Files

### Raw Data (`data_raw/`)
- `molla_durissima_discopieno_*.csv`: Data from experiments with a solid disk (linear damping)
- `molla_dura_discopieno_*.csv`: Data from different spring constants and masses
- `molla_morbida_discopieno_*.csv`: Data with softer spring constants

### Simulated Data (`data_simulated/`)
- `molla_dura_discoforato_*.csv`: Simulated data for perforated disk (quadratic damping)
- `molla_dura_discoforato_long_*.csv`: Extended 100-second simulations with various noise levels

## Scripts

### Data Generation (`scripts/`)
- `generate_standard_quadratic_damping.py`: Basic quadratic damping simulation
- `generate_multiple_quadratic_damping.py`: Multiple parameter configurations
- `generate_long_quadratic_damping.py`: Long-duration simulations
- `generate_high_noise_quadratic_damping.py`: High-noise simulations

### Analysis Tools (`scripts/`)
- `analyze_exponential_damping.py`: Analysis for linear damping (exp decay)
- `analyze_quadratic_damping.py`: Analysis for quadratic damping (1/t decay)
- `analyze_long_quadratic_damping.py`: Tools for long-duration signals
- `analyze_quadratic_damping_auto.py`: Automated analysis with reduced interaction
- `run_quadratic_damping_analysis.py`: CLI wrapper for analysis scripts

## Physics Background

The project studies two damping models:

1. **Linear Damping** (Solid Disk): `F_d = -c·v`
   - Amplitude follows exponential decay: `A(t) = A₀·e^(-t/τ)`
   
2. **Quadratic Damping** (Perforated Disk): `F_d = -b·v²·sign(v)`
   - Amplitude follows reciprocal decay: `A(t) = A₀/(1 + β·A₀·t)`
   - Damping parameter `β = (8·b·ω)/(3·π·m)`

## Usage

To analyze existing data:
```
python scripts/analyze_quadratic_damping.py
```

To generate new simulated data:
```
python scripts/generate_multiple_quadratic_damping.py
```

For long-duration analysis:
```
python scripts/analyze_long_quadratic_damping.py
```

To generate the interactive visualization gallery:
```
python scripts/generate_bootstrap_gallery_with_template.py
```

## Results

The analysis produces several outputs, organized by type:

### Interactive Bootstrap Gallery
- Professional gallery with Montserrat font
- Filter by plot type and damping model
- Responsive design for all devices
- [Open Gallery](results/gallery/index.html)

### Raw Data (`plots/raw_data/`)
- Position, velocity, and acceleration time series plots
- Common format for all data files

### Amplitude Decay (`plots/amplitude_decay/`)
- Displays amplitude values extracted from peaks and valleys
- Shows both quadratic and linear damping model fits
- Includes R² values for model comparison

### Full Oscillation (`plots/full_oscillation/`)
- Complete oscillation with decay envelope
- Marked peaks and valleys
- Reference lines for half-life or time constant

### Residual Analysis (`plots/residuals/`)
- Side-by-side comparison of model residuals
- Helps identify systematic deviations from models

### Model Comparison (`plots/model_comparison/`)
- Direct comparison of different damping models
- Statistical metrics like R² and AIC values

### Analysis Results (`analysis/`)
- Parameter estimation (spring constant, damping coefficient)
- Numerical values of fitted parameters
- Statistical comparison metrics

## Dependencies

- NumPy
- SciPy
- Pandas
- Matplotlib
- scikit-learn (optional for advanced regression)