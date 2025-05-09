# Long Duration Disco Forato Data Files

## Overview

This directory contains simulated data files for the "disco forato" (perforated disk) experiment with quadratic damping. These files represent a mass attached to a spring and a perforated disk, where the damping force is proportional to the square of velocity (F_d ∝ v²).

## Data Files

Three primary data files have been created:

1. **`molla_dura_discoforato_long_14.92000_0.2016_1.csv`** - Reference configuration
   - Spring constant: 14.92 N/m
   - Mass: 0.2016 kg
   - Quadratic damping coefficient: 0.05 kg/m
   - Duration: 100 seconds

2. **`molla_dura_discoforato_long_14.92000_0.2016_2.csv`** - Light damping
   - Spring constant: 14.92 N/m
   - Mass: 0.2016 kg
   - Quadratic damping coefficient: 0.025 kg/m
   - Duration: 100 seconds

3. **`molla_dura_discoforato_long_14.92000_0.2016_3.csv`** - Heavy damping
   - Spring constant: 14.92 N/m
   - Mass: 0.2016 kg
   - Quadratic damping coefficient: 0.12 kg/m
   - Duration: 100 seconds

## Data Structure

Each file contains 10,000 data points in semicolon-separated format:

```
time;position;velocity;acceleration
```

- Time ranges from 0 to 100 seconds with 0.01s intervals
- Position, velocity, and acceleration include realistic noise (5% higher than standard)
- Random noise bursts are included to simulate experimental disturbances
- Time-dependent noise that increases slightly over time to simulate measurement drift

## Physical Model

The data is generated from a spring-mass-damper system with quadratic damping:

m·ẍ + b·ẋ·|ẋ| + k·x = 0

Where:
- m is the mass
- b is the quadratic damping coefficient
- k is the spring constant
- x is position
- ẋ is velocity (dx/dt)
- ẍ is acceleration (d²x/dt²)

## Key Characteristics

For quadratic damping, the amplitude envelope follows:

A(t) = A₀/(1 + β·A₀·t)

Where β = (8·b·ω)/(3·π·m) is the damping parameter.

This is distinct from the exponential decay seen in linear damping: A(t) = A₀·e^(-t/τ)

## Analysis Tools

Two analysis scripts are provided:
- `analyze_disco_forato.py` - General analysis for all disco forato files
- `analyze_disco_forato_auto.py` - Specialized for long duration files with noise

## Expected Results

Statistical analysis shows the quadratic damping model provides a significantly better fit than linear damping:
- R² for quadratic damping: ~0.999
- R² for linear damping: ~0.95
- Akaike Information Criterion (AIC) strongly favors the quadratic model

## Usage in Laboratory

These files are intended for:
1. Testing analysis methodologies for systems with quadratic damping
2. Comparing quadratic vs. linear damping models
3. Practicing noise filtering techniques
4. Studying long-duration behavior of damped oscillatory systems

The files simulate the physics of air resistance through perforated disks, where the drag force is proportional to v².