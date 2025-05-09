# Quadratic vs. Linear Damping

## Physical Models

When studying oscillatory systems with damping, two models are commonly encountered:

### Linear Damping

**Differential equation**: $m\ddot{x} + c\dot{x} + kx = 0$

**Damping force**: $F_d = -c\dot{x}$ (proportional to velocity)

**Solution**: $x(t) = A_0 e^{-\frac{t}{2\tau}} \cos(\omega' t + \phi)$

**Amplitude decay**: $A(t) = A_0 e^{-\frac{t}{2\tau}}$

Where:
- $\tau = \frac{m}{c}$ is the characteristic time
- $\omega' = \sqrt{\omega_0^2 - \frac{1}{4\tau^2}}$ is the damped angular frequency
- $\omega_0 = \sqrt{\frac{k}{m}}$ is the natural angular frequency

### Quadratic Damping

**Differential equation**: $m\ddot{x} + b|\dot{x}|\dot{x} + kx = 0$

**Damping force**: $F_d = -b|\dot{x}|\dot{x}$ (proportional to velocity squared)

**Amplitude decay**: $A(t) = \frac{A_0}{1 + \beta A_0 t}$

Where:
- $\beta = \frac{8b\omega_0}{3\pi m}$ is the damping parameter
- $b$ is the quadratic damping coefficient

## Experimental Setups

### Linear Damping ("Disco Pieno")
- Solid disk moving through air
- Primarily viscous drag at low velocities
- Exponential amplitude decay
- Examples: small pendulums, electrical RLC circuits

### Quadratic Damping ("Disco Forato") 
- Perforated disk moving through air
- Air resistance dominates at higher velocities
- Reciprocal amplitude decay
- Examples: large pendulums, objects falling through fluid

## Model Comparison

| Feature | Linear Damping | Quadratic Damping |
|---------|----------------|-------------------|
| Force relation | $F \propto v$ | $F \propto v^2$ |
| Amplitude decay | Exponential | Reciprocal |
| Energy dissipation | Constant fraction per cycle | Proportional to kinetic energy |
| Half-life | Independent of initial amplitude | Inversely proportional to initial amplitude |
| Appearance in nature | Low Reynolds number flows | High Reynolds number flows |

## Distinguishing Between Models

### Statistical Analysis
When fitting data to both models, several metrics help determine which model is more appropriate:
- R² value (higher is better)
- Residual analysis (should be random, not systematic)
- Akaike Information Criterion (lower AIC indicates better model)

### Velocity-Acceleration Relationship
- For linear damping: acceleration component from damping ∝ v
- For quadratic damping: acceleration component from damping ∝ v²

### Amplitude-Time Relationship
- Linear (exponential): ln(A) vs t will be linear
- Quadratic (reciprocal): 1/A vs t will be linear

## Practical Implications

Understanding the correct damping model is crucial for:
1. Accurate prediction of system behavior
2. Correct extrapolation of future states
3. Proper physical interpretation of the damping mechanism
4. Accurate calculation of energy dissipation
5. Proper design of control systems

## Analysis Methodology

Our analysis approach involves:
1. Data preprocessing and noise filtering
2. Peak and amplitude detection
3. Fitting both decay models to amplitude data
4. Statistical comparison of model performance
5. Physical parameter extraction
6. Validation through v² relationship testing

The scripts in this project provide a comprehensive toolset for analyzing both types of damping and determining which model best describes a given oscillatory system.