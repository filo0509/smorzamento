# Setup Instructions for Smorzamento Project

## Prerequisites

- Python 3.8 or newer
- Git
- A virtual environment manager (optional but recommended)

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/smorzamento.git
   cd smorzamento
   ```

2. **Create and activate a virtual environment (optional but recommended)**

   Using venv:
   ```bash
   python -m venv env
   ```

   - On Windows:
     ```bash
     .\env\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source env/bin/activate
     ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Verifying Installation

You can verify that everything is installed correctly by running one of the analysis scripts:

```bash
python scripts/analyze_quadratic_damping.py
```

If the script runs without errors and generates plots in the `results/plots` directory, the installation was successful.

## Project Organization

The project is organized as follows:

- `data_raw/`: Contains the original experimental data files
- `data_simulated/`: Contains generated simulation data
- `scripts/`: Contains all analysis and data generation scripts
- `results/`: Contains output from analysis scripts (plots and analysis results)
- `docs/`: Contains documentation

## Running the Analysis

The project includes several analysis scripts:

1. **Basic Analysis**

   ```bash
   # For quadratic damping analysis
   python scripts/analyze_quadratic_damping.py
   
   # For linear damping analysis
   python scripts/analyze_exponential_damping.py
   ```

2. **Resonance Analysis**

   ```bash
   # Basic resonance analysis
   python scripts/analyze_resonance.py
   
   # Enhanced resonance analysis
   python scripts/analyze_resonance_enhanced.py
   ```

3. **Data Generation**

   ```bash
   # Generate simulated data for quadratic damping
   python scripts/generate_multiple_quadratic_damping.py
   ```

4. **Visualization Gallery**

   ```bash
   # Launch the visualization gallery in a web browser
   python view_gallery.py
   ```

## Troubleshooting

If you encounter any issues:

1. Make sure all dependencies are correctly installed:
   ```bash
   pip install -r requirements.txt --upgrade
   ```

2. Check that the project directories are correctly structured:
   ```bash
   ls -la
   ```

3. For plotting issues, ensure that matplotlib is correctly configured for your environment.

4. For data loading issues, check that the CSV files are in the correct directories and have the expected format.