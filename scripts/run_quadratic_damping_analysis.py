#!/usr/bin/env python3
import os
import sys
from analyze_quadratic_damping import analyze_disco_forato_data

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        # If filename provided as argument
        filename = sys.argv[1]
        data_dir = "data_simulated"
        
        # Check if file exists in simulated data directory
        if not os.path.exists(os.path.join(data_dir, filename)):
            # Try raw data directory
            data_dir = "data_raw"
            if not os.path.exists(os.path.join(data_dir, filename)):
                print(f"File not found: {filename}")
                print("Available files in data_simulated directory:")
                for file in os.listdir("data_simulated"):
                    if file.endswith(".csv"):
                        print(f" - {file}")
                print("\nAvailable files in data_raw directory:")
                for file in os.listdir("data_raw"):
                    if file.endswith(".csv"):
                        print(f" - {file}")
                sys.exit(1)
        
        # If mass is also provided as argument
        mass = None
        if len(sys.argv) > 2:
            try:
                mass = float(sys.argv[2])
                print(f"Using provided mass: {mass} kg")
            except ValueError:
                print(f"Invalid mass value: {sys.argv[2]}")
                sys.exit(1)
        
        # Run analysis with the specified file
        analyze_disco_forato_data(
            filename=os.path.join(data_dir, filename),
            mass=mass
        )
    else:
        # Run with interactive file selection
        print("Starting quadratic damping analysis...")
        print("The script will help you select a data file.")
        analyze_disco_forato_data()
    
    print("\nAnalysis complete. Check the generated plots in the results/plots directory.")