import os
import shutil
import glob

def cleanup_plots():
    """
    Clean up the plots directory by:
    1. Removing inconsistently named files
    2. Organizing plots into subdirectories by type
    3. Ensuring consistent naming conventions
    """
    print("Cleaning up plots directory...")
    
    # Ensure plots directory exists
    plots_dir = "results/plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir, exist_ok=True)
    
    # Create subdirectories for different plot types
    plot_types = ["raw_data", "amplitude_decay", "full_oscillation", "residuals", "model_comparison", "other"]
    subdirs = {}
    
    for plot_type in plot_types:
        subdir = os.path.join(plots_dir, plot_type)
        os.makedirs(subdir, exist_ok=True)
        subdirs[plot_type] = subdir
    
    # Get all files in the plots directory
    all_files = glob.glob(os.path.join(plots_dir, "*.png"))
    
    # Files to keep as is (standardized files)
    standardized_patterns = [
        "linear_damping_*",
        "quadratic_damping_*"
    ]
    
    # Identify standardized files to keep
    standardized_files = []
    for pattern in standardized_patterns:
        standardized_files.extend(glob.glob(os.path.join(plots_dir, pattern)))
    
    # Non-standardized files to remove
    files_to_remove = []
    for file in all_files:
        if file not in standardized_files:
            files_to_remove.append(file)
    
    # Remove non-standardized files
    for file in files_to_remove:
        try:
            os.remove(file)
            print(f"Removed inconsistent file: {os.path.basename(file)}")
        except Exception as e:
            print(f"Error removing {os.path.basename(file)}: {e}")
    
    # Organize standardized files into subdirectories
    for file in standardized_files:
        basename = os.path.basename(file)
        
        # Determine destination subdirectory based on filename
        destination_subdir = None
        if "_raw_data" in basename:
            destination_subdir = subdirs["raw_data"]
        elif "_amplitude_decay" in basename:
            destination_subdir = subdirs["amplitude_decay"]
        elif "_full_oscillation" in basename:
            destination_subdir = subdirs["full_oscillation"]
        elif "_residuals" in basename:
            destination_subdir = subdirs["residuals"]
        elif "_model_comparison" in basename:
            destination_subdir = subdirs["model_comparison"]
        else:
            destination_subdir = subdirs["other"]
        
        # Move the file
        destination = os.path.join(destination_subdir, basename)
        try:
            shutil.move(file, destination)
            print(f"Moved {basename} to {os.path.relpath(destination_subdir, 'results')}")
        except Exception as e:
            print(f"Error moving {basename}: {e}")
    
    print("\nCleanup complete! Files organized into the following categories:")
    for plot_type, subdir in subdirs.items():
        count = len(glob.glob(os.path.join(subdir, "*.png")))
        print(f"  - {plot_type}: {count} files")

if __name__ == "__main__":
    cleanup_plots()