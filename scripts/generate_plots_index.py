import os
import glob
import datetime

def generate_html_index():
    """Generate an HTML index file for browsing the plots directory."""
    # Define paths
    results_dir = "results"
    plots_dir = os.path.join(results_dir, "plots")
    index_path = os.path.join(results_dir, "plots_index.html")
    
    # Ensure plots directory exists
    if not os.path.exists(plots_dir):
        print(f"Error: Plots directory '{plots_dir}' not found.")
        return
    
    # Get plot categories (subdirectories)
    categories = [d for d in os.listdir(plots_dir) 
                 if os.path.isdir(os.path.join(plots_dir, d))]
    categories.sort()
    
    # Start HTML content
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quadratic and Linear Damping Analysis Plots</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #2980b9;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }
        .category {
            margin-bottom: 40px;
        }
        .plots-container {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }
        .plot-card {
            border: 1px solid #ddd;
            border-radius: 5px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .plot-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .plot-img {
            width: 100%;
            height: 200px;
            object-fit: cover;
            border-bottom: 1px solid #eee;
        }
        .plot-info {
            padding: 10px;
            background-color: #f9f9f9;
        }
        .plot-title {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .plot-desc {
            font-size: 0.9em;
            color: #666;
        }
        a {
            text-decoration: none;
            color: inherit;
        }
        .nav {
            background-color: #2c3e50;
            padding: 10px;
            position: sticky;
            top: 0;
            border-radius: 5px;
            margin-bottom: 20px;
            z-index: 100;
        }
        .nav ul {
            list-style: none;
            display: flex;
            padding: 0;
            margin: 0;
            flex-wrap: wrap;
        }
        .nav li {
            margin-right: 15px;
        }
        .nav a {
            color: white;
            padding: 5px 10px;
            border-radius: 3px;
            transition: background-color 0.2s;
        }
        .nav a:hover {
            background-color: #3498db;
        }
        footer {
            margin-top: 50px;
            text-align: center;
            color: #777;
            font-size: 0.9em;
            border-top: 1px solid #eee;
            padding-top: 20px;
        }
    </style>
</head>
<body>
    <h1>Quadratic and Linear Damping Analysis Plots</h1>
    
    <div class="nav">
        <ul>
            <li><a href="#top">Top</a></li>
"""
    
    # Add navigation links for each category
    for category in categories:
        category_name = category.replace('_', ' ').title()
        html += f'            <li><a href="#{category}">{category_name}</a></li>\n'
    
    html += """        </ul>
    </div>
    
    <p>
        This page provides an overview of all generated plots for the damping analysis project. 
        The plots are organized by type and show comparisons between quadratic damping (perforated disk)
        and linear damping (solid disk) models.
    </p>
"""
    
    # Generate section for each category
    for category in categories:
        category_name = category.replace('_', ' ').title()
        category_path = os.path.join(plots_dir, category)
        plot_files = glob.glob(os.path.join(category_path, "*.png"))
        
        if not plot_files:
            continue
            
        # Sort plot files: linear damping first, then quadratic damping
        plot_files.sort(key=lambda x: 
            "0" + os.path.basename(x) if "linear_damping" in os.path.basename(x) else 
            "1" + os.path.basename(x))
        
        html += f"""
    <div class="category" id="{category}">
        <h2>{category_name} ({len(plot_files)} plots)</h2>
        <div class="plots-container">
"""
        
        # Add each plot in this category
        for plot_file in plot_files:
            rel_path = os.path.relpath(plot_file, results_dir)
            filename = os.path.basename(plot_file)
            
            # Generate a nice display name
            display_name = filename.replace('.png', '').replace('_', ' ').title()
            
            # Determine if it's quadratic or linear damping
            damping_type = "Linear Damping"
            if "quadratic" in filename:
                damping_type = "Quadratic Damping"
            
            # Extract any numbering from the filename
            plot_number = ""
            for part in filename.split('_'):
                if part.isdigit():
                    plot_number = f" #{part}"
                    break
            
            html += f"""
            <a href="{rel_path}" target="_blank" class="plot-card">
                <img src="{rel_path}" alt="{display_name}" class="plot-img">
                <div class="plot-info">
                    <div class="plot-title">{display_name}</div>
                    <div class="plot-desc">{damping_type}{plot_number}</div>
                </div>
            </a>
"""
        
        html += """        </div>
    </div>
"""
    
    # Add footer
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html += f"""
    <footer>
        <p>Generated on {current_time}</p>
        <p>Smorzamento Project - Quadratic and Linear Damping Analysis</p>
    </footer>
</body>
</html>
"""
    
    # Write the HTML file
    with open(index_path, "w") as f:
        f.write(html)
    
    print(f"HTML index generated at: {index_path}")
    return index_path

if __name__ == "__main__":
    generate_html_index()