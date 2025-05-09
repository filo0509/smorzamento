import os
import glob
import datetime

def generate_gallery():
    """Generate a simple but elegant HTML gallery for the plot files."""
    # Define paths
    results_dir = "results"
    plots_dir = os.path.join(results_dir, "plots")
    index_path = os.path.join(results_dir, "gallery.html")
    
    # Ensure plots directory exists
    if not os.path.exists(plots_dir):
        print(f"Error: Plots directory '{plots_dir}' not found.")
        return
    
    # Get all category directories
    categories = [d for d in os.listdir(plots_dir) 
                 if os.path.isdir(os.path.join(plots_dir, d))]
    categories.sort()
    
    # HTML header and styles
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Damping Analysis Gallery</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Roboto', sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f8fa;
        }
        
        header {
            background: linear-gradient(135deg, #4a6fa5, #166088);
            color: white;
            padding: 2rem 0;
            text-align: center;
        }
        
        h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            font-weight: 300;
            max-width: 800px;
            margin: 0 auto;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 2rem;
        }
        
        nav {
            background: white;
            border-radius: 8px;
            padding: 1rem;
            margin: 1.5rem 0;
            position: sticky;
            top: 1rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            z-index: 100;
        }
        
        .nav-links {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 1rem;
        }
        
        .nav-link {
            color: #166088;
            text-decoration: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            transition: all 0.2s;
        }
        
        .nav-link:hover, .nav-link.active {
            background-color: #166088;
            color: white;
        }
        
        section {
            margin-bottom: 3rem;
        }
        
        h2 {
            color: #166088;
            margin-bottom: 1rem;
            font-size: 1.8rem;
            border-bottom: 2px solid #eaeaea;
            padding-bottom: 0.5rem;
        }
        
        .description {
            margin-bottom: 1.5rem;
            color: #666;
        }
        
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 1.5rem;
        }
        
        .gallery-item {
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }
        
        .gallery-item:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        }
        
        .gallery-img {
            width: 100%;
            height: 220px;
            object-fit: cover;
        }
        
        .gallery-caption {
            padding: 1rem;
        }
        
        .gallery-title {
            font-weight: 500;
            margin-bottom: 0.5rem;
            font-size: 1.1rem;
        }
        
        .gallery-meta {
            display: flex;
            justify-content: space-between;
            color: #666;
            font-size: 0.9rem;
        }
        
        .badge {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            color: white;
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        .badge-linear {
            background-color: #6a4c93;
        }
        
        .badge-quadratic {
            background-color: #f72585;
        }
        
        footer {
            background: #166088;
            color: white;
            text-align: center;
            padding: 2rem 0;
            margin-top: 2rem;
        }
        
        .footer-content {
            font-size: 0.9rem;
            opacity: 0.8;
        }
        
        @media (max-width: 768px) {
            .gallery {
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            }
            
            h1 {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>Damping Analysis Visualization Gallery</h1>
            <p class="subtitle">
                An interactive gallery for comparing linear and quadratic damping models in mechanical oscillations
            </p>
        </div>
    </header>
    
    <div class="container">
        <nav>
            <div class="nav-links">
                <a href="#" class="nav-link active" data-category="all">All</a>
"""
    
    # Add navigation links
    for category in categories:
        category_display = category.replace('_', ' ').title()
        html += f'                <a href="#{category}" class="nav-link" data-category="{category}">{category_display}</a>\n'
    
    html += """            </div>
        </nav>
        
"""
    
    # Category descriptions
    descriptions = {
        'raw_data': 'Position, velocity, and acceleration time series data from the original oscillation experiments.',
        'amplitude_decay': 'Analysis of how oscillation amplitude decreases over time under different damping models.',
        'full_oscillation': 'Complete oscillation visualizations showing the damping envelope over time.',
        'residuals': 'Differences between measured data points and model predictions to evaluate fit quality.',
        'model_comparison': 'Direct comparison between linear and quadratic damping models with statistical metrics.',
        'other': 'Additional analysis plots and visualizations for the oscillation experiments.'
    }
    
    # Generate sections for each category
    for category in categories:
        category_path = os.path.join(plots_dir, category)
        plot_files = glob.glob(os.path.join(category_path, "*.png"))
        
        if not plot_files:
            continue
        
        # Sort plots (linear damping first, then quadratic)
        plot_files.sort(key=lambda x: 
            "0" + os.path.basename(x) if "linear" in os.path.basename(x).lower() else 
            "1" + os.path.basename(x))
        
        # Get category description
        category_display = category.replace('_', ' ').title()
        description = descriptions.get(category, 'Visualization plots for oscillation analysis.')
        
        html += f"""        <section id="{category}">
            <h2>{category_display} ({len(plot_files)})</h2>
            <p class="description">{description}</p>
            
            <div class="gallery" data-category="{category}">
"""
        
        # Add each plot in this category
        for plot_file in plot_files:
            rel_path = os.path.relpath(plot_file, results_dir)
            filename = os.path.basename(plot_file)
            display_name = filename.replace('.png', '').replace('_', ' ').title()
            
            # Determine damping type
            damping_type = "Linear"
            badge_class = "badge-linear"
            if "quadratic" in filename.lower():
                damping_type = "Quadratic"
                badge_class = "badge-quadratic"
            
            # Extract number if present
            plot_num = ""
            for part in filename.split('_'):
                if part.isdigit():
                    plot_num = f"#{part}"
                    break
            
            html += f"""                <a href="{rel_path}" target="_blank" class="gallery-item" data-type="{damping_type.lower()}">
                    <img src="{rel_path}" alt="{display_name}" class="gallery-img">
                    <div class="gallery-caption">
                        <div class="gallery-title">{display_name}</div>
                        <div class="gallery-meta">
                            <span class="badge {badge_class}">{damping_type}</span>
                            <span>{plot_num}</span>
                        </div>
                    </div>
                </a>
"""
        
        html += """            </div>
        </section>
        
"""
    
    # Add footer and JavaScript
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    html += f"""    </div>
    
    <footer>
        <div class="container">
            <div class="footer-content">
                <p>Smorzamento Project - Quadratic & Linear Damping Analysis</p>
                <p>Generated on {current_date}</p>
            </div>
        </div>
    </footer>
    
    <script>
        // Navigation functionality
        document.querySelectorAll('.nav-link').forEach(link => {{
            link.addEventListener('click', function(e) {{
                e.preventDefault();
                
                // Update active class
                document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
                this.classList.add('active');
                
                const category = this.getAttribute('data-category');
                
                // Show all items or just selected category
                document.querySelectorAll('.gallery-item').forEach(item => {{
                    if (category === 'all') {{
                        item.style.display = 'block';
                    }} else {{
                        const parentGallery = item.closest('.gallery');
                        if (parentGallery && parentGallery.getAttribute('data-category') === category) {{
                            item.style.display = 'block';
                        }} else {{
                            item.style.display = 'none';
                        }}
                    }}
                }});
                
                // Scroll to section if applicable
                if (category !== 'all') {{
                    const section = document.getElementById(category);
                    if (section) {{
                        section.scrollIntoView({{ behavior: 'smooth' }});
                    }}
                }}
            }});
        }});
    </script>
</body>
</html>
"""
    
    # Write the HTML file
    with open(index_path, "w") as f:
        f.write(html)
    
    print(f"Simple gallery generated at: {index_path}")
    return index_path

if __name__ == "__main__":
    generate_gallery()