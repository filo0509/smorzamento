import os
import glob
import datetime

def generate_bootstrap_gallery():
    """Generate a professional Bootstrap-powered gallery with Montserrat font."""
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
    
    # Count plots
    total_plots = 0
    linear_count = 0
    quadratic_count = 0
    
    for category in categories:
        category_path = os.path.join(plots_dir, category)
        plot_files = glob.glob(os.path.join(category_path, "*.png"))
        total_plots += len(plot_files)
        
        for plot_file in plot_files:
            filename = os.path.basename(plot_file).lower()
            if "linear" in filename:
                linear_count += 1
            elif "quadratic" in filename:
                quadratic_count += 1
    
    # Category descriptions and icons
    category_info = {
        'raw_data': {
            'icon': 'bi-graph-up',
            'desc': 'Time series data showing position, velocity, and acceleration measurements.'
        },
        'amplitude_decay': {
            'icon': 'bi-reception-4',
            'desc': 'Analysis of oscillation amplitude reduction over time across different models.'
        },
        'full_oscillation': {
            'icon': 'bi-activity',
            'desc': 'Complete oscillation visualizations with the decay envelope.'
        },
        'residuals': {
            'icon': 'bi-distribute-vertical',
            'desc': 'Error analysis between measured data and model predictions.'
        },
        'model_comparison': {
            'icon': 'bi-check2-square',
            'desc': 'Direct comparison between linear and quadratic damping models.'
        },
        'other': {
            'icon': 'bi-pie-chart-fill',
            'desc': 'Additional visualization and analysis plots.'
        }
    }
    
    # Build HTML content
    html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Damping Analysis Gallery</title>
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Bootstrap Icons -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
    <!-- Google Fonts - Montserrat -->
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Montserrat', sans-serif;
            background-color: #f8f9fa;
        }
        .hero-section {
            background: linear-gradient(120deg, #1e3c72, #2a5298);
            color: white;
            padding: 4rem 0;
        }
        .hero-title {
            font-weight: 700;
        }
        .hero-subtitle {
            font-weight: 300;
        }
        .stats-card {
            background-color: rgba(255, 255, 255, 0.15);
            border-radius: 10px;
            padding: 1.5rem;
            transition: transform 0.3s;
            backdrop-filter: blur(5px);
        }
        .stats-card:hover {
            transform: translateY(-5px);
        }
        .category-section {
            background: white;
            border-radius: 10px;
            box-shadow: 0 3px 15px rgba(0,0,0,0.05);
            margin-bottom: 2rem;
        }
        .plot-card {
            height: 100%;
            transition: transform 0.3s, box-shadow 0.3s;
            border: none;
            overflow: hidden;
        }
        .plot-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }
        .plot-img-container {
            position: relative;
            overflow: hidden;
            padding-top: 75%;
        }
        .plot-img {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.3s;
        }
        .plot-card:hover .plot-img {
            transform: scale(1.05);
        }
        .plot-badge {
            position: absolute;
            top: 10px;
            right: 10px;
        }
        .linear-badge {
            background-color: #6f42c1;
        }
        .quadratic-badge {
            background-color: #e83e8c;
        }
        footer {
            background: #212529;
            color: white;
            padding: 3rem 0;
            margin-top: 3rem;
        }
    </style>
</head>
<body>
    <!-- Navbar -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="#">Smorzamento</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link active" href="#">Gallery</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="../docs/quadratic_vs_linear_damping.md">Documentation</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <!-- Hero Section -->
    <section class="hero-section">
        <div class="container text-center">
            <h1 class="hero-title display-4 mb-3">Oscillation Analysis Gallery</h1>
            <p class="hero-subtitle lead mx-auto" style="max-width: 700px;">
                Visual exploration of linear and quadratic damping in mechanical oscillatory systems
            </p>
            
            <div class="row justify-content-center mt-5">
                <div class="col-md-3 col-6 mb-4">
                    <div class="stats-card">
                        <div class="h1 mb-2">''' + str(total_plots) + '''</div>
                        <div class="text-uppercase small">Total Plots</div>
                    </div>
                </div>
                <div class="col-md-3 col-6 mb-4">
                    <div class="stats-card">
                        <div class="h1 mb-2">''' + str(len(categories)) + '''</div>
                        <div class="text-uppercase small">Categories</div>
                    </div>
                </div>
                <div class="col-md-3 col-6 mb-4">
                    <div class="stats-card">
                        <div class="h1 mb-2">''' + str(linear_count) + '''</div>
                        <div class="text-uppercase small">Linear Damping</div>
                    </div>
                </div>
                <div class="col-md-3 col-6 mb-4">
                    <div class="stats-card">
                        <div class="h1 mb-2">''' + str(quadratic_count) + '''</div>
                        <div class="text-uppercase small">Quadratic Damping</div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Main Content -->
    <section class="py-5">
        <div class="container">
            <!-- Navigation Pills -->
            <div class="d-flex justify-content-center mb-4">
                <ul class="nav nav-pills">
                    <li class="nav-item">
                        <button class="nav-link active" data-bs-toggle="pill" data-bs-target="#all-tab">All</button>
                    </li>'''
    
    # Add navigation tabs
    for category in categories:
        display_name = category.replace('_', ' ').title()
        html += f'''
                    <li class="nav-item">
                        <button class="nav-link" data-bs-toggle="pill" data-bs-target="#{category}-tab">{display_name}</button>
                    </li>'''

    html += '''
                </ul>
            </div>
            
            <!-- Filter Bar -->
            <div class="bg-white p-3 rounded shadow-sm mb-4">
                <div class="row align-items-center">
                    <div class="col-md-4">
                        <h5 class="mb-0">Filter By:</h5>
                    </div>
                    <div class="col-md-8">
                        <div class="d-flex gap-3">
                            <div class="form-check form-check-inline">
                                <input class="form-check-input" type="checkbox" id="filterLinear" checked>
                                <label class="form-check-label" for="filterLinear">
                                    <span class="badge linear-badge">Linear</span>
                                </label>
                            </div>
                            <div class="form-check form-check-inline">
                                <input class="form-check-input" type="checkbox" id="filterQuadratic" checked>
                                <label class="form-check-label" for="filterQuadratic">
                                    <span class="badge quadratic-badge">Quadratic</span>
                                </label>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Tab Content -->
            <div class="tab-content">
                <!-- All Plots Tab -->
                <div class="tab-pane fade show active" id="all-tab">
                    <div class="row">'''
    
    # Add all plots
    all_plots = []
    for category in categories:
        category_path = os.path.join(plots_dir, category)
        for plot_file in glob.glob(os.path.join(category_path, "*.png")):
            all_plots.append((category, plot_file))
    
    # Process all plots
    for category, plot_file in all_plots:
        rel_path = os.path.relpath(plot_file, results_dir)
        filename = os.path.basename(plot_file)
        display_name = filename.replace('.png', '').replace('_', ' ').title()
        
        # Determine type
        damping_type = "linear"
        badge_class = "linear-badge"
        if "quadratic" in filename.lower():
            damping_type = "quadratic"
            badge_class = "quadratic-badge"
            
        html += f'''
                        <div class="col-md-6 col-lg-4 mb-4 plot-item" data-type="{damping_type}">
                            <div class="card plot-card h-100">
                                <div class="plot-img-container">
                                    <img src="{rel_path}" class="plot-img" alt="{display_name}">
                                    <span class="badge plot-badge {badge_class}">{damping_type.title()}</span>
                                </div>
                                <div class="card-body">
                                    <h5 class="card-title fs-6">{display_name}</h5>
                                    <p class="card-text small text-muted">{category.replace('_', ' ').title()}</p>
                                </div>
                                <div class="card-footer bg-white border-top-0">
                                    <a href="{rel_path}" class="btn btn-sm btn-primary w-100" target="_blank">
                                        View Full Size <i class="bi bi-arrows-fullscreen"></i>
                                    </a>
                                </div>
                            </div>
                        </div>'''
    
    html += '''
                    </div>
                </div>'''
    
    # Add category tabs
    for category in categories:
        category_path = os.path.join(plots_dir, category)
        plot_files = glob.glob(os.path.join(category_path, "*.png"))
        
        if not plot_files:
            continue
            
        display_name = category.replace('_', ' ').title()
        icon = category_info.get(category, {}).get('icon', 'bi-graph-up')
        description = category_info.get(category, {}).get('desc', 'Visualization plots.')
        
        html += f'''
                <!-- {display_name} Tab -->
                <div class="tab-pane fade" id="{category}-tab">
                    <div class="category-section mb-4">
                        <div class="p-4 border-bottom">
                            <h2 class="h4 mb-3">
                                <i class="bi {icon} me-2"></i>
                                {display_name}
                            </h2>
                            <p class="text-muted mb-0">{description}</p>
                        </div>
                        <div class="p-4">
                            <div class="row">'''
        
        # Add plots for this category
        for plot_file in plot_files:
            rel_path = os.path.relpath(plot_file, results_dir)
            filename = os.path.basename(plot_file)
            display_name = filename.replace('.png', '').replace('_', ' ').title()
            
            damping_type = "linear"
            badge_class = "linear-badge"
            if "quadratic" in filename.lower():
                damping_type = "quadratic"
                badge_class = "quadratic-badge"
                
            html += f'''
                                <div class="col-md-6 col-lg-4 mb-4 plot-item" data-type="{damping_type}">
                                    <div class="card plot-card h-100">
                                        <div class="plot-img-container">
                                            <img src="{rel_path}" class="plot-img" alt="{display_name}">
                                            <span class="badge plot-badge {badge_class}">{damping_type.title()}</span>
                                        </div>
                                        <div class="card-body">
                                            <h5 class="card-title fs-6">{display_name}</h5>
                                        </div>
                                        <div class="card-footer bg-white border-top-0">
                                            <a href="{rel_path}" class="btn btn-sm btn-primary w-100" target="_blank">
                                                View Full Size <i class="bi bi-arrows-fullscreen"></i>
                                            </a>
                                        </div>
                                    </div>
                                </div>'''
                
        html += '''
                            </div>
                        </div>
                    </div>
                </div>'''
    
    # Add footer
    current_date = datetime.datetime.now().strftime("%B %d, %Y")
    current_year = datetime.datetime.now().year
    
    html += f'''
            </div>
        </div>
    </section>

    <!-- Footer -->
    <footer>
        <div class="container">
            <div class="row">
                <div class="col-lg-5 mb-4">
                    <h5 class="mb-4 fw-bold">Smorzamento Project</h5>
                    <p class="mb-4 opacity-75">
                        A comprehensive analysis system for studying damped mechanical oscillations, 
                        featuring both linear and quadratic damping models.
                    </p>
                </div>
                <div class="col-lg-3 col-md-6 mb-4">
                    <h5 class="mb-4 fw-bold">Quick Links</h5>
                    <ul class="list-unstyled">
                        <li class="mb-2"><a href="#" class="text-white-50">Home</a></li>
                        <li class="mb-2"><a href="../docs/quadratic_damping.md" class="text-white-50">Documentation</a></li>
                        <li class="mb-2"><a href="../docs/quadratic_vs_linear_damping.md" class="text-white-50">Model Comparison</a></li>
                    </ul>
                </div>
                <div class="col-lg-4 col-md-6 mb-4">
                    <h5 class="mb-4 fw-bold">About This Gallery</h5>
                    <p class="mb-4 opacity-75">
                        This visualization gallery provides intuitive access to analysis results from
                        experiments investigating mechanical oscillations with different damping mechanisms.
                    </p>
                    <p class="mb-0 opacity-75">
                        <i class="bi bi-calendar-event me-2"></i> Generated on {current_date}
                    </p>
                </div>
            </div>
            <div class="border-top border-secondary pt-4 mt-4 text-center opacity-50">
                <p class="mb-0">&copy; {current_year} Smorzamento Project. All rights reserved.</p>
            </div>
        </div>
    </footer>

    <!-- Bootstrap Bundle with Popper -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    
    <!-- Custom Script -->
    <script>
        // Filtering functionality
        document.getElementById('filterLinear').addEventListener('change', updateFilters);
        document.getElementById('filterQuadratic').addEventListener('change', updateFilters);
        
        function updateFilters() {
            const showLinear = document.getElementById('filterLinear').checked;
            const showQuadratic = document.getElementById('filterQuadratic').checked;
            
            document.querySelectorAll('.plot-item').forEach(item => {
                const type = item.getAttribute('data-type');
                
                if ((type === 'linear' && showLinear) || (type === 'quadratic' && showQuadratic)) {
                    item.style.display = '';
                } else {
                    item.style.display = 'none';
                }
            });
        }
    </script>
</body>
</html>'''
    
    # Write the HTML file
    with open(index_path, "w") as f:
        f.write(html)
    
    print(f"Bootstrap gallery with Montserrat font generated at: {index_path}")
    return index_path

if __name__ == "__main__":
    generate_bootstrap_gallery()