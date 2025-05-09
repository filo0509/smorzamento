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
    
    # Count total plots and by damping type
    total_plots = 0
    linear_count = 0
    quadratic_count = 0
    
    for category in categories:
        category_path = os.path.join(plots_dir, category)
        plot_files = glob.glob(os.path.join(category_path, "*.png"))
        total_plots += len(plot_files)
        
        for plot_file in plot_files:
            if "linear" in os.path.basename(plot_file).lower():
                linear_count += 1
            elif "quadratic" in os.path.basename(plot_file).lower():
                quadratic_count += 1
    
    # Category descriptions
    descriptions = {
        'raw_data': 'Time series data showing position, velocity, and acceleration measurements from oscillation experiments.',
        'amplitude_decay': 'Analysis of oscillation amplitude reduction over time, comparing different damping models.',
        'full_oscillation': 'Complete oscillation visualizations with the corresponding decay envelope.',
        'residuals': 'Error analysis between measured data and theoretical model predictions.',
        'model_comparison': 'Direct comparison between linear and quadratic damping models with statistical metrics.',
        'other': 'Additional visualization and analysis plots for the oscillation experiments.'
    }
    
    # Category icons from Bootstrap icons
    icons = {
        'raw_data': 'bi-graph-up',
        'amplitude_decay': 'bi-reception-4',
        'full_oscillation': 'bi-activity',
        'residuals': 'bi-distribute-vertical',
        'model_comparison': 'bi-check2-square',
        'other': 'bi-pie-chart-fill'
    }
    
    # HTML header and Bootstrap styling
    html = """<!DOCTYPE html>
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
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #0d6efd;
            --primary-dark: #0a58ca;
            --secondary: #6c757d;
            --secondary-dark: #5a6268;
            --success: #198754;
            --linear-color: #6f42c1;
            --quadratic-color: #e83e8c;
            --light-gray: #f8f9fa;
            --dark-gray: #343a40;
        }
        
        body {
            font-family: 'Montserrat', sans-serif;
            background-color: #f8f9fa;
            color: #212529;
        }
        
        .navbar-brand {
            font-weight: 700;
            letter-spacing: 0.5px;
        }
        
        h1, h2, h3, h4, h5, h6 {
            font-weight: 600;
        }
        
        .hero-section {
            background: linear-gradient(120deg, #1e3c72, #2a5298);
            color: white;
            padding: 4rem 0;
        }
        
        .hero-title {
            font-weight: 700;
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        
        .hero-subtitle {
            font-weight: 300;
            max-width: 700px;
            margin: 0 auto 2rem;
        }
        
        .stats-card {
            background-color: rgba(255, 255, 255, 0.15);
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
            backdrop-filter: blur(5px);
            transition: transform 0.3s;
        }
        
        .stats-card:hover {
            transform: translateY(-5px);
        }
        
        .stats-number {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        .stats-label {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            opacity: 0.8;
        }
        
        .category-section {
            background: white;
            border-radius: 10px;
            box-shadow: 0 3px 15px rgba(0,0,0,0.05);
            margin-bottom: 2rem;
            overflow: hidden;
        }
        
        .category-header {
            padding: 1.5rem;
            border-bottom: 1px solid rgba(0,0,0,0.05);
        }
        
        .category-title {
            font-size: 1.5rem;
            margin-bottom: 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .category-description {
            margin-top: 1rem;
            color: #6c757d;
        }
        
        .category-content {
            padding: 1.5rem;
        }
        
        .plot-card {
            height: 100%;
            transition: transform 0.3s, box-shadow 0.3s;
            border: none;
            border-radius: 10px;
            overflow: hidden;
        }
        
        .plot-card:hover {
            transform: translateY(-7px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }
        
        .plot-img-container {
            position: relative;
            overflow: hidden;
            padding-top: 75%; /* 4:3 aspect ratio */
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
        
        .badge-linear {
            background-color: var(--linear-color);
            color: white;
            position: absolute;
            top: 10px;
            right: 10px;
        }
        
        .badge-quadratic {
            background-color: var(--quadratic-color);
            color: white;
            position: absolute;
            top: 10px;
            right: 10px;
        }
        
        .plot-title {
            font-weight: 500;
            margin-top: 0.5rem;
            font-size: 1rem;
            line-height: 1.4;
        }
        
        .plot-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 0.75rem;
        }
        
        .plot-category {
            font-size: 0.75rem;
            color: #6c757d;
        }
        
        .nav-pills .nav-link.active {
            background-color: var(--primary);
        }
        
        .sticky-top {
            top: 20px;
        }
        
        .sticky-filter {
            background: white;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 3px 15px rgba(0,0,0,0.05);
        }
        
        .btn-view {
            font-size: 0.85rem;
        }
        
        footer {
            background: #212529;
            color: white;
            padding: 3rem 0;
            margin-top: 3rem;
        }
        
        .footer-title {
            font-weight: 700;
            margin-bottom: 1.5rem;
        }
        
        .footer-description {
            opacity: 0.7;
            margin-bottom: 1.5rem;
        }
        
        .footer-copyright {
            opacity: 0.5;
            font-size: 0.9rem;
            text-align: center;
            margin-top: 2rem;
            padding-top: 1.5rem;
            border-top: 1px solid rgba(255,255,255,0.1);
        }
        
        .social-link {
            display: inline-block;
            width: 40px;
            height: 40px;
            line-height: 40px;
            text-align: center;
            background: rgba(255,255,255,0.1);
            border-radius: 50%;
            margin-right: 0.5rem;
            transition: background 0.3s;
        }
        
        .social-link:hover {
            background: rgba(255,255,255,0.2);
        }
        
        @media (max-width: 768px) {
            .hero-title {
                font-size: 2rem;
            }
            
            .stats-number {
                font-size: 1.8rem;
            }
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
                    <li class="nav-item">
                        <a class="nav-link" href="https://github.com/yourusername/smorzamento">GitHub</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <!-- Hero Section -->
    <section class="hero-section">
        <div class="container text-center">
            <h1 class="hero-title">Oscillation Analysis Gallery</h1>
            <p class="hero-subtitle">
                Visual exploration of linear and quadratic damping in mechanical oscillatory systems
            </p>
            
            <div class="row justify-content-center mt-5">
                <div class="col-md-3 col-6 mb-4">
                    <div class="stats-card">
                        <div class="stats-number">{total_plots}</div>
                        <div class="stats-label">Total Plots</div>
                    </div>
                </div>
                <div class="col-md-3 col-6 mb-4">
                    <div class="stats-card">
                        <div class="stats-number">{len(categories)}</div>
                        <div class="stats-label">Categories</div>
                    </div>
                </div>
                <div class="col-md-3 col-6 mb-4">
                    <div class="stats-card">
                        <div class="stats-number">{linear_count}</div>
                        <div class="stats-label">Linear Damping</div>
                    </div>
                </div>
                <div class="col-md-3 col-6 mb-4">
                    <div class="stats-card">
                        <div class="stats-number">{quadratic_count}</div>
                        <div class="stats-label">Quadratic Damping</div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Main Content -->
    <section class="py-5">
        <div class="container">
            <div class="row">
                <!-- Sidebar Filters -->
                <div class="col-lg-3 mb-4">
                    <div class="sticky-top sticky-filter">
                        <h5 class="mb-3">Categories</h5>
                        <div class="nav flex-column nav-pills">
                            <button class="nav-link active mb-2" data-bs-toggle="pill" data-bs-target="#all">All Categories</button>"""

    # Add navigation pills for each category
    for category in categories:
        category_display = category.replace('_', ' ').title()
        html += f"""
                            <button class="nav-link mb-2" data-bs-toggle="pill" data-bs-target="#{category}">{category_display}</button>"""
                            
    html += """
                        </div>
                        
                        <hr>
                        
                        <h5 class="mb-3">Damping Type</h5>
                        <div class="form-check mb-2">
                            <input class="form-check-input filter-check" type="checkbox" id="showLinear" checked data-filter="linear">
                            <label class="form-check-label" for="showLinear">
                                <span class="badge bg-primary me-1">Linear</span> Linear Damping
                            </label>
                        </div>
                        <div class="form-check">
                            <input class="form-check-input filter-check" type="checkbox" id="showQuadratic" checked data-filter="quadratic">
                            <label class="form-check-label" for="showQuadratic">
                                <span class="badge bg-danger me-1">Quadratic</span> Quadratic Damping
                            </label>
                        </div>
                    </div>
                </div>
                
                <!-- Main Gallery -->
                <div class="col-lg-9">
                    <div class="tab-content">
                        <!-- All Plots Tab -->
                        <div class="tab-pane fade show active" id="all">
                            <div class="alert alert-primary mb-4">
                                <i class="bi bi-info-circle-fill me-2"></i>
                                Showing all {total_plots} plots across {len(categories)} categories. Use the filters to narrow your view.
                            </div>
                            
                            <div class="row">"""
    
    # Add all plots to the "All" tab
    all_plots = []
    for category in categories:
        category_path = os.path.join(plots_dir, category)
        for plot_file in glob.glob(os.path.join(category_path, "*.png")):
            all_plots.append((category, plot_file))
    
    # Sort all plots (linear first, then quadratic)
    all_plots.sort(key=lambda x: "linear" in os.path.basename(x[1]).lower())
    
    for category, plot_file in all_plots:
        rel_path = os.path.relpath(plot_file, results_dir)
        filename = os.path.basename(plot_file)
        display_name = filename.replace('.png', '').replace('_', ' ').title()
        
        # Determine damping type
        damping_type = "linear"
        badge_class = "badge-linear"
        if "quadratic" in filename.lower():
            damping_type = "quadratic"
            badge_class = "badge-quadratic"
        
        html += f"""
                                <div class="col-md-6 col-lg-4 mb-4 plot-item" data-category="{category}" data-type="{damping_type}">
                                    <div class="card plot-card">
                                        <div class="plot-img-container">
                                            <img src="{rel_path}" class="plot-img" alt="{display_name}">
                                            <span class="badge {badge_class}">{damping_type.title()}</span>
                                        </div>
                                        <div class="card-body">
                                            <h5 class="plot-title">{display_name}</h5>
                                            <div class="plot-footer">
                                                <span class="plot-category">{category.replace('_', ' ').title()}</span>
                                                <a href="{rel_path}" target="_blank" class="btn btn-sm btn-outline-primary btn-view">
                                                    View <i class="bi bi-box-arrow-up-right ms-1"></i>
                                                </a>
                                            </div>
                                        </div>
                                    </div>
                                </div>"""
    
    html += """
                            </div>
                        </div>"""
    
    # Generate tabs for each category
    for category in categories:
        category_path = os.path.join(plots_dir, category)
        plot_files = glob.glob(os.path.join(category_path, "*.png"))
        
        if not plot_files:
            continue
        
        # Sort plot files (linear first, then quadratic)
        plot_files.sort(key=lambda x: "0" + os.path.basename(x) if "linear" in os.path.basename(x).lower() else "1" + os.path.basename(x))
        
        category_display = category.replace('_', ' ').title()
        icon = icons.get(category, 'bi-graph-up')
        description = descriptions.get(category, 'Visualization plots for oscillation analysis.')
        
        html += f"""
                        <!-- {category_display} Tab -->
                        <div class="tab-pane fade" id="{category}">
                            <div class="category-section mb-4">
                                <div class="category-header">
                                    <h2 class="category-title">
                                        <i class="bi {icon} me-2"></i>
                                        {category_display}
                                    </h2>
                                    <p class="category-description">{description}</p>
                                </div>
                                <div class="category-content">
                                    <div class="row">"""
        
        for plot_file in plot_files:
            rel_path = os.path.relpath(plot_file, results_dir)
            filename = os.path.basename(plot_file)
            display_name = filename.replace('.png', '').replace('_', ' ').title()
            
            # Determine damping type
            damping_type = "linear"
            badge_class = "badge-linear"
            if "quadratic" in filename.lower():
                damping_type = "quadratic"
                badge_class = "badge-quadratic"
            
            html += f"""
                                        <div class="col-md-6 col-lg-4 mb-4 plot-item" data-category="{category}" data-type="{damping_type}">
                                            <div class="card plot-card">
                                                <div class="plot-img-container">
                                                    <img src="{rel_path}" class="plot-img" alt="{display_name}">
                                                    <span class="badge {badge_class}">{damping_type.title()}</span>
                                                </div>
                                                <div class="card-body">
                                                    <h5 class="plot-title">{display_name}</h5>
                                                    <div class="plot-footer">
                                                        <span class="plot-category">{category_display}</span>
                                                        <a href="{rel_path}" target="_blank" class="btn btn-sm btn-outline-primary btn-view">
                                                            View <i class="bi bi-box-arrow-up-right ms-1"></i>
                                                        </a>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>"""
        
        html += """
                                    </div>
                                </div>
                            </div>
                        </div>"""
    
    # Footer and JavaScript
    current_date = datetime.datetime.now().strftime("%B %d, %Y")
    current_year = datetime.datetime.now().year
    
    html += f"""
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Footer -->
    <footer>
        <div class="container">
            <div class="row">
                <div class="col-lg-5 mb-4">
                    <h5 class="footer-title">Smorzamento Project</h5>
                    <p class="footer-description">
                        A comprehensive analysis system for studying damped mechanical oscillations, 
                        featuring both linear and quadratic damping models.
                    </p>
                    <div class="social-links">
                        <a href="#" class="social-link text-white">
                            <i class="bi bi-github"></i>
                        </a>
                        <a href="#" class="social-link text-white">
                            <i class="bi bi-linkedin"></i>
                        </a>
                        <a href="#" class="social-link text-white">
                            <i class="bi bi-twitter"></i>
                        </a>
                    </div>
                </div>
                <div class="col-lg-3 col-md-6 mb-4">
                    <h5 class="footer-title">Quick Links</h5>
                    <ul class="list-unstyled">
                        <li class="mb-2"><a href="#" class="text-white-50">Home</a></li>
                        <li class="mb-2"><a href="../docs/quadratic_damping.md" class="text-white-50">Documentation</a></li>
                        <li class="mb-2"><a href="../docs/long_quadratic_damping.md" class="text-white-50">Long Duration Analysis</a></li>
                        <li class="mb-2"><a href="../docs/quadratic_vs_linear_damping.md" class="text-white-50">Model Comparison</a></li>
                    </ul>
                </div>
                <div class="col-lg-4 col-md-6 mb-4">
                    <h5 class="footer-title">About This Gallery</h5>
                    <p class="footer-description">
                        This visualization gallery provides intuitive access to analysis results from
                        experiments investigating mechanical oscillations with different damping mechanisms.
                    </p>
                    <p class="mb-0">
                        <i class="bi bi-calendar-event me-2"></i> Generated on {current_date}
                    </p>
                </div>
            </div>
            <div class="footer-copyright">
                <p>&copy; {current_year} Smorzamento Project. All rights reserved.</p>
            </div>
        </div>
    </footer>

    <!-- Bootstrap Bundle with Popper -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    
    <!-- Custom Script -->
    <script>
        // Filter functionality
        document.querySelectorAll('.filter-check').forEach(check => {
            check.addEventListener('change', updateFilters);
        });
        
        function updateFilters() {
            const showLinear = document.getElementById('showLinear').checked;
            const showQuadratic = document.getElementById('showQuadratic').checked;
            
            document.querySelectorAll('.plot-item').forEach(item => {
                const type = item.getAttribute('data-type');
                
                if ((type === 'linear' && showLinear) || (type === 'quadratic' && showQuadratic)) {
                    item.style.display = 'block';
                } else {
                    item.style.display = 'none';
                }
            });
        }
        
        // Add smooth scrolling
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function(e) {
                e.preventDefault();
                const targetId = this.getAttribute('href');
                const targetElement = document.querySelector(targetId);
                
                if (targetElement) {
                    window.scrollTo({
                        top: targetElement.offsetTop - 20,
                        behavior: 'smooth'
                    });
                }
            });
        });
    </script>
</body>
</html>"""
    
    # Write the HTML file
    with open(index_path, "w") as f:
        f.write(html)
    
    print(f"Bootstrap gallery with Montserrat font generated at: {index_path}")
    return index_path

if __name__ == "__main__":
    generate_bootstrap_gallery()