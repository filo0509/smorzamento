import os
import glob
import datetime
import random

def generate_html_index():
    """Generate an enhanced, visually stunning HTML index for browsing plots."""
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
    
    # Counters for stats
    total_plots = 0
    linear_plots = 0
    quadratic_plots = 0
    
    # Count all plots
    for category in categories:
        category_path = os.path.join(plots_dir, category)
        plot_files = glob.glob(os.path.join(category_path, "*.png"))
        
        total_plots += len(plot_files)
        linear_plots += len([f for f in plot_files if "linear" in os.path.basename(f).lower()])
        quadratic_plots += len([f for f in plot_files if "quadratic" in os.path.basename(f).lower()])
    
    # Define category icons and descriptions
    category_info = {
        'raw_data': {
            'icon': 'fas fa-wave-square',
            'description': 'Position, velocity, and acceleration time series from oscillation experiments. These plots show the raw measurement data before analysis.'
        },
        'amplitude_decay': {
            'icon': 'fas fa-tachometer-alt',
            'description': 'Analysis of how oscillation amplitude decreases over time, comparing quadratic damping (proportional to v²) with linear damping (proportional to v) models.'
        },
        'full_oscillation': {
            'icon': 'fas fa-compress-alt',
            'description': 'Complete oscillation with amplitude decay envelope showing the damping effect over time. These plots demonstrate how the overall motion evolves.'
        },
        'residuals': {
            'icon': 'fas fa-random',
            'description': 'Difference between measured data points and model predictions, helping identify how well each damping model fits the experimental data.'
        },
        'model_comparison': {
            'icon': 'fas fa-balance-scale',
            'description': 'Direct comparison between different damping models, showing statistical metrics like R² and AIC to determine which model best describes the physical system.'
        },
        'other': {
            'icon': 'fas fa-chart-bar',
            'description': 'Additional visualizations and analysis plots that provide further insights into the oscillation behavior.'
        }
    }
    
    # Start building HTML
    html = "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
    html += "    <meta charset=\"UTF-8\">\n"
    html += "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
    html += "    <title>Damping Analysis Visualization Gallery</title>\n"
    html += "    <link rel=\"stylesheet\" href=\"https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css\">\n"
    
    # Add CSS styles
    html += """    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Playfair+Display:wght@700&display=swap');
        
        :root {
            --primary: #4361ee;
            --primary-light: #4895ef;
            --primary-dark: #3f37c9;
            --secondary: #f72585;
            --light: #f8f9fa;
            --dark: #212529;
            --gray: #6c757d;
            --success: #4cc9f0;
            --warning: #f8961e;
            --linear: #9d4edd;
            --quadratic: #f72585;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', sans-serif;
            line-height: 1.6;
            color: var(--dark);
            background-color: #f5f7fa;
            overflow-x: hidden;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 20px;
        }
        
        header {
            background: linear-gradient(135deg, var(--primary-dark), var(--primary), var(--primary-light));
            color: white;
            padding: 60px 0 100px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        
        header::before {
            content: "";
            position: absolute;
            top: -50%;
            right: -50%;
            width: 100%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 70%);
            transform: rotate(30deg);
        }
        
        .wavy-bg {
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 70px;
            background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 320'%3E%3Cpath fill='%23f5f7fa' fill-opacity='1' d='M0,224L40,208C80,192,160,160,240,144C320,128,400,128,480,149.3C560,171,640,213,720,213.3C800,213,880,171,960,165.3C1040,160,1120,192,1200,186.7C1280,181,1360,139,1400,117.3L1440,96L1440,320L1400,320C1360,320,1280,320,1200,320C1120,320,1040,320,960,320C880,320,800,320,720,320C640,320,560,320,480,320C400,320,320,320,240,320C160,320,80,320,40,320L0,320Z'%3E%3C/path%3E%3C/svg%3E");
            background-size: cover;
        }
        
        h1 {
            font-family: 'Playfair Display', serif;
            font-size: 3.5rem;
            margin-bottom: 15px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            text-align: center;
        }
        
        .subtitle {
            text-align: center;
            font-weight: 300;
            font-size: 1.2rem;
            margin-bottom: 30px;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
        }
        
        .header-stats {
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-top: 40px;
        }
        
        .stat-card {
            background: rgba(255,255,255,0.15);
            backdrop-filter: blur(5px);
            padding: 15px 25px;
            border-radius: 10px;
            text-align: center;
            transition: transform 0.3s;
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            background: rgba(255,255,255,0.25);
        }
        
        .stat-value {
            font-size: 2.5rem;
            font-weight: 700;
        }
        
        .stat-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .main-content {
            position: relative;
            margin-top: -50px;
            background: transparent;
            z-index: 10;
        }
        
        .sticky-nav {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 15px;
            border-radius: 15px;
            box-shadow: 0 5px 30px rgba(0, 0, 0, 0.1);
            position: sticky;
            top: 20px;
            z-index: 100;
            margin-bottom: 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border: 1px solid rgba(0,0,0,0.05);
        }
        
        .nav-title {
            font-weight: 600;
            font-size: 1.1rem;
            color: var(--primary-dark);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .nav-title i {
            font-size: 1.3rem;
        }
        
        .nav-links {
            display: flex;
            gap: 10px;
            overflow-x: auto;
            max-width: 70%;
            padding: 5px;
        }
        
        .nav-links::-webkit-scrollbar {
            height: 4px;
        }
        
        .nav-links::-webkit-scrollbar-track {
            background: rgba(0,0,0,0.05);
            border-radius: 20px;
        }
        
        .nav-links::-webkit-scrollbar-thumb {
            background: var(--primary-light);
            border-radius: 20px;
        }
        
        .nav-link {
            color: var(--gray);
            padding: 8px 15px;
            border-radius: 30px;
            text-decoration: none;
            white-space: nowrap;
            transition: all 0.2s;
            font-size: 0.9rem;
            font-weight: 500;
        }
        
        .nav-link:hover, .nav-link.active {
            background-color: var(--primary);
            color: white;
        }
        
        .category-section {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 40px;
            box-shadow: 0 5px 30px rgba(0, 0, 0, 0.05);
            border: 1px solid rgba(0,0,0,0.03);
            overflow: hidden;
        }
        
        .category-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 25px;
            border-bottom: 1px solid rgba(0,0,0,0.05);
            padding-bottom: 15px;
        }
        
        .category-title {
            font-family: 'Playfair Display', serif;
            font-size: 1.8rem;
            color: var(--dark);
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .category-icon {
            background: linear-gradient(135deg, var(--primary), var(--primary-light));
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 10px;
            color: white;
            font-size: 1.2rem;
        }
        
        .category-count {
            background-color: var(--light);
            color: var(--primary);
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: 600;
        }
        
        .category-description {
            color: var(--gray);
            margin-bottom: 25px;
            line-height: 1.7;
            font-size: 0.95rem;
        }
        
        .plots-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 25px;
        }
        
        .plot-card {
            border-radius: 12px;
            overflow: hidden;
            background: white;
            transition: all 0.3s;
            text-decoration: none;
            color: inherit;
            height: 100%;
            display: flex;
            flex-direction: column;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            border: 1px solid rgba(0,0,0,0.03);
        }
        
        .plot-card:hover {
            transform: translateY(-7px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }
        
        .plot-img-container {
            position: relative;
            padding-top: 75%; /* 4:3 aspect ratio */
            overflow: hidden;
            background: #f9f9f9;
        }
        
        .plot-img {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.5s;
        }
        
        .plot-card:hover .plot-img {
            transform: scale(1.03);
        }
        
        .plot-badge {
            position: absolute;
            top: 15px;
            right: 15px;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            z-index: 10;
            backdrop-filter: blur(5px);
        }
        
        .badge-linear {
            background-color: rgba(157, 78, 221, 0.9);
            color: white;
        }
        
        .badge-quadratic {
            background-color: rgba(247, 37, 133, 0.9);
            color: white;
        }
        
        .plot-content {
            padding: 20px;
            flex-grow: 1;
            display: flex;
            flex-direction: column;
        }
        
        .plot-title {
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 10px;
            line-height: 1.4;
            color: var(--dark);
        }
        
        .plot-info {
            color: var(--gray);
            font-size: 0.85rem;
            margin-top: auto;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .plot-detail {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .plot-expand {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-top: 15px;
            padding: 8px;
            background-color: #f8f9fa;
            border-radius: 8px;
            color: var(--primary);
            font-weight: 500;
            font-size: 0.85rem;
            transition: background-color 0.2s;
        }
        
        .plot-card:hover .plot-expand {
            background-color: var(--primary);
            color: white;
        }
        
        footer {
            background: linear-gradient(135deg, #232526, #414345);
            color: white;
            padding: 60px 0 30px;
            margin-top: 80px;
            position: relative;
        }
        
        .footer-wave {
            position: absolute;
            top: -70px;
            left: 0;
            width: 100%;
            height: 70px;
            background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 320'%3E%3Cpath fill='%23232526' fill-opacity='1' d='M0,96L48,128C96,160,192,224,288,224C384,224,480,160,576,149.3C672,139,768,181,864,170.7C960,160,1056,96,1152,96C1248,96,1344,160,1392,192L1440,224L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z'%3E%3C/path%3E%3C/svg%3E");
            background-size: cover;
        }
        
        .footer-content {
            display: flex;
            justify-content: space-between;
            margin-bottom: 30px;
            flex-wrap: wrap;
            gap: 40px;
        }
        
        .footer-info {
            flex: 1;
            min-width: 300px;
        }
        
        .footer-logo {
            font-family: 'Playfair Display', serif;
            font-size: 1.8rem;
            margin-bottom: 15px;
        }
        
        .footer-description {
            color: rgba(255,255,255,0.7);
            line-height: 1.7;
            margin-bottom: 20px;
            font-size: 0.9rem;
        }
        
        .footer-links {
            flex: 1;
            min-width: 200px;
        }
        
        .footer-links h3 {
            font-size: 1.2rem;
            margin-bottom: 20px;
            position: relative;
            display: inline-block;
        }
        
        .footer-links h3::after {
            content: "";
            position: absolute;
            bottom: -8px;
            left: 0;
            width: 40px;
            height: 3px;
            background: var(--primary);
            border-radius: 3px;
        }
        
        .footer-links ul {
            list-style: none;
        }
        
        .footer-links li {
            margin-bottom: 10px;
        }
        
        .footer-links a {
            color: rgba(255,255,255,0.7);
            text-decoration: none;
            transition: color 0.2s;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .footer-links a:hover {
            color: white;
        }
        
        .footer-links i {
            font-size: 0.8rem;
        }
        
        .footer-copyright {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid rgba(255,255,255,0.1);
            color: rgba(255,255,255,0.5);
            font-size: 0.85rem;
        }
        
        .animate-wave {
            animation: wave 25s linear infinite;
            transform-origin: center bottom;
        }
        
        @keyframes wave {
            0% {
                transform: translateX(0);
            }
            50% {
                transform: translateX(-25%);
            }
            100% {
                transform: translateX(0);
            }
        }
        
        .damping-animation {
            animation: damped-oscillation 4s ease-out infinite;
            transform-origin: center;
        }
        
        @keyframes damped-oscillation {
            0% { transform: translateX(20px); }
            10% { transform: translateX(-15px); }
            20% { transform: translateX(10px); }
            30% { transform: translateX(-6px); }
            40% { transform: translateX(3px); }
            50% { transform: translateX(-1px); }
            100% { transform: translateX(0); }
        }
        
        /* Dark mode toggle */
        .dark-mode-toggle {
            position: fixed;
            bottom: 30px;
            right: 30px;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: var(--primary);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            z-index: 1000;
            border: none;
            transition: all 0.3s;
        }
        
        .dark-mode-toggle:hover {
            transform: scale(1.1);
        }
        
        /* Responsive styles */
        @media (max-width: 768px) {
            header {
                padding: 40px 0 80px;
            }
            
            h1 {
                font-size: 2.5rem;
            }
            
            .header-stats {
                flex-direction: column;
                gap: 15px;
            }
            
            .category-section {
                padding: 20px;
            }
            
            .plots-grid {
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            }
            
            .sticky-nav {
                flex-direction: column;
                align-items: flex-start;
                gap: 15px;
            }
            
            .nav-links {
                max-width: 100%;
            }
        }
        
        /* Loading animation */
        .loader-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: white;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 9999;
            transition: opacity 0.5s;
        }
        
        .loader {
            width: 80px;
            height: 80px;
            position: relative;
        }
        
        .loader:before, .loader:after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            border-radius: 50%;
            width: 100%;
            height: 100%;
        }
        
        .loader:before {
            background: var(--primary);
            animation: pulse 2s ease-out infinite;
        }
        
        .loader:after {
            background: var(--secondary);
            animation: pulse 2s 0.5s ease-out infinite;
            opacity: 0.5;
        }
        
        @keyframes pulse {
            0% {
                transform: scale(0);
                opacity: 1;
            }
            100% {
                transform: scale(1.5);
                opacity: 0;
            }
        }
    </style>
</head>
<body>
    <!-- Loading animation -->
    <div class="loader-overlay" id="loader">
        <div class="loader"></div>
    </div>

    <header>
        <div class="container">
            <h1 class="damping-animation">Oscillation Analysis</h1>
            <p class="subtitle">Interactive visualization gallery comparing linear and quadratic damping models for mechanical oscillations, featuring detailed amplitude decay analysis and model comparisons.</p>
            
            <div class="header-stats">
                <div class="stat-card">
                    <div class="stat-value" id="total-plots">0</div>
                    <div class="stat-label">Total Plots</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="categories-count">0</div>
                    <div class="stat-label">Categories</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="linear-count">0</div>
                    <div class="stat-label">Linear Damping</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="quadratic-count">0</div>
                    <div class="stat-label">Quadratic Damping</div>
                </div>
            </div>
        </div>
        <div class="wavy-bg animate-wave"></div>
    </header>

    <div class="main-content">
        <div class="container">
            <nav class="sticky-nav">
                <div class="nav-title">
                    <i class="fas fa-chart-line"></i>
                    <span>Plot Categories</span>
                </div>
                <div class="nav-links" id="category-nav">
                    <a href="#" class="nav-link active" data-target="all">All</a>
"""
    
    # Add navigation links for each category
    for category in categories:
        category_name = category.replace('_', ' ').title()
        html += f'                    <a href="#{category}" class="nav-link" data-target="{category}">{category_name}</a>\n'
    
    html += """                </div>
            </nav>
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
        
        # Get category icon and description
        icon = category_info.get(category, {}).get('icon', 'fas fa-chart-line')
        description = category_info.get(category, {}).get('description', 'Visualization plots for oscillation analysis.')
        
        html += f"""
            <section class="category-section" id="{category}">
                <div class="category-header">
                    <h2 class="category-title">
                        <span class="category-icon"><i class="{icon}"></i></span>
                        {category_name}
                    </h2>
                    <span class="category-count">{len(plot_files)} plots</span>
                </div>
                
                <p class="category-description">{description}</p>
                
                <div class="plots-grid">
"""
        
        # Add each plot in this category
        for plot_file in plot_files:
            rel_path = os.path.relpath(plot_file, results_dir)
            filename = os.path.basename(plot_file)
            
            # Generate a nice display name
            raw_name = filename.replace('.png', '')
            display_name = raw_name.replace('_', ' ').title()
            
            # Determine if it's quadratic or linear damping
            damping_type = "Linear"
            badge_class = "badge-linear"
            if "quadratic" in filename.lower():
                damping_type = "Quadratic"
                badge_class = "badge-quadratic"
            
            # Extract any numbering from the filename
            plot_number = ""
            for part in raw_name.split('_'):
                if part.isdigit():
                    plot_number = part
                    break
            
            # Random image dimensions for variety
            img_height = random.choice(['200px', '180px', '220px'])
            
            html += f"""
                    <a href="{rel_path}" target="_blank" class="plot-card" data-damping="{damping_type.lower()}">
                        <div class="plot-img-container">
                            <img src="{rel_path}" alt="{display_name}" class="plot-img">
                            <span class="plot-badge {badge_class}">{damping_type}</span>
                        </div>
                        <div class="plot-content">
                            <h3 class="plot-title">{display_name}</h3>
                            <div class="plot-info">
                                <div class="plot-detail">
                                    <i class="fas fa-chart-line"></i>
                                    <span>{category_name}</span>
                                </div>
                                {f'<div class="plot-detail"><i class="fas fa-hashtag"></i><span>{plot_number}</span></div>' if plot_number else ''}
                            </div>
                            <div class="plot-expand">
                                <i class="fas fa-external-link-alt"></i>
                                <span>View Full Size</span>
                            </div>
                        </div>
                    </a>
"""
        
        html += """                </div>
            </section>
"""
    
    # Add footer
    current_time = datetime.datetime.now().strftime("%B %d, %Y at %H:%M")
    html += f"""
        </div>
    </div>

    <footer>
        <div class="footer-wave"></div>
        <div class="container">
            <div class="footer-content">
                <div class="footer-info">
                    <div class="footer-logo">Smorzamento</div>
                    <p class="footer-description">
                        A comprehensive analysis system for studying damped mechanical oscillations, featuring both linear and quadratic damping models. This visualization gallery provides insights into the different damping behaviors.
                    </p>
                </div>
                
                <div class="footer-links">
                    <h3>Quick Links</h3>
                    <ul>
                        <li><a href="#"><i class="fas fa-home"></i> Home</a></li>
"""
    
    # Add category links to footer
    for category in categories:
        category_name = category.replace('_', ' ').title()
        icon = category_info.get(category, {}).get('icon', 'fas fa-chart-line')
        html += f'                        <li><a href="#{category}"><i class="{icon}"></i> {category_name}</a></li>\n'
    
    html += f"""                    </ul>
                </div>
                
                <div class="footer-links">
                    <h3>Resources</h3>
                    <ul>
                        <li><a href="#"><i class="fas fa-file-alt"></i> Documentation</a></li>
                        <li><a href="#"><i class="fas fa-code"></i> Source Code</a></li>
                        <li><a href="#"><i class="fas fa-book"></i> Physics Background</a></li>
                        <li><a href="#"><i class="fas fa-download"></i> Download Data</a></li>
                    </ul>
                </div>
            </div>
            
            <div class="footer-copyright">
                <p>Generated on {current_time}</p>
                <p>Smorzamento Project © {datetime.