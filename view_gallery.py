import os
import webbrowser
import platform
import subprocess

def open_gallery():
    """Open the gallery HTML file in the default web browser."""
    # Check if the gallery file exists, if not generate it
    gallery_path = "results/gallery/index.html"
    if not os.path.exists(gallery_path):
        try:
            print("Gallery file not found. Generating new gallery...")
            import scripts.fix_bootstrap_gallery
            scripts.fix_bootstrap_gallery.fix_bootstrap_gallery()
        except Exception as e:
            print(f"Error generating gallery: {e}")
            return

    # Convert to absolute path
    current_dir = os.path.abspath(os.path.dirname(__file__))
    gallery_path = os.path.join(current_dir, gallery_path)
    
    # Convert to file URL format
    if platform.system() == 'Windows':
        file_url = 'file:///' + gallery_path.replace('\\', '/')
    else:
        file_url = 'file://' + gallery_path
    
    # Try to open in browser
    print(f"Opening gallery in your default web browser...")
    try:
        webbrowser.open(file_url)
    except Exception as e:
        print(f"Error opening browser: {e}")
        # Alternative method
        try:
            if platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', gallery_path])
            elif platform.system() == 'Windows':
                os.startfile(gallery_path)
            elif platform.system() == 'Linux':
                subprocess.run(['xdg-open', gallery_path])
        except Exception as e:
            print(f"Failed to open gallery: {e}")
            print(f"Please open this file manually: {gallery_path}")

if __name__ == "__main__":
    open_gallery()