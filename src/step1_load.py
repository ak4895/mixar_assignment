import os
import sys
from pathlib import Path
from datetime import datetime

def check_dependencies():
    required_packages = {
        'numpy': 'numpy',
        'trimesh': 'trimesh',
        'open3d': 'open3d',
        'matplotlib': 'matplotlib'
    }
    
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("ERROR: Missing required packages:", ", ".join(missing_packages))
        print("\nPlease install with:")
        print("pip install numpy trimesh open3d matplotlib")
        sys.exit(1)

def create_output_directories():
    directories = [
        'outputs',
        'logs',
        'outputs/visuals'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def list_mesh_files(input_dir='8samples'):
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"\nERROR: Input directory '{input_dir}' does not exist!")
        print(f"Please create the folder and add .obj files.")
        sys.exit(1)
    
    obj_files = list(input_path.glob('*.obj'))
    obj_files = [str(f) for f in obj_files]
    
    return obj_files

def log_setup(obj_files):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path('logs') / f'run_{timestamp}.log'
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"Setup Run - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Found {len(obj_files)} .obj files:\n")
        for obj_file in obj_files:
            f.write(f"  - {obj_file}\n")
        f.write("\n")

def main():
    print("STEP 1 - Environment & Inputs (Setup)")
    
    check_dependencies()
    create_output_directories()
    
    print("Scanning for .obj files in 8samples/...")
    obj_files = list_mesh_files('8samples')
    
    if len(obj_files) == 0:
        print("\nERROR: No .obj files found in 8samples/ folder!")
        print("Please add at least one .obj file to the 8samples/ directory.")
        sys.exit(1)
    
    print(f"Found {len(obj_files)} .obj files: {obj_files}")
    
    log_setup(obj_files)
    
    print("\nStep 1 completed successfully!")
    print("Ready to proceed to Step 2.")

if __name__ == "__main__":
    main()
