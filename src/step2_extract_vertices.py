import os
import sys
import json
import numpy as np
import trimesh
from pathlib import Path
from datetime import datetime

def list_mesh_files(input_dir='8samples'):
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"\nERROR: Input directory '{input_dir}' does not exist!")
        sys.exit(1)
    
    obj_files = list(input_path.glob('*.obj'))
    obj_files = [str(f) for f in obj_files]
    
    if len(obj_files) == 0:
        print(f"\nERROR: No .obj files found in {input_dir}/ folder!")
        sys.exit(1)
    
    return obj_files

def load_mesh(path):
    try:
        mesh = trimesh.load(path, process=False)
        vertices = mesh.vertices
        faces = mesh.faces
        
        return mesh, vertices, faces
    except Exception as e:
        print(f"ERROR loading {path}: {e}")
        sys.exit(1)

def compute_stats(vertices):
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Invalid vertices shape: {vertices.shape}. Expected (V, 3)")
    
    n_vertices = vertices.shape[0]
    
    v_min = vertices.min(axis=0)
    v_max = vertices.max(axis=0)
    v_mean = vertices.mean(axis=0)
    v_std = vertices.std(axis=0)
    
    stats = {
        'n_vertices': int(n_vertices),
        'min': {
            'x': float(v_min[0]),
            'y': float(v_min[1]),
            'z': float(v_min[2])
        },
        'max': {
            'x': float(v_max[0]),
            'y': float(v_max[1]),
            'z': float(v_max[2])
        },
        'mean': {
            'x': float(v_mean[0]),
            'y': float(v_mean[1]),
            'z': float(v_mean[2])
        },
        'std': {
            'x': float(v_std[0]),
            'y': float(v_std[1]),
            'z': float(v_std[2])
        }
    }
    
    return stats

def print_stats(filename, stats, vertices):
    print(f"\nFile: {filename}")
    print(f"Number of vertices: {stats['n_vertices']}")
    print(f"\nPer-axis statistics:")
    print(f"  X-axis: min={stats['min']['x']:.6f}, max={stats['max']['x']:.6f}, "
          f"mean={stats['mean']['x']:.6f}, std={stats['std']['x']:.6f}")
    print(f"  Y-axis: min={stats['min']['y']:.6f}, max={stats['max']['y']:.6f}, "
          f"mean={stats['mean']['y']:.6f}, std={stats['std']['y']:.6f}")
    print(f"  Z-axis: min={stats['min']['z']:.6f}, max={stats['max']['z']:.6f}, "
          f"mean={stats['mean']['z']:.6f}, std={stats['std']['z']:.6f}")
    
    print(f"\nFirst 5 vertices (sample):")
    sample_vertices = vertices[:5]
    for i, v in enumerate(sample_vertices):
        print(f"  [{i}]: ({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f})")

def save_stats(output_path, filename, stats):
    base_name = Path(filename).stem
    output_file = output_path / f"{base_name}_stats.json"
    
    output_data = {
        'filename': filename,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'statistics': stats
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

def update_log(obj_files, all_stats):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path('logs') / f'step2_{timestamp}.log'
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"Step 2 - Extract Vertices - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Processed {len(obj_files)} .obj files:\n\n")
        
        for obj_file in obj_files:
            base_name = Path(obj_file).name
            if base_name in all_stats:
                stats = all_stats[base_name]
                f.write(f"File: {base_name}\n")
                f.write(f"  Vertices: {stats['n_vertices']}\n")
                f.write(f"  X range: [{stats['min']['x']:.6f}, {stats['max']['x']:.6f}]\n")
                f.write(f"  Y range: [{stats['min']['y']:.6f}, {stats['max']['y']:.6f}]\n")
                f.write(f"  Z range: [{stats['min']['z']:.6f}, {stats['max']['z']:.6f}]\n")
                f.write("\n")

def main():
    print("STEP 2 - Load Mesh and Extract Vertices")
    
    obj_files = list_mesh_files('8samples')
    print(f"Found {len(obj_files)} .obj files to process.")
    
    stats_path = Path('outputs') / 'stats'
    stats_path.mkdir(parents=True, exist_ok=True)
    
    all_stats = {}
    
    for obj_file in obj_files:
        mesh, vertices, faces = load_mesh(obj_file)
        
        stats = compute_stats(vertices)
        
        filename = Path(obj_file).name
        print_stats(filename, stats, vertices)
        
        save_stats(stats_path, filename, stats)
        
        all_stats[filename] = stats
    
    update_log(obj_files, all_stats)
    
    print(f"\nStep 2 completed successfully!")
    print(f"Processed {len(obj_files)} mesh files")

if __name__ == "__main__":
    main()
