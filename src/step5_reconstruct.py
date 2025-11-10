import os
import sys
import json
import numpy as np
import trimesh
from pathlib import Path
from datetime import datetime

from normalization import denorm_minmax, denorm_sphere

BINS = 1024

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

def load_quantized_array(output_path, base_name, method):
    npy_file = output_path / f"{base_name}_quantized_{method}.npy"
    if not npy_file.exists():
        print(f"ERROR: Quantized array not found: {npy_file}")
        sys.exit(1)
    
    q = np.load(npy_file)
    return q

def load_normalization_params(output_path, base_name, method):
    json_file = output_path / f"{base_name}_params_{method}.json"
    if not json_file.exists():
        print(f"ERROR: Parameters file not found: {json_file}")
        sys.exit(1)
    
    with open(json_file, 'r') as f:
        params = json.load(f)
    
    if 'v_min' in params:
        params['v_min'] = np.array(params['v_min'])
    if 'v_max' in params:
        params['v_max'] = np.array(params['v_max'])
    if 'center' in params:
        params['center'] = np.array(params['center'])
    
    return params

def dequantize(q, bins=BINS, method='minmax'):
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError(f"Invalid quantized shape: {q.shape}. Expected (V, 3)")
    dequantized01 = q.astype(np.float64) / (bins - 1)
    
    if method == 'minmax':
        normalized = dequantized01
    elif method == 'sphere':
        normalized = dequantized01 * 2.0 - 1.0
    else:
        raise ValueError(f"Unknown method: {method}. Expected 'minmax' or 'sphere'")
    
    return normalized

def denormalize(normalized, params):
    method = params['method']
    if method == 'minmax':
        return denorm_minmax(normalized, params)
    elif method == 'sphere':
        return denorm_sphere(normalized, params)
    else:
        raise ValueError(f"Unknown method: {method}")

def save_reconstructed_mesh(vertices, faces, output_path, base_name, method):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    output_file = output_path / f"{base_name}_reconstructed_{method}.ply"
    mesh.export(output_file)
    
    print(f"  ✓ Saved reconstructed mesh: {output_file}")
    return output_file

def save_sample_vertices(vertices, output_path, base_name, method):
    output_file = output_path / f"{base_name}_reconstructed_{method}_sample.txt"
    with open(output_file, 'w') as f:
        f.write(f"Reconstructed vertices sample (first 5 vertices)\n")
        f.write(f"Method: {method}\n")
        f.write("=" * 60 + "\n\n")
        
        sample_size = min(5, vertices.shape[0])
        for i in range(sample_size):
            v = vertices[i]
            f.write(f"[{i}]: ({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f})\n")
    
    print(f"  ✓ Saved reconstructed sample: {output_file}")

def process_mesh_reconstruction(obj_file, output_path):
    filename = Path(obj_file).name
    base_name = Path(obj_file).stem
    print(f"\n{'='*60}")
    print(f"Reconstructing: {filename}")
    print(f"{'='*60}")
    
    mesh, original_vertices, faces = load_mesh(obj_file)
    print(f"✓ Loaded original mesh with {original_vertices.shape[0]} vertices")
    
    stats = {
        'filename': filename,
        'n_vertices': original_vertices.shape[0],
        'methods': {}
    }
    
    methods = ['minmax', 'sphere']
    
    for method_name in methods:
        print(f"\n--- Method: {method_name.upper()} ---")
        
        q = load_quantized_array(output_path, base_name, method_name)
        print(f"  ✓ Loaded quantized array (shape: {q.shape})")
        
        if q.shape != original_vertices.shape:
            print(f"  ✗ ERROR: Shape mismatch!")
            print(f"    Original: {original_vertices.shape}, Quantized: {q.shape}")
            sys.exit(1)
        
        params = load_normalization_params(output_path, base_name, method_name)
        print(f"  ✓ Loaded normalization parameters (method: {params['method']})")
        
        normalized = dequantize(q, bins=BINS, method=method_name)
        print(f"  ✓ Dequantized to normalized coordinates (shape: {normalized.shape})")
        
        if method_name == 'minmax':
            n_min, n_max = normalized.min(), normalized.max()
            print(f"    Normalized range: [{n_min:.6f}, {n_max:.6f}] (expected: [0, 1])")
        else:
            n_min, n_max = normalized.min(), normalized.max()
            max_dist = np.max(np.linalg.norm(normalized, axis=1))
            print(f"    Normalized range: [{n_min:.6f}, {n_max:.6f}]")
            print(f"    Max distance from origin: {max_dist:.6f} (expected: ≤ 1)")
        
        reconstructed = denormalize(normalized, params)
        print(f"  ✓ Denormalized to original scale (shape: {reconstructed.shape})")
        
        if reconstructed.shape != original_vertices.shape:
            print(f"  ✗ ERROR: Reconstructed shape mismatch!")
            sys.exit(1)
        
        save_reconstructed_mesh(reconstructed, faces, output_path, base_name, method_name)
        
        save_sample_vertices(reconstructed, output_path, base_name, method_name)
        
        stats['methods'][method_name] = {
            'reconstructed_shape': reconstructed.shape,
            'sample_vertices': reconstructed[:5].tolist()
        }
    
    return stats

def update_log(all_stats):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path('logs') / f'step5_{timestamp}.log'
    with open(log_path, 'w') as f:
        f.write(f"Step 5 - Reconstruct Meshes - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Reconstructed {len(all_stats)} mesh files\n\n")
        
        for stat in all_stats:
            f.write(f"File: {stat['filename']}\n")
            f.write(f"  Vertices: {stat['n_vertices']}\n")
            
            for method_name, method_stats in stat['methods'].items():
                f.write(f"  Method: {method_name}\n")
                f.write(f"    Reconstructed shape: {method_stats['reconstructed_shape']}\n")
                f.write(f"    First vertex: {method_stats['sample_vertices'][0]}\n")
            
            f.write("\n")
    
    print(f"\n✓ Log file created: {log_path}")

def main():
    print("=" * 60)
    print("STEP 5 — Reverse Transforms: Dequantize & Denormalize")
    print("=" * 60)
    print(f"\nReconstruction process:")
    print(f"  1. Load quantized data (.npy) and parameters (.json)")
    print(f"  2. Dequantize: integer bins → normalized coordinates")
    print(f"  3. Denormalize: normalized → original world coordinates")
    print(f"  4. Save reconstructed meshes (.ply)")
    obj_files = list_mesh_files('8samples')
    print(f"\nFound {len(obj_files)} .obj files to reconstruct.")
    
    output_path = Path('outputs')
    
    if not output_path.exists():
        print(f"\nERROR: Output directory not found!")
        print(f"Please run Step 4 first to generate quantized data.")
        sys.exit(1)
    
    all_stats = []
    
    for obj_file in obj_files:
        stats = process_mesh_reconstruction(obj_file, output_path)
        all_stats.append(stats)
    
    update_log(all_stats)
    
    print("\n" + "=" * 60)
    print("✓ Step 5 completed successfully!")
    print(f"✓ Reconstructed {len(obj_files)} mesh files with both methods")
    print(f"✓ Reconstructed meshes saved in outputs/ directory")
    print("=" * 60)
    print("\nReady to proceed to Step 6 (Compute error metrics).")

if __name__ == "__main__":
    main()
