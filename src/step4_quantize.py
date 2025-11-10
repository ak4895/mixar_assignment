import os
import sys
import json
import numpy as np
import trimesh
from pathlib import Path
from datetime import datetime
from normalization import norm_minmax, norm_sphere

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

def quantize(normalized, bins=BINS, method='minmax'):
    if normalized.ndim != 2 or normalized.shape[1] != 3:
        raise ValueError(f"Invalid normalized shape: {normalized.shape}. Expected (V, 3)")
    
    if method == 'minmax':
        normalized01 = normalized.copy()
    elif method == 'sphere':
        normalized01 = (normalized + 1.0) / 2.0
    else:
        raise ValueError(f"Unknown method: {method}. Expected 'minmax' or 'sphere'")
    
    q = np.floor(normalized01 * (bins - 1)).astype(np.int32)
    q = np.clip(q, 0, bins - 1)
    
    return q

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

def save_quantized_mesh(vertices, faces, output_path, base_name, method):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    output_file = output_path / f"{base_name}_quantized_{method}.ply"
    mesh.export(output_file)

def save_quantized_array(q, output_path, base_name, method):
    output_file = output_path / f"{base_name}_quantized_{method}.npy"
    np.save(output_file, q)

def save_quantized_sample(q, output_path, base_name, method):
    output_file = output_path / f"{base_name}_quantized_{method}_sample.txt"
    
    with open(output_file, 'w') as f:
        f.write(f"Quantized coordinates sample (first 10 vertices)\n")
        f.write(f"Method: {method}, BINS: {BINS}\n")
        f.write(f"Valid range: [0, {BINS-1}]\n")
        f.write("=" * 60 + "\n\n")
        
        sample_size = min(10, q.shape[0])
        for i in range(sample_size):
            f.write(f"[{i}]: ({q[i, 0]:4d}, {q[i, 1]:4d}, {q[i, 2]:4d})\n")

def save_normalization_params(params, output_path, base_name, method):
    output_file = output_path / f"{base_name}_params_{method}.json"
    
    params_serializable = {}
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            params_serializable[key] = value.tolist()
        else:
            params_serializable[key] = value
    
    with open(output_file, 'w') as f:
        json.dump(params_serializable, f, indent=2)

def process_mesh_file(obj_file, output_path):
    filename = Path(obj_file).name
    base_name = Path(obj_file).stem
    
    print(f"\nProcessing: {filename}")
    
    mesh, vertices, faces = load_mesh(obj_file)
    
    stats = {
        'filename': filename,
        'n_vertices': vertices.shape[0],
        'methods': {}
    }
    
    methods = [
        ('minmax', norm_minmax),
        ('sphere', norm_sphere)
    ]
    
    for method_name, norm_func in methods:
        normalized, params = norm_func(vertices)
        q = quantize(normalized, bins=BINS, method=method_name)
        
        q_min = q.min()
        q_max = q.max()
        
        if q_min < 0 or q_max >= BINS:
            print(f"ERROR: Quantized values out of range!")
            sys.exit(1)
        
        dequantized = dequantize(q, bins=BINS, method=method_name)
        
        save_quantized_mesh(dequantized, faces, output_path, base_name, method_name)
        save_quantized_array(q, output_path, base_name, method_name)
        save_quantized_sample(q, output_path, base_name, method_name)
        save_normalization_params(params, output_path, base_name, method_name)
        
        stats['methods'][method_name] = {
            'q_min': int(q_min),
            'q_max': int(q_max),
            'params': params
        }
    
    return stats

def update_log(all_stats):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path('logs') / f'step4_{timestamp}.log'
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"Step 4 - Quantize Meshes - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"BINS: {BINS}\n")
        f.write(f"Processed {len(all_stats)} mesh files\n\n")
        
        for stat in all_stats:
            f.write(f"File: {stat['filename']}\n")
            f.write(f"  Vertices: {stat['n_vertices']}\n")
            
            for method_name, method_stats in stat['methods'].items():
                f.write(f"  Method: {method_name}\n")
                f.write(f"    Quantized range: [{method_stats['q_min']}, {method_stats['q_max']}]\n")
            
            f.write("\n")

def main():
    print("STEP 4 - Quantize & Save Quantized Meshes")
    print(f"\nQuantization parameter: BINS = {BINS}")
    
    obj_files = list_mesh_files('8samples')
    print(f"Found {len(obj_files)} .obj files to process.")
    
    output_path = Path('outputs')
    output_path.mkdir(exist_ok=True)
    
    all_stats = []
    
    for obj_file in obj_files:
        stats = process_mesh_file(obj_file, output_path)
        all_stats.append(stats)
    
    update_log(all_stats)
    
    print(f"\nStep 4 completed successfully!")
    print(f"Processed {len(obj_files)} mesh files with both methods")

if __name__ == "__main__":
    main()
