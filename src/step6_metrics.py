import os
import sys
import json
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

ERROR_THRESHOLD = 1e-3

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

def load_reconstructed_mesh(output_path, base_name, method):
    from step4_quantize import dequantize
    from normalization import denorm_minmax, denorm_sphere
    npy_file = output_path / f"{base_name}_quantized_{method}.npy"
    if not npy_file.exists():
        print(f"ERROR: Quantized array not found: {npy_file}")
        sys.exit(1)
    
    q = np.load(npy_file)
    
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
    
    normalized = dequantize(q, bins=1024, method=method)
    
    if method == 'minmax':
        reconstructed = denorm_minmax(normalized, params)
    elif method == 'sphere':
        reconstructed = denorm_sphere(normalized, params)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return reconstructed

def compute_errors(original, reconstructed):
    if original.shape != reconstructed.shape:
        raise ValueError(f"Shape mismatch: {original.shape} vs {reconstructed.shape}")
    diff = original - reconstructed
    
    mse_total = np.mean(diff ** 2)
    mae_total = np.mean(np.abs(diff))
    
    mse_per_axis = np.mean(diff ** 2, axis=0)
    mae_per_axis = np.mean(np.abs(diff), axis=0)
    
    per_vertex_error = np.linalg.norm(diff, axis=1)
    
    per_vertex_abs_error = np.abs(diff)
    
    max_error = np.max(per_vertex_error)
    min_error = np.min(per_vertex_error)
    std_error = np.std(per_vertex_error)
    
    metrics = {
        'mse_total': float(mse_total),
        'mae_total': float(mae_total),
        'mse_per_axis': {
            'x': float(mse_per_axis[0]),
            'y': float(mse_per_axis[1]),
            'z': float(mse_per_axis[2])
        },
        'mae_per_axis': {
            'x': float(mae_per_axis[0]),
            'y': float(mae_per_axis[1]),
            'z': float(mae_per_axis[2])
        },
        'max_error': float(max_error),
        'min_error': float(min_error),
        'std_error': float(std_error),
        'per_vertex_error': per_vertex_error,  # Keep as array for plotting
        'per_vertex_abs_error': per_vertex_abs_error  # Keep as array for plotting
    }
    
    return metrics

def save_metrics(output_path, base_name, method, metrics):
    output_file = output_path / f"{base_name}_metrics_{method}.json"
    metrics_serializable = {
        'mse_total': metrics['mse_total'],
        'mae_total': metrics['mae_total'],
        'mse_per_axis': metrics['mse_per_axis'],
        'mae_per_axis': metrics['mae_per_axis'],
        'max_error': metrics['max_error'],
        'min_error': metrics['min_error'],
        'std_error': metrics['std_error']
    }
    
    with open(output_file, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    print(f"  ✓ Saved metrics: {output_file}")

def plot_mse_per_axis(metrics, output_path, base_name, method):
    output_file = output_path / f"{base_name}_mse_axis_{method}.png"
    axes = ['X', 'Y', 'Z']
    mse_values = [
        metrics['mse_per_axis']['x'],
        metrics['mse_per_axis']['y'],
        metrics['mse_per_axis']['z']
    ]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(axes, mse_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    
    for bar, value in zip(bars, mse_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.6f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Axis', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Squared Error (MSE)', fontsize=12, fontweight='bold')
    plt.title(f'Reconstruction MSE per Axis\n{base_name} - {method.upper()}', 
              fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved MSE plot: {output_file}")

def plot_error_histogram(metrics, output_path, base_name, method):
    output_file = output_path / f"{base_name}_error_hist_{method}.png"
    errors = metrics['per_vertex_error']
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, color='#6C5CE7', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    plt.axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.6f}')
    plt.axvline(median_error, color='green', linestyle='--', linewidth=2, label=f'Median: {median_error:.6f}')
    
    plt.xlabel('Per-Vertex Error (Euclidean Distance)', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title(f'Per-Vertex Error Distribution\n{base_name} - {method.upper()}', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved histogram: {output_file}")

def plot_error_sorted(metrics, output_path, base_name, method):
    output_file = output_path / f"{base_name}_error_sorted_{method}.png"
    errors = metrics['per_vertex_error']
    sorted_errors = np.sort(errors)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_errors, color='#FF6B6B', linewidth=1.5, alpha=0.8)
    
    mean_error = np.mean(errors)
    p95_error = np.percentile(errors, 95)
    plt.axhline(mean_error, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.6f}')
    plt.axhline(p95_error, color='orange', linestyle='--', linewidth=2, label=f'95th percentile: {p95_error:.6f}')
    
    plt.xlabel('Vertex Index (sorted by error)', fontsize=12, fontweight='bold')
    plt.ylabel('Per-Vertex Error', fontsize=12, fontweight='bold')
    plt.title(f'Sorted Per-Vertex Error\n{base_name} - {method.upper()}', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved sorted error plot: {output_file}")

def print_metrics(filename, method, metrics):
    print(f"\n  --- Metrics for {method.upper()} ---")
    print(f"  MSE (total):  {metrics['mse_total']:.10f}")
    print(f"  MAE (total):  {metrics['mae_total']:.10f}")
    print(f"  MSE per axis: X={metrics['mse_per_axis']['x']:.10f}, "
          f"Y={metrics['mse_per_axis']['y']:.10f}, Z={metrics['mse_per_axis']['z']:.10f}")
    print(f"  MAE per axis: X={metrics['mae_per_axis']['x']:.10f}, "
          f"Y={metrics['mae_per_axis']['y']:.10f}, Z={metrics['mae_per_axis']['z']:.10f}")
    print(f"  Max error:    {metrics['max_error']:.10f}")
    print(f"  Min error:    {metrics['min_error']:.10f}")
    print(f"  Std error:    {metrics['std_error']:.10f}")
    if metrics['mse_total'] > ERROR_THRESHOLD:
        print(f"  ⚠ WARNING: MSE ({metrics['mse_total']:.6f}) exceeds threshold ({ERROR_THRESHOLD})")
    else:
        print(f"  ✓ MSE is within acceptable threshold")

def process_mesh_metrics(obj_file, output_path):
    filename = Path(obj_file).name
    base_name = Path(obj_file).stem
    print(f"\n{'='*60}")
    print(f"Computing metrics: {filename}")
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
        
        reconstructed_vertices = load_reconstructed_mesh(output_path, base_name, method_name)
        print(f"  ✓ Loaded reconstructed mesh (shape: {reconstructed_vertices.shape})")
        
        metrics = compute_errors(original_vertices, reconstructed_vertices)
        
        print_metrics(filename, method_name, metrics)
        
        save_metrics(output_path, base_name, method_name, metrics)
        
        print(f"\n  Creating visualizations...")
        plot_mse_per_axis(metrics, output_path, base_name, method_name)
        plot_error_histogram(metrics, output_path, base_name, method_name)
        plot_error_sorted(metrics, output_path, base_name, method_name)
        
        stats['methods'][method_name] = {
            'mse_total': metrics['mse_total'],
            'mae_total': metrics['mae_total'],
            'max_error': metrics['max_error']
        }
    
    return stats

def create_comparison_summary(all_stats, output_path):
    output_file = output_path / "comparison_summary.txt"
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RECONSTRUCTION ERROR COMPARISON SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"{'Mesh':<20} {'Method':<10} {'MSE (total)':<15} {'MAE (total)':<15} {'Max Error':<15}\n")
        f.write("-" * 80 + "\n")
        
        for stat in all_stats:
            for method_name, method_stats in stat['methods'].items():
                f.write(f"{stat['filename']:<20} {method_name.upper():<10} "
                       f"{method_stats['mse_total']:<15.10f} "
                       f"{method_stats['mae_total']:<15.10f} "
                       f"{method_stats['max_error']:<15.10f}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"\n✓ Created comparison summary: {output_file}")

def update_log(all_stats):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path('logs') / f'step6_{timestamp}.log'
    with open(log_path, 'w') as f:
        f.write(f"Step 6 - Error Metrics - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Computed metrics for {len(all_stats)} mesh files\n\n")
        
        for stat in all_stats:
            f.write(f"File: {stat['filename']}\n")
            f.write(f"  Vertices: {stat['n_vertices']}\n")
            
            for method_name, method_stats in stat['methods'].items():
                f.write(f"  Method: {method_name}\n")
                f.write(f"    MSE: {method_stats['mse_total']:.10f}\n")
                f.write(f"    MAE: {method_stats['mae_total']:.10f}\n")
                f.write(f"    Max Error: {method_stats['max_error']:.10f}\n")
            
            f.write("\n")
    
    print(f"\n✓ Log file created: {log_path}")

def main():
    print("=" * 60)
    print("STEP 6 — Compute Error Metrics and Per-Axis Error Plots")
    print("=" * 60)
    print(f"\nError threshold: {ERROR_THRESHOLD}")
    print(f"\nMetrics to compute:")
    print(f"  - MSE (Mean Squared Error) - total and per-axis")
    print(f"  - MAE (Mean Absolute Error) - total and per-axis")
    print(f"  - Per-vertex error statistics")
    print(f"\nVisualizations to generate:")
    print(f"  - Bar chart: MSE per axis")
    print(f"  - Histogram: Per-vertex error distribution")
    print(f"  - Line plot: Sorted errors")
    obj_files = list_mesh_files('8samples')
    print(f"\nFound {len(obj_files)} .obj files to analyze.")
    
    output_path = Path('outputs')
    
    if not output_path.exists():
        print(f"\nERROR: Output directory not found!")
        print(f"Please run Step 5 first to generate reconstructed meshes.")
        sys.exit(1)
    
    all_stats = []
    
    for obj_file in obj_files:
        stats = process_mesh_metrics(obj_file, output_path)
        all_stats.append(stats)
    
    create_comparison_summary(all_stats, output_path)
    
    update_log(all_stats)
    
    print("\n" + "=" * 60)
    print("✓ Step 6 completed successfully!")
    print(f"✓ Computed metrics for {len(obj_files)} mesh files")
    print(f"✓ Generated {len(obj_files) * 2 * 3} plots (3 plots × 2 methods × {len(obj_files)} meshes)")
    print(f"✓ All outputs saved in outputs/ directory")
    print("=" * 60)
    print("\nReady to proceed to Step 7 (Visualization).")

if __name__ == "__main__":
    main()
