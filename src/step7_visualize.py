import os
import sys
import json
import numpy as np
import trimesh
import matplotlib.pyplot as plt
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
        return mesh
    except Exception as e:
        print(f"ERROR loading {path}: {e}")
        sys.exit(1)

def load_reconstructed_vertices(output_path, base_name, method):
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

def create_mesh_info_plot(mesh, title, output_file):
    vertices = mesh.vertices
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    ax.hist(vertices[:, 0], bins=30, alpha=0.7, label='X', color='#FF6B6B')
    ax.hist(vertices[:, 1], bins=30, alpha=0.7, label='Y', color='#4ECDC4')
    ax.hist(vertices[:, 2], bins=30, alpha=0.7, label='Z', color='#45B7D1')
    ax.set_xlabel('Coordinate Value', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Vertex Distribution per Axis', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[0, 1]
    scatter = ax.scatter(vertices[:, 0], vertices[:, 1], c=vertices[:, 2], 
                        cmap='viridis', s=1, alpha=0.6)
    ax.set_xlabel('X', fontweight='bold')
    ax.set_ylabel('Y', fontweight='bold')
    ax.set_title('XY Projection (colored by Z)', fontweight='bold')
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Z value')
    
    ax = axes[1, 0]
    scatter = ax.scatter(vertices[:, 0], vertices[:, 2], c=vertices[:, 1], 
                        cmap='plasma', s=1, alpha=0.6)
    ax.set_xlabel('X', fontweight='bold')
    ax.set_ylabel('Z', fontweight='bold')
    ax.set_title('XZ Projection (colored by Y)', fontweight='bold')
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Y value')
    
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"Mesh Statistics:\n"
    stats_text += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    stats_text += f"Vertices: {len(vertices):,}\n"
    stats_text += f"Faces: {len(mesh.faces):,}\n\n"
    stats_text += "Bounding Box:\n"
    stats_text += f"  X: [{vertices[:, 0].min():.4f}, {vertices[:, 0].max():.4f}]\n"
    stats_text += f"  Y: [{vertices[:, 1].min():.4f}, {vertices[:, 1].max():.4f}]\n"
    stats_text += f"  Z: [{vertices[:, 2].min():.4f}, {vertices[:, 2].max():.4f}]\n\n"
    stats_text += "Centroid:\n"
    stats_text += f"  ({vertices[:, 0].mean():.4f}, {vertices[:, 1].mean():.4f}, {vertices[:, 2].mean():.4f})\n\n"
    stats_text += f"Volume: {mesh.volume:.6f}\n"
    stats_text += f"Surface Area: {mesh.area:.6f}"
    
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

def create_comparison_plot(original_vertices, reconstructed_vertices, method, base_name, output_file):
    error = np.abs(original_vertices - reconstructed_vertices)
    error_magnitude = np.linalg.norm(error, axis=1)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Comparison: {base_name} - {method.upper()} Method', 
                 fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    scatter = ax.scatter(original_vertices[:, 0], original_vertices[:, 1], 
                        c=original_vertices[:, 2], cmap='viridis', s=1, alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Original Mesh (XY)', fontweight='bold')
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Z')
    
    ax = axes[0, 1]
    scatter = ax.scatter(reconstructed_vertices[:, 0], reconstructed_vertices[:, 1], 
                        c=reconstructed_vertices[:, 2], cmap='viridis', s=1, alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Reconstructed Mesh (XY)', fontweight='bold')
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Z')
    
    ax = axes[0, 2]
    scatter = ax.scatter(reconstructed_vertices[:, 0], reconstructed_vertices[:, 1], 
                        c=error_magnitude, cmap='Reds', s=1, alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Error Magnitude (XY)', fontweight='bold')
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Error')
    
    ax = axes[1, 0]
    ax.hist(error[:, 0], bins=30, alpha=0.7, label='X Error', color='#FF6B6B')
    ax.hist(error[:, 1], bins=30, alpha=0.7, label='Y Error', color='#4ECDC4')
    ax.hist(error[:, 2], bins=30, alpha=0.7, label='Z Error', color='#45B7D1')
    ax.set_xlabel('Error Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution per Axis', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[1, 1]
    ax.hist(error_magnitude, bins=50, alpha=0.7, color='coral')
    ax.set_xlabel('Error Magnitude')
    ax.set_ylabel('Frequency')
    ax.set_title('Overall Error Magnitude', fontweight='bold')
    ax.grid(alpha=0.3)
    
    ax = axes[1, 2]
    ax.axis('off')
    
    mse = np.mean(error ** 2)
    mae = np.mean(np.abs(error))
    max_error = error_magnitude.max()
    
    error_text = f"Error Statistics:\n"
    error_text += "━━━━━━━━━━━━━━━━━━━━━━\n"
    error_text += f"Method: {method.upper()}\n"
    error_text += f"Bins: 1024\n\n"
    error_text += f"MSE: {mse:.6e}\n"
    error_text += f"MAE: {mae:.6e}\n"
    error_text += f"Max Error: {max_error:.6e}\n\n"
    error_text += "Per-Axis MAE:\n"
    error_text += f"  X: {np.mean(np.abs(error[:, 0])):.6e}\n"
    error_text += f"  Y: {np.mean(np.abs(error[:, 1])):.6e}\n"
    error_text += f"  Z: {np.mean(np.abs(error[:, 2])):.6e}"
    
    ax.text(0.1, 0.5, error_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_mesh(obj_file, output_path, visuals_path):
    filename = Path(obj_file).name
    base_name = Path(obj_file).stem
    
    print(f"\nVisualizing: {filename}")
    
    mesh = load_mesh(obj_file)
    original_vertices = mesh.vertices
    
    original_plot_path = visuals_path / f"{base_name}_original.png"
    create_mesh_info_plot(mesh, f'Original Mesh: {base_name}', original_plot_path)
    
    stats = {
        'filename': filename,
        'visualizations_created': 1,
        'methods': {}
    }
    
    methods = ['minmax', 'sphere']
    
    for method_name in methods:
        reconstructed_vertices = load_reconstructed_vertices(output_path, base_name, method_name)
        
        if reconstructed_vertices.shape != original_vertices.shape:
            print(f"ERROR: Shape mismatch for {method_name}")
            continue
        
        comparison_plot_path = visuals_path / f"{base_name}_{method_name}_comparison.png"
        create_comparison_plot(original_vertices, reconstructed_vertices, 
                             method_name, base_name, comparison_plot_path)
        
        stats['visualizations_created'] += 1
        stats['methods'][method_name] = {
            'comparison_plot': str(comparison_plot_path.name)
        }
    
    return stats

def create_summary_plot(all_stats, output_path):
    metrics_data = {}
    
    for metrics_file in output_path.glob('*_metrics_*.json'):
        filename = metrics_file.stem
        parts = filename.split('_metrics_')
        mesh_name = parts[0]
        method = parts[1]
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        if mesh_name not in metrics_data:
            metrics_data[mesh_name] = {}
        
        metrics_data[mesh_name][method] = metrics
    
    if not metrics_data:
        return
    
    mesh_names = sorted(metrics_data.keys())
    methods = ['minmax', 'sphere']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Reconstruction Quality Summary - All Meshes', 
                 fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    x = np.arange(len(mesh_names))
    width = 0.35
    minmax_mse = [metrics_data[m]['minmax']['mse_total'] for m in mesh_names]
    sphere_mse = [metrics_data[m]['sphere']['mse_total'] for m in mesh_names]
    ax.bar(x - width/2, minmax_mse, width, label='Min-Max', alpha=0.8, color='#FF6B6B')
    ax.bar(x + width/2, sphere_mse, width, label='Sphere', alpha=0.8, color='#4ECDC4')
    ax.set_xlabel('Mesh')
    ax.set_ylabel('MSE')
    ax.set_title('Mean Squared Error Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(mesh_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    ax = axes[0, 1]
    minmax_mae = [metrics_data[m]['minmax']['mae_total'] for m in mesh_names]
    sphere_mae = [metrics_data[m]['sphere']['mae_total'] for m in mesh_names]
    ax.bar(x - width/2, minmax_mae, width, label='Min-Max', alpha=0.8, color='#FF6B6B')
    ax.bar(x + width/2, sphere_mae, width, label='Sphere', alpha=0.8, color='#4ECDC4')
    ax.set_xlabel('Mesh')
    ax.set_ylabel('MAE')
    ax.set_title('Mean Absolute Error Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(mesh_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    ax = axes[1, 0]
    improvement = [(metrics_data[m]['sphere']['mse_total'] - metrics_data[m]['minmax']['mse_total']) / 
                   metrics_data[m]['sphere']['mse_total'] * 100 for m in mesh_names]
    colors = ['green' if i > 0 else 'red' for i in improvement]
    ax.bar(x, improvement, alpha=0.8, color=colors)
    ax.set_xlabel('Mesh')
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Min-Max vs Sphere (% MSE Reduction)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(mesh_names, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(alpha=0.3, axis='y')
    
    ax = axes[1, 1]
    ax.axis('off')
    
    avg_mse_minmax = np.mean(minmax_mse)
    avg_mse_sphere = np.mean(sphere_mse)
    avg_mae_minmax = np.mean(minmax_mae)
    avg_mae_sphere = np.mean(sphere_mae)
    
    summary_text = f"Overall Summary:\n"
    summary_text += "━━━━━━━━━━━━━━━━━━━━━━\n"
    summary_text += f"Total Meshes: {len(mesh_names)}\n"
    summary_text += f"Quantization: 1024 bins\n\n"
    summary_text += "Average MSE:\n"
    summary_text += f"  Min-Max: {avg_mse_minmax:.6e}\n"
    summary_text += f"  Sphere:  {avg_mse_sphere:.6e}\n\n"
    summary_text += "Average MAE:\n"
    summary_text += f"  Min-Max: {avg_mae_minmax:.6e}\n"
    summary_text += f"  Sphere:  {avg_mae_sphere:.6e}\n\n"
    if avg_mse_minmax < avg_mse_sphere:
        summary_text += "Winner: Min-Max"
    else:
        summary_text += "Winner: Sphere"
    
    ax.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
            verticalalignment='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
    plt.tight_layout()
    summary_path = output_path / 'visuals' / 'summary_all_meshes.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()

def update_log(all_stats):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path('logs') / f'step7_{timestamp}.log'
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"Step 7 - Visualize Meshes - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Created visualizations for {len(all_stats)} mesh files\n\n")
        
        for stat in all_stats:
            f.write(f"File: {stat['filename']}\n")
            f.write(f"  Visualizations created: {stat['visualizations_created']}\n")
            for method_name in stat['methods']:
                f.write(f"    {method_name}: {stat['methods'][method_name]['comparison_plot']}\n")
            f.write("\n")

def main():
    print("STEP 7 - Visualize Original vs Reconstructed Meshes")
    
    obj_files = list_mesh_files('8samples')
    print(f"Found {len(obj_files)} .obj files to visualize.")
    
    output_path = Path('outputs')
    visuals_path = output_path / 'visuals'
    visuals_path.mkdir(parents=True, exist_ok=True)
    
    if not output_path.exists():
        print(f"\nERROR: Output directory not found!")
        print(f"Please run previous steps first.")
        sys.exit(1)
    
    all_stats = []
    
    for obj_file in obj_files:
        stats = visualize_mesh(obj_file, output_path, visuals_path)
        all_stats.append(stats)
    
    create_summary_plot(all_stats, output_path)
    update_log(all_stats)
    
    print(f"\nStep 7 completed successfully!")
    print(f"Created visualizations for {len(obj_files)} mesh files")
    print(f"Visualizations saved in outputs/visuals/")

if __name__ == "__main__":
    main()
