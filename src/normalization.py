import numpy as np

def norm_minmax(vertices):
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Invalid vertices shape: {vertices.shape}. Expected (V, 3)")
    
    v_min = vertices.min(axis=0)
    v_max = vertices.max(axis=0)
    
    range_per_axis = v_max - v_min
    range_per_axis[range_per_axis == 0] = 1.0
    
    normalized = (vertices - v_min) / range_per_axis
    
    params = {
        'method': 'minmax',
        'v_min': v_min.copy(),
        'v_max': v_max.copy()
    }
    
    return normalized, params

def denorm_minmax(normalized, params):
    if params['method'] != 'minmax':
        raise ValueError(f"Invalid method in params: {params['method']}. Expected 'minmax'")
    
    if normalized.ndim != 2 or normalized.shape[1] != 3:
        raise ValueError(f"Invalid normalized shape: {normalized.shape}. Expected (V, 3)")
    
    v_min = params['v_min']
    v_max = params['v_max']
    
    reconstructed = normalized * (v_max - v_min) + v_min
    
    return reconstructed

def norm_sphere(vertices):
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Invalid vertices shape: {vertices.shape}. Expected (V, 3)")
    
    center = vertices.mean(axis=0)
    shifted = vertices - center
    distances = np.linalg.norm(shifted, axis=1)
    scale = np.max(distances)
    
    if scale == 0:
        scale = 1.0
    
    normalized = shifted / scale
    
    params = {
        'method': 'sphere',
        'center': center.copy(),
        'scale': float(scale)
    }
    
    return normalized, params

def denorm_sphere(normalized, params):
    if params['method'] != 'sphere':
        raise ValueError(f"Invalid method in params: {params['method']}. Expected 'sphere'")
    
    if normalized.ndim != 2 or normalized.shape[1] != 3:
        raise ValueError(f"Invalid normalized shape: {normalized.shape}. Expected (V, 3)")
    
    center = params['center']
    scale = params['scale']
    
    reconstructed = normalized * scale + center
    
    return reconstructed

def validate_normalization(vertices, normalized, params, method_name):
    print(f"\nValidating {method_name} normalization...")
    
    if vertices.shape != normalized.shape:
        print(f"  ✗ Shape mismatch: {vertices.shape} vs {normalized.shape}")
        return False
    print(f"  ✓ Shapes match: {vertices.shape}")
    
    if method_name == "Min-Max":
        n_min = normalized.min(axis=0)
        n_max = normalized.max(axis=0)
        print(f"  ✓ Normalized range: x=[{n_min[0]:.6f}, {n_max[0]:.6f}], "
              f"y=[{n_min[1]:.6f}, {n_max[1]:.6f}], z=[{n_min[2]:.6f}, {n_max[2]:.6f}]")
        
        if np.any(n_min < -1e-6) or np.any(n_max > 1 + 1e-6):
            print(f"  ✗ Warning: Values outside [0,1] range")
    
    elif method_name == "Unit Sphere":
        distances = np.linalg.norm(normalized, axis=1)
        max_dist = np.max(distances)
        print(f"  ✓ Max distance from origin: {max_dist:.6f} (should be ≤ 1.0)")
        
        if max_dist > 1 + 1e-6:
            print(f"  ✗ Warning: Points outside unit sphere")
    
    return True

def test_normalization_methods():
    print("=" * 60)
    print("Testing Normalization Methods")
    print("=" * 60)
    
    test_vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 0.5, 1.5],
        [-1.0, 2.0, 0.5]
    ], dtype=np.float64)
    
    print(f"\nTest vertices (shape {test_vertices.shape}):")
    print(test_vertices)
    
    print("\n" + "-" * 60)
    print("Testing Min-Max Normalization")
    print("-" * 60)
    normalized_mm, params_mm = norm_minmax(test_vertices)
    print(f"Normalized (Min-Max):")
    print(normalized_mm)
    print(f"\nParameters: v_min={params_mm['v_min']}, v_max={params_mm['v_max']}")
    
    validate_normalization(test_vertices, normalized_mm, params_mm, "Min-Max")
    
    reconstructed_mm = denorm_minmax(normalized_mm, params_mm)
    error_mm = np.max(np.abs(test_vertices - reconstructed_mm))
    print(f"\n✓ Denormalization max error: {error_mm:.10f}")
    
    print("\n" + "-" * 60)
    print("Testing Unit Sphere Normalization")
    print("-" * 60)
    normalized_sp, params_sp = norm_sphere(test_vertices)
    print(f"Normalized (Unit Sphere):")
    print(normalized_sp)
    print(f"\nParameters: center={params_sp['center']}, scale={params_sp['scale']}")
    
    validate_normalization(test_vertices, normalized_sp, params_sp, "Unit Sphere")
    
    reconstructed_sp = denorm_sphere(normalized_sp, params_sp)
    error_sp = np.max(np.abs(test_vertices - reconstructed_sp))
    print(f"\n✓ Denormalization max error: {error_sp:.10f}")
    
    print("\n" + "=" * 60)
    print("✓ All normalization tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_normalization_methods()
