import torch
import numpy as np
import matplotlib.pyplot as plt
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_voxel_spacing(physical_dims, SOD_list, SID_list, pixel_spacing_list, angles_list):
    """Calculate required voxel spacing based on theory in Section 3.1
    Args:
        physical_dims: physical size (w_x, w_y, w_z) mm
        SOD_list: list of SOD values or single SOD value
        SID_list: list of SID values or single SID value  
        pixel_spacing_list: list of pixel spacing tuples or single tuple
        angles_list: list of (alpha, beta) angle pairs
    Returns:
        voxel_spacing: (d_x, d_y, d_z) mm
        resolution: (res_x, res_y, res_z) 
    """
    EPS = 1e-8
    
    # Convert single values to lists for uniform processing
    if not isinstance(SOD_list, list):
        SOD_list = [SOD_list]
    if not isinstance(SID_list, list):
        SID_list = [SID_list]
    if not isinstance(pixel_spacing_list, list):
        pixel_spacing_list = [pixel_spacing_list]
    if not isinstance(angles_list, list):
        angles_list = [angles_list]
    
    # Check lengths
    num_angles = len(angles_list)
    
    # If single values provided, replicate them for all angles
    if len(SOD_list) == 1 and num_angles > 1:
        SOD_list = SOD_list * num_angles
    if len(SID_list) == 1 and num_angles > 1:
        SID_list = SID_list * num_angles
    if len(pixel_spacing_list) == 1 and num_angles > 1:
        pixel_spacing_list = pixel_spacing_list * num_angles
        
    # Check if lists have the same length
    if len(SOD_list) != num_angles or len(SID_list) != num_angles or len(pixel_spacing_list) != num_angles:
        raise ValueError(f"Lists' lengths should be the same. Got: SOD={len(SOD_list)}, SID={len(SID_list)}, pixel_spacing={len(pixel_spacing_list)}, angles={num_angles}")
    
    w_x, w_y, w_z = physical_dims
    
    # initialize bound
    d_x_bound = float('inf')
    d_y_bound = float('inf')
    d_z_bound = float('inf')
    
    # create voxel vertices
    half_x = w_x / 2
    half_y = w_y / 2
    half_z = w_z / 2
    vertices = torch.tensor([
        [-half_x, -half_y, -half_z],
        [-half_x, -half_y, half_z],
        [-half_x, half_y, -half_z],
        [-half_x, half_y, half_z],
        [half_x, -half_y, -half_z],
        [half_x, -half_y, half_z],
        [half_x, half_y, -half_z],
        [half_x, half_y, half_z]
    ], device=device)  
    
    # traverse all angles and parameters
    for idx in range(num_angles):
        alpha, beta = angles_list[idx]
        SOD = SOD_list[idx]
        SID = SID_list[idx]
        pixel_spacing = pixel_spacing_list[idx]  # pixel spacing is tuple (δx, δy)
        
        cos_a = math.cos(alpha)
        sin_a = math.sin(alpha)
        cos_b = math.cos(beta)
        sin_b = math.sin(beta)
        
        # rotation matrix
        Rx = torch.tensor([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ], device=device)
        
        Ry = torch.tensor([
            [cos_b, 0, sin_b],
            [0, 1, 0],
            [-sin_b, 0, cos_b]
        ], device=device)
        
        # composite rotation matrix
        R = torch.matmul(Ry, Rx)
        
        # rotation vertices
        rotated_vertices = torch.matmul(vertices, R.T) 
        
        # transform to X-ray source (z-coord plus an SOD)
        z_prime = rotated_vertices[:, 2] + SOD
        
        # find min(Qz')
        min_z_prime = z_prime.min().item()
        
        # avoid divided by too small min(Qz') (e.g., 0)
        min_z_prime = max(min_z_prime, EPS)
        
        # calculate the angle bound (equation 9 and 11)
        # bound on x
        d_x_ang = float('inf')
        if abs(cos_b) > EPS:
            d_x_ang = (pixel_spacing[0] * min_z_prime) / (abs(cos_b) * SID)
        
        # bound on y
        d_y_ang1 = float('inf')
        d_y_ang2 = float('inf')
        if abs(sin_b * sin_a) > EPS:
            d_y_ang1 = (pixel_spacing[0] * min_z_prime) / (abs(sin_b * sin_a) * SID)
        if abs(cos_a) > EPS:
            d_y_ang2 = (pixel_spacing[1] * min_z_prime) / (abs(cos_a) * SID)
        d_y_ang = min(d_y_ang1, d_y_ang2)
        
        # bound on z
        d_z_ang1 = float('inf')
        d_z_ang2 = float('inf')
        if abs(sin_b * cos_a) > EPS:
            d_z_ang1 = (pixel_spacing[0] * min_z_prime) / (abs(sin_b * cos_a) * SID)
        if abs(sin_a) > EPS:
            d_z_ang2 = (pixel_spacing[1] * min_z_prime) / (abs(sin_a) * SID)
        d_z_ang = min(d_z_ang1, d_z_ang2)
        
        # update bound
        d_x_bound = min(d_x_bound, d_x_ang)
        d_y_bound = min(d_y_bound, d_y_ang)
        d_z_bound = min(d_z_bound, d_z_ang)
    
    # avoid infinity
    d_x_bound = d_x_bound if d_x_bound != float('inf') else 1.0
    d_y_bound = d_y_bound if d_y_bound != float('inf') else 1.0
    d_z_bound = d_z_bound if d_z_bound != float('inf') else 1.0
    
    # calculate resolution
    res_x = math.ceil(w_x / d_x_bound) if d_x_bound > 0 else 1
    res_y = math.ceil(w_y / d_y_bound) if d_y_bound > 0 else 1
    res_z = math.ceil(w_z / d_z_bound) if d_z_bound > 0 else 1
    
    return (d_x_bound, d_y_bound, d_z_bound), (res_x, res_y, res_z)

def volume_statistics(volume):
    # Convert PyTorch tensor to numpy array if needed
    if isinstance(volume, torch.Tensor):
        volume_np = volume.cpu().numpy()  # Move to CPU first, then convert
    else:
        volume_np = volume
    
    # Calculate total number of voxels in the volume
    total_voxels = volume_np.size
    
    # Count non-zero voxels (active/occupied voxels)
    active_voxels = np.sum(volume_np > 0)
    
    # Calculate fill ratio (density measure)
    # Low values indicate sparse volumes, high values indicate dense volumes
    # Useful for determining if sparse data structures would be beneficial
    fill_ratio = active_voxels / total_voxels
    
    # Return comprehensive statistics dictionary
    return {
        'total_voxels': total_voxels,           # Total memory footprint indicator
        'active_voxels': int(active_voxels),    # Actual object size
        'fill_ratio': fill_ratio,               # Sparsity measure (0=empty, 1=full)
        'volume_shape': volume_np.shape         # 3D dimensions (Z, Y, X)
    }