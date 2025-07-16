import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
from config import OUTPUT_DIR, ANGLES

def create_output_dir():
    """Create output directory if it doesn't exist"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def visualize_projections(projections, sample_id, save=True):
    """Visualize the input projection images"""
    fig, axes = plt.subplots(1, len(projections), figsize=(15, 5))
    fig.suptitle(f'Input Projections for Sample {sample_id}', fontsize=16)
    
    for i, proj in enumerate(projections):
        ax = axes[i] if len(projections) > 1 else axes
        ax.imshow(proj.cpu().numpy(), cmap='gray')
        ax.set_title(f'Projection {i}')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save:
        create_output_dir()
        plt.savefig(os.path.join(OUTPUT_DIR, f'projections_sample_{sample_id}.png'), 
                   dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig

def visualize_3d_volume(volume, title, save_path=None, max_points=3000):
    """Visualize 3D volume as 3D scatter plot with angles matching config ANGLES"""
    # Dynamically calculate number of subplots based on config ANGLES
    num_views = len(ANGLES)
    fig_width = min(5 * num_views, 20)  # Limit maximum width
    fig = plt.figure(figsize=(fig_width, 5))
    
    # Get coordinates of non-zero voxels
    coords = torch.nonzero(volume).cpu().numpy()
    
    if len(coords) == 0:
        print(f"Warning: No voxels found in volume for {title}")
        return None
    
    # Subsample if too many points
    if len(coords) > max_points:
        indices = np.random.choice(len(coords), max_points, replace=False)
        coords = coords[indices]
    
    # Generate views based on config ANGLES
    views = []
    for i, (theta, phi) in enumerate(ANGLES):
        # Convert radians to degrees
        elev_deg = np.degrees(theta)
        azim_deg = np.degrees(phi)
        
        # For better 3D visualization, add some elevation if theta is 0
        if theta == 0:
            elev_deg = 20  # Add slight elevation for better perspective
        
        # Generate view name based on actual angles
        view_name = f'View {i+1}: phi={azim_deg:.0f}°'
        if theta != 0:
            view_name += f', theta={np.degrees(theta):.0f}°'
        
        views.append((elev_deg, azim_deg, view_name))
    
    for i, (elev, azim, view_name) in enumerate(views):
        ax = fig.add_subplot(1, num_views, i+1, projection='3d')
        
        # Plot points with color based on Z coordinate for depth perception
        scatter = ax.scatter(coords[:, 2], coords[:, 0], coords[:, 1],
                           c=coords[:, 0], cmap='viridis', s=1, alpha=0.8)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Z') 
        ax.set_zlabel('Y')
        ax.set_title(view_name)
        ax.view_init(elev=elev, azim=azim)
        
        # Set equal aspect ratio
        max_range = np.array([coords[:, 2].max()-coords[:, 2].min(),
                             coords[:, 0].max()-coords[:, 0].min(),
                             coords[:, 1].max()-coords[:, 1].min()]).max() / 2.0
        mid_x = (coords[:, 2].max()+coords[:, 2].min()) * 0.5
        mid_y = (coords[:, 0].max()+coords[:, 0].min()) * 0.5
        mid_z = (coords[:, 1].max()+coords[:, 1].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig

def create_volume_slices(volume, sample_id, save_path=None):
    """Create volume slices with orientations matching config ANGLES"""
    volume_np = volume.cpu().numpy() if isinstance(volume, torch.Tensor) else volume
    
    # Dynamically determine number of slices based on config ANGLES
    num_slices = min(len(ANGLES), 3)  # Limit to 3 slices for display
    fig_width = 5 * num_slices
    fig, axes = plt.subplots(1, num_slices, figsize=(fig_width, 5))
    
    # If only one slice, make axes iterable
    if num_slices == 1:
        axes = [axes]
    
    # Get middle indices
    z_mid = volume_np.shape[0] // 2
    y_mid = volume_np.shape[1] // 2
    x_mid = volume_np.shape[2] // 2
    
    # Define slice types
    slice_types = [
        (volume_np[z_mid, :, :], 'Axial', z_mid, 'Z'),
        (volume_np[:, y_mid, :], 'Coronal', y_mid, 'Y'),
        (volume_np[:, :, x_mid], 'Sagittal', x_mid, 'X')
    ]
    
    for i in range(num_slices):
        slice_data, slice_type, slice_idx, axis = slice_types[i % len(slice_types)]
        
        # Get corresponding angle from config
        if i < len(ANGLES):
            theta, phi = ANGLES[i]
            angle_info = f'phi={np.degrees(phi):.0f}°'
            if theta != 0:
                angle_info += f', theta={np.degrees(theta):.0f}°'
            title = f'{slice_type} Slice ({axis}={slice_idx})\nAngle: {angle_info}'
        else:
            title = f'{slice_type} Slice ({axis}={slice_idx})'
        
        axes[i].imshow(slice_data, cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.suptitle(f'Volume Slices - Sample {sample_id}\n(Orientations based on config ANGLES)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig

def plot_loss_curve(loss_history, sample_id, save=True):
    """Plot the optimization loss curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Reconstruction Loss for Sample {sample_id}')
    plt.grid(True)
    plt.yscale('log')
    
    if save:
        create_output_dir()
        plt.savefig(os.path.join(OUTPUT_DIR, f'loss_curve_sample_{sample_id}.png'), 
                   dpi=150, bbox_inches='tight')
    
    plt.show()

def visualize_reconstruction_results(result, projections, sample_id):
    """Comprehensive visualization of reconstruction results - Fixed: removed duplicate 3D visualization"""
    create_output_dir()
    
    # Print angle information from config
    print(f"\nUsing {len(ANGLES)} projection angles from config:")
    for i, (theta, phi) in enumerate(ANGLES):
        print(f"  Angle {i}: theta={np.degrees(theta):.1f}°, phi={np.degrees(phi):.1f}°")
    
    # 1. Input projections (already called in main)
    # visualize_projections(projections, sample_id)
    
    # 2. Fixed: removed duplicate 3D visualization - already called in main.py through reconstruction.py
    
    # 3. Volume slices with angle-based orientations
    #save_path_slices = os.path.join(OUTPUT_DIR, f'reconstruction_slices_sample_{sample_id}.png')
    #create_volume_slices(result['reconstructed'], sample_id, save_path_slices)
    
    # 4. Loss curve
    plot_loss_curve(result['loss_history'], sample_id)
    
    # 5. Summary statistics
    print(f"\nReconstruction Summary for Sample {sample_id}:")
    print(f"  Final Loss: {result['final_loss']:.6f}")
    print(f"  Non-zero voxels: {torch.sum(result['reconstructed']).item():.0f}")
    print(f"  Volume fill ratio: {torch.sum(result['reconstructed']).item() / result['reconstructed'].numel() * 100:.2f}%")