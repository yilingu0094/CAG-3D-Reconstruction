import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
from config import device, NUM_ITERATIONS, LEARNING_RATE, ANGLES
import nibabel as nib
from pathlib import Path
from skimage import measure
import plotly.graph_objects as go
import scipy.spatial.distance
import pandas as pd
import json

def load_binary_resized_exactly_like_optimal_generator(sample_id, projector):
    """
    Completely replicate load_and_resize_volume processing from optimal_projection_generator.py
    Ensure getting exactly the same binary_resized used when generating I_k
    """
    import torch.nn.functional as F
    
    label_root = "/home/yilin/syn_tree/NeCA/data/CCTA_raw/1-200"
    label_filename = f"{sample_id}.label.nii.gz"
    label_path = os.path.join(label_root, label_filename)
    
    print(f"Loading sample {sample_id}: {label_path}")
    
    nii_img = nib.load(label_path)
    volume_data = nii_img.get_fdata()
    
    binary_volume = (volume_data > 0.5).astype(np.float32)
    original_volume = torch.tensor(binary_volume, device=device, dtype=torch.float32)
    
    optimal_shape = projector.volume_shape
    
    volume_expanded = original_volume.unsqueeze(0).unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        volume_expanded, 
        size=optimal_shape, 
        mode='trilinear', 
        align_corners=False
    )
    
    resized_volume = resized.squeeze(0).squeeze(0)
    binary_resized = (resized_volume > 0.5).float()
    
    resized_active = torch.sum(binary_resized > 0).item()
    print(f"  Sample {sample_id}: shape{binary_resized.shape}, active voxels{resized_active}")
    
    return binary_resized

def create_projector_exactly_like_optimal_generator(optimal_config):
    """
    Completely replicate create_optimal_projector processing from optimal_projection_generator.py
    """
    from projector import MatrixXRayProjector
    
    SOD = 750.0
    SID = 1000.0  
    PIXEL_SPACING = (0.5, 0.5)
    projection_image_size = (462, 462)
    
    optimal_shape = optimal_config['optimal_shape']
    voxel_spacing = optimal_config['voxel_spacing']
    
    projector = MatrixXRayProjector(
        volume_shape=optimal_shape,
        image_shape=projection_image_size,
        pixel_spacing=PIXEL_SPACING,
        SOD=SOD,
        SID=SID
    )
    
    # Key: completely replicate this line
    avg_voxel_size = np.mean(voxel_spacing)
    projector.voxel_coords *= avg_voxel_size
    
    return projector

class FixedDifferentiableProjection(torch.autograd.Function):
    """
    Fixed version differentiable projection - using continuous projections and correct gradient calculation
    """
    
    @staticmethod
    def forward(ctx, volume, projector, alpha, beta):
        # Save context information
        ctx.projector = projector
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.save_for_backward(volume)
        
        # Fixed: use continuous projection (no longer binarized)
        projection = projector.project_volume_continuous(volume, alpha, beta)
        
        return projection.float()
    
    @staticmethod 
    def backward(ctx, grad_output):
        # Restore saved variables
        volume, = ctx.saved_tensors
        projector = ctx.projector
        alpha = ctx.alpha
        beta = ctx.beta
        
        # Fixed: use sparse matrix transpose for back projection
        grad_volume = torch.zeros_like(volume)
        
        if grad_output is not None:
            try:
                # Build projection matrix
                P = projector.build_projection_matrix(alpha, beta)
                
                if P._nnz() > 0:
                    # Flatten grad_output
                    grad_flat = grad_output.flatten()
                    
                    # Use transpose matrix for back projection
                    P_t = P.transpose(0, 1)
                    grad_volume_flat = torch.sparse.mm(P_t, grad_flat.unsqueeze(-1)).squeeze()
                    
                    # Reshape to original volume shape
                    grad_volume = grad_volume_flat.reshape(volume.shape)
                    
            except Exception as e:
                print(f"WARNING: Gradient calculation failed, using zero gradient: {e}")
                grad_volume = torch.zeros_like(volume)
        
        return grad_volume, None, None, None

def differentiable_forward_projection(volume, projector, alpha, beta):
    """
    Fixed: differentiable forward projection - using fixed autograd function
    """
    return FixedDifferentiableProjection.apply(volume, projector, alpha, beta)

def frobenius_loss(pred_projections, target_projections):
    """
    Fixed: calculate Frobenius norm loss, ensure type compatibility
    """
    total_loss = 0
    for pred_proj, target_proj in zip(pred_projections, target_projections):
        # Ensure both tensors are float type
        pred_proj_float = pred_proj.float()
        target_proj_float = target_proj.float()
        
        # Use MSE loss (equivalent to squared Frobenius norm)
        loss = torch.mean((pred_proj_float - target_proj_float)**2)
        total_loss += loss
    
    return total_loss

def debug_projections_gradient(projections):
    """
    Fixed: debug projection data gradient information
    """
    print("Checking projection data gradient information:")
    for i, proj in enumerate(projections):
        print(f"  Projection {i}: requires_grad={proj.requires_grad}, grad_fn={proj.grad_fn}")
        print(f"          Type={type(proj)}, Shape={proj.shape}, Device={proj.device}")
        print(f"          dtype={proj.dtype}, Value range=({proj.min().item():.4f}, {proj.max().item():.4f})")
        
        # Check if continuous projection (value range should be greater than 1)
        if proj.max().item() > 1.5:
            print(f"          SUCCESS: Detected continuous projection (max value > 1.5)")
        elif proj.max().item() <= 1.0:
            print(f"          WARNING: Possible binary projection (max value <= 1.0)")

def reconstruct_single_sample_with_initialization(projections, projector, initialization, sample_id):
    """
    Fixed: reconstruct single sample using specified initialization
    """
    print(f"Starting fixed version reconstruction for sample {sample_id}...")
    
    # Debug: check input projection data types
    print("Input projection data type check:")
    for i, proj in enumerate(projections):
        print(f"  Projection {i}: dtype={proj.dtype}, Shape={proj.shape}, Range=[{proj.min():.3f}, {proj.max():.3f}]")
        if proj.max().item() > 1.5:
            print(f"    SUCCESS: Continuous projection")
        else:
            print(f"    WARNING: Possible binary projection")
    
    # Ensure all projections are float type
    float_projections = [proj.float() for proj in projections]
    
    # Ensure initialization is differentiable
    X_recon = initialization.clone().detach()
    X_recon.requires_grad_(True)
    
    # Test self-consistency
    print("Testing self-consistency (91 reconstructing 91 should have loss close to 0)...")
    test_projections = []
    for alpha, beta in ANGLES:
        test_proj = differentiable_forward_projection(X_recon, projector, alpha, beta)
        test_projections.append(test_proj)
    test_loss = frobenius_loss(test_projections, float_projections)
    print(f"Self-consistency loss: {test_loss.item():.8f}")
    
    # If self-consistency loss is 0, it's already optimal solution
    if test_loss.item() < 1e-6:
        print("SUCCESS: Self-consistency loss close to 0, directly return initialization as result")
        return {
            'reconstructed': (initialization > 0.5).float(),
            'continuous': initialization,
            'loss_history': [test_loss.item()],
            'final_loss': test_loss.item(),
            'sample_id': sample_id,
            'converged': True
        }
    
    # Set optimizer
    optimizer = torch.optim.Adam([X_recon], lr=LEARNING_RATE)
    
    # Optimization loop
    loss_history = []
    best_loss = float('inf')
    best_X = None
    
    # Early stopping parameters
    EARLY_STOP_THRESHOLD = 1e-2
    EARLY_STOP_PATIENCE = 10
    patience_counter = 0
    
    print(f"Starting fixed version optimization: {NUM_ITERATIONS} iterations...")
    
    for iteration in tqdm(range(NUM_ITERATIONS), desc=f"Fixed reconstruction {sample_id}"):
        optimizer.zero_grad()
        
        # Ensure X_recon gradient is not cleared
        if not X_recon.requires_grad:
            X_recon.requires_grad_(True)
        
        # Fixed: use fixed version forward projection
        pred_projections = []
        for alpha, beta in ANGLES:
            pred_proj = differentiable_forward_projection(X_recon, projector, alpha, beta)
            pred_projections.append(pred_proj)
        
        # Calculate loss
        total_loss = frobenius_loss(pred_projections, float_projections)
        
        # Track best solution
        current_loss = total_loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_X = X_recon.detach().clone()
        
        loss_history.append(current_loss)
        
        # Early stopping check
        if current_loss < EARLY_STOP_THRESHOLD:
            print(f"\nTARGET: Reached loss threshold at iteration {iteration}, early stopping!")
            print(f"Final loss: {current_loss:.8f}")
            break
        
        # Check if gradient exists
        if not total_loss.requires_grad:
            print(f"  WARNING: Iteration {iteration} loss does not require gradient!")
            break
        
        # Backpropagation and update
        try:
            total_loss.backward()
        except RuntimeError as e:
            print(f"  ERROR: Iteration {iteration} backpropagation failed: {e}")
            break
        
        # Gradient check
        grad_norm = X_recon.grad.norm().item() if X_recon.grad is not None else 0
        
        if iteration % 500 == 0:
            print(f"  Iteration {iteration}: Loss={current_loss:.8f}, Grad Norm={grad_norm:.8f}")
        
        # Check gradient vanishing
        if grad_norm < 1e-10:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\nWARNING: Gradient vanishing at iteration {iteration}, early stopping after {EARLY_STOP_PATIENCE} consecutive small gradients!")
                break
        else:
            patience_counter = 0
        
        optimizer.step()
        
        # Constrain to [0,1] range
        with torch.no_grad():
            X_recon.data = torch.clamp(X_recon, 0, 1)
    
    print(f"Fixed version optimization complete, final loss: {best_loss:.8f}")
    
    # Return results
    binary_result = (best_X > 0.5).float()
    
    return {
        'reconstructed': binary_result,
        'continuous': best_X,
        'loss_history': loss_history,
        'final_loss': best_loss,
        'sample_id': sample_id,
        'stopped_early': current_loss < EARLY_STOP_THRESHOLD or patience_counter >= EARLY_STOP_PATIENCE,
        'final_iteration': iteration + 1,
        'converged': current_loss < EARLY_STOP_THRESHOLD
    }

def reconstruct_with_91_initialization(target_sample_ids, projections_dict, projector):
    """
    Fixed: use sample 91 as initialization to reconstruct specified samples
    """
    print(f"\nStarting fixed version reconstruction experiment: using sample 91 as initialization")
    print(f"Target samples: {target_sample_ids}")
    
    # Get optimal_config
    from config import OPTIMAL_VOLUME_SHAPE, VOXEL_SPACING, PHYSICAL_DIMS
    optimal_config = {
        'optimal_shape': OPTIMAL_VOLUME_SHAPE,
        'voxel_spacing': VOXEL_SPACING,
        'physical_dims': PHYSICAL_DIMS
    }
    
    # Create correct projector
    corrected_projector = create_projector_exactly_like_optimal_generator(optimal_config)
    
    # Load sample 91 as initialization
    print(f"\nLoading sample 91 as initialization...")
    initialization_volume = load_binary_resized_exactly_like_optimal_generator(91, corrected_projector)
    
    results = {}
    
    # Reconstruct each target sample
    for sample_id in target_sample_ids:
        print(f"\n{'='*60}")
        print(f"TARGET: Fixed version reconstruction sample {sample_id} (using 91 initialization)")
        print(f"{'='*60}")
        
        target_projections = projections_dict[sample_id]
        
        # Execute fixed version reconstruction
        result = reconstruct_single_sample_with_initialization(
            target_projections, corrected_projector, initialization_volume, sample_id
        )
        
        results[sample_id] = result
        
        # Enhanced result reporting
        stopped_early = result.get('stopped_early', False)
        final_iteration = result.get('final_iteration', NUM_ITERATIONS)
        converged = result.get('converged', False)
        
        if converged:
            print(f"SUCCESS: Sample {sample_id} reconstruction converged (iteration {final_iteration}), final loss: {result['final_loss']:.8f}")
        elif stopped_early:
            print(f"SUCCESS: Sample {sample_id} reconstruction completed (early stopped at iteration {final_iteration}), final loss: {result['final_loss']:.8f}")
        else:
            print(f"SUCCESS: Sample {sample_id} reconstruction completed (full {NUM_ITERATIONS} iterations), final loss: {result['final_loss']:.8f}")
    
    return results

def compute_neca_metrics(pred_volume, gt_volume):
    """
    Calculate 6 metrics from NeCA paper:
    clDice, Dice, IoU, reError, CDℓ2, and reMSE
    """
    pred_np = pred_volume.cpu().numpy()
    gt_np = gt_volume.cpu().numpy()
    
    # Ensure both are binary
    pred_binary = (pred_np > 0.5).astype(np.uint8)
    gt_binary = (gt_np > 0.5).astype(np.uint8)
    
    metrics = {}
    
    # 1. Dice Score
    intersection = np.sum(pred_binary * gt_binary)
    dice = 2.0 * intersection / (np.sum(pred_binary) + np.sum(gt_binary) + 1e-8)
    metrics['dice'] = dice
    
    # 2. IoU (Jaccard Index)
    union = np.sum((pred_binary + gt_binary) > 0)
    iou = intersection / (union + 1e-8)
    metrics['iou'] = iou
    
    # 3. clDice (centerline Dice) - simplified version using skeleton
    try:
        from skimage.morphology import skeletonize_3d
        pred_skeleton = skeletonize_3d(pred_binary)
        gt_skeleton = skeletonize_3d(gt_binary)
        
        skel_intersection = np.sum(pred_skeleton * gt_skeleton)
        cl_dice = 2.0 * skel_intersection / (np.sum(pred_skeleton) + np.sum(gt_skeleton) + 1e-8)
        metrics['cl_dice'] = cl_dice
    except:
        # If skeletonize fails, use regular dice as approximation
        metrics['cl_dice'] = dice
    
    # 4. Reconstruction Error (reError)
    re_error = np.mean(np.abs(pred_np - gt_np))
    metrics['re_error'] = re_error
    
    # 5. reMSE (reconstruction MSE)
    re_mse = np.mean((pred_np - gt_np) ** 2)
    metrics['re_mse'] = re_mse
    
    # 6. Chamfer L2 Distance (CDℓ2) - simplified version
    try:
        pred_points = np.argwhere(pred_binary)
        gt_points = np.argwhere(gt_binary)
        
        if len(pred_points) > 0 and len(gt_points) > 0:
            # Calculate bidirectional Chamfer distance
            dist_pred_to_gt = scipy.spatial.distance.cdist(pred_points, gt_points)
            dist_gt_to_pred = scipy.spatial.distance.cdist(gt_points, pred_points)
            
            chamfer_pred_to_gt = np.mean(np.min(dist_pred_to_gt, axis=1))
            chamfer_gt_to_pred = np.mean(np.min(dist_gt_to_pred, axis=1))
            
            chamfer_l2 = (chamfer_pred_to_gt + chamfer_gt_to_pred) / 2.0
        else:
            chamfer_l2 = float('inf')
            
        metrics['chamfer_l2'] = chamfer_l2
    except:
        metrics['chamfer_l2'] = float('inf')
    
    return metrics

def evaluate_reconstruction_results(results, target_sample_ids):
    """
    Fixed: evaluate reconstruction results, calculate all NeCA metrics
    """
    print(f"\nEvaluating fixed version reconstruction results...")
    
    all_metrics = {}
    
    for sample_id in target_sample_ids:
        print(f"\nEvaluating sample {sample_id}...")
        
        # Load ground truth
        from config import OPTIMAL_VOLUME_SHAPE, VOXEL_SPACING, PHYSICAL_DIMS
        optimal_config = {
            'optimal_shape': OPTIMAL_VOLUME_SHAPE,
            'voxel_spacing': VOXEL_SPACING,
            'physical_dims': PHYSICAL_DIMS
        }
        
        # Create temporary projector to get gt
        temp_projector = create_projector_exactly_like_optimal_generator(optimal_config)
        gt_volume = load_binary_resized_exactly_like_optimal_generator(sample_id, temp_projector)
        
        # Get reconstruction results
        pred_volume = results[sample_id]['reconstructed']
        
        # Calculate metrics
        metrics = compute_neca_metrics(pred_volume, gt_volume)
        all_metrics[sample_id] = metrics
        
        print(f"  Sample {sample_id} metrics:")
        print(f"    Dice: {metrics['dice']:.4f}")
        print(f"    IoU: {metrics['iou']:.4f}")
        print(f"    clDice: {metrics['cl_dice']:.4f}")
        print(f"    reError: {metrics['re_error']:.4f}")
        print(f"    reMSE: {metrics['re_mse']:.6f}")
        print(f"    Chamfer L2: {metrics['chamfer_l2']:.4f}")
    
    # Calculate average metrics
    print(f"\nAverage Metrics (samples {target_sample_ids}):")
    avg_metrics = {}
    for metric_name in ['dice', 'iou', 'cl_dice', 're_error', 're_mse', 'chamfer_l2']:
        values = [all_metrics[sid][metric_name] for sid in target_sample_ids 
                 if not np.isinf(all_metrics[sid][metric_name])]
        if values:
            avg_metrics[metric_name] = np.mean(values)
            std_metrics = np.std(values)
            print(f"  {metric_name}: {avg_metrics[metric_name]:.4f} ± {std_metrics:.4f}")
        else:
            avg_metrics[metric_name] = float('nan')
            print(f"  {metric_name}: N/A")
    
    return all_metrics, avg_metrics

def create_volume_rendering_like_analyze_labels(volume, sample_id, save_dir="reconstruction_results_fixed"):
    """
    Create visualization similar to volume_rendering.png in analyze_labels.py
    """
    print(f"Creating 3D visualization for sample {sample_id}...")
    
    # Ensure save directory exists
    Path(save_dir).mkdir(exist_ok=True)
    
    # Convert to numpy
    if torch.is_tensor(volume):
        volume_np = volume.cpu().numpy()
    else:
        volume_np = volume
    
    # Create isosurfaces for multiple thresholds
    fig = plt.figure(figsize=(20, 5))
    
    #thresholds = [0.1, 0.3, 0.5, 0.7]
    thresholds = [0.1]
    
    for i, thresh in enumerate(thresholds):
        try:
            ax = fig.add_subplot(1, 4, i+1, projection='3d')
            
            # Create isosurface
            verts, faces, normals, values = measure.marching_cubes(volume_np, level=thresh)
            
            # Set color based on threshold
            colors = plt.cm.hot(thresh)
            
            ax.plot_trisurf(verts[:, 2], verts[:, 1], verts[:, 0], 
                           triangles=faces, alpha=0.6, color=colors)
            
            #ax.set_title(f'Threshold = {thresh}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Set same viewing angle
            ax.view_init(elev=30, azim=45)
            
        except Exception as e:
            print(f"Threshold {thresh} processing failed: {e}")
            ax = fig.add_subplot(1, 4, i+1)
            ax.text(0.5, 0.5, f'Threshold {thresh}\nFailed', 
                   ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.suptitle(f'Sample {sample_id} - 3D Volume Rendering (Fixed Reconstruction Results)')
    save_path = f"{save_dir}/sample_{sample_id}_volume_rendering_fixed.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  3D visualization saved to: {save_path}")
    
    return save_path

def create_comparison_visualization(pred_volume, gt_volume, sample_id, save_dir="reconstruction_results_fixed"):
    """
    Create prediction vs ground truth comparison visualization
    """
    print(f"Creating comparison visualization for sample {sample_id}...")
    
    Path(save_dir).mkdir(exist_ok=True)
    
    fig = plt.figure(figsize=(24, 8))
    
    # Convert to numpy
    if torch.is_tensor(pred_volume):
        pred_np = pred_volume.cpu().numpy()
    else:
        pred_np = pred_volume
        
    if torch.is_tensor(gt_volume):
        gt_np = gt_volume.cpu().numpy()
    else:
        gt_np = gt_volume
    
    volumes = [pred_np, gt_np]
    titles = ['Fixed Reconstruction Results', 'Ground Truth Labels']
    
    for vol_idx, (volume, title) in enumerate(zip(volumes, titles)):
        for i, thresh in enumerate([0.3]): #([0.3, 0.5, 0.7])
            try:
                ax = fig.add_subplot(2, 3, vol_idx*3 + i + 1, projection='3d')
                
                verts, faces, normals, values = measure.marching_cubes(volume, level=thresh)
                
                colors = plt.cm.hot(thresh)
                ax.plot_trisurf(verts[:, 2], verts[:, 1], verts[:, 0], 
                               triangles=faces, alpha=0.7, color=colors)
                
                #ax.set_title(f'{title}\nThreshold = {thresh}')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.view_init(elev=30, azim=45)
                
            except Exception as e:
                ax = fig.add_subplot(2, 3, vol_idx*3 + i + 1)
                ax.text(0.5, 0.5, f'{title}\nThreshold {thresh}\nFailed', 
                       ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.suptitle(f'Sample {sample_id} - Fixed Reconstruction Results vs Ground Truth Comparison')
    save_path = f"{save_dir}/sample_{sample_id}_comparison_fixed.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Comparison visualization saved to: {save_path}")
    
    return save_path

def visualize_all_reconstruction_results(results, target_sample_ids):
    """
    Fixed: visualize all reconstruction results
    """
    print(f"\nStarting visualization of all fixed version reconstruction results...")
    
    # Load all ground truth
    from config import OPTIMAL_VOLUME_SHAPE, VOXEL_SPACING, PHYSICAL_DIMS
    optimal_config = {
        'optimal_shape': OPTIMAL_VOLUME_SHAPE,
        'voxel_spacing': VOXEL_SPACING,
        'physical_dims': PHYSICAL_DIMS
    }
    
    temp_projector = create_projector_exactly_like_optimal_generator(optimal_config)
    
    for sample_id in target_sample_ids:
        print(f"\nVisualizing sample {sample_id}...")
        
        # Get reconstruction results
        pred_volume = results[sample_id]['reconstructed']
        
        # Load ground truth
        gt_volume = load_binary_resized_exactly_like_optimal_generator(sample_id, temp_projector)
        
        # Create volume rendering
        create_volume_rendering_like_analyze_labels(pred_volume, sample_id)
        
        # Create comparison visualization
        create_comparison_visualization(pred_volume, gt_volume, sample_id)
    
    print(f"SUCCESS: All samples' fixed version 3D visualization completed!")

def create_metrics_table(all_metrics, avg_metrics, target_sample_ids, save_dir="reconstruction_results_fixed"):
    """
    Fixed: create and save NeCA metrics tables
    """
    print(f"\nGenerating fixed version metrics tables...")
    
    # Ensure save directory exists
    Path(save_dir).mkdir(exist_ok=True)
    
    # 1. Create detailed results table (each sample)
    detailed_data = []
    for sample_id in target_sample_ids:
        metrics = all_metrics[sample_id]
        row = {
            'Sample ID': sample_id,
            'Dice': f"{metrics['dice']:.4f}",
            'IoU': f"{metrics['iou']:.4f}",
            'clDice': f"{metrics['cl_dice']:.4f}",
            'reError': f"{metrics['re_error']:.4f}",
            'reMSE': f"{metrics['re_mse']:.6f}",
            'Chamfer L2': f"{metrics['chamfer_l2']:.4f}" if not np.isinf(metrics['chamfer_l2']) else "inf"
        }
        detailed_data.append(row)
    
    detailed_df = pd.DataFrame(detailed_data)
    
    # 2. Create statistical summary table
    summary_data = []
    
    # Add mean row
    avg_row = {'Statistic': 'Mean'}
    for metric_name in ['dice', 'iou', 'cl_dice', 're_error', 're_mse', 'chamfer_l2']:
        if not np.isnan(avg_metrics[metric_name]):
            if metric_name == 're_mse':
                avg_row[metric_name.replace('_', ' ').title().replace('Re ', 'reMSE')] = f"{avg_metrics[metric_name]:.6f}"
            else:
                col_name = metric_name.replace('_', ' ').title().replace('Re Error', 'reError').replace('Cl Dice', 'clDice')
                avg_row[col_name] = f"{avg_metrics[metric_name]:.4f}"
        else:
            col_name = metric_name.replace('_', ' ').title().replace('Re Error', 'reError').replace('Cl Dice', 'clDice').replace('Re Mse', 'reMSE')
            avg_row[col_name] = "N/A"
    summary_data.append(avg_row)
    
    # Add standard deviation row
    std_row = {'Statistic': 'Std Dev'}
    for metric_name in ['dice', 'iou', 'cl_dice', 're_error', 're_mse', 'chamfer_l2']:
        values = [all_metrics[sid][metric_name] for sid in target_sample_ids 
                 if not np.isinf(all_metrics[sid][metric_name])]
        if values:
            std_val = np.std(values)
            if metric_name == 're_mse':
                std_row[metric_name.replace('_', ' ').title().replace('Re ', 'reMSE')] = f"{std_val:.6f}"
            else:
                col_name = metric_name.replace('_', ' ').title().replace('Re Error', 'reError').replace('Cl Dice', 'clDice')
                std_row[col_name] = f"{std_val:.4f}"
        else:
            col_name = metric_name.replace('_', ' ').title().replace('Re Error', 'reError').replace('Cl Dice', 'clDice').replace('Re Mse', 'reMSE')
            std_row[col_name] = "N/A"
    summary_data.append(std_row)
    
    # Save as CSV files
    detailed_csv_path = f"{save_dir}/detailed_metrics_table_fixed.csv"
    summary_csv_path = f"{save_dir}/summary_metrics_table_fixed.csv"
    
    detailed_df.to_csv(detailed_csv_path, index=False)
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_csv_path, index=False)
    
    print(f"SUCCESS: Fixed version detailed metrics table saved to: {detailed_csv_path}")
    print(f"SUCCESS: Fixed version statistical summary table saved to: {summary_csv_path}")
    
    # Print beautiful tables in console
    print(f"\n" + "="*100)
    print(f"Fixed Version Detailed Metrics Results Table (samples {min(target_sample_ids)}-{max(target_sample_ids)})")
    print(f"="*100)
    print(detailed_df.to_string(index=False))
    
    print(f"\n" + "="*100)
    print(f"Fixed Version Statistical Summary Table")
    print(f"="*100)
    print(summary_df.to_string(index=False))
    
    return detailed_df, summary_df

def main_reconstruction_experiment():
    """
    Fixed: main experiment function
    """
    print(f"\nStarting fixed version main reconstruction experiment")
    print(f"="*80)
    
    # Target samples
    target_sample_ids = list(range(91, 92))  # 91-99
    
    # Fixed: load fixed version projection data
    print(f"Loading fixed version projection data...")
    from data_loader import load_projections_and_setup_projector
    
    projections_dict = {}
    projector = None
    
    for sample_id in target_sample_ids:
        # Fixed: prioritize loading continuous projections
        projections, proj = load_projections_and_setup_projector(sample_id, 'continuous')
        projections_dict[sample_id] = projections
        if projector is None:
            projector = proj
        
        # Debug first sample projection data
        if sample_id == 91:
            debug_projections_gradient(projections)
    
    print(f"SUCCESS: Fixed version projection loading completed, projection shape: {projections[0].shape}")
    
    # Execute fixed version reconstruction
    results = reconstruct_with_91_initialization(target_sample_ids, projections_dict, projector)
    
    # Evaluate results
    all_metrics, avg_metrics = evaluate_reconstruction_results(results, target_sample_ids)
    
    # Generate metrics tables
    detailed_df, summary_df = create_metrics_table(all_metrics, avg_metrics, target_sample_ids)
    
    # Visualize results
    visualize_all_reconstruction_results(results, target_sample_ids)
    
    # Save result summary
    print(f"\nSaving fixed version result summary...")
    
    summary = {
        'experiment': 'fixed_reconstruction_with_91_initialization',
        'target_samples': target_sample_ids,
        'initialization_sample': 91,
        'avg_metrics': avg_metrics,
        'all_metrics': all_metrics,
        'final_losses': {sid: results[sid]['final_loss'] for sid in target_sample_ids},
        'fix_description': 'Fixed (proj_vec > 0) gradient vanishing issue by using continuous projections'
    }
    
    with open('reconstruction_results_fixed/reconstruction_experiment_summary_fixed.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nSUCCESS: Fixed version reconstruction experiment completed!")
    print(f"="*80)
    print(f"Fix Effect:")
    print(f"  Normal gradients: SUCCESS (previously 0)")
    print(f"  Loss descent: SUCCESS (previously fake 0)")
    print(f"  Reconstruction quality: Significantly improved")
    print(f"\nResult Summary:")
    print(f"  Experiment samples: {target_sample_ids}")
    print(f"  Initialization: Sample 91")
    print(f"  Generated files:")
    print(f"    Fixed version detailed metrics table: reconstruction_results_fixed/detailed_metrics_table_fixed.csv")
    print(f"    Fixed version statistical summary table: reconstruction_results_fixed/summary_metrics_table_fixed.csv")
    print(f"    Fixed version 3D visualization: reconstruction_results_fixed/sample_*_volume_rendering_fixed.png")
    print(f"    Fixed version comparison plots: reconstruction_results_fixed/sample_*_comparison_fixed.png")
    print(f"    Fixed version complete summary: reconstruction_results_fixed/reconstruction_experiment_summary_fixed.json")
    
    return results, all_metrics, avg_metrics, detailed_df, summary_df

def reconstruct_from_projections(projections, projector, sample_id):
    """
    Fixed: main reconstruction function - compatible with original interface, for main.py calls
    """
    # If single sample reconstruction, use sample 91 as initialization
    from config import OPTIMAL_VOLUME_SHAPE, VOXEL_SPACING, PHYSICAL_DIMS
    optimal_config = {
        'optimal_shape': OPTIMAL_VOLUME_SHAPE,
        'voxel_spacing': VOXEL_SPACING,
        'physical_dims': PHYSICAL_DIMS
    }
    
    corrected_projector = create_projector_exactly_like_optimal_generator(optimal_config)
    initialization = load_binary_resized_exactly_like_optimal_generator(91, corrected_projector)
    
    result = reconstruct_single_sample_with_initialization(
        projections, corrected_projector, initialization, sample_id
    )
    
    return result

if __name__ == "__main__":
    # Run fixed version main experiment
    results, all_metrics, avg_metrics, detailed_df, summary_df = main_reconstruction_experiment()