# optimal_projection_generator.py - Fixed version: generate continuous projections
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import math
import os
import json
import glob
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from utils import calculate_voxel_spacing
from projector import MatrixXRayProjector

class OptimalProjectionGenerator:
    def __init__(self):
        # Fixed geometric parameters
        self.SOD = 750.0  # Source to Object Distance (mm)
        self.SID = 1000.0  # Source to Image Distance (mm)
        self.PIXEL_SPACING = (0.5, 0.5)  # mm per pixel
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Improved angle configuration - maximize angle separation
        self.medical_angles = [
            (math.radians(0.0), math.radians(0.0)),      # Front AP
            (math.radians(90.0), math.radians(0.0)),     # Side LAT  
            (math.radians(0.0), math.radians(90.0))      # Top CRAN
        ]
        
        self.angle_names = ["AP_0deg", "LAT_90deg", "CRAN_90deg"]
        
        # Other parameters remain unchanged
        self.default_physical_size = 120.0
        self.projection_image_size = (462, 462)
        self.label_root = "/home/yilin/syn_tree/NeCA/data/CCTA_raw/1-200"
        self.output_root = "/home/yilin/syn_tree/NeCA/data/CCTA_optimal_proj"
        self.config_output_dir = "/home/yilin/syn_tree/3d_recon"
        
        print(f"Fixed optimal resolution projection generator initialization completed")
        print(f"Device: {self.device}")
        print(f"Fixed: will generate continuous projections (no longer binarized)")
        print(f"Geometric parameters: SOD={self.SOD}mm, SID={self.SID}mm, pixel spacing={self.PIXEL_SPACING}mm")
        print(f"Angle configuration:")
        for i, (alpha, beta) in enumerate(self.medical_angles):
            print(f"  {self.angle_names[i]}: alpha={math.degrees(alpha):.1f}°, beta={math.degrees(beta):.1f}°")
    
    def analyze_angle_separation(self):
        """Analyze angle separation"""
        print(f"\nAngle separation analysis:")
        
        angles = self.medical_angles
        min_separation = float('inf')
        
        for i in range(len(angles)):
            for j in range(i+1, len(angles)):
                alpha1, beta1 = angles[i]
                alpha2, beta2 = angles[j]
                
                # Calculate angle differences
                alpha_diff = abs(alpha1 - alpha2)
                beta_diff = abs(beta1 - beta2)
                
                # 3D angle separation (simplified calculation)
                separation = math.sqrt(alpha_diff**2 + beta_diff**2)
                separation_deg = math.degrees(separation)
                
                min_separation = min(min_separation, separation_deg)
                
                print(f"  {self.angle_names[i]} ↔ {self.angle_names[j]}: {separation_deg:.1f}°")
        
        print(f"  Minimum angle separation: {min_separation:.1f}° {'Good' if min_separation > 45 else 'Recommend increase'}")
        
        return min_separation
    
    def find_all_label_files(self):
        """Automatically retrieve all .label.nii.gz files"""
        print(f"\nRetrieving label files...")
        print(f"Search directory: {self.label_root}")
        
        label_pattern = os.path.join(self.label_root, "*.label.nii.gz")
        label_files = glob.glob(label_pattern)
        
        print(f"Found files:")
        for file in sorted(label_files):
            print(f"  {os.path.basename(file)}")
        
        sample_ids = []
        for file in label_files:
            basename = os.path.basename(file)
            if basename.endswith('.label.nii.gz'):
                sample_id = basename.replace('.label.nii.gz', '')
                try:
                    int(sample_id)
                    sample_ids.append(sample_id)
                except ValueError:
                    print(f"  WARNING: Skip non-numeric file: {basename}")
        
        sample_ids = sorted(sample_ids, key=int)
        print(f"\nSUCCESS: Found {len(sample_ids)} valid label files:")
        print(f"Sample IDs: {sample_ids}")
        
        return sample_ids, label_files
    
    def calculate_optimal_resolution(self, physical_dims=None):
        """Calculate optimal 3D resolution"""
        if physical_dims is None:
            physical_dims = (self.default_physical_size,) * 3
        
        print(f"\nCalculating optimal resolution...")
        print(f"Input parameters:")
        print(f"  Physical dimensions: {physical_dims} mm")
        print(f"  Geometric parameters: SOD={self.SOD}, SID={self.SID}")
        print(f"  Pixel spacing: {self.PIXEL_SPACING}")
        print(f"  Number of projection angles: {len(self.medical_angles)}")
        
        try:
            (d_x, d_y, d_z), (res_x, res_y, res_z) = calculate_voxel_spacing(
                physical_dims, self.SOD, self.SID, self.PIXEL_SPACING, self.medical_angles
            )
            
            print(f"Calculation results:")
            print(f"  Voxel spacing: dx={d_x:.4f}, dy={d_y:.4f}, dz={d_z:.4f} mm")
            print(f"  Theoretical resolution: {res_x} x {res_y} x {res_z}")
            
            total_voxels = res_x * res_y * res_z
            memory_gb = total_voxels * 4 / (1024**3)
            
            print(f"  Total voxels: {total_voxels:,}")
            print(f"  Estimated memory: {memory_gb:.2f} GB")
            
            if max(res_x, res_y, res_z) > 512 or memory_gb > 2.0:
                print(f"  WARNING: Resolution too high, applying reasonable limits...")
                max_res = 400
                res_x = min(res_x, max_res)
                res_y = min(res_y, max_res)
                res_z = min(res_z, max_res)
                print(f"  Limited resolution: {res_x} x {res_y} x {res_z}")
            
            optimal_shape = (int(res_z), int(res_y), int(res_x))
            print(f"SUCCESS: Optimal shape: {optimal_shape}")
            
            return {
                'optimal_shape': optimal_shape,
                'voxel_spacing': (d_x, d_y, d_z),
                'physical_dims': physical_dims,
                'total_voxels': res_x * res_y * res_z,
                'memory_gb': res_x * res_y * res_z * 4 / (1024**3)
            }
            
        except Exception as e:
            print(f"ERROR: Calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'optimal_shape': (200, 200, 200),
                'voxel_spacing': (0.6, 0.6, 0.6),
                'physical_dims': physical_dims,
                'total_voxels': 8000000,
                'memory_gb': 0.03
            }
    
    def load_and_resize_volume(self, sample_id, optimal_config):
        """Load and resize volume to optimal resolution"""
        label_filename = f"{sample_id}.label.nii.gz"
        label_path = os.path.join(self.label_root, label_filename)
        
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file does not exist: {label_path}")
        
        print(f"Loading label: {label_filename}")
        
        nii_img = nib.load(label_path)
        volume_data = nii_img.get_fdata()
        
        print(f"Original data: shape{volume_data.shape}, value range[{volume_data.min():.3f}, {volume_data.max():.3f}]")
        
        binary_volume = (volume_data > 0.5).astype(np.float32)
        original_volume = torch.tensor(binary_volume, device=self.device, dtype=torch.float32)
        
        original_active = torch.sum(original_volume > 0).item()
        print(f"Original active voxels: {original_active}")
        
        optimal_shape = optimal_config['optimal_shape']
        print(f"Resize to optimal resolution: {original_volume.shape} -> {optimal_shape}")
        
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
        print(f"Resized active voxels: {resized_active}")
        
        return binary_resized, nii_img
    
    def create_optimal_projector(self, optimal_config):
        """Create optimal configuration projector"""
        optimal_shape = optimal_config['optimal_shape']
        voxel_spacing = optimal_config['voxel_spacing']
        
        print(f"Creating optimal projector:")
        print(f"  Volume shape: {optimal_shape}")
        print(f"  Image shape: {self.projection_image_size}")
        print(f"  Voxel spacing: {voxel_spacing}")
        
        projector = MatrixXRayProjector(
            volume_shape=optimal_shape,
            image_shape=self.projection_image_size,
            pixel_spacing=self.PIXEL_SPACING,
            SOD=self.SOD,
            SID=self.SID
        )
        
        avg_voxel_size = np.mean(voxel_spacing)
        projector.voxel_coords *= avg_voxel_size
        
        print(f"  SUCCESS: Projector creation completed, using average voxel size: {avg_voxel_size:.4f} mm")
        
        return projector
    
    def generate_projections(self, volume, projector, sample_id):
        """Fixed: generate continuous projections (no longer binarized)"""
        print(f"Generating fixed version projections for sample {sample_id}...")
        
        projections_continuous = []
        projections_binary = []
        projection_info = []
        
        for i, (alpha, beta) in enumerate(self.medical_angles):
            angle_name = self.angle_names[i]
            print(f"  Generating projection {i}: {angle_name} (alpha={math.degrees(alpha):.1f}°, beta={math.degrees(beta):.1f}°)")
            
            # Fixed: generate continuous projection (preserve ray intensity)
            projection_continuous = projector.project_volume_continuous(volume, alpha, beta)
            projections_continuous.append(projection_continuous)
            
            # Fixed: generate binary projection (for comparison)
            projection_binary = projector.project_volume_binary(volume, alpha, beta)
            projections_binary.append(projection_binary)
            
            # Statistics for continuous projection
            nonzero_pixels = torch.sum(projection_continuous > 0).item()
            max_val = torch.max(projection_continuous).item()
            sum_val = torch.sum(projection_continuous).item()
            
            if nonzero_pixels > 0:
                coords = torch.nonzero(projection_continuous)
                center_y = torch.mean(coords[:, 0].float()).item()
                center_x = torch.mean(coords[:, 1].float()).item()
                img_center_y, img_center_x = self.projection_image_size[0]/2, self.projection_image_size[1]/2
                
                offset_y = center_y - img_center_y
                offset_x = center_x - img_center_x
            else:
                offset_y, offset_x = 0, 0
            
            proj_info = {
                'index': i,
                'name': angle_name,
                'alpha_deg': math.degrees(alpha),
                'beta_deg': math.degrees(beta),
                'alpha_rad': alpha,
                'beta_rad': beta,
                'continuous': {
                    'nonzero_pixels': nonzero_pixels,
                    'max_value': max_val,
                    'sum_value': sum_val,
                    'mean_value': sum_val / nonzero_pixels if nonzero_pixels > 0 else 0
                },
                'binary': {
                    'nonzero_pixels': torch.sum(projection_binary > 0).item(),
                    'max_value': 1.0,
                    'sum_value': torch.sum(projection_binary).item()
                },
                'center_offset_y': offset_y,
                'center_offset_x': offset_x,
                'info_loss_percent': (max_val - 1.0) / max_val * 100 if max_val > 0 else 0
            }
            projection_info.append(proj_info)
            
            print(f"    Fixed continuous result: non-zero pixels={nonzero_pixels}, max value={max_val:.1f}, info loss={proj_info['info_loss_percent']:.1f}%")
            print(f"    Binary result: non-zero pixels={proj_info['binary']['nonzero_pixels']}, max value=1.0")
        
        return projections_continuous, projections_binary, projection_info
    
    def save_projections(self, projections_continuous, projections_binary, projection_info, sample_id, optimal_config):
        """Fixed: save continuous and binary projection data"""
        sample_output_dir = os.path.join(self.output_root, f"sample_{sample_id}")
        os.makedirs(sample_output_dir, exist_ok=True)
        
        print(f"Saving sample {sample_id} fixed version projection data...")
        
        saved_files = []
        
        # Fixed: save continuous projections (for reconstruction)
        for i, projection in enumerate(projections_continuous):
            # Save as PyTorch tensor
            tensor_filename = f"proj_continuous_{sample_id}_{i:02d}.pt"
            tensor_filepath = os.path.join(sample_output_dir, tensor_filename)
            torch.save(projection.cpu(), tensor_filepath)
            saved_files.append(tensor_filepath)
            
            # Save as normalized PNG (for visualization)
            proj_np = projection.cpu().numpy()
            if proj_np.max() > 0:
                proj_normalized = (proj_np / proj_np.max() * 255).astype(np.uint8)
            else:
                proj_normalized = proj_np.astype(np.uint8)
            
            png_filename = f"proj_continuous_{sample_id}_{i:02d}.png"
            png_filepath = os.path.join(sample_output_dir, png_filename)
            Image.fromarray(proj_normalized).save(png_filepath)
            saved_files.append(png_filepath)
            
            print(f"  Saved continuous projection {i}: {tensor_filename} + {png_filename}")
        
        # Fixed: save binary projections (for comparison)
        for i, projection in enumerate(projections_binary):
            proj_np = projection.cpu().numpy()
            proj_normalized = (proj_np * 255).astype(np.uint8)
            
            filename = f"proj_binary_{sample_id}_{i:02d}.png"
            filepath = os.path.join(sample_output_dir, filename)
            Image.fromarray(proj_normalized).save(filepath)
            saved_files.append(filepath)
            
            print(f"  Saved binary projection {i}: {filename}")
        
        # Fixed: save traditional format projections (compatibility)
        for i, projection in enumerate(projections_binary):
            proj_np = projection.cpu().numpy()
            proj_normalized = (proj_np * 255).astype(np.uint8)
            
            traditional_filename = f"proj_{sample_id}_{i:02d}.png"
            traditional_filepath = os.path.join(sample_output_dir, traditional_filename)
            Image.fromarray(proj_normalized).save(traditional_filepath)
            saved_files.append(traditional_filepath)
            
            print(f"  Saved traditional format {i}: {traditional_filename}")
        
        # Save metadata
        metadata = {
            'sample_id': sample_id,
            'projection_type': 'continuous_and_binary_fixed',
            'optimal_config': optimal_config,
            'geometry_parameters': {
                'SOD': self.SOD,
                'SID': self.SID,
                'pixel_spacing': self.PIXEL_SPACING,
                'projection_image_size': self.projection_image_size
            },
            'improved_angles': [
                {
                    'index': i,
                    'alpha_rad': alpha,
                    'beta_rad': beta,
                    'alpha_deg': math.degrees(alpha),
                    'beta_deg': math.degrees(beta),
                    'name': self.angle_names[i]
                }
                for i, (alpha, beta) in enumerate(self.medical_angles)
            ],
            'projection_info': projection_info,
            'generation_timestamp': str(np.datetime64('now')),
            'fix_description': 'Fixed (proj_vec > 0) issue by generating continuous projections alongside binary ones'
        }
        
        info_filename = f"optimal_projection_info_{sample_id}.json"
        info_filepath = os.path.join(sample_output_dir, info_filename)
        
        with open(info_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        saved_files.append(info_filepath)
        print(f"  Saved metadata: {info_filename}")
        
        return saved_files, sample_output_dir
    
    def visualize_sample_projections(self, projections_continuous, projections_binary, projection_info, sample_id, optimal_config, save_dir):
        """Fixed: visualize continuous vs binary projection comparison"""
        print(f"Visualizing sample {sample_id} fixed version projection comparison...")
        
        fig, axes = plt.subplots(2, len(projections_continuous), figsize=(18, 12))
        if len(projections_continuous) == 1:
            axes = axes.reshape(2, 1)
        
        optimal_shape = optimal_config['optimal_shape']
        voxel_spacing = optimal_config['voxel_spacing']
        
        for i, (proj_cont, proj_bin, info) in enumerate(zip(projections_continuous, projections_binary, projection_info)):
            # Continuous projection
            proj_cont_np = proj_cont.cpu().numpy()
            im1 = axes[0, i].imshow(proj_cont_np, cmap='viridis', vmin=0, vmax=proj_cont_np.max())
            axes[0, i].set_title(
                f'Continuous Projection {i}: {info["name"]}\n'
                f'Range: [0, {proj_cont_np.max():.0f}]\n'
                f'Info Loss: {info["info_loss_percent"]:.1f}%',
                fontsize=10
            )
            axes[0, i].axis('off')
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
            
            # Binary projection
            proj_bin_np = proj_bin.cpu().numpy()
            im2 = axes[1, i].imshow(proj_bin_np, cmap='viridis', vmin=0, vmax=1)
            axes[1, i].set_title(
                f'Binary Projection {i}: {info["name"]}\n'
                f'Range: [0, 1]\n'
                f'Non-zero Pixels: {info["binary"]["nonzero_pixels"]}',
                fontsize=10
            )
            axes[1, i].axis('off')
            plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
        
        plt.suptitle(
            f'Fixed Version Projection Comparison - Sample {sample_id}\n'
            f'Fixed: Continuous projections preserve ray intensity information, binary projections for compatibility\n'
            f'Resolution: {optimal_shape}, Voxel spacing: ({voxel_spacing[0]:.3f}, {voxel_spacing[1]:.3f}, {voxel_spacing[2]:.3f}) mm',
            fontsize=14
        )
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'projections_fixed_sample_{sample_id}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"  Fixed version projection visualization: {save_path}")
        return save_path
    
    def generate_config_file(self, optimal_config, processed_samples):
        """Fixed: generate fixed version configuration file"""
        print(f"\nGenerating fixed version configuration file...")
        
        os.makedirs(self.config_output_dir, exist_ok=True)
        
        config_content = f'''# config.py - Fixed version optimal resolution configuration file
import torch
import math

# Device configuration  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Projection parameters
SOD = {self.SOD}  # Source to Object Distance (mm)
SID = {self.SID}  # Source to Image Distance (mm)  
PIXEL_SPACING = {self.PIXEL_SPACING}  # mm per pixel

# Optimal Volume parameters
OPTIMAL_VOLUME_SHAPE = {optimal_config['optimal_shape']}  # Z, Y, X
PHYSICAL_DIMS = {optimal_config['physical_dims']}  # mm
VOXEL_SPACING = {optimal_config['voxel_spacing']}  # mm
PROJECTION_IMAGE_SIZE = {self.projection_image_size}  # pixels

# Improved projection angles - maximize angle separation
ANGLES = [
    (math.radians({math.degrees(self.medical_angles[0][0]):.1f}), math.radians({math.degrees(self.medical_angles[0][1]):.1f})),  # {self.angle_names[0]}
    (math.radians({math.degrees(self.medical_angles[1][0]):.1f}), math.radians({math.degrees(self.medical_angles[1][1]):.1f})),  # {self.angle_names[1]}
    (math.radians({math.degrees(self.medical_angles[2][0]):.1f}), math.radians({math.degrees(self.medical_angles[2][1]):.1f}))   # {self.angle_names[2]}
]

# Reconstruction parameters
NUM_ITERATIONS = 5000
LEARNING_RATE = 0.01

# Data paths
DATA_ROOT = "{self.output_root}"
LABEL_ROOT = "{self.label_root}"
SAMPLE_RANGE = range({min([int(s) for s in processed_samples])}, {max([int(s) for s in processed_samples]) + 1})
PROJECTION_INDICES = ["00", "01", "02"]

# Output paths
OUTPUT_DIR = "reconstruction_results_fixed"

# Fixed information
FIX_STATUS = "FIXED"
FIX_DESCRIPTION = """
Fixed content:
1. Solved (proj_vec > 0) gradient vanishing issue
2. Projector now generates continuous projections by default, preserving ray intensity information
3. Data generator saves both continuous and binary projections
4. Reconstruction process uses continuous projections for optimization

Expected effects:
- Gradient norm > 1000 (no longer 0)
- Loss properly descends (no longer "fake 0")
- Reconstruction quality significantly improved
"""

# Performance expectations
EXPECTED_MEMORY_GB = {optimal_config['memory_gb']:.2f}
EXPECTED_GRADIENT_NORM = ">1000"
EXPECTED_LOSS_BEHAVIOR = "Normal descent"
EXPECTED_RECONSTRUCTION_QUALITY = "Significantly improved"

# Angle information
ANGLE_NAMES = {self.angle_names}
PROCESSED_SAMPLE_IDS = {[int(s) for s in processed_samples]}

# Usage instructions
USAGE_NOTES = """
Using fixed version configuration:
1. Projector uses continuous projections by default (project_volume())
2. Data loader can choose to load continuous or binary projections
3. Reconstruction process automatically uses continuous projection optimization
4. Expected loss to properly descend close to 0

File structure:
- proj_continuous_XX_YY.pt: Continuous projection tensor (for reconstruction)
- proj_binary_XX_YY.png: Binary projection image (for display)
- proj_XX_YY.png: Traditional format (compatibility)
"""
'''
        
        config_path = os.path.join(self.config_output_dir, 'config.py')
        
        # Backup original configuration
        if os.path.exists(config_path):
            backup_path = os.path.join(self.config_output_dir, 'config_original_backup.py')
            print(f"Backing up original configuration file: config.py -> config_original_backup.py")
            import shutil
            shutil.copy2(config_path, backup_path)
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"SUCCESS: Fixed version configuration file generated: {config_path}")
        return config_path
    
    def process_all_samples(self, physical_dims=None):
        """Fixed: process all samples, generate continuous projections"""
        print(f"\nStarting to process all samples - fixed version (generate continuous projections)")
        
        # Analyze angle separation
        self.analyze_angle_separation()
        
        sample_ids, label_files = self.find_all_label_files()
        if not sample_ids:
            print("ERROR: No .label.nii.gz files found")
            return []
        
        optimal_config = self.calculate_optimal_resolution(physical_dims)
        projector = self.create_optimal_projector(optimal_config)
        
        processed_samples = []
        all_results = []
        
        for sample_id in tqdm(sample_ids, desc="Fixed version processing"):
            try:
                print(f"\n{'='*50}")
                print(f"Fixed version processing sample {sample_id}")
                print(f"{'='*50}")
                
                volume, nii_img = self.load_and_resize_volume(sample_id, optimal_config)
                projections_continuous, projections_binary, projection_info = self.generate_projections(volume, projector, sample_id)
                saved_files, sample_dir = self.save_projections(projections_continuous, projections_binary, projection_info, sample_id, optimal_config)
                viz_path = self.visualize_sample_projections(projections_continuous, projections_binary, projection_info, sample_id, optimal_config, sample_dir)
                
                processed_samples.append(sample_id)
                
                result = {
                    'sample_id': sample_id,
                    'optimal_config': optimal_config,
                    'projection_info': projection_info,
                    'saved_files': saved_files,
                    'visualization': viz_path,
                    'fix_status': 'FIXED'
                }
                all_results.append(result)
                
                print(f"SUCCESS: Sample {sample_id} fixed version processing completed")
                
            except Exception as e:
                print(f"ERROR: Sample {sample_id} processing failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if processed_samples:
            config_path = self.generate_config_file(optimal_config, processed_samples)
            print(f"Fixed version configuration file generation completed: {config_path}")
        
        self.print_summary(optimal_config, processed_samples, all_results)
        return all_results
    
    def print_summary(self, optimal_config, processed_samples, all_results):
        """Fixed: print fixed version processing summary"""
        print(f"\n{'='*70}")
        print("Fixed Version Processing Summary")
        print(f"{'='*70}")
        
        print(f"Fix Status: Fixed (proj_vec > 0) gradient vanishing issue")
        
        print(f"\nAngle configuration:")
        for i, (alpha, beta) in enumerate(self.medical_angles):
            print(f"  {self.angle_names[i]}: alpha={math.degrees(alpha):.1f}°, beta={math.degrees(beta):.1f}°")
        
        print(f"\nOptimal configuration:")
        print(f"  Resolution: {optimal_config['optimal_shape']}")
        print(f"  Voxel spacing: ({optimal_config['voxel_spacing'][0]:.4f}, {optimal_config['voxel_spacing'][1]:.4f}, {optimal_config['voxel_spacing'][2]:.4f}) mm")
        print(f"  Memory requirement: {optimal_config['memory_gb']:.2f} GB")
        
        print(f"\nProcessing results:")
        print(f"  Successfully processed: {len(processed_samples)} samples")
        print(f"  Sample list: {processed_samples}")
        
        print(f"\nFixed effect expectations:")
        print(f"  Gradient norm: >1000 (previously 0)")
        print(f"  Loss behavior: Normal descent (previously fake 0)")
        print(f"  Reconstruction quality: Significantly improved")
        
        print(f"\nGenerated files:")
        print(f"  proj_continuous_XX_YY.pt: Continuous projection tensor (for reconstruction)")
        print(f"  proj_continuous_XX_YY.png: Continuous projection visualization")
        print(f"  proj_binary_XX_YY.png: Binary projection comparison")
        print(f"  proj_XX_YY.png: Traditional format (compatibility)")
        
        print(f"\nNext steps:")
        print(f"  1. Run reconstruction: python main.py")
        print(f"  2. Observe whether gradients and losses are normal")
        print(f"  3. Compare reconstruction quality before and after fix")

def main():
    """Fixed: main function"""
    print("Fixed Optimal Resolution Projection Generator")
    print("="*70)
    print("Fix objective: Solve (proj_vec > 0) gradient vanishing issue")
    
    generator = OptimalProjectionGenerator()
    results = generator.process_all_samples()
    
    print(f"\nFixed version processing completed!")
    print(f"Generated fixed version projections for {len(results)} samples")
    print(f"Now normal gradient optimization is possible!")

if __name__ == "__main__":
    main()