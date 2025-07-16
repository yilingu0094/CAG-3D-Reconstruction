# data_loader.py - Fixed version: support continuous projection loading
import os
import torch
from PIL import Image
import numpy as np
import glob
from config import device, SAMPLE_RANGE, PROJECTION_INDICES, ANGLES, DATA_ROOT

def get_image_shape():
    """Get actual size of projection images"""
    if not os.path.exists(DATA_ROOT):
        raise FileNotFoundError(f"Data root directory not found: {DATA_ROOT}")
    
    # Adapt to new file structure: DATA_ROOT/sample_XX/proj_XX_YY.png
    for sample_id in SAMPLE_RANGE:
        sample_dir = os.path.join(DATA_ROOT, f"sample_{sample_id}")
        if not os.path.exists(sample_dir):
            continue
            
        for proj_idx in PROJECTION_INDICES:
            # Fixed: prioritize checking continuous projection tensor files
            tensor_filename = f"proj_continuous_{sample_id}_{proj_idx}.pt"
            tensor_filepath = os.path.join(sample_dir, tensor_filename)
            if os.path.exists(tensor_filepath):
                print(f"Detecting image size from continuous projection: {tensor_filepath}")
                projection = torch.load(tensor_filepath, map_location='cpu')
                detected_size = projection.shape  # (height, width)
                print(f"Detected continuous projection size: {detected_size}")
                return detected_size
            
            # Fallback: check traditional PNG files
            filename = f"proj_{sample_id}_{proj_idx}.png"
            filepath = os.path.join(sample_dir, filename)
            if os.path.exists(filepath):
                print(f"Detecting image size from traditional projection: {filepath}")
                image = Image.open(filepath)
                detected_size = image.size  # (width, height)
                converted_size = detected_size[::-1]  # (height, width)
                print(f"Detected traditional projection size: {detected_size} -> {converted_size}")
                return converted_size
    
    raise FileNotFoundError("No valid projection images found to determine size")

def load_continuous_projection(sample_id, proj_idx):
    """Fixed: load continuous projection (for reconstruction)"""
    sample_dir = os.path.join(DATA_ROOT, f"sample_{sample_id}")
    tensor_filename = f"proj_continuous_{sample_id}_{proj_idx}.pt"
    tensor_filepath = os.path.join(sample_dir, tensor_filename)
    
    if not os.path.exists(tensor_filepath):
        raise FileNotFoundError(f"Continuous projection not found: {tensor_filepath}")
    
    # Load continuous projection tensor
    projection = torch.load(tensor_filepath, map_location=device)
    
    print(f"Loaded continuous projection: {tensor_filename}, range: [{projection.min():.2f}, {projection.max():.2f}]")
    
    return projection

def load_binary_projection(sample_id, proj_idx):
    """Fixed: load binary projection (for display)"""
    sample_dir = os.path.join(DATA_ROOT, f"sample_{sample_id}")
    
    # Prioritize loading binary PNG
    binary_filename = f"proj_binary_{sample_id}_{proj_idx}.png"
    binary_filepath = os.path.join(sample_dir, binary_filename)
    
    if os.path.exists(binary_filepath):
        image = Image.open(binary_filepath).convert('L')
        image_array = np.array(image)
        binary_image = (image_array > 5).astype(np.float32)
        projection = torch.tensor(binary_image, device=device)
        print(f"Loaded binary projection: {binary_filename}")
        return projection
    
    # Fallback: load traditional format
    traditional_filename = f"proj_{sample_id}_{proj_idx}.png"
    traditional_filepath = os.path.join(sample_dir, traditional_filename)
    
    if os.path.exists(traditional_filepath):
        image = Image.open(traditional_filepath).convert('L')
        image_array = np.array(image)
        binary_image = (image_array > 5).astype(np.float32)
        projection = torch.tensor(binary_image, device=device)
        print(f"Loaded traditional projection: {traditional_filename}")
        return projection
    
    raise FileNotFoundError(f"No binary projection found for sample {sample_id}, projection {proj_idx}")

def load_projection_image(sample_id, proj_idx, mode='continuous'):
    """
    Fixed: unified projection loading interface
    mode: 'continuous' | 'binary' | 'auto'
    """
    if mode == 'continuous':
        return load_continuous_projection(sample_id, proj_idx)
    elif mode == 'binary':
        return load_binary_projection(sample_id, proj_idx)
    elif mode == 'auto':
        # Auto select: prioritize continuous, fallback to binary
        try:
            return load_continuous_projection(sample_id, proj_idx)
        except FileNotFoundError:
            print(f"WARNING: Continuous projection not available, using binary projection")
            return load_binary_projection(sample_id, proj_idx)
    else:
        raise ValueError(f"Unknown projection mode: {mode}")

def load_sample_projections(sample_id, mode='continuous'):
    """Fixed: load all projections for specified sample"""
    print(f"Loading sample {sample_id} {mode} projection data...")
    
    projections = []
    
    for proj_idx in PROJECTION_INDICES:
        proj_image = load_projection_image(sample_id, proj_idx, mode)
        projections.append(proj_image)
    
    print(f"SUCCESS: Loading completed: {len(projections)} projections, type: {mode}")
    
    return projections

def get_sample_list():
    """Get available sample list"""
    if not os.path.exists(DATA_ROOT):
        return []
    
    available_samples = []
    
    for sample_id in SAMPLE_RANGE:
        sample_dir = os.path.join(DATA_ROOT, f"sample_{sample_id}")
        if not os.path.exists(sample_dir):
            continue
            
        # Fixed: check if has continuous projections or traditional projections
        has_continuous = True
        has_binary = True
        
        for proj_idx in PROJECTION_INDICES:
            # Check continuous projections
            continuous_file = os.path.join(sample_dir, f"proj_continuous_{sample_id}_{proj_idx}.pt")
            if not os.path.exists(continuous_file):
                has_continuous = False
            
            # Check binary projections (traditional format)
            binary_file = os.path.join(sample_dir, f"proj_{sample_id}_{proj_idx}.png")
            if not os.path.exists(binary_file):
                has_binary = False
        
        if has_continuous or has_binary:
            available_samples.append(sample_id)
            status = "Fixed continuous" if has_continuous else "Binary"
            print(f"  Sample {sample_id}: {status}")
        else:
            print(f"WARNING: Sample {sample_id}: missing projection files")
    
    return available_samples

def verify_data_structure():
    """Fixed: verify data structure"""
    if not os.path.exists(DATA_ROOT):
        raise FileNotFoundError(f"Data root directory not found: {DATA_ROOT}")
    
    # Get image size
    image_shape = get_image_shape()
    print(f"Fixed version data verification:")
    print(f"Data root: {DATA_ROOT}")
    print(f"Projection image shape: {image_shape}")
    print(f"Looking for samples: {list(SAMPLE_RANGE)}")
    print(f"Expected projection indices: {PROJECTION_INDICES}")
    
    available_samples = get_sample_list()
    print(f"Available samples: {available_samples}")
    
    # Fixed: check fix status
    sample_91_dir = os.path.join(DATA_ROOT, "sample_91")
    if os.path.exists(sample_91_dir):
        continuous_exists = os.path.exists(os.path.join(sample_91_dir, "proj_continuous_91_00.pt"))
        if continuous_exists:
            print(f"SUCCESS: Detected fixed version data (continuous projections exist)")
        else:
            print(f"WARNING: Fixed version data not detected (only traditional projections)")
    
    return available_samples, image_shape

def load_reference_volume(sample_id=91):
    """
    Fixed: load sample 91's real 3D coronary tree as initialization
    Load from original .label.nii.gz file, adjust to optimal resolution
    """
    import nibabel as nib
    from config import OPTIMAL_VOLUME_SHAPE, VOXEL_SPACING
    
    # Original label file path
    label_root = "/home/yilin/syn_tree/NeCA/data/CCTA_raw/1-200"
    label_path = os.path.join(label_root, f"{sample_id}.label.nii.gz")
    
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Reference volume not found: {label_path}")
    
    print(f"Fixed: Loading reference volume: {label_path}")
    
    # Load NII file
    nii_img = nib.load(label_path)
    volume_data = nii_img.get_fdata()
    
    # Convert to binary volume
    binary_volume = (volume_data > 0.5).astype(np.float32)
    original_volume = torch.tensor(binary_volume, device=device, dtype=torch.float32)
    
    print(f"Original volume shape: {original_volume.shape}")
    
    # Fixed: adjust to optimal resolution (consistent with projection generator)
    volume_expanded = original_volume.unsqueeze(0).unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        volume_expanded, 
        size=OPTIMAL_VOLUME_SHAPE, 
        mode='trilinear', 
        align_corners=False
    )
    
    resized_volume = resized.squeeze(0).squeeze(0)
    binary_resized = (resized_volume > 0.5).float()
    
    print(f"Adjusted volume shape: {binary_resized.shape}")
    print(f"Active voxels: {torch.sum(binary_resized > 0).item()}")
    
    return binary_resized

def compare_projection_types(sample_id):
    """Fixed: compare different types of projections"""
    print(f"\nComparing sample {sample_id} projection types...")
    
    try:
        continuous_projections = load_sample_projections(sample_id, 'continuous')
        binary_projections = load_sample_projections(sample_id, 'binary')
        
        print(f"\nComparison results:")
        for i, (cont, bin_proj) in enumerate(zip(continuous_projections, binary_projections)):
            print(f"  Projection {i}:")
            print(f"    Continuous: range[{cont.min():.3f}, {cont.max():.3f}], non-zero pixels: {torch.sum(cont > 0).item()}")
            print(f"    Binary: range[{bin_proj.min():.3f}, {bin_proj.max():.3f}], non-zero pixels: {torch.sum(bin_proj > 0).item()}")
            
            # Check information loss
            info_loss = cont.max().item() - 1.0
            if info_loss > 0:
                print(f"    Information preserved: continuous projection retains additional information {info_loss:.1f}")
        
        return continuous_projections, binary_projections
        
    except Exception as e:
        print(f"ERROR: Comparison failed: {e}")
        return None, None

# Fixed: backward compatible main loading function
def load_projections_and_setup_projector(sample_id, projection_type='continuous'):
    """
    Fixed: load projection data and setup projector
    projection_type: 'continuous' | 'binary' | 'auto'
    """
    from projector import MatrixXRayProjector
    from config import OPTIMAL_VOLUME_SHAPE, PROJECTION_IMAGE_SIZE, PIXEL_SPACING, SOD, SID, VOXEL_SPACING
    
    print(f"Loading sample {sample_id} {projection_type} projection data...")
    
    projections = load_sample_projections(sample_id, projection_type)
    
    # Create projector
    projector = MatrixXRayProjector(
        volume_shape=OPTIMAL_VOLUME_SHAPE,
        image_shape=PROJECTION_IMAGE_SIZE,
        pixel_spacing=PIXEL_SPACING,
        SOD=SOD,
        SID=SID
    )
    
    # Apply voxel spacing
    avg_voxel_size = np.mean(VOXEL_SPACING)
    projector.voxel_coords *= avg_voxel_size
    
    print(f"SUCCESS: Loading completed: {len(projections)} projections, type: {projection_type}")
    
    return projections, projector

if __name__ == "__main__":
    # Fixed: test fixed version data loader
    print("Testing fixed version data loader")
    print("="*50)
    
    # Verify data structure
    available_samples, image_shape = verify_data_structure()
    
    if available_samples:
        sample_id = available_samples[0]
        print(f"\nTesting sample {sample_id}...")
        
        # Compare projection types
        compare_projection_types(sample_id)
        
        # Test reference volume loading
        try:
            ref_volume = load_reference_volume(sample_id)
            print(f"SUCCESS: Reference volume loading successful")
        except Exception as e:
            print(f"ERROR: Reference volume loading failed: {e}")
    else:
        print("ERROR: No available samples for testing")