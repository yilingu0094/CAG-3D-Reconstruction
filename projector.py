import torch
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MatrixXRayProjector:
    def __init__(self, volume_shape, image_shape, pixel_spacing, SOD, SID):
        self.volume_shape = volume_shape
        self.image_shape = image_shape
        self.pixel_spacing = torch.tensor(pixel_spacing, device=device)
        self.SOD = SOD
        self.SID = SID
        
        # Create voxel coordinates grid (Z,Y,X)
        z, y, x = torch.meshgrid(
            torch.arange(volume_shape[0], device=device),
            torch.arange(volume_shape[1], device=device),
            torch.arange(volume_shape[2], device=device),
            indexing='ij'
        )
        # Convert to physical coordinates (mm) centered at isocenter
        self.voxel_coords = torch.stack((x, y, z), dim=-1).float()
        self.voxel_coords[..., 0] -= volume_shape[2] // 2  # X
        self.voxel_coords[..., 1] -= volume_shape[1] // 2  # Y
        self.voxel_coords[..., 2] -= volume_shape[0] // 2  # Z
        
    def get_rotation_matrix(self, alpha, beta):
        """Create rotation matrix as defined in the paper (equation 1)"""
        cos_a, sin_a = math.cos(alpha), math.sin(alpha)
        cos_b, sin_b = math.cos(beta), math.sin(beta)
        return torch.tensor([
            [cos_b,         sin_b*sin_a,      sin_b*cos_a],
            [0,             cos_a,            -sin_a],
            [-sin_b,        cos_b*sin_a,      cos_b*cos_a]
        ], device=device)
    
    def build_projection_matrix(self, alpha, beta):
        h, w = self.image_shape
        m, n, p = self.volume_shape
        total_pixels = h * w
        total_voxels = m * n * p
        
        row_indices = []
        col_indices = []
        
        R = self.get_rotation_matrix(alpha, beta)
        
        # Apply rotation to all voxels
        rotated_coords = torch.matmul(self.voxel_coords, R.T)
        
        # Translate to X-ray source coordinate system
        # Z-coordinate: distance from source to point
        proj_coords = rotated_coords.clone()
        proj_coords[..., 2] += self.SOD  # Add SOD to z-coordinate
        
        # Only consider points in front of the detector (z > 0)
        valid_mask = proj_coords[..., 2] > 0
        valid_voxels = torch.nonzero(valid_mask)
        
        if len(valid_voxels) == 0:
            return torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long), 
                                         torch.empty((0,)), 
                                         size=(total_pixels, total_voxels),
                                         device=device)
        
        # Perspective projection (equation 3)
        det_coords = (proj_coords[valid_mask][..., :2] / 
                     proj_coords[valid_mask][..., 2].unsqueeze(-1)) * self.SID
        
        # Convert to pixel coordinates
        pixel_coords = det_coords / self.pixel_spacing + torch.tensor([w/2, h/2], device=device)
        pixel_coords = pixel_coords.round().long()
        
        # Filter out-of-bound pixels
        pixel_valid = (
            (pixel_coords[:, 0] >= 0) & 
            (pixel_coords[:, 0] < w) & 
            (pixel_coords[:, 1] >= 0) & 
            (pixel_coords[:, 1] < h))
        
        if torch.sum(pixel_valid) == 0:
            return torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long), 
                                         torch.empty((0,)), 
                                         size=(total_pixels, total_voxels),
                                         device=device)
        
        valid_pixel_coords = pixel_coords[pixel_valid]
        valid_voxel_indices = valid_voxels[pixel_valid]
        
        # Convert voxel indices to linear indices
        voxel_linear_indices = (
            valid_voxel_indices[:, 0] * n * p + 
            valid_voxel_indices[:, 1] * p + 
            valid_voxel_indices[:, 2])
        
        # Convert pixel coordinates to linear indices
        pixel_linear_indices = (
            valid_pixel_coords[:, 1] * w + 
            valid_pixel_coords[:, 0])
        
        # Create sparse matrix
        indices = torch.stack([pixel_linear_indices, voxel_linear_indices], dim=0)
        values = torch.ones(len(pixel_linear_indices), device=device)
        
        P = torch.sparse_coo_tensor(
            indices, 
            values, 
            size=(total_pixels, total_voxels),
            device=device
        )
        
        return P
    
    def project_volume_continuous(self, X, alpha, beta):
        """Fixed: continuous projection (for optimization) - preserve ray intensity information"""
        P = self.build_projection_matrix(alpha, beta)
        if P._nnz() == 0:
            return torch.zeros(self.image_shape, device=device)
        
        vec_X = X.flatten().float()
        proj_vec = torch.sparse.mm(P, vec_X.unsqueeze(-1)).squeeze()
        proj_image = proj_vec.reshape(self.image_shape)
        return proj_image
    
    def project_volume_normalized(self, X, alpha, beta):
        """Fixed: normalized projection (for optimization) - ray intensity normalized to [0,1]"""
        P = self.build_projection_matrix(alpha, beta)
        if P._nnz() == 0:
            return torch.zeros(self.image_shape, device=device)
        
        vec_X = X.flatten().float()
        proj_vec = torch.sparse.mm(P, vec_X.unsqueeze(-1)).squeeze()
        
        # Normalize to [0,1]
        if proj_vec.max() > 0:
            proj_vec = proj_vec / proj_vec.max()
        
        proj_image = proj_vec.reshape(self.image_shape)
        return proj_image
    
    def project_volume_binary(self, X, alpha, beta):
        """Fixed: binary projection (for display) - only preserve ray penetration information"""
        proj_continuous = self.project_volume_continuous(X, alpha, beta)
        return (proj_continuous > 0).float()
    
    def project_volume(self, X, alpha, beta, mode='continuous'):
        """
        Fixed: unified projection interface
        mode: 'binary' | 'continuous' | 'normalized'
        """
        if mode == 'binary':
            return self.project_volume_binary(X, alpha, beta)
        elif mode == 'continuous':
            return self.project_volume_continuous(X, alpha, beta)
        elif mode == 'normalized':
            return self.project_volume_normalized(X, alpha, beta)
        else:
            # Fixed: default use continuous projection (fix original binarization issue)
            return self.project_volume_continuous(X, alpha, beta)