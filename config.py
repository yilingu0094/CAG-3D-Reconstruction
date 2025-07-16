# config.py 
import torch
import math

# Device configuration  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Projection parameters
SOD = 750.0  # Source to Object Distance (mm)
SID = 1000.0  # Source to Image Distance (mm)  
PIXEL_SPACING = (0.5, 0.5)  # mm per pixel

# optimal Volume parameter
OPTIMAL_VOLUME_SHAPE = (348, 348, 348)  # Z, Y, X
PHYSICAL_DIMS = (120.0, 120.0, 120.0)  # mm
VOXEL_SPACING = (0.345, 0.345, 0.345)  # mm
PROJECTION_IMAGE_SIZE = (462, 462)  # pixels

# angles 
ANGLES = [
    (math.radians(0.0), math.radians(0.0)),  # AP_0deg
    (math.radians(90.0), math.radians(0.0)),  # LAT_90deg
    (math.radians(0.0), math.radians(90.0))   # CRAN_90deg
]

# Reconstruction parameters
NUM_ITERATIONS = 5000
LEARNING_RATE = 0.01

# Data paths
DATA_ROOT = "/home/yilin/syn_tree/NeCA/data/CCTA_optimal_proj"
LABEL_ROOT = "/home/yilin/syn_tree/NeCA/data/CCTA_raw/1-200"
SAMPLE_RANGE = range(91, 100)
PROJECTION_INDICES = ["00", "01", "02"]

# Output paths
OUTPUT_DIR = "reconstruction_results_fixed"

FIX_STATUS = "FIXED"

# Expectation
EXPECTED_MEMORY_GB = 0.16
EXPECTED_GRADIENT_NORM = ">1000"
EXPECTED_LOSS_BEHAVIOR = "normal gradient"
EXPECTED_RECONSTRUCTION_QUALITY = "significant improve"

# angle information
ANGLE_NAMES = ['AP_0deg', 'LAT_90deg', 'CRAN_90deg']
PROCESSED_SAMPLE_IDS = [91, 92, 93, 94, 95, 96, 97, 98, 99]
