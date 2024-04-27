import torch
import os

os.system('export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'resources/data')

DEFAULT_DATASET_ARGS = {
    'speckle_mean_add': 0.0,
    'speckle_std_add': 0.1,
    'speckle_mean_mul': 0.0,
    'speckle_std_mul': 0.1,
    'transform': None,
    'mult_noise_magnitude': 1.0,
    'addi_noise_magnitude': 1.0,
}