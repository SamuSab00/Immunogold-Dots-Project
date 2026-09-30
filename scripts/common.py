import os
import re
import datetime
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.feature import blob_log
from IPython.display import display
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

base = '/content/drive/MyDrive/progetto_immunogold'

paths = {
    'raw_images': os.path.join(base, 'raw_images'),
    'annotations': os.path.join(base, 'annotations'),
    'patches_images': os.path.join(base, 'patches', 'images'),
    'patches_coordinates': os.path.join(base, 'patches', 'coordinates'),
    'patches_density': os.path.join(base, 'patches', 'density_maps'),
    'checkpoints': os.path.join(base, 'checkpoints'),
    'logs': os.path.join(base, 'logs'),
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log(msg):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    log_path = os.path.join(paths['logs'], 'project_log.txt')
    with open(log_path, 'a') as f:
        f.write(f'[{timestamp}] {msg}\n')
    print(f'[{timestamp}] {msg}')

__all__ = [
    'os', 're', 'datetime', 'np', 'pd', 'torch', 'Path', 'Image',
    'gaussian_filter', 'blob_log', 'display',
    'paths', 'device', 'log', 'base', 'nn', 'F', 'plt'
]