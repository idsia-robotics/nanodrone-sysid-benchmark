import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os

# === Import your model and dataset classes ===
from new.models import QuadModel, QuadMultiStepModel
from new.dataset import QuadDataset, QuadMultiStepDataset
from quadrotor_sys import quad_dynamics

# === Example: set GPU 1 only ===
os.environ["CUDA_VISIBLE_DEVICES"] = "0"