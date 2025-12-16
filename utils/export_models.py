import torch
import numpy as np
from pathlib import Path
from collections import namedtuple
import torchinfo
from thop import profile
from thop.fx_profile import fx_profile

import sys
sys.path.append('.')
sys.path.append('..')
from models.models import PhysQuadModel, ResidualQuadModel, PhysResQuadModel, QuadLSTM

out_dir = Path('out')

dt = 0.01
phys_params = {
    "g": 9.81,
    "m": 0.045,
    "J": np.diag([2.3951e-5, 2.3951e-5, 3.2347e-6]),
    "thrust_to_weight": 2.0,
    "max_torque": np.array([1e-2, 1e-2, 3e-3]),
}

device = 'cpu'
Scaler = namedtuple('Scaler', ['mean_', 'scale_'])

# Physical model
# phys_model = PhysQuadModel(phys_params, dt)

# Residual model
model_path = out_dir / 'models' / 'residual_random_square_chirp.pt'
ckpt = torch.load(model_path, map_location=device)
cfg = ckpt["config"]
residual_model = ResidualQuadModel(**cfg)
residual_model.load_state_dict(ckpt["model_state"])
residual_model.eval()

# Physical + Residual model
# model_path = out_dir / 'models' / 'phys+res_random_square_chirp.pt'
# ckpt = torch.load(model_path, weights_only=True, map_location=device)
# state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
# phys_model = PhysQuadModel(phys_params, dt)
# res_model = ResidualQuadModel(**ckpt["config"])
# phys_res_model = PhysResQuadModel(
#     phys=phys_model,
#     residual=res_model,
#     x_scaler=Scaler(1.0, 1.0),
#     u_scaler=Scaler(1.0, 1.0)
# )
# phys_res_model.load_state_dict(state)
# phys_res_model.eval()

# LSTM model
# ckpt = torch.load(model_path, map_location=device)
# cfg = ckpt["config"]
# lstm_model = QuadLSTM(**cfg)
# lstm_model.load_state_dict(ckpt['model_state'])
# lstm_model.eval()

## NOTE: the physical, phys+residual and lstm models require changes in the model archicture for compatibility with 
## ST Edge AI Developer Cloud, i.e. to hardcode single-step prediction.
## Please refer to the export_models branch, for the exact export code.
models = {
    # 'physical': phys_model,
    'residual': residual_model,
    # 'phys+residual': phys_res_model,
    # 'lstm': lstm_model
}

for model_name, model in models.items():
    onnx_dir = out_dir / 'export' / model_name
    onnx_dir.mkdir(parents=True, exist_ok=True)

    x0, u = (torch.zeros((1, 12)), torch.zeros((1, 1, 4)))

    macs, params = profile(model, inputs=(x0, u), verbose=True, report_missing=True)
    print(f"THOP profiling: {macs} MACS, {params} params")

    # Doesn't handle multi-input models
    try:
        flops = fx_profile(model, input=(x0, u), verbose=False)
        print(f"THOP FX profiling: {flops} FLOPs")
    except Exception as e:
        print(f"THOP FX profiling failed")
        print(e)
        pass

    torchinfo.summary(
        model,
        input_data=(x0, u),
        col_names=[
            "input_size",
            "output_size",
            "num_params",
            "params_percent",
            "kernel_size",
            "mult_adds",
            "trainable",
        ],
        depth=4
    )

    onnx_path = onnx_dir / f'{model_name}.onnx'
    torch.onnx.export(
        model,
        (x0, u),
        onnx_path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=['x0', 'u'],
        output_names=['x1'],
        training=torch.onnx.TrainingMode.TRAINING, # Disable op fusion optimizations
    )
