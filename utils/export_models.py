import torch
import numpy as np
from pathlib import Path
from collections import namedtuple
import torchinfo
from thop import profile
from thop.fx_profile import fx_profile

import sys
sys.path.append("..")
from identification.models import *

sys_params = {
    "g": 9.81,                                   # Gravitational acceleration [m/s²]
    "m": 0.032,                                  # Mass [kg]
    "J": np.diag([1.43e-6, 1.43e-6, 2.89e-6]),   # Inertia matrix [kg·m²], diagonal: [Jx, Jy, Jz]
    "thrust_to_weight": 2.0,                     # Max thrust / weight ratio []
    # "thrust_to_weight": 3.0,                     # Max thrust / weight ratio []
    # "max_torque": jnp.array([1e-6, 1e-6, 2e-6]),  # Maximum torques [Nm], for [roll, pitch, yaw]
    "max_torque": np.array([1e-4, 1e-4, 3e-5]),  # Maximum torques [Nm], for [roll, pitch, yaw]
}
phys_model = PhysQuadModel(sys_params, 1/100)

neural_model = NeuralQuadModel(num_layers=5, hidden_dim=64)

Scaler = namedtuple('Scaler', ['mean_', 'scale_'])
residual_model = ResidualQuadModel(phys_model, neural_model, Scaler(1.0, 1.0), Scaler(1.0, 1.0))

lstm_model = QuadLSTM(hidden_dim=64, num_layers=1)

models = {
    'physical': phys_model,
    'neural': neural_model,
    'residual': residual_model,
    'lstm': lstm_model
}

out_dir = Path('export')

for model_name, model in models.items():
    onnx_dir = out_dir / model_name
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
        # dynamic_axes={
        #     'x0': {0: 'batch_size'},
        #     'u': {0: 'batch_size'},
        #     'x1': {0: 'batch_size'}
        # },
        training=torch.onnx.TrainingMode.TRAINING, # Disable op fusion optimizations
        # dynamo=True
    )

    if not hasattr(model, 'make_init'):
        continue

    onnx_path = onnx_dir / f'{model_name}-init.onnx'
    torch.onnx.export(
        model.make_init(),
        (x0,),
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['x0'],
        output_names=['h0', 'c0'],
        dynamic_axes={
            'x0': {0: 'batch_size'},
            'h0': {0: 'batch_size'},
            'c0': {0: 'batch_size'}
        },
        training=torch.onnx.TrainingMode.TRAINING, # Disable op fusion optimizations
        # dynamo=True
    )

    onnx_path = onnx_dir / f'{model_name}-loop.onnx'
    torch.onnx.export(
        model.make_loop(),
        (x0, u),
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['x0', 'u'],
        output_names=['x1'],
        dynamic_axes={
            'x0': {0: 'batch_size'},
            'u': {0: 'batch_size'},
            'x1': {0: 'batch_size'}
        },
        training=torch.onnx.TrainingMode.TRAINING, # Disable op fusion optimizations
        # dynamo=True
    )