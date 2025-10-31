import torch

if not torch.cuda.is_available():
    print("❌ No CUDA devices available.")
else:
    num_devices = torch.cuda.device_count()
    print(f"✅ {num_devices} CUDA device(s) available:\n")

    for i in range(num_devices):
        name = torch.cuda.get_device_name(i)
        total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3  # in GB
        print(f"  [{i}] {name} — {total_mem:.1f} GB")
