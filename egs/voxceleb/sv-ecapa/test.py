import torch

# 检查CUDA是否可用
cuda_available = torch.cuda.is_available()

if cuda_available:
    # 获取当前GPU的数量
    gpu_count = torch.cuda.device_count()
    print(f"当前有 {gpu_count} 块GPU可用.")

    # 获取当前GPU的名称
    current_gpu_name = torch.cuda.get_device_name(0)  # 假设使用第一块GPU
    print(f"当前GPU名称为: {current_gpu_name}.")
else:
    print("CUDA 不可用，将使用 CPU 进行计算.")

