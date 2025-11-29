import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    
    # 检查是否支持 bfloat16
    print(f"\n支持 bfloat16: {torch.cuda.is_bf16_supported()}")
    
    # 测试 float16
    try:
        x = torch.tensor([1.0], dtype=torch.float16).cuda()
        print(f"支持 float16: True")
    except:
        print(f"支持 float16: False")
    
    # 测试 bfloat16
    try:
        x = torch.tensor([1.0], dtype=torch.bfloat16).cuda()
        print(f"支持 bfloat16 (实测): True")
    except:
        print(f"支持 bfloat16 (实测): False")
