#!/bin/bash

echo "======================================"
echo "安装 Qwen2.5-Omni 会议总结系统依赖"
echo "======================================"

# 检查 Python 版本
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python 版本: $python_version"

# 检查 CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "检测到 NVIDIA GPU"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    echo ""
fi

# 升级 pip
echo "升级 pip..."
pip install --upgrade pip

# 安装基础依赖
echo ""
echo "安装基础依赖..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install accelerate

# 安装特殊版本的 transformers
echo ""
echo "安装 Qwen2.5-Omni 版本的 transformers..."
pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview

# 安装 qwen-omni-utils
echo ""
echo "安装 qwen-omni-utils..."
pip install qwen-omni-utils

# 安装音频处理库
echo ""
echo "安装音频处理库..."
pip install librosa soundfile numpy scipy

# 尝试安装 Flash Attention
echo ""
echo "尝试安装 Flash Attention（需要 CUDA 11.6+）..."
pip install flash-attn --no-build-isolation || echo "Flash Attention 安装失败，将使用标准注意力机制"

# 安装其他依赖
echo ""
echo "安装其他依赖..."
pip install tqdm ffmpeg-python coloredlogs

echo ""
echo "======================================"
echo "安装完成！"
echo "======================================"
echo ""
echo "运行测试脚本检查环境:"
echo "  python test_qwen25_model.py"
echo ""
echo "运行会议总结:"
echo "  python run_qwen25_summary.py --input ../eval-F8N/M005/M005-F8N --with-attribution"
echo ""
