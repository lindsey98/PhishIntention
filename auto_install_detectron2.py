#!/usr/bin/env python3
"""auto_install_detectron2.py - 自动检测环境并安装合适的 PyTorch 和 detectron2"""

import platform
import subprocess
import sys
import os

def get_system_info():
    """检测系统和GPU信息"""
    os_name = platform.system()  # 'Windows', 'Linux', 'Darwin'
    
    # 检测 CUDA 可用性
    has_cuda = False
    cuda_version = None
    
    # 首先通过 nvidia-smi 检测是否有 NVIDIA GPU
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            has_cuda = True
            # 尝试从 nvidia-smi 输出中提取 CUDA 版本
            output = result.stdout
            for line in output.split('\n'):
                if 'CUDA Version' in line:
                    # 格式类似: "| NVIDIA-SMI 535.104.05   Driver Version: 535.104.05   CUDA Version: 12.2 |"
                    parts = line.split('CUDA Version:')
                    if len(parts) > 1:
                        cuda_version = parts[1].strip().split()[0].strip('|').strip()
                    break
    except FileNotFoundError:
        pass
    
    # 如果已安装 PyTorch，使用 PyTorch 报告的 CUDA 版本
    try:
        import torch
        if torch.cuda.is_available():
            has_cuda = True
            cuda_version = torch.version.cuda
    except ImportError:
        pass
    
    if args.cpu:
        print("You have set the --cpu flag, so CPU version of PyTorch and Detectron2 will be installed.")
        has_cuda = False
        cuda_version = None

    return {
        'os': os_name,
        'has_cuda': has_cuda,
        'cuda_version': cuda_version,
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}"
    }

def get_torch_version():
    """获取已安装的 PyTorch 版本"""
    try:
        import torch
        return torch.__version__.split('+')[0]  # 去掉 +cu118 等后缀
    except ImportError:
        return None

def install_pytorch(info):
    """根据环境信息安装 PyTorch"""
    os_name = info['os']
    has_cuda = info['has_cuda']
    cuda_version = info['cuda_version']
    
    print(f"正在安装 PyTorch...")
    
    if os_name == 'Darwin':  # macOS
        # macOS 只支持 CPU 版本（或 MPS for Apple Silicon）
        install_cmd = [
            sys.executable, '-m', 'pip', 'install',
            'torch', 'torchvision'
        ]
    elif has_cuda and cuda_version:
        # 根据 CUDA 版本选择合适的 PyTorch
        cuda_major_minor = cuda_version.split('.')[:2]
        cuda_ver = ''.join(cuda_major_minor)  # "12.2" -> "122"
        
        # PyTorch 支持的 CUDA 版本映射
        # 根据 https://pytorch.org/get-started/locally/
        if cuda_ver.startswith('12'):
            cuda_tag = 'cu124'  # CUDA 12.x 使用 cu124
        elif cuda_ver.startswith('11'):
            cuda_tag = 'cu118'  # CUDA 11.x 使用 cu118
        else:
            print(f"警告：CUDA {cuda_version} 可能不被支持，尝试使用 cu124")
            cuda_tag = 'cu124'
        
        install_cmd = [
            sys.executable, '-m', 'pip', 'install',
            'torch', 'torchvision',
            '--index-url', f'https://download.pytorch.org/whl/{cuda_tag}'
        ]
    else:
        # CPU 版本
        install_cmd = [
            sys.executable, '-m', 'pip', 'install',
            'torch', 'torchvision',
            '--index-url', 'https://download.pytorch.org/whl/cpu'
        ]
    
    print(f"执行: {' '.join(install_cmd)}")
    result = subprocess.run(install_cmd)
    
    if result.returncode != 0:
        print("PyTorch 安装失败！")
        return False
    
    # 重新导入以获取版本
    print("✓ PyTorch 安装成功！")
    return True

def install_detectron2(info):
    """根据环境信息安装 detectron2"""
    os_name = info['os']
    has_cuda = info['has_cuda']
    torch_version = get_torch_version()
    
    if torch_version is None:
        print("错误：PyTorch 未安装")
        return False
    
    # 重新检测 CUDA（因为 PyTorch 可能刚安装）
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            info['cuda_version'] = torch.version.cuda
    except ImportError:
        pass
    
    # 简化版本号 (2.8.0 -> 2.8)
    torch_short = '.'.join(torch_version.split('.')[:2])
    
    print(f"\n正在安装 Detectron2...")
    print(f"系统: {os_name}, CUDA: {has_cuda}, PyTorch: {torch_version}")
    
    if os_name == 'Linux':
        if has_cuda and info.get('cuda_version'):
            # Linux + CUDA: 使用官方预编译包
            cuda_ver = info['cuda_version'].replace('.', '')[:3]  # "11.8" -> "118"
            install_cmd = [
                sys.executable, '-m', 'pip', 'install',
                'detectron2',
                '-f', f'https://dl.fbaipublicfiles.com/detectron2/wheels/cu{cuda_ver}/torch{torch_short}/index.html'
            ]
        else:
            # Linux + CPU
            install_cmd = [
                sys.executable, '-m', 'pip', 'install',
                'detectron2',
                '-f', f'https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch{torch_short}/index.html'
            ]
    
    elif os_name == 'Windows':
        # Windows: 使用第三方预编译包
        # https://github.com/miroatgithub/torch_packages_builder
        base_url = 'https://miropsota.github.io/torch_packages_builder'
        
        # 构建包名
        if has_cuda and info.get('cuda_version'):
            cuda_ver = info['cuda_version'].replace('.', '')[:3]  # "11.8" -> "118"
            # 格式: detectron2==0.6+18f6958pt{torch_version}cu{cuda_version}
            pkg = f'detectron2==0.6+18f6958pt{torch_version}cu{cuda_ver}'
        else:
            pkg = f'detectron2==0.6+18f6958pt{torch_version}cpu'
        
        install_cmd = [
            sys.executable, '-m', 'pip', 'install',
            '--extra-index-url', base_url, pkg
        ]
    
    elif os_name == 'Darwin':  # macOS
        # macOS: 从源码编译（不支持 CUDA）
        print("macOS 检测到，将从源码编译（仅支持 CPU）...")
        install_cmd = [
            sys.executable, '-m', 'pip', 'install',
            'git+https://github.com/facebookresearch/detectron2.git'
        ]
    
    else:
        print(f"不支持的操作系统: {os_name}")
        return False
    
    print(f"执行: {' '.join(install_cmd)}")
    result = subprocess.run(install_cmd)
    
    if result.returncode != 0:
        # 如果预编译包失败，尝试从源码编译
        print("\n预编译包安装失败，尝试从源码编译...")
        fallback_cmd = [
            sys.executable, '-m', 'pip', 'install',
            'git+https://github.com/facebookresearch/detectron2.git'
        ]
        print(f"执行: {' '.join(fallback_cmd)}")
        result = subprocess.run(fallback_cmd)
    
    return result.returncode == 0

def main():
    """主函数：自动检测环境并安装 PyTorch 和 detectron2"""
    print("=" * 60)
    print("自动安装脚本 - PyTorch 和 Detectron2")
    print("=" * 60)
    
    # 检测环境
    info = get_system_info()
    print(f"\n检测到环境:")
    print(f"  - 操作系统: {info['os']}")
    print(f"  - Python: {info['python_version']}")
    print(f"  - CUDA 可用: {info['has_cuda']}")
    if info['cuda_version']:
        print(f"  - CUDA 版本: {info['cuda_version']}")
    
    # 检查是否已安装 PyTorch
    torch_version = get_torch_version()
    if torch_version:
        print(f"\n已检测到 PyTorch {torch_version}")
        # 验证 CUDA 支持
        try:
            import torch
            if info['has_cuda'] and not torch.cuda.is_available():
                print("警告：系统有 CUDA 但 PyTorch 是 CPU 版本")
                response = input("是否重新安装 CUDA 版 PyTorch？(y/n): ").strip().lower()
                if response == 'y':
                    if not install_pytorch(info):
                        sys.exit(1)
        except ImportError:
            pass
    else:
        print("\n未检测到 PyTorch，开始安装...")
        if not install_pytorch(info):
            sys.exit(1)
    
    # 更新 torch 版本信息
    torch_version = get_torch_version()
    print(f"\n当前 PyTorch 版本: {torch_version}")
    
    # 检查是否已安装 detectron2
    try:
        import detectron2
        print(f"已检测到 Detectron2 {detectron2.__version__}")
        response = input("是否重新安装 Detectron2？(y/n): ").strip().lower()
        if response != 'y':
            print("跳过 Detectron2 安装")
            return
    except ImportError:
        pass
    
    # 安装 detectron2
    # 重新获取系统信息（PyTorch 可能已更新）
    info = get_system_info()
    success = install_detectron2(info)
    
    if success:
        print("\n" + "=" * 60)
        print("✓ 安装完成！")
        print("=" * 60)
        
        # 验证安装
        try:
            import torch
            import detectron2
            print(f"\n验证安装:")
            print(f"  - PyTorch: {torch.__version__}")
            print(f"  - CUDA 可用: {torch.cuda.is_available()}")
            print(f"  - Detectron2: {detectron2.__version__}")
        except ImportError as e:
            print(f"\n警告：验证时出错: {e}")
    else:
        print("\n" + "=" * 60)
        print("✗ 安装失败，请查看错误信息")
        print("=" * 60)
        sys.exit(1)

def auto_install():
    """非交互式自动安装（用于脚本调用）"""
    print("=" * 60)
    print("自动安装脚本 - PyTorch 和 Detectron2 (非交互模式)")
    print("=" * 60)
    
    # 检测环境
    info = get_system_info()
    print(f"\n检测到环境:")
    print(f"  - 操作系统: {info['os']}")
    print(f"  - Python: {info['python_version']}")
    print(f"  - CUDA 可用: {info['has_cuda']}")
    if info['cuda_version']:
        print(f"  - CUDA 版本: {info['cuda_version']}")
    
    # 检查是否已安装 PyTorch
    torch_version = get_torch_version()
    need_reinstall_pytorch = False
    
    if torch_version:
        print(f"\n已检测到 PyTorch {torch_version}")
        # 验证 CUDA 支持
        try:
            import torch
            if info['has_cuda'] and not torch.cuda.is_available():
                print("警告：系统有 CUDA 但 PyTorch 是 CPU 版本，将重新安装")
                need_reinstall_pytorch = True
        except ImportError:
            need_reinstall_pytorch = True
    else:
        need_reinstall_pytorch = True
    
    if need_reinstall_pytorch:
        print("\n开始安装 PyTorch...")
        if not install_pytorch(info):
            sys.exit(1)
    
    # 更新 torch 版本信息
    torch_version = get_torch_version()
    print(f"\n当前 PyTorch 版本: {torch_version}")
    
    # 检查是否已安装 detectron2
    need_install_detectron2 = True
    try:
        import detectron2
        print(f"已检测到 Detectron2 {detectron2.__version__}")
        need_install_detectron2 = False
    except ImportError:
        pass
    
    if need_install_detectron2:
        # 重新获取系统信息
        info = get_system_info()
        success = install_detectron2(info)
        
        if not success:
            print("\n✗ Detectron2 安装失败")
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ 安装完成！")
    print("=" * 60)
    
    # 验证安装
    try:
        import torch
        import detectron2
        print(f"\n验证安装:")
        print(f"  - PyTorch: {torch.__version__}")
        print(f"  - CUDA 可用: {torch.cuda.is_available()}")
        print(f"  - Detectron2: {detectron2.__version__}")
    except ImportError as e:
        print(f"\n警告：验证时出错: {e}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='自动安装 PyTorch 和 Detectron2')
    parser.add_argument('--auto', action='store_true', help='非交互式自动安装')
    args = parser.parse_args()
    
    if args.auto:
        auto_install()
    else:
        main()
