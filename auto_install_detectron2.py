#!/usr/bin/env python3
"""auto_install_detectron2.py - 自动检测环境并安装合适的 PyTorch 和 detectron2

使用第三方预编译包源: https://miropsota.github.io/torch_packages_builder/detectron2/
支持 Linux, Windows, macOS 三平台
"""

import platform
import subprocess
import sys
import re
import urllib.request
import urllib.error

# 第三方预编译包的基础 URL
PREBUILT_BASE_URL = "https://github.com/MiroPsota/torch_packages_builder/releases/download"
PREBUILT_INDEX_URL = "https://miropsota.github.io/torch_packages_builder/detectron2/"

# 已知的 commit hash 列表（按优先级排序，较新的在前）
KNOWN_COMMIT_HASHES = ['fd27788', '864913f', '18f6958', '2a420ed', '']

# PyTorch 版本到 commit hash 的映射（基于实际可用性）
# fd27788: 支持 PyTorch 2.3.0 - 2.9.1
# 864913f: 支持 PyTorch 2.0.0 - 2.4.0
# 18f6958: 支持 PyTorch 2.1.0 - 2.8.0
# 2a420ed: 支持 PyTorch 2.0.0 - 2.7.0
# 无前缀: 支持 PyTorch 2.0.0 - 2.3.1
TORCH_VERSION_COMMIT_MAP = {
    # PyTorch 2.9.x
    '2.9.1': ['fd27788'],
    '2.9.0': ['fd27788'],
    # PyTorch 2.8.x
    '2.8.0': ['fd27788', '18f6958'],
    # PyTorch 2.7.x
    '2.7.1': ['fd27788'],
    '2.7.0': ['fd27788', '18f6958', '2a420ed'],
    # PyTorch 2.6.x
    '2.6.0': ['fd27788', '18f6958', '2a420ed'],
    # PyTorch 2.5.x
    '2.5.1': ['fd27788', '18f6958', '2a420ed'],
    '2.5.0': ['fd27788', '18f6958', '2a420ed'],
    # PyTorch 2.4.x
    '2.4.1': ['fd27788', '18f6958', '2a420ed'],
    '2.4.0': ['fd27788', '864913f', '18f6958', '2a420ed'],
    # PyTorch 2.3.x
    '2.3.1': ['fd27788', '864913f', '18f6958', '2a420ed', ''],
    '2.3.0': ['fd27788', '864913f', '18f6958', '2a420ed', ''],
    # PyTorch 2.2.x
    '2.2.2': ['864913f', '18f6958', '2a420ed', ''],
    '2.2.1': ['864913f', '18f6958', '2a420ed', ''],
    '2.2.0': ['864913f', '18f6958', '2a420ed', ''],
    # PyTorch 2.1.x
    '2.1.2': ['864913f', '18f6958', '2a420ed', ''],
    '2.1.1': ['864913f', '18f6958', '2a420ed', ''],
    '2.1.0': ['864913f', '18f6958', '2a420ed', ''],
    # PyTorch 2.0.x
    '2.0.1': ['864913f', '2a420ed', ''],
    '2.0.0': ['864913f', '2a420ed', ''],
}


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
    
    print("正在安装 PyTorch...")
    
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

def get_platform_tag():
    """获取当前平台的 wheel 标签"""
    os_name = platform.system()
    machine = platform.machine().lower()
    
    if os_name == 'Linux':
        if machine in ('x86_64', 'amd64'):
            return 'linux_x86_64'
        elif machine == 'aarch64':
            return 'linux_aarch64'
    elif os_name == 'Windows':
        if machine in ('x86_64', 'amd64', 'x64'):
            return 'win_amd64'
    elif os_name == 'Darwin':
        # macOS 使用 universal2 格式
        return 'macosx_universal2'  # 实际文件名可能是 macosx_14_0_universal2 或 macosx_11_0_universal2
    
    return None


def get_cuda_tag(cuda_version):
    """根据 CUDA 版本获取对应的标签"""
    if not cuda_version:
        return 'cpu'
    
    # 解析 CUDA 版本
    match = re.match(r'(\d+)\.(\d+)', cuda_version)
    if not match:
        return 'cpu'
    
    major, minor = int(match.group(1)), int(match.group(2))
    
    # 映射到可用的 CUDA 标签
    if major == 12:
        if minor >= 4:
            return 'cu124'
        elif minor >= 1:
            return 'cu121'
        else:
            return 'cu121'  # CUDA 12.0 也使用 cu121
    elif major == 11:
        return 'cu118'  # 所有 CUDA 11.x 使用 cu118
    else:
        return 'cpu'  # 不支持的 CUDA 版本，回退到 CPU


def build_wheel_url(torch_version, cuda_tag, python_version, platform_tag, commit_hash='fd27788'):
    """构建预编译 wheel 包的下载 URL
    
    命名格式: detectron2-0.6+{commit_hash}pt{torch_version}{cuda_tag}-cp{py_ver}-cp{py_ver}-{platform}.whl
    例如: detectron2-0.6+fd27788pt2.5.0cu121-cp310-cp310-linux_x86_64.whl
    """
    # 解析 Python 版本
    py_match = re.match(r'(\d+)\.(\d+)', python_version)
    if not py_match:
        return None
    py_major, py_minor = py_match.group(1), py_match.group(2)
    py_tag = f"cp{py_major}{py_minor}"
    
    # 构建包名
    if commit_hash:
        pkg_version = f"0.6+{commit_hash}pt{torch_version}{cuda_tag}"
    else:
        pkg_version = f"0.6+pt{torch_version}{cuda_tag}"
    
    # macOS 平台需要特殊处理
    if 'macosx' in platform_tag:
        # 尝试不同的 macOS 版本标签
        macos_versions = ['14_0', '11_0']
        urls = []
        for macos_ver in macos_versions:
            actual_platform = f"macosx_{macos_ver}_universal2"
            filename = f"detectron2-{pkg_version}-{py_tag}-{py_tag}-{actual_platform}.whl"
            if commit_hash:
                release_tag = f"detectron2-0.6%2B{commit_hash}"
            else:
                release_tag = f"detectron2-0.6%2Bpt{torch_version}{cuda_tag}"
            url = f"{PREBUILT_BASE_URL}/{release_tag}/{filename}"
            urls.append(url)
        return urls
    else:
        filename = f"detectron2-{pkg_version}-{py_tag}-{py_tag}-{platform_tag}.whl"
        if commit_hash:
            release_tag = f"detectron2-0.6%2B{commit_hash}"
        else:
            # 无 commit hash 的情况，release tag 格式不同
            release_tag = f"detectron2-0.6%2Bpt{torch_version}{cuda_tag}"
        url = f"{PREBUILT_BASE_URL}/{release_tag}/{filename}"
        return url


def check_url_exists(url):
    """检查 URL 是否存在"""
    try:
        req = urllib.request.Request(url, method='HEAD')
        urllib.request.urlopen(req, timeout=10)
        return True
    except (urllib.error.HTTPError, urllib.error.URLError):
        return False


def find_best_wheel_url(torch_version, cuda_tag, python_version, platform_tag):
    """查找最佳的预编译 wheel 包 URL
    
    按优先级尝试不同的 commit hash
    """
    # 获取该 PyTorch 版本支持的 commit hash 列表
    commit_hashes = TORCH_VERSION_COMMIT_MAP.get(torch_version, KNOWN_COMMIT_HASHES)
    
    for commit_hash in commit_hashes:
        url = build_wheel_url(torch_version, cuda_tag, python_version, platform_tag, commit_hash)
        
        if isinstance(url, list):
            # macOS 情况，尝试多个 URL
            for u in url:
                print(f"  尝试: {u}")
                if check_url_exists(u):
                    return u
        else:
            print(f"  尝试: {url}")
            if check_url_exists(url):
                return url
    
    return None


def install_detectron2(info):
    """根据环境信息安装 detectron2
    
    使用第三方预编译包: https://miropsota.github.io/torch_packages_builder/detectron2/
    支持 Linux, Windows, macOS 三平台
    """
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
    
    print("\n正在安装 Detectron2...")
    print(f"系统: {os_name}, CUDA: {has_cuda}, PyTorch: {torch_version}")
    
    # 获取平台标签
    platform_tag = get_platform_tag()
    if not platform_tag:
        print(f"警告：不支持的平台 {os_name} {platform.machine()}")
        platform_tag = None
    
    # 获取 CUDA 标签
    if os_name == 'Darwin':
        # macOS 只支持 CPU
        cuda_tag = 'cpu'
    elif has_cuda and info.get('cuda_version'):
        cuda_tag = get_cuda_tag(info['cuda_version'])
    else:
        cuda_tag = 'cpu'
    
    # 获取 Python 版本
    python_version = info['python_version']
    
    print(f"平台: {platform_tag}, CUDA标签: {cuda_tag}, Python: {python_version}")
    
    # 尝试使用第三方预编译包
    install_cmd = None
    wheel_url = None
    
    if platform_tag and torch_version in TORCH_VERSION_COMMIT_MAP:
        print("\n正在查找预编译包...")
        wheel_url = find_best_wheel_url(torch_version, cuda_tag, python_version, platform_tag)
        
        if wheel_url:
            print(f"\n找到预编译包: {wheel_url}")
            install_cmd = [
                sys.executable, '-m', 'pip', 'install', wheel_url
            ]
    
    if not install_cmd:
        print("\n未找到预编译包，将从源码编译...")
        install_cmd = [
            sys.executable, '-m', 'pip', 'install',
            'git+https://github.com/facebookresearch/detectron2.git'
        ]
    
    print(f"执行: {' '.join(install_cmd)}")
    result = subprocess.run(install_cmd)
    
    if result.returncode != 0 and wheel_url:
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
    print("\n检测到环境:")
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
            print("\n验证安装:")
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
    print("\n检测到环境:")
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
        print("\n验证安装:")
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
