#!/usr/bin/env python3
"""auto_install_detectron2.py - Automatically detect environment and install appropriate PyTorch and detectron2

Uses third-party prebuilt package source: https://miropsota.github.io/torch_packages_builder/detectron2/
Supports Linux, Windows, macOS platforms
"""

import platform
import subprocess
import sys
import re
import urllib.request
import urllib.error

# Base URL for third-party prebuilt packages
PREBUILT_BASE_URL = "https://github.com/MiroPsota/torch_packages_builder/releases/download"
PREBUILT_INDEX_URL = "https://miropsota.github.io/torch_packages_builder/detectron2/"

# List of known commit hashes (sorted by priority, newer ones first)
KNOWN_COMMIT_HASHES = ['fd27788', '864913f', '18f6958', '2a420ed', '']

# Mapping from PyTorch version to commit hash (based on actual availability)
# fd27788: Supports PyTorch 2.3.0 - 2.9.1
# 864913f: Supports PyTorch 2.0.0 - 2.4.0
# 18f6958: Supports PyTorch 2.1.0 - 2.8.0
# 2a420ed: Supports PyTorch 2.0.0 - 2.7.0
# No prefix: Supports PyTorch 2.0.0 - 2.3.1
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
    """Detect system and GPU information"""
    os_name = platform.system()  # 'Windows', 'Linux', 'Darwin'
    
    # Detect CUDA availability
    has_cuda = False
    cuda_version = None
    
    # First detect NVIDIA GPU through nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            has_cuda = True
            # Try to extract CUDA version from nvidia-smi output
            output = result.stdout
            for line in output.split('\n'):
                if 'CUDA Version' in line:
                    # Format similar to: "| NVIDIA-SMI 535.104.05   Driver Version: 535.104.05   CUDA Version: 12.2 |"
                    parts = line.split('CUDA Version:')
                    if len(parts) > 1:
                        cuda_version = parts[1].strip().split()[0].strip('|').strip()
                    break
    except FileNotFoundError:
        pass
    
    # If PyTorch is installed, use CUDA version reported by PyTorch
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
    """Get installed PyTorch version"""
    try:
        import torch
        return torch.__version__.split('+')[0]  # Remove +cu118 and other suffixes
    except ImportError:
        return None

def install_pytorch(info):
    """Install PyTorch based on environment information"""
    os_name = info['os']
    has_cuda = info['has_cuda']
    cuda_version = info['cuda_version']
    
    print("Installing PyTorch...")
    
    if os_name == 'Darwin':  # macOS
        # macOS only supports CPU version (or MPS for Apple Silicon)
        install_cmd = [
            sys.executable, '-m', 'pip', 'install',
            'torch', 'torchvision'
        ]
    elif has_cuda and cuda_version:
        # Choose appropriate PyTorch based on CUDA version
        cuda_major_minor = cuda_version.split('.')[:2]
        cuda_ver = ''.join(cuda_major_minor)  # "12.2" -> "122"
        
        # Mapping of CUDA versions supported by PyTorch
        # Based on https://pytorch.org/get-started/locally/
        if cuda_ver.startswith('12'):
            cuda_tag = 'cu124'  # CUDA 12.x uses cu124
        elif cuda_ver.startswith('11'):
            cuda_tag = 'cu118'  # CUDA 11.x uses cu118
        else:
            print(f"Warning: CUDA {cuda_version} may not be supported, trying cu124")
            cuda_tag = 'cu124'
        
        install_cmd = [
            sys.executable, '-m', 'pip', 'install',
            'torch', 'torchvision',
            '--index-url', f'https://download.pytorch.org/whl/{cuda_tag}'
        ]
    else:
        # CPU version
        install_cmd = [
            sys.executable, '-m', 'pip', 'install',
            'torch', 'torchvision',
            '--index-url', 'https://download.pytorch.org/whl/cpu'
        ]
    
    print(f"Executing: {' '.join(install_cmd)}")
    result = subprocess.run(install_cmd)
    
    if result.returncode != 0:
        print("PyTorch installation failed!")
        return False
    
    # Re-import to get version
    print("✓ PyTorch installed successfully!")
    return True

def get_platform_tag():
    """Get wheel tag for current platform"""
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
        # macOS uses universal2 format
        return 'macosx_universal2'  # Actual filename may be macosx_14_0_universal2 or macosx_11_0_universal2
    
    return None


def get_cuda_tag(cuda_version):
    """Get corresponding tag based on CUDA version"""
    if not cuda_version:
        return 'cpu'
    
    # Parse CUDA version
    match = re.match(r'(\d+)\.(\d+)', cuda_version)
    if not match:
        return 'cpu'
    
    major, minor = int(match.group(1)), int(match.group(2))
    
    # Map to available CUDA tags
    if major == 12:
        if minor >= 4:
            return 'cu124'
        elif minor >= 1:
            return 'cu121'
        else:
            return 'cu121'  # CUDA 12.0 also uses cu121
    elif major == 11:
        return 'cu118'  # All CUDA 11.x use cu118
    else:
        return 'cpu'  # Unsupported CUDA version, fallback to CPU


def build_wheel_url(torch_version, cuda_tag, python_version, platform_tag, commit_hash='fd27788'):
    """Build download URL for prebuilt wheel package
    
    Naming format: detectron2-0.6+{commit_hash}pt{torch_version}{cuda_tag}-cp{py_ver}-cp{py_ver}-{platform}.whl
    Example: detectron2-0.6+fd27788pt2.5.0cu121-cp310-cp310-linux_x86_64.whl
    """
    # Parse Python version
    py_match = re.match(r'(\d+)\.(\d+)', python_version)
    if not py_match:
        return None
    py_major, py_minor = py_match.group(1), py_match.group(2)
    py_tag = f"cp{py_major}{py_minor}"
    
    # Build package name
    if commit_hash:
        pkg_version = f"0.6+{commit_hash}pt{torch_version}{cuda_tag}"
    else:
        pkg_version = f"0.6+pt{torch_version}{cuda_tag}"
    
    # macOS platform requires special handling
    if 'macosx' in platform_tag:
        # Try different macOS version tags
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
            # No commit hash case, release tag format is different
            release_tag = f"detectron2-0.6%2Bpt{torch_version}{cuda_tag}"
        url = f"{PREBUILT_BASE_URL}/{release_tag}/{filename}"
        return url


def check_url_exists(url):
    """Check if URL exists"""
    try:
        req = urllib.request.Request(url, method='HEAD')
        urllib.request.urlopen(req, timeout=10)
        return True
    except (urllib.error.HTTPError, urllib.error.URLError):
        return False


def find_best_wheel_url(torch_version, cuda_tag, python_version, platform_tag):
    """Find the best prebuilt wheel package URL
    
    Try different commit hashes by priority
    """
    # Get list of commit hashes supported by this PyTorch version
    commit_hashes = TORCH_VERSION_COMMIT_MAP.get(torch_version, KNOWN_COMMIT_HASHES)
    
    for commit_hash in commit_hashes:
        url = build_wheel_url(torch_version, cuda_tag, python_version, platform_tag, commit_hash)
        
        if isinstance(url, list):
            # macOS case, try multiple URLs
            for u in url:
                print(f"  Trying: {u}")
                if check_url_exists(u):
                    return u
        else:
            print(f"  Trying: {url}")
            if check_url_exists(url):
                return url
    
    return None


def install_detectron2(info):
    """Install detectron2 based on environment information
    
    Uses third-party prebuilt packages: https://miropsota.github.io/torch_packages_builder/detectron2/
    Supports Linux, Windows, macOS platforms
    """
    os_name = info['os']
    has_cuda = info['has_cuda']
    torch_version = get_torch_version()
    
    if torch_version is None:
        print("Error: PyTorch is not installed")
        return False
    
    # Re-detect CUDA (PyTorch may have just been installed)
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            info['cuda_version'] = torch.version.cuda
    except ImportError:
        pass
    
    print("\nInstalling Detectron2...")
    print(f"System: {os_name}, CUDA: {has_cuda}, PyTorch: {torch_version}")
    
    # Get platform tag
    platform_tag = get_platform_tag()
    if not platform_tag:
        print(f"Warning: Unsupported platform {os_name} {platform.machine()}")
        platform_tag = None
    
    # Get CUDA tag
    if os_name == 'Darwin':
        # macOS only supports CPU
        cuda_tag = 'cpu'
    elif has_cuda and info.get('cuda_version'):
        cuda_tag = get_cuda_tag(info['cuda_version'])
    else:
        cuda_tag = 'cpu'
    
    # Get Python version
    python_version = info['python_version']
    
    print(f"Platform: {platform_tag}, CUDA tag: {cuda_tag}, Python: {python_version}")
    
    # Try using third-party prebuilt packages
    install_cmd = None
    wheel_url = None
    
    if platform_tag and torch_version in TORCH_VERSION_COMMIT_MAP:
        print("\nSearching for prebuilt packages...")
        wheel_url = find_best_wheel_url(torch_version, cuda_tag, python_version, platform_tag)
        
        if wheel_url:
            print(f"\nFound prebuilt package: {wheel_url}")
            install_cmd = [
                sys.executable, '-m', 'pip', 'install', wheel_url
            ]
    
    if not install_cmd:
        print("\nPrebuilt package not found, compiling from source...")
        install_cmd = [
            sys.executable, '-m', 'pip', 'install',
            'git+https://github.com/facebookresearch/detectron2.git'
        ]
    
    print(f"Executing: {' '.join(install_cmd)}")
    result = subprocess.run(install_cmd)
    
    if result.returncode != 0 and wheel_url:
        # If prebuilt package fails, try compiling from source
        print("\nPrebuilt package installation failed, trying to compile from source...")
        fallback_cmd = [
            sys.executable, '-m', 'pip', 'install',
            'git+https://github.com/facebookresearch/detectron2.git'
        ]
        print(f"Executing: {' '.join(fallback_cmd)}")
        result = subprocess.run(fallback_cmd)
    
    return result.returncode == 0

def main():
    """Main function: automatically detect environment and install PyTorch and detectron2"""
    print("=" * 60)
    print("Auto Installation Script - PyTorch and Detectron2")
    print("=" * 60)
    
    # Detect environment
    info = get_system_info()
    print("\nDetected environment:")
    print(f"  - Operating System: {info['os']}")
    print(f"  - Python: {info['python_version']}")
    print(f"  - CUDA Available: {info['has_cuda']}")
    if info['cuda_version']:
        print(f"  - CUDA Version: {info['cuda_version']}")
    
    # Check if PyTorch is already installed
    torch_version = get_torch_version()
    if torch_version:
        print(f"\nDetected PyTorch {torch_version}")
        # Verify CUDA support
        try:
            import torch
            if info['has_cuda'] and not torch.cuda.is_available():
                print("Warning: System has CUDA but PyTorch is CPU version")
                response = input("Reinstall CUDA version of PyTorch? (y/n): ").strip().lower()
                if response == 'y':
                    if not install_pytorch(info):
                        sys.exit(1)
        except ImportError:
            pass
    else:
        print("\nPyTorch not detected, starting installation...")
        if not install_pytorch(info):
            sys.exit(1)
    
    # Update torch version information
    torch_version = get_torch_version()
    print(f"\nCurrent PyTorch version: {torch_version}")
    
    # Check if detectron2 is already installed
    try:
        import detectron2
        print(f"Detected Detectron2 {detectron2.__version__}")
        response = input("Reinstall Detectron2? (y/n): ").strip().lower()
        if response != 'y':
            print("Skipping Detectron2 installation")
            return
    except ImportError:
        pass
    
    # Install detectron2
    # Re-get system information (PyTorch may have been updated)
    info = get_system_info()
    success = install_detectron2(info)
    
    if success:
        print("\n" + "=" * 60)
        print("✓ Installation completed!")
        print("=" * 60)
        
        # Verify installation
        try:
            import torch
            import detectron2
            print("\nVerifying installation:")
            print(f"  - PyTorch: {torch.__version__}")
            print(f"  - CUDA Available: {torch.cuda.is_available()}")
            print(f"  - Detectron2: {detectron2.__version__}")
        except ImportError as e:
            print(f"\nWarning: Error during verification: {e}")
    else:
        print("\n" + "=" * 60)
        print("✗ Installation failed, please check error messages")
        print("=" * 60)
        sys.exit(1)

def auto_install():
    """Non-interactive automatic installation (for script calls)"""
    print("=" * 60)
    print("Auto Installation Script - PyTorch and Detectron2 (Non-interactive Mode)")
    print("=" * 60)
    
    # Detect environment
    info = get_system_info()
    print("\nDetected environment:")
    print(f"  - Operating System: {info['os']}")
    print(f"  - Python: {info['python_version']}")
    print(f"  - CUDA Available: {info['has_cuda']}")
    if info['cuda_version']:
        print(f"  - CUDA Version: {info['cuda_version']}")
    
    # Check if PyTorch is already installed
    torch_version = get_torch_version()
    need_reinstall_pytorch = False
    
    if torch_version:
        print(f"\nDetected PyTorch {torch_version}")
        # Verify CUDA support
        try:
            import torch
            if info['has_cuda'] and not torch.cuda.is_available():
                print("Warning: System has CUDA but PyTorch is CPU version, will reinstall")
                need_reinstall_pytorch = True
        except ImportError:
            need_reinstall_pytorch = True
    else:
        need_reinstall_pytorch = True
    
    if need_reinstall_pytorch:
        print("\nStarting PyTorch installation...")
        if not install_pytorch(info):
            sys.exit(1)
    
    # Update torch version information
    torch_version = get_torch_version()
    print(f"\nCurrent PyTorch version: {torch_version}")
    
    # Check if detectron2 is already installed
    need_install_detectron2 = True
    try:
        import detectron2
        print(f"Detected Detectron2 {detectron2.__version__}")
        need_install_detectron2 = False
    except ImportError:
        pass
    
    if need_install_detectron2:
        # Re-get system information
        info = get_system_info()
        success = install_detectron2(info)
        
        if not success:
            print("\n✗ Detectron2 installation failed")
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ Installation completed!")
    print("=" * 60)
    
    # Verify installation
    try:
        import torch
        import detectron2
        print("\nVerifying installation:")
        print(f"  - PyTorch: {torch.__version__}")
        print(f"  - CUDA Available: {torch.cuda.is_available()}")
        print(f"  - Detectron2: {detectron2.__version__}")
    except ImportError as e:
        print(f"\nWarning: Error during verification: {e}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Automatically install PyTorch and Detectron2')
    parser.add_argument('--auto', action='store_true', help='Non-interactive automatic installation')
    args = parser.parse_args()
    
    if args.auto:
        auto_install()
    else:
        main()
