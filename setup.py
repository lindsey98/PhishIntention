from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import sys
import subprocess  # This module is used for executing shell commands

class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def run(self):
        install.run(self)
        self.download_models()

    def download_models(self):
        # Define the directory to store model files
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phishintention', 'models')
        os.makedirs(model_dir, exist_ok=True)
        os.chdir(model_dir)

        # Model files and their Google Drive IDs
        model_files = {
            "layout_detector.pth": "1HWjE5Fv-c3nCDzLCBc7I3vClP1IeuP_I",
            "crp_classifier.pth.tar": "1igEMRz0vFBonxAILeYMRWTyd7A9sRirO",
            "crp_locator.pth": "1_O5SALqaJqvWoZDrdIVpsZyCnmSkzQcm",
            "ocr_pretrained.pth.tar": "15pfVWnZR-at46gqxd50cWhrXemP8oaxp",
            "ocr_siamese.pth.tar": "1BxJf5lAcNEnnC0In55flWZ89xwlYkzPk",
            "expand_targetlist.zip": "1fr5ZxBKyDiNZ_1B6rRAfZbAHBBoUjZ7I",
            "domain_map.pkl": "1qSdkSSoCYUkZMKs44Rup_1DPBxHnEKl1"
        }

        for file_name, file_id in model_files.items():
            destination_path = os.path.join(model_dir, file_name)
            if not os.path.exists(destination_path):
                print(f"Downloading {file_name}...")
                self.download_file_from_google_drive(file_id, destination_path)

    def download_file_from_google_drive(self, file_id, destination):
        """Download a file from Google Drive using gdown"""
        command = f"gdown --id {file_id} -O {destination}"
        # Check for HTTP proxy in the environment
        if 'http_proxy' in os.environ or 'https_proxy' in os.environ:
            proxy = os.environ.get('https_proxy') or os.environ.get('http_proxy')
            command += f" --proxy {proxy}"
        subprocess.call(command, shell=True)

def check_cuda():
    """Check if CUDA is available"""
    cuda_available = False
    try:
        subprocess.check_call(['nvcc', '--version'])
        cuda_available = True
    except subprocess.CalledProcessError:
        pass
    try:
        subprocess.check_call(['nvidia-smi'])
        cuda_available = True
    except subprocess.CalledProcessError:
        pass
    return cuda_available


class CustomInstall(install):
    """Custom installation script to handle special dependency cases."""

    def run(self):
        install.run(self)
        # Handle Detectron2 installation based on the system
        if os.uname().sysname == 'Darwin':
            detectron2_url = "https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.9/index.html"
            torch_urls = ""
        elif check_cuda():
            detectron2_url = "https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html"
            torch_url = "https://download.pytorch.org/whl/torch_stable.html"
        else:
            detectron2_url = "https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.9/index.html"
            torch_url =

        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'detectron2', '-f', detectron2_url])

def get_dependencies():
    base_dependencies = [
        'scipy',
        'tldextract',
        'opencv-python',
        'pandas',
        'numpy',
        'tqdm',
        'Pillow==8.4.0',
        'pathlib',
        'fvcore',
        'pycocotools',
        'scikit-learn',
        'lxml',
        'gdown',
        'editdistance',
        'lxml',
        'cryptography==38.0.4',
        'httpcore==0.15.0',
        'h11',
        'h2',
        'hyperframe',
        'selenium==4.0.0',
        'selenium-wire',
        'helium',
        'webdriver-manager',
        'gdown'
    ]
    if os.uname().sysname == 'Darwin':
        # Specific dependencies for MacOS
        return base_dependencies + [
            'torch==1.9.0',
            'torchvision==0.10.0',
            'torchaudio==0.9.0',
        ]
    elif check_cuda():
        # Dependencies for systems with CUDA
        return base_dependencies + [
            'torch==1.9.0+cu111',
            'torchvision==0.10.0+cu111',
            'torchaudio==0.9.0',
        ]
    else:
        # Dependencies for CPU-only systems
        return base_dependencies + [
            'torch==1.9.0+cpu',
            'torchvision==0.10.0+cpu',
            'torchaudio==0.9.0',
        ]



setup(
    name='PhishIntention',
    version='0.1.0',
    author='Ruofan Liu',
    author_email='kelseyliu1998@gmail.com',
    description='PhishIntention',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/lindsey98/PhishIntention',
    packages=find_packages(),
    install_requires=get_dependencies(),
    cmdclass={
        'install': CustomInstall,
    #     'install': PostInstallCommand,
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
)