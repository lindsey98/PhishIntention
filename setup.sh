#!/bin/bash

FILEDIR=$(pwd)

# Install pytorch, torchvision, detectron2
OS=$(uname -s)

if [[ "$OS" == "Darwin" ]]; then
  echo "Installing PyTorch and torchvision for macOS."
  conda run -n "$ENV_NAME" pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0
  conda run -n "$ENV_NAME" python -m pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.9/index.html"
else
  # Check if NVIDIA GPU is available for Linux and Windows
  if command -v nvcc || command -v nvidia-smi &> /dev/null; then
    echo "CUDA is detected, installing GPU-supported PyTorch and torchvision."
    conda run -n "$ENV_NAME" pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f "https://download.pytorch.org/whl/torch_stable.html"
    conda run -n "$ENV_NAME" python -m pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html"
  else
    echo "No CUDA detected, installing CPU-only PyTorch and torchvision."
    conda run -n "$ENV_NAME" pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f "https://download.pytorch.org/whl/torch_stable.html"
    conda run -n "$ENV_NAME" python -m pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.9/index.html"
  fi
fi

# Install other requirements
conda run -n "$ENV_NAME" python -m pip install -r requirements.txt
conda run -n "$ENV_NAME" python -m pip install helium
conda run -n "$ENV_NAME" python -m pip install webdriver-manager
conda run -n "$ENV_NAME" python -m pip install gdown

# Install the package with verbose output using conda run
conda run -n "$ENV_NAME" pip install -v .

# Get the package location
package_location=$(conda run -n "$ENV_NAME" pip show phishintention | grep Location | awk '{print $2}')

# Check if the package location is found and not empty
if [ -z "$package_location" ]; then
  echo "Package phishintention not found in the Conda environment $ENV_NAME."
  exit 1
else
  echo "Going to the directory of package phishintention in Conda environment $ENV_NAME."
  cd "$package_location/phishintention" || exit
  conda run -n "$ENV_NAME" pip install gdown
  conda run -n "$ENV_NAME" gdown --id 1zw2MViLSZRemrEsn2G-UzHRTPTfZpaEd
  sudo apt-get install unzip
  unzip src.zip
fi

# Replace the placeholder in the YAML template
sed "s|CONDA_ENV_PATH_PLACEHOLDER|$package_location/phishintention|g" "$FILEDIR/phishintention/configs_template.yaml" > "$package_location/phishintention/configs.yaml"
cd "$FILEDIR"
