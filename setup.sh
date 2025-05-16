#!/bin/bash
# Check if ENV_NAME is set
if [ -z "$ENV_NAME" ]; then
  echo "ENV_NAME is not set. Please set the environment name and try again."
  exit 1
fi

retry_count=3  # Number of retries

download_with_retry() {
  local file_id=$1
  local file_name=$2
  local count=0

  until [ $count -ge $retry_count ]
  do
    conda run -n "$ENV_NAME" gdown --id "$file_id" -O "$file_name" && break  # attempt to download and break if successful
    count=$((count+1))
    echo "Retry $count of $retry_count..."
    sleep 1  # wait for 1 second before retrying
  done

  if [ $count -ge $retry_count ]; then
    echo "Failed to download $file_name after $retry_count attempts."
    exit 1
  fi
}

FILEDIR=$(pwd)
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Check if Conda environment exists
conda info --envs | grep -q "^$ENV_NAME "
if [ $? -eq 0 ]; then
   echo "Activating existing Conda environment $ENV_NAME"
   conda activate "$ENV_NAME"
else
   echo "Creating and activating new Conda environment with Python 3.8"
   conda create -n "$ENV_NAME" python=3.8 -y
   conda activate "$ENV_NAME"
fi

OS=$(uname -s)

if [[ "$OS" == "Darwin" ]]; then
  echo "Installing PyTorch and torchvision for macOS."
  conda run -n "$ENV_NAME" pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0
  conda run -n "$ENV_NAME"  python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
else
  # Check if NVIDIA GPU is available for Linux and Windows
  if command -v nvcc &> /dev/null || command -v nvidia-smi &> /dev/null; then
    echo "CUDA is detected, installing GPU-supported PyTorch and torchvision."
    conda run -n "$ENV_NAME" pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f "https://download.pytorch.org/whl/torch_stable.html"
    conda run -n "$ENV_NAME" python -m pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html"
  else
    echo "No CUDA detected, installing CPU-only PyTorch and torchvision."
    conda run -n "$ENV_NAME" pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f "https://download.pytorch.org/whl/torch_stable.html"
    conda run -n "$ENV_NAME" python -m pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.9/index.html"
  fi
fi

# Install additional packages
conda run -n "$ENV_NAME" python -m pip install -r requirements.txt
conda run -n "$ENV_NAME" pip install --upgrade pip setuptools wheel

## Download models
echo "Going to the directory of package Phishpedia in Conda environment myenv."
mkdir -p models/
cd models/


# RCNN model weights
if [ -f "layout_detector.pth" ]; then
  echo "layout_detector weights exists... skip"
else
  download_with_retry 1HWjE5Fv-c3nCDzLCBc7I3vClP1IeuP_I layout_detector.pth
fi


# Faster RCNN config
if [ -f "crp_classifier.pth.tar" ]; then
  echo "CRP classifier weights exists... skip"
else
  download_with_retry 1igEMRz0vFBonxAILeYMRWTyd7A9sRirO crp_classifier.pth.tar
fi


# Siamese model weights
if [ -f "crp_locator.pth" ]; then
  echo "crp_locator weights exists... skip"
else
  download_with_retry 1_O5SALqaJqvWoZDrdIVpsZyCnmSkzQcm crp_locator.pth
fi


if [ -f "ocr_pretrained.pth.tar" ]; then
  echo "OCR pretrained model weights exists... skip"
else
  download_with_retry 15pfVWnZR-at46gqxd50cWhrXemP8oaxp ocr_pretrained.pth.tar
fi
if [ -f "ocr_siamese.pth.tar" ]; then
  echo "OCR-siamese weights exists... skip"
else
  download_with_retry 1BxJf5lAcNEnnC0In55flWZ89xwlYkzPk ocr_siamese.pth.tar
fi

# Reference list
if [ -f "expand_targetlist.zip" ]; then
  echo "Reference list exists... skip"
else
  download_with_retry 1fr5ZxBKyDiNZ_1B6rRAfZbAHBBoUjZ7I expand_targetlist.zip
fi

# Unzip the file
unzip expand_targetlist.zip -d expand_targetlist

# Change to the extracted directory
cd expand_targetlist || exit 1  # Exit if the directory doesn't exist

# Check if there's a nested 'expand_targetlist/' directory
if [ -d "expand_targetlist" ]; then
  echo "Nested directory 'expand_targetlist/' detected. Moving contents up..."

  # Enable dotglob to include hidden files
  shopt -s dotglob

  # Move everything from the nested directory to the current directory
  mv expand_targetlist/* .

  # Disable dotglob to revert back to normal behavior
  shopt -u dotglob

  # Remove the now-empty nested directory
  rmdir expand_targetlist
  cd ../
else
  echo "No nested 'expand_targetlist/' directory found. No action needed."
fi

echo "Extraction completed successfully."

# Domain map
if [ -f "domain_map.pkl" ]; then
  echo "Domain map exists... skip"
else
  download_with_retry 1qSdkSSoCYUkZMKs44Rup_1DPBxHnEKl1 domain_map.pkl
fi


# Replace the placeholder in the YAML template
echo "All packages installed successfully!"
