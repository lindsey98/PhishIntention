#!/bin/bash
retry_count=3  # Number of retries

download_with_retry() {
  local file_id=$1
  local file_name=$2
  local count=0

  until [ $count -ge $retry_count ]
  do
    gdown --id "$file_id" -O "$file_name" && break  # attempt to download and break if successful
    count=$((count+1))
    echo "Retry $count of $retry_count..."
    sleep 1  # wait for 5 seconds before retrying
  done

  if [ $count -ge $retry_count ]; then
    echo "Failed to download $file_name after $retry_count attempts."
  fi
}

FILEDIR=$(pwd)
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda info --envs | grep -w "phishintention" > /dev/null

if [ $? -eq 0 ]; then
   echo "Activating Conda environment phishintention"
   conda activate phishintention
else
   echo "Creating and activating new Conda environment phishintention with Python 3.8"
   conda create -n phishintention python=3.8
   conda activate phishintention
fi


OS=$(uname -s)

if [[ "$OS" == "Darwin" ]]; then
  echo "Installing PyTorch and torchvision for macOS."
  pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0
  python -m pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.9/index.html"
else
  # Check if NVIDIA GPU is available for Linux and Windows
  if command -v nvcc || command -v nvidia-smi &> /dev/null; then
    echo "CUDA is detected, installing GPU-supported PyTorch and torchvision."
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f "https://download.pytorch.org/whl/torch_stable.html"
    python -m pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html"
  else
    echo "No CUDA detected, installing CPU-only PyTorch and torchvision."
    pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f "https://download.pytorch.org/whl/torch_stable.html"
    python -m pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.9/index.html"
  fi
fi

python -m pip install -r requirements.txt
python -m pip install helium
python -m pip install webdriver-manager
python -m pip install gdown

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

# Domain map
if [ -f "domain_map.pkl" ]; then
  echo "Domain map exists... skip"
else
  download_with_retry 1qSdkSSoCYUkZMKs44Rup_1DPBxHnEKl1 domain_map.pkl
fi


# Replace the placeholder in the YAML template
echo "All packages installed successfully!"