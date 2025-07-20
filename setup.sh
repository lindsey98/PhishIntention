#!/bin/bash
# 1. Set ENV_NAME with default value "phishintention" if not already set
ENV_NAME="${ENV_NAME:-phishintention}"
# Double check if ENV_NAME is set (it always will be now, but kept for flexibility)
if [ -z "$ENV_NAME" ]; then
  error_exit "ENV_NAME is not set. Please set the environment name and try again."
fi

# 2. Ensure Conda is installed
if ! command -v conda &> /dev/null; then
  error_exit "Conda is not installed. Please install Conda and try again."
fi

# 3. Create or activate conda environment
FILEDIR=$(pwd)
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 4. Check if Conda environment exists
conda info --envs | grep -q "^$ENV_NAME "
if [ $? -eq 0 ]; then
   echo "Activating existing Conda environment $ENV_NAME"
   conda activate "$ENV_NAME"
else
   echo "Creating and activating new Conda environment with Python 3.9"
   conda create -n "$ENV_NAME" python=3.9 -y
   conda activate "$ENV_NAME"
fi


# 5. Install pytorch, torchvision, detectron2
OS=$(uname -s)
if [[ "$OS" == "Darwin" ]]; then
  echo "Detected macOS. Installing PyTorch and torchvision for macOS..."
  conda run -n "$ENV_NAME" pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0
  conda run -n "$ENV_NAME" pip install 'git+https://github.com/facebookresearch/detectron2.git'
else
  # Check for NVIDIA GPU by looking for 'nvcc' or 'nvidia-smi'
  if command -v nvcc > /dev/null 2>&1 || command -v nvidia-smi > /dev/null 2>&1; then
    echo "CUDA detected. Installing GPU-supported PyTorch and torchvision..."
    conda run -n "$ENV_NAME" pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f "https://download.pytorch.org/whl/torch_stable.html"
    conda run -n "$ENV_NAME" pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html"
  else
    echo "No CUDA detected. Installing CPU-only PyTorch and torchvision..."
    conda run -n "$ENV_NAME" pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f "https://download.pytorch.org/whl/torch_stable.html"
    conda run -n "$ENV_NAME" pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.9/index.html"
  fi
fi

# 6. Ensure gdown is installed in the environment
if ! conda run -n "$ENV_NAME" pip show gdown > /dev/null 2>&1; then
  echo "Installing gdown in the Conda environment..."
  conda run -n "$ENV_NAME" pip install gdown
fi

# 7. Install other requirements
if [ -f "requirements.txt" ]; then
  echo "Installing additional Python dependencies from requirements.txt..."
  conda run -n "$ENV_NAME" pip install -r requirements.txt
else
  error_exit "requirements.txt not found in the current directory."
fi

# 8. Install the package with verbose output using conda run
echo "Installing the package..."
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
  conda run -n "$ENV_NAME" gdown --id 1zw2MViLSZRemrEsn2G-UzHRTPTfZpaEd
  if ! command -v unzip &> /dev/null; then
    echo "unzip could not be found, installing it..."
    sudo apt-get install unzip -y
  fi
  unzip src.zip
  cd src || exit 1  # Exit if the directory doesn't exist
  # Check if there's a nested 'src/' directory
  if [ -d "src" ]; then
    echo "Nested directory 'src/' detected. Moving contents up..."
    shopt -s dotglob
    # Move everything from the nested directory to the current directory
    mv src/* .
    # Disable dotglob to revert back to normal behavior
    shopt -u dotglob
    # Remove the now-empty nested directory
    rmdir src
    cd ../
  else
    echo "No nested 'src/' directory found. No action needed."
  fi
fi

# Replace the placeholder in the YAML template
sed "s|CONDA_ENV_PATH_PLACEHOLDER|$package_location/phishintention|g" "$FILEDIR/phishintention/configs_template.yaml" > "$package_location/phishintention/configs.yaml"
cd "$FILEDIR"
echo "All packages installed and models downloaded successfully!"
