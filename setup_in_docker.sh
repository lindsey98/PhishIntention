#!/bin/bash

set -euo pipefail  # Safer bash behavior
IFS=$'\n\t'

# Set up model directory
FILEDIR="$(pwd)"
MODELS_DIR="$FILEDIR/models"
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

# RCNN model weights
if [ -f "layout_detector.pth" ]; then
  echo "layout_detector weights exists... skip"
else
  gdown --id "1HWjE5Fv-c3nCDzLCBc7I3vClP1IeuP_I" -O "layout_detector.pth"
fi

# Faster RCNN config
if [ -f "crp_classifier.pth.tar" ]; then
  echo "CRP classifier weights exists... skip"
else
  gdown --id "1igEMRz0vFBonxAILeYMRWTyd7A9sRirO" -O "crp_classifier.pth.tar"
fi


# Siamese model weights
if [ -f "crp_locator.pth" ]; then
  echo "crp_locator weights exists... skip"
else
  gdown --id  "1_O5SALqaJqvWoZDrdIVpsZyCnmSkzQcm" -O "crp_locator.pth"
fi

# Siamese model pretrained weights
if [ -f "ocr_pretrained.pth.tar" ]; then
  echo "OCR pretrained model weights exists... skip"
else
  gdown --id "15pfVWnZR-at46gqxd50cWhrXemP8oaxp" -O "ocr_pretrained.pth.tar"
fi

# Siamese model finetuned weights
if [ -f "ocr_siamese.pth.tar" ]; then
  echo "OCR-siamese weights exists... skip"
else
  gdown --id  "1BxJf5lAcNEnnC0In55flWZ89xwlYkzPk" -O "ocr_siamese.pth.tar"
fi

# Reference list
if [ -f "expand_targetlist.zip" ]; then
  echo "Reference list exists... skip"
else
  gdown --id "1fr5ZxBKyDiNZ_1B6rRAfZbAHBBoUjZ7I" -O "expand_targetlist.zip"
fi

# Domain map
if [ -f "domain_map.pkl" ]; then
  echo "Domain map exists... skip"
else
  gdown --id "1qSdkSSoCYUkZMKs44Rup_1DPBxHnEKl1" -O "domain_map.pkl"
fi

# Extract and flatten expand_targetlist
echo "Extracting expand_targetlist.zip..."
unzip -o expand_targetlist.zip -d expand_targetlist
cd expand_targetlist || error_exit "Extraction directory missing."

if [ -d "expand_targetlist" ]; then
  echo "Flattening nested expand_targetlist/ directory..."
  mv expand_targetlist/* .
  rm -r expand_targetlist
fi

echo "Extraction completed successfully."
echo "All packages installed successfully!"
