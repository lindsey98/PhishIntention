@echo off
setlocal enabledelayedexpansion

:: ------------------------------------------------------------------------------
:: Install Detectron2
:: ------------------------------------------------------------------------------
echo [%DATE% %TIME%] Installing detectron2...
pixi run pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2.git
if errorlevel 1 (
    echo [ERROR] Failed to install detectron2.
    exit /b 1
)

REM Create models directory and change into it
if not exist models (
    mkdir models
)
cd models

REM RCNN model weights
if exist "layout_detector.pth" (
    echo layout_detector weights exist... skip
) else (
    call pixi run gdown --id 1HWjE5Fv-c3nCDzLCBc7I3vClP1IeuP_I -O layout_detector.pth
)

REM Faster RCNN config
if exist "crp_classifier.pth.tar" (
    echo CRP classifier weights exist... skip
) else (
    call pixi run gdown --id 1igEMRz0vFBonxAILeYMRWTyd7A9sRirO -O crp_classifier.pth.tar
)

REM Siamese model weights
if exist "crp_locator.pth" (
    echo crp_locator weights exist... skip
) else (
    call pixi run gdown --id 1_O5SALqaJqvWoZDrdIVpsZyCnmSkzQcm -O crp_locator.pth
)

REM Siamese pretrained weights
if exist "ocr_pretrained.pth.tar" (
    echo OCR pretrained model weights exist... skip
) else (
    call pixi run gdown --id 15pfVWnZR-at46gqxd50cWhrXemP8oaxp -O ocr_pretrained.pth.tar
)

REM Siamese fine-tuned weights
if exist "ocr_siamese.pth.tar" (
    echo OCR-siamese weights exist... skip
) else (
    call pixi run gdown --id 1BxJf5lAcNEnnC0In55flWZ89xwlYkzPk -O ocr_siamese.pth.tar
)

REM Reference list
if exist "expand_targetlist.zip" (
    echo Reference list exists... skip
) else (
    call pixi run gdown --id 1fr5ZxBKyDiNZ_1B6rRAfZbAHBBoUjZ7I -O expand_targetlist.zip
)

REM Domain map
if exist "domain_map.pkl" (
    echo Domain map exists... skip
) else (
    call pixi run gdown --id 1qSdkSSoCYUkZMKs44Rup_1DPBxHnEKl1 -O domain_map.pkl
)

REM Unzip the file
powershell -Command "Expand-Archive -Force expand_targetlist.zip expand_targetlist"
if errorlevel 1 (
    echo [ERROR] Failed to unzip expand_targetlist.zip
    exit /b 1
)

REM Change into extracted directory
if exist "expand_targetlist" (
    cd expand_targetlist
) else (
    echo [ERROR] Directory expand_targetlist not found after extraction.
    exit /b 1
)

REM Handle nested expand_targetlist/ directory
if exist "expand_targetlist" (
    echo Nested directory 'expand_targetlist\' detected. Moving contents up...
    xcopy expand_targetlist\*.* . /E /H /Y
    rmdir /S /Q expand_targetlist
    cd ..
) else (
    echo No nested 'expand_targetlist\' directory found. No action needed.
    cd ..
)

echo Extraction completed successfully.
echo All packages installed successfully!
