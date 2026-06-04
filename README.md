# PhishIntention

<div align="center">

![Protected Brands](https://img.shields.io/badge/Protected_Brands-277-green?style=flat-square)

<a href="https://www.usenix.org/conference/usenixsecurity22/presentation/liu-ruofan">Paper</a> •
<a href="https://sites.google.com/view/phishintention">Website</a> •
<a href="https://www.youtube.com/watch?v=yU7FrlSJ818">Video</a> •
<a href="#citation">Citation</a>

</div>

Official implementation of **"Inferring Phishing Intention via Webpage Appearance and Dynamics: A Deep Vision-Based Approach"** (USENIX Security 2022) — [paper](http://linyun.info/publications/usenix22.pdf), [project website](https://sites.google.com/view/phishintention/home).

## Motivation

Existing reference-based phishing detectors only capture **brand intention**, which makes them prone to false positives. PhishIntention is, to our knowledge, the first system to analyze both **brand intention** and **credential-taking intention** in a systematic way, and it powers a phishing-monitoring pipeline that reports phishing webpages daily with state-of-the-art precision.

## Framework

<img src="big_pic/Screenshot 2021-08-13 at 9.15.56 PM.png" style="width:2000px;height:350px"/>

`Input`: a screenshot (and optionally the HTML source). `Output`: `Phish`/`Benign` and the predicted phishing target.

1. **Abstract Layout Detector (AWL)** — detect page elements (logos, inputs, buttons, …).
2. **OCR-aided Siamese Logo Comparison** — if no brand is matched, return `Benign`; otherwise continue with the matched target.
3. **CRP Classifier** — decide whether the page requests credentials (CRP). If yes, go to step 5; if no and the CRP locator has not run yet, go to step 4; otherwise return `Benign`.
4. **CRP Locator (dynamic analysis)** — click login/signup links to reach a credential page. On success, restart from step 1 with the updated URL and screenshot; on failure, return `Benign`.
5. **Decision** — a credential page with a matched brand inconsistent with the domain ⇒ `Phish` + target; otherwise `Benign`.

## Project Structure

```
PhishIntention/
├── configs/                  # Configuration
│   ├── configs.yaml          #   Global config: weight paths, thresholds, brand count
│   └── detectron2/           #   Detectron2 architecture configs for the Faster R-CNN detectors
├── modules/                  # Pipeline components
│   ├── awl_detector.py       #   Abstract layout detector (Faster R-CNN)
│   ├── crp_classifier.py     #   Credential-requiring-page classifier + HTML heuristic
│   ├── crp_locator.py        #   Dynamic analysis to locate credential pages
│   └── logo_matching.py      #   OCR-aided Siamese logo matcher
├── networks/                 # Neural-network architectures
│   ├── bit_backbone.py       #   Shared BiT ResNet-v2 building blocks
│   ├── crp_models.py         #   CRP classifier networks
│   ├── siamese_models.py     #   Siamese logo-matching network
│   └── ocr/                  #   Vendored OCR encoder (ASTER text recognizer)
├── utils/                    # Image and Selenium/WebDriver helpers
├── scripts/                  # Install/setup scripts (PyTorch, Detectron2, Chrome, weights)
├── configs.py                # Loads configs and builds all models
└── phishintention.py         # Entry point / pipeline orchestrator

# Created at setup time (downloaded by scripts/setup.sh|setup.bat, not committed):
models/                       # Model weights (*.pth) + reference data
                              #   (expand_targetlist = brand logos, domain_map.pkl = brand→domain)
```

## Installation

The setup scripts **auto-detect your system and GPU** and install the appropriate PyTorch and Detectron2 builds (CUDA if an NVIDIA GPU is present, otherwise CPU). They also download all model weights and the reference list into `models/`.

### Option A — Docker (recommended for Linux/Windows)

```bash
git clone https://github.com/lindsey98/PhishIntention.git
cd PhishIntention
docker build -t phishintention .

docker run --rm phishintention \
  pixi run python phishintention.py --folder datasets/test_sites --output_fn test.json
```

### Option B — Native install

Prerequisite: [Pixi](https://pixi.sh/latest/).

<details>
<summary><b>Linux</b></summary>

```bash
git clone https://github.com/lindsey98/PhishIntention.git
cd PhishIntention
export KMP_DUPLICATE_LIB_OK=TRUE

# Install pixi (restart your terminal afterwards)
curl -fsSL https://pixi.sh/install.sh | sh

# Install Chrome (Ubuntu/Debian)
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo dpkg -i google-chrome-stable_current_amd64.deb
sudo apt-get install -f

pixi install
chmod +x scripts/setup.sh && ./scripts/setup.sh
```
</details>

<details>
<summary><b>macOS</b></summary>

```bash
git clone https://github.com/lindsey98/PhishIntention.git
cd PhishIntention
export KMP_DUPLICATE_LIB_OK=TRUE

# Install pixi (restart your terminal afterwards)
curl -fsSL https://pixi.sh/install.sh | sh

# Install Chrome and expose it on PATH (use ~/.zshrc for zsh)
brew install --cask google-chrome
echo 'export CHROME_BIN="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"' >> ~/.bash_profile
echo 'export PATH="/Applications/Google Chrome.app/Contents/MacOS:$PATH"' >> ~/.bash_profile
source ~/.bash_profile

pixi install
chmod +x scripts/setup.sh && ./scripts/setup.sh
```
</details>

<details>
<summary><b>Windows</b></summary>

```bash
git clone https://github.com/lindsey98/PhishIntention.git
cd PhishIntention

# Install latest Chrome and ChromeDriver
.\scripts\chrome_setup.bat

# Install pixi (restart your terminal afterwards)
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"

pixi install
.\scripts\setup.bat
```
</details>

If the automatic PyTorch/Detectron2 install fails, run it interactively with `pixi run python scripts/auto_install_detectron2.py`, or install [PyTorch](https://pytorch.org/get-started/locally/) and [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) manually.

### ChromeDriver

The setup scripts install a matching ChromeDriver automatically. To pin it manually, check your Chrome version (`google-chrome --version` or `chrome://version`), download the matching driver from [this repository](https://github.com/dreamshao/chromedriver/tree/main), and place the binary under `./chromedriver/`.

## Usage

```bash
pixi run python phishintention.py --folder datasets/test_sites --output_fn test.json
```

On the first run, the reference list is embedded and cached to `LOGO_FEATS.npy`, which can take a few minutes.

The input `--folder` must contain one sub-directory per site:

```
test_site_1/
├── info.txt    # the URL (required)
├── shot.png    # the screenshot (required)
└── html.txt    # the HTML source (optional)
```

Results are written to the `--output_fn` JSON file; a `predict.png` visualization is saved next to the screenshot when phishing is detected.

## Baselines

The phishing detection and identification baselines from our paper are available [here](https://github.com/lindsey98/PhishingBaseline).

## Citation

```bibtex
@inproceedings{liu2022inferring,
  title={Inferring Phishing Intention via Webpage Appearance and Dynamics: A Deep Vision Based Approach},
  author={Liu, Ruofan and Lin, Yun and Yang, Xianglin and Ng, Siang Hwee and Divakaran, Dinil Mon and Dong, Jin Song},
  booktitle={31st USENIX Security Symposium (USENIX Security 22)},
  year={2022}
}
```

## Contact

For questions, open an issue or email [liu.ruofan16@u.nus.edu](mailto:liu.ruofan16@u.nus.edu), [lin_yun@sjtu.edu.cn](mailto:lin_yun@sjtu.edu.cn), or [dcsdjs@nus.edu.sg](mailto:dcsdjs@nus.edu.sg).
