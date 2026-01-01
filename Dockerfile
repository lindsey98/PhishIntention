# PhishIntention container
# Base image chosen for Python 3.10 compatibility with torch 1.13 and detectron2
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
PYTHONUNBUFFERED=1 \
KMP_DUPLICATE_LIB_OK=TRUE

# Use Tsinghua University APT mirror for faster package installs in China
RUN set -eux; \
MIRROR="mirrors.tuna.tsinghua.edu.cn"; \
if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
sed -i "s|http://deb.debian.org/debian|https://$MIRROR/debian|g; \
s|https://deb.debian.org/debian|https://$MIRROR/debian|g; \
s|http://security.debian.org/debian-security|https://$MIRROR/debian-security|g; \
s|https://security.debian.org/debian-security|https://$MIRROR/debian-security|g" \
/etc/apt/sources.list.d/debian.sources; \
else \
sed -i "s|http://deb.debian.org/debian|https://$MIRROR/debian|g; \
s|https://deb.debian.org/debian|https://$MIRROR/debian|g; \
s|http://security.debian.org/debian-security|https://$MIRROR/debian-security|g; \
s|https://security.debian.org/debian-security|https://$MIRROR/debian-security|g" \
/etc/apt/sources.list; \
fi; \
apt-get update; \
apt-get install -y --no-install-recommends ca-certificates curl; \
rm -rf /var/lib/apt/lists/*

# Use Tsinghua University PyPI mirror for faster package installs in China
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# System deps for OpenCV, Detectron2 build, and Selenium/Chrome driver runtime
RUN apt-get update \ 
    && apt-get install -y --no-install-recommends \ 
       build-essential \ 
       git \ 
       curl \ 
       ca-certificates \ 
       unzip \ 
       wget \ 
       libglib2.0-0 \ 
       libsm6 \ 
       libxrender1 \ 
       libxext6 \ 
       libgl1 \ 
       chromium \ 
       chromium-driver \ 
       fonts-liberation \ 
       libasound2 \ 
       libatk-bridge2.0-0 \ 
       libatk1.0-0 \ 
       libatspi2.0-0 \ 
       libcups2 \ 
       libdbus-1-3 \ 
       libdrm2 \ 
       libgbm1 \ 
       libgtk-3-0 \ 
       libnspr4 \ 
       libnss3 \ 
       libwayland-client0 \ 
       libxcomposite1 \ 
       libxdamage1 \ 
       libxfixes3 \ 
       libxkbcommon0 \ 
       libxrandr2 \ 
       xdg-utils \ 
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Upgrade pip tooling
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir numpy==1.23.0

# Torch + torchvision (CPU builds) pinned for Detectron2 compatibility
RUN pip install --no-cache-dir \ 
    --index-url https://download.pytorch.org/whl/cpu \ 
    torch==1.13.1+cpu \ 
    torchvision==0.14.1+cpu

# Core Python deps
RUN pip install --no-cache-dir \ 
    numpy==1.23.0 \ 
    requests \ 
    scikit-learn \ 
    spacy \ 
    beautifulsoup4 \ 
    matplotlib \ 
    pandas \ 
    nltk \ 
    tqdm \ 
    unidecode \ 
    gdown \ 
    tldextract \ 
    scipy \ 
    pathlib \ 
    fvcore \ 
    lxml \ 
    psutil \ 
    Pillow==8.4.0 \ 
    editdistance \ 
    cryptography==38.0.4 \ 
    httpcore==0.15.0 \ 
    h11 \ 
    h2 \ 
    blinker==1.7.0 \ 
    hyperframe \ 
    selenium-wire \ 
    helium \ 
    selenium \ 
    webdriver-manager \ 
    flask \ 
    flask-cors \ 
    pycocotools \ 
    opencv-python \ 
    opencv-contrib-python

# Detectron2 from source (CPU build)
RUN pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2.git

# Ensure chromedriver has execute permissions
RUN if [ -f ./chromedriver-linux64/chromedriver ]; then chmod +x ./chromedriver-linux64/chromedriver; fi

RUN cp /usr/bin/chromedriver ./chromedriver-linux64/chromedriver

# Prepare models during build
RUN chmod +x setup_in_docker.sh
RUN ./setup_in_docker.sh

# Default command can be overridden; keep shell for interactive runs
CMD ["bash"]
