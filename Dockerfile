FROM ubuntu:22.04

# Avoid interactive installation prompts
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install base tools and Python
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python3-venv \
    wget \
    curl \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libnspr4 \
    libnss3 \
    libgconf-2-4 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .
# Install Google Chrome and chromedriver
RUN export KMP_DUPLICATE_LIB_OK=TRUE \
    && curl -fsSL https://pixi.sh/install.sh | sh \
    && echo 'export PATH="/root/.pixi/bin:$PATH"' >> /etc/profile.d/pixi.sh \
    && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && apt-get install -y dos2unix && dos2unix scripts/chrome_setup.sh && chmod +x scripts/chrome_setup.sh && ./scripts/chrome_setup.sh linux \
    && apt-get clean

ENV PATH="/opt/google/chrome:/usr/local/bin:/root/.pixi/bin:$PATH"
ENV CHROME_BIN="/opt/google/chrome/chrome"

RUN echo "=== test Chrome installation ===" \
    && which google-chrome-stable \
    && google-chrome-stable --version \
    && pixi install \
    && dos2unix scripts/setup.sh \
    && chmod +x scripts/setup.sh \
    && ./scripts/setup.sh