FROM ubuntu:22.04

# 避免交互式安装提示
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# 安装基础工具和Python
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

# 复制项目文件
COPY . .

COPY chrome.deb /tmp/chrome.deb

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcairo2 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libexpat1 \
    libgbm1 \
    libglib2.0-0 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    wget \
    xdg-utils && \
    dpkg -i /tmp/chrome.deb || apt-get install -yf && \
    rm /tmp/chrome.deb

RUN export KMP_DUPLICATE_LIB_OK=TRUE \
    && curl -fsSL https://pixi.sh/install.sh | sh \
    && echo 'export PATH="/root/.pixi/bin:$PATH"' >> /etc/profile.d/pixi.sh

RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - && \
    echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update && \
    apt-get install -y google-chrome-stable

RUN apt-get install -y dos2unix && dos2unix chrome_setup.sh && chmod +x chrome_setup.sh && ./chrome_setup.sh linux

# RUN chmod +x chrome_setup.sh \
#     && ./chrome_setup.sh linux

ENV PATH="/opt/google/chrome:/usr/local/bin:/root/.pixi/bin:$PATH"
ENV CHROME_BIN="/opt/google/chrome/chrome"

RUN echo "=== test Chrome installation ===" && \
    which google-chrome-stable && \
    google-chrome-stable --version

RUN pixi install

RUN dos2unix setup.sh \
    && chmod +x setup.sh \
    && ./setup.sh