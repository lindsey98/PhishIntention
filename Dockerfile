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

# 安装pixi和google chrome
RUN export KMP_DUPLICATE_LIB_OK=TRUE \
    && curl -fsSL https://pixi.sh/install.sh | sh \
    && echo 'export PATH="/root/.pixi/bin:$PATH"' >> /etc/profile.d/pixi.sh \
    && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable dos2unix \
    && apt-get clean

ENV PATH="/opt/google/chrome:/usr/local/bin:/root/.pixi/bin:$PATH"
ENV CHROME_BIN="/opt/google/chrome/chrome"

# 项目文件通过挂载方式加载，运行容器时使用:
# docker run -v /path/to/PhishIntention:/app -it <image_name> bash
# 首次运行需执行: dos2unix chrome_setup.sh setup.sh && chmod +x chrome_setup.sh setup.sh && ./chrome_setup.sh linux && pixi install && ./setup.sh