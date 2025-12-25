#!/bin/bash

if [ "$#" -ne 1 ] || { [ "$1" != "linux" ] && [ "$1" != "macos" ]; }; then
    echo "用法: $0 <linux|macos>"
    echo "示例: $0 linux"
    echo "示例: $0 macos"
    exit 1
fi

if [ "$1" = "linux" ]; then
    platform="linux64"
    archive_dir="chromedriver-linux64"
elif [ "$1" = "macos" ]; then
    platform="mac-arm64"
    archive_dir="chromedriver-mac-arm64"
fi

curl -O https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions.json

grep -o '"version":"[^"]*"' last-known-good-versions.json | head -1 | cut -d'"' -f4 > chrome_version.txt

stableVersion=$(cat chrome_version.txt)

curl -O "https://storage.googleapis.com/chrome-for-testing-public/$stableVersion/$platform/chromedriver-$platform.zip"

unzip "chromedriver-$platform.zip"

mkdir -p chromedriver

cp "$archive_dir/chromedriver" chromedriver/
