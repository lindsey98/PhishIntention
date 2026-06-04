@echo off
setlocal enabledelayedexpansion

curl -O https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions.json

powershell -Command "$json=Get-Content -Path 'last-known-good-versions.json' -Raw | ConvertFrom-Json; $stableVersion=$json.channels.Stable.version; Write-Output $stableVersion" > chrome_version.txt

set /p stableVersion=<chrome_version.txt

curl -O "https://storage.googleapis.com/chrome-for-testing-public/%stableVersion%/win64/chrome-win64.zip"
unzip chrome-win64.zip

curl -O "https://storage.googleapis.com/chrome-for-testing-public/%stableVersion%/win64/chromedriver-win64.zip"
unzip chromedriver-win64.zip

mkdir chromedriver

@REM copy chromedriver-win64\chromedriver.exe chromedriver\chromedriver.exe
copy chromedriver-win64\chromedriver.exe chromedriver\chromedriver.exe