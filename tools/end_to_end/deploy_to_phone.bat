@echo off
echo Deploying to phone

REM checking adb is installed
where adb >nul 2>&1
if %errorlevel% neq 0 (
    echo adb could not be found
    exit /b 1
)

REM checking adb devices has output other than "List of devices attached"
for /f "tokens=1 delims=" %%a in ('adb devices ^| findstr /v /c:"List of devices attached"') do (
    set "ADB_DEVICES_OUTPUT=%%a"
)

if "%ADB_DEVICES_OUTPUT%"=="" (
    echo No devices found
    exit /b 1
)

REM 定义目标路径变量
set TARGET_PATH=/data/local/tmp/smartserving

REM 如果TARGET_PATH不存在，则创建
adb shell mkdir -p %TARGET_PATH%

adb push .\proj %TARGET_PATH%/

adb shell "%TARGET_PATH%/proj/bin/smart-run -d ./Llama-3.1-8B-PowerServe-QNN"

echo Deployment finished.