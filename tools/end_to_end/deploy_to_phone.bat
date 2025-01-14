@echo off
echo Deploying to phone

REM Check if adb is installed
where adb >nul 2>&1
if %errorlevel% neq 0 (
    echo adb could not be found
    exit /b 1
)

REM Check if any devices are connected
set "ADB_DEVICES_OUTPUT="
for /f "tokens=1 delims=" %%a in ('adb devices ^| findstr /v /c:"List of devices attached"') do (
    set "ADB_DEVICES_OUTPUT=%%a"
)

if "%ADB_DEVICES_OUTPUT%"=="" (
    echo No devices found
    exit /b 1
)

REM Count PowerServe folders
set "LINES=0"
for /f %%a in ('dir /b .\proj ^| findstr /i "PowerServe"') do (
    set /a LINES+=1
)

REM Determine if speculation is enabled
set "SPECULATION_FLAG="
if %LINES%==2 (
    echo Speculation enabled
    set "SPECULATION_FLAG=-s"
)

REM Define target path
set "TARGET_PATH=/data/local/tmp/powerserve"

REM Create target path if it doesn't exist
adb shell mkdir -p %TARGET_PATH%

REM Push project to target path
adb push .\proj %TARGET_PATH%\

REM Run the appropriate command based on speculation flag
if "%SPECULATION_FLAG%"=="-s" (
    adb shell "%TARGET_PATH%/proj/bin/powerserve-run -d %TARGET_PATH%/proj"
) else (
    adb shell "%TARGET_PATH%/proj/bin/powerserve-run -d %TARGET_PATH%/proj --use-spec"
)

echo Deployment finished.
