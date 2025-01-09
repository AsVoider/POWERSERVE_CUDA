echo "Deploying to phone"

if ! command -v adb &> /dev/null
then
    echo "adb could not be found"
    exit
fi

if [ "$(adb devices)" == "List of devices attached" ]
then
    echo "No devices found"
    exit
fi

# 如果./proj里面有两个名称含PowerServe的文件夹，那么就是投机执行
# 如果./proj里面有一个名称含PowerServe的文件夹，那么就是正常执行
lines=$(ls ./proj | grep "PowerServe" | wc -l)
if [ $lines -eq 2 ]
then
    echo "Speculation enabled"
    speculation_flag="-s"
fi

TARGET_PATH="/data/local/tmp/powerserve"

adb shell mkdir -p $TARGET_PATH

adb push ./proj $TARGET_PATH/

# TODO: add speculation flag
# If speculation is not enabled, then we powerserve-run, else powerserve-speculate
if [ "$speculation_flag" == "-s" ]
    adb shell "$TARGET_PATH/proj/bin/powerserve-run -d $TARGET_PATH/proj"
else
    adb shell "$TARGET_PATH/proj/bin/powerserve-speculate -d $TARGET_PATH/proj"
fi
