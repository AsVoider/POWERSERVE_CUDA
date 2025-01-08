echo "Deploying to phone"

# checking adb is installed
if ! command -v adb &> /dev/null
then
    echo "adb could not be found"
    exit
fi

# checking adb devices has output other than "List of devices attached"
if [ "$(adb devices)" == "List of devices attached" ]
then
    echo "No devices found"
    exit
fi

# 定义目标路径变量
TARGET_PATH="/data/local/tmp/powerserve"

# 如果TARGET_PATH不存在，则创建
adb shell mkdir -p $TARGET_PATH

adb push ./proj $TARGET_PATH/

# 如果adb shell "su -c "echo 123""的结果是123说明有root
if [ "$(adb shell "su -c \"echo 123\"")" == "123" ]
then
    echo "Device is rooted"
    adb shell "su -c \"$TARGET_PATH/proj/bin/powerserve-run -d $TARGET_PATH/proj\""
else
    echo "Device is not rooted"
    adb shell "$TARGET_PATH/proj/bin/powerserve-run -d $TARGET_PATH/proj"
    exit
fi
