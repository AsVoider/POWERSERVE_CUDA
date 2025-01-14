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
    adb shell "$TARGET_PATH/proj/bin/powerserve-run -d $TARGET_PATH/proj -n 100"
else
    adb shell "$TARGET_PATH/proj/bin/powerserve-run -d $TARGET_PATH/proj -n 100 --use-spec"
fi
