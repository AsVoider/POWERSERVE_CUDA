#!/bin/bash

soc_name_list=("8G3" "8G4")
# adb shell getprop ro.board.platform
# then check the result, pineapple means 8G3, sun means 8G4
platform_name_8G3_list=("pineapple")
platform_name_8G4_list=("sun")
platform_name_list=("${platform_name_8G3_list[@]}" "${platform_name_8G4_list[@]}")


# Check whether there are devices connected to adb
adb start-server
# adb devices
if [ "$(adb devices | wc -l)" -le 2 ]; then
    echo -e "\033[31mNo devices found in ADB\033[0m"
    echo -e "\033[31mCheck whether the Phone is connected with suitable wire.\033[0m"
    echo -e "\033[31mIf you don't know how to apply ADB on your phone. You can go to link https://developer.android.google.cn/tools/adb \033[0m"
    exit 1
fi

# Check if too many devices are connected
if [ "$(adb devices | wc -l)" -gt 3 ]; then
    echo -e "\033[31mToo many devices found in ADB\033[0m"
    echo -e "\033[31mPlease disconnect the unnecessary devices.\033[0m"
    exit 1
fi

# Determine the soc name through adb shell getprop ro.board.platform
platform_name=$(adb shell getprop ro.board.platform | tr -d '\r')
# Check whether the platform name is in the platform name list
if [[ ! " ${platform_name_list[@]} " =~ " ${platform_name} " ]]; then
    echo -e "\033[31mPlatform name $platform_name is not supported.\033[0m"
    echo -e "\033[31mSupported platform names: ${platform_name_list[@]} (which means ${soc_name_list[@]})\033[0m"
    exit 1
fi

if [[ " ${platform_name_8G3_list[@]} " =~ " ${platform_name} " ]]; then
    soc_name="8G3"
elif [[ " ${platform_name_8G4_list[@]} " =~ " ${platform_name} " ]]; then
    soc_name="8G4"
fi

echo -e "\033[32mSoC          : $soc_name\033[0m"

# pass the soc information to the next script
echo $soc_name > tmpfile
