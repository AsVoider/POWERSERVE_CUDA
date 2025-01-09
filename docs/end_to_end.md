1. `git clone --recurse-submodules https://ipads.se.sjtu.edu.cn:1312/smartserving/smartserving.git`
<!-- TODO: change this link to github -->
If you have cloned the repository without `--recurse-submodules`, you can run `git submodule update --init --recursive` to get the submodules.

2. `cd smartserving`
<!-- TODO: change this to powerserve -->
Please make sure you locates in the powerserve directory.

3. `sudo docker run -v "$(pwd):/code" -w /code -it santoxin/mobile-build:v1.1 /bin/bash`

Now you are interact with the bash in the docker container. Please make sure you have docker installed in your computer.

4. `git config --global --add safe.directory /code && ./tools/end_to_end/end_to_end_run.sh`

You need to choose your model chip and the model you want to run during the process.

5. press `ctrl+d` to quit the docker container. `pwd` should tell you that you are in the powerserve directory.

6. Please ensure that your phone has connected to your computer and adb is all installed. Then:
- for **Linux & MacOS** : run `./tools/end_to_end/deploy_to_phone.sh`
- for **Windows** : run `./tools/end_to_end/deploy_to_phone.bat`
