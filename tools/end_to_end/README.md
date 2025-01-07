1. `git clone --recurse-submodules https://ipads.se.sjtu.edu.cn:1312/smartserving/smartserving.git`

2. `cd smartserving`

Please make sure you locates in the smartserving directory.

3. `sudo docker run -v "$(pwd):/code" -w /code -it santoxin/mobile-build:v1.1 /bin/bash`

Now you are interact with the bash in the docker container.

4. `git config --global --add safe.directory /code && ./tools/end_to_end/end_to_end_run.sh`

5. press `ctrl+d` to quit the docker container. `pwd` should tell you that you are in the smartserving directory.

6. Please ensure that your phone has connected to your computer and adb is all installed. `./tools/end_to_end/deploy_to_phone.sh`
