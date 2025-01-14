# Get the code

`git clone https://github.com/powerserve-project/PowerServe`

# Enter the directory

`cd PowerServe`

Please make sure you locates at the PowerServe directory.

# Run the Powerserve script.

Note that we have a few prerequisites for the end-to-end script:
- Your computer should have docker installed. [How to install docker](https://docs.docker.com/get-started/get-docker/)
- Your phone should open ADB debugging and connect to the computer. [How to enable ADB debugging](https://developer.android.google.cn/tools/adb) When you typing `adb shell` in your shell, you should see the shell of your phone.
- Your Internet connection with github, docker and huggingface should be good. If you need to use a proxy, you can set the `https_proxy` at your host shell environment. If `https_proxy` is set, the end-to-end script will use the proxy to download the model and docker images automatically.

These prerequisites are necessary for the end-to-end script to run successfully, no matter whhat operating system you are using.

## Linux or MacOS or WSL(Windows Subsystem for Linux)

For example, `./tools/end_to_end/powerserve.sh smallthinker-3b` will run an end-to-end example with the smallthinker-3b model.

You can modify the prompt, e.g. `./tools/end_to_end/powerserve -p "What is the meaning of life?" smallthinker-3b`.

You can also open speculation flag to improve the speed, e.g. `./tools/end_to_end/powerserve -s smallthinker-3b`.

If you only want to use CPU to run the model, you can add the `-c` flag, e.g. `./tools/end_to_end/powerserve -c smallthinker-3b`.

## Windows

If your host is Windows, we strongly recommend you to use WSL(Windows Subsystem for Linux) to run the end-to-end script. You can follow the Linux or MacOS or WSL instructions to run the script.

But we also provide you with a powershell end-to-end script. For example, `.\tools\end_to_end\powerserve.ps1 smallthinker-3b` will run an end-to-end example with the smallthinker-3b model.

Other flags are the same as Linux or MacOS.