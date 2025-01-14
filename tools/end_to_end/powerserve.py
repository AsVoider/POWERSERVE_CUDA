#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys
import requests
from huggingface_hub import snapshot_download
from subprocess import DEVNULL


# Define model mappings and speculation support
MODEL_MAP = {
    "smallthinker-3b": "PowerServe/SmallThinker-3B-PowerServe-QNN29-{soc_name}",
    "llama-3.1-8b": "PowerServe/Llama-3.1-8B-PowerServe-QNN29-{soc_name}",
    "llama-3.2-1b": "PowerServe/Llama-3.2-1B-PowerServe-QNN29-{soc_name}"
}

# Updated SPECULATION_MAP to directly store model_name to {target_model, draft_model} mapping
SPECULATION_MAP = {
    "smallthinker-3b": {
        "target_model": "PowerServe/SmallThinker-3B-PowerServe-QNN29-{soc_name}",
        "draft_model": "PowerServe/SmallThinker-0.5B-PowerServe-QNN29-{soc_name}"
    },
    "llama-3.1-8b": {
        "target_model": "PowerServe/Llama-3.1-8B-PowerServe-Speculate-QNN29-{soc_name}",
        "draft_model": "PowerServe/Llama-3.2-1B-PowerServe-QNN29-{soc_name}"
    }
}

SUPPORTED_MODELS = list(MODEL_MAP.keys())
MODELS_WITH_SPECULATION = list(SPECULATION_MAP.keys())
PHONE_TMP_FILE = "/data/local/tmp/powerserve_pushed_models.txt"  # TMP_FILE stored on the phone

def check_network_connectivity(url):
    try:
        response = requests.head(url, timeout=10)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException:
        return False

def get_soc():
    # Check for ADB devices
    adb_devices = subprocess.check_output(['adb', 'devices']).decode().splitlines()
    if len(adb_devices) <= 2:
        print("\033[31mNo devices found in ADB\033[0m")
        print("\033[31mCheck whether the Phone is connected with suitable wire.\033[0m")
        print("\033[31mIf you don't know how to apply ADB on your phone. You can go to link https://developer.android.google.cn/tools/adb \033[0m")
        sys.exit(1)
    if len(adb_devices) > 3:
        print("\033[31mToo many devices found in ADB\033[0m")
        print("\033[31mPlease disconnect the unnecessary devices.\033[0m")
        sys.exit(1)
    
    print("\033[32mADB found and working.\033[0m")
    
    # Determine SoC
    platform_name_list = ["pineapple", "sun"]
    platform_name_8G3_list = ["pineapple"]
    platform_name_8G4_list = ["sun"]
    
    platform_name = subprocess.check_output(['adb', 'shell', 'getprop', 'ro.board.platform']).decode().strip()
    if platform_name not in platform_name_list:
        print(f"\033[31mPlatform name {platform_name} is not supported.\033[0m")
        print(f"\033[31mSupported platform names: {platform_name_list} (which means 8G3 or 8G4)\033[0m")
        sys.exit(1)
    
    soc_name = "8G3" if platform_name in platform_name_8G3_list else "8G4"
    print(f"\033[32mSoC          : {soc_name}\033[0m")
    return soc_name

def compile_binary():    
    # Check network connectivity to GitHub
    if not check_network_connectivity("https://github.com"):
        print("\033[31mGitHub is not reachable. Please check your internet connection.\033[0m")
        sys.exit(1)
    
    # Download submodules
    print("\033[36mDownloading the submodule from GitHub.\033[0m")
    subprocess.run(['git', 'submodule', 'update', '--init', '--recursive'], check=True)
    
    # Construct Docker command
    print("\033[36mCompiling the binary using Docker.\033[0m")
    docker_command = [
        'sudo', 'docker', 'run', '--platform', 'linux/amd64', '--rm', '--name', 'powerserve_container',
        '-v', f'{os.getcwd()}:/code', '-w', '/code',
        '-e', f'https_proxy={os.environ.get("https_proxy", "")}',
        '-e', f'http_proxy={os.environ.get("http_proxy", "")}',
        '-e', f'socks_proxy={os.environ.get("socks_proxy", "")}',
        '--network', 'host', 'santoxin/mobile-build:v1.1',
        '/bin/bash', '-c', 
        './tools/end_to_end/compile.sh'
    ]
    
    subprocess.run(docker_command, check=True)
    print("\033[32mSuccessfully compiled the binary.\033[0m")

def get_pushed_models_from_phone():
    """Read the list of pushed models from the phone's TMP_FILE."""
    try:
        result = subprocess.run(['adb', 'shell', f'cat {PHONE_TMP_FILE}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return set(result.stdout.decode().strip().splitlines())
    except subprocess.CalledProcessError:
        pass
    return set()

def update_pushed_models_on_phone(model_name):
    """Update the list of pushed models on the phone's TMP_FILE."""
    pushed_models = get_pushed_models_from_phone()
    pushed_models.add(model_name)
    with open('temp_pushed_models.txt', 'w') as f:
        f.write('\n'.join(pushed_models))
    subprocess.run(['adb', 'push', 'temp_pushed_models.txt', PHONE_TMP_FILE], check=True)
    os.remove('temp_pushed_models.txt')

def run_model(args):
    print("\033[32mStarting to run the model on the phone.\033[0m")
    
    # Check if ./build_android exists
    if not os.path.exists("./build_android"):
        print("\033[31m./build_android directory not found. Did you forget to run `./tools/end_to_end/powerserve.sh compile`?\033[0m")
        sys.exit(1)
    
    soc_name = get_soc()
    
    if args.model_name not in SUPPORTED_MODELS:
        print(f"\033[31mModel {args.model_name} is not supported.\033[0m")
        print(f"\033[31mSupported models: {', '.join(SUPPORTED_MODELS)}\033[0m")
        sys.exit(1)
    
    if args.speculation:
        if args.model_name in SPECULATION_MAP:
            model_info = SPECULATION_MAP[args.model_name]
            target_model = model_info["target_model"].format(soc_name=soc_name)
            draft_model = model_info["draft_model"].format(soc_name=soc_name)
            models = [target_model, draft_model]
            # print("DEBUG: models", models)
        else:
            print(f"\033[31mSpeculation not supported for model {args.model_name}.\033[0m")
            print(f"\033[31mModels supporting speculation: {', '.join(MODELS_WITH_SPECULATION)}.\033[0m")
            sys.exit(1)
    else:
        target_model = MODEL_MAP[args.model_name].format(soc_name=soc_name)
        models = [target_model]
        
    # Check network connectivity to Hugging Face
    if not check_network_connectivity("https://huggingface.co"):
        print("\033[31mHugging Face is not reachable. Please check your internet connection.\033[0m")
        sys.exit(1)
    
    # Set cache directory to project's cache folder
    cache_dir = os.path.join(os.getcwd(), '.cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Download models using huggingface_hub
    for model_repo in models:
        model_dir = os.path.join("models", model_repo.split("/")[-1])
        print(f"\033[36mDownloading model {model_repo} to {model_dir}.\033[0m")
        snapshot_download(repo_id=model_repo, local_dir=model_dir, local_dir_use_symlinks=False, cache_dir=cache_dir)
        
    if args.prompt_file:
        try:
            with open(args.prompt_file, 'r') as f:
                prompt = f.read().strip()
            if not prompt:
                print("Prompt is empty. Please provide a valid prompt.")
                sys.exit(1)
        except FileNotFoundError:
            print(f"\033[31mFile {args.prompt_file} not found.\033[0m")
            sys.exit(1)
        except IOError:
            print(f"\033[31mError reading file {args.prompt_file}.\033[0m")
            sys.exit(1)
    else:
        prompt = args.prompt  # from -p option
    
    # Set the prompt for the arguments
    args.prompt = prompt
        
    # Print configuration
    print("\033[32mModel        : ", args.model_name, "\033[0m")
    print("\033[32mPrompt       : ", args.prompt, "\033[0m")
    if args.cpu_only:
        print("\033[32mCPU only     : ", args.cpu_only, "\033[0m")
    if args.speculation:
        print("\033[32mSpeculation  : ", args.speculation, "\033[0m")
    
    # If model_name include smallthinker-3b, then add <|im_start|>user\n and <|im_end|>\n<|im_start|>assistant\n to the prompt
    if "smallthinker" in args.model_name:
        args.prompt = f"<|im_start|>user\n{args.prompt}<|im_end|>\n<|im_start|>assistant\n"
    elif "llama" in args.model_name:
        args.prompt = f"<|start_header_id|>user<|end_header_id|>\n{args.prompt}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"
    
    docker_dir = "/code"
    if args.speculation:
        create_command = f'python3 {docker_dir}/powerserve create -m "{docker_dir}/models/{models[0].split("/")[-1]}" -d "{docker_dir}/models/{models[1].split("/")[-1]}" --exe-path {docker_dir}/build_android/out'
    else:
        create_command = f'python3 {docker_dir}/powerserve create -m "{docker_dir}/models/{target_model.split("/")[-1]}" --exe-path {docker_dir}/build_android/out'
    
    # 由你不同的cpu型号导致
    print("\033[36mCreating the workspace using Docker.(It may takes 30s ~ 5min to finish, depending on your CPU model)\033[0m")
    all_indocker_command = [
        "bash /lib/qnn/unzip_qnn.sh",
        "source /qnn/bin/envsetup.sh",
        create_command
    ]
    # use && to run multiple commands in a single line
    all_indocker_command = " && ".join(all_indocker_command)
    
    docker_command = [
        'sudo', 'docker', 'run', '--platform', 'linux/amd64', '--rm', '--name', 'powerserve_container',
        '-v', f'{os.getcwd()}:/code', '-w', '/code',
        '-e', f'https_proxy={os.environ.get("https_proxy", "")}',
        '-e', f'http_proxy={os.environ.get("http_proxy", "")}',
        '-e', f'socks_proxy={os.environ.get("socks_proxy", "")}',
        '--network', 'host', 'santoxin/mobile-build:v1.1',
        '/bin/bash', '-c', 
        all_indocker_command
    ]
    subprocess.run(docker_command, check=True, stdout=DEVNULL, stderr=DEVNULL)
    
    print("\033[32mSuccessfully create the workspace.\033[0m")
    
    # Deploy to phone
    deploy_to_phone(args, models)
    
    print("\033[32mSuccessfully finished running the model on the phone.\033[0m")

def deploy_to_phone(args, models):
    # Set target path
    print("\033[36mDeploying the workspace to the phone.\033[0m")
    target_path = "/data/local/tmp"
    
    # Check if model has been pushed before
    model_name = models[0].split("/")[-1]
    pushed_models = get_pushed_models_from_phone()
    
    subprocess.run(['adb', 'shell', 'mkdir', '-p', f"{target_path}/proj"], check=True)
    
    if model_name not in pushed_models:
        # Push the model directory
        print(f"\033[36mPushing model {model_name} to the phone.\033[0m")
        subprocess.run(['adb', 'push', '--sync', f'./models/{model_name}', f"{target_path}/proj/"], check=True)
        update_pushed_models_on_phone(model_name)
    else:
        print(f"\033[36mModel {model_name} already exists on the phone. Skipping model push.\033[0m")
    
    # Always push the bin directory and config files
    subprocess.run(['adb', 'push', '--sync', './proj/bin', f"{target_path}/proj/"], check=True)
    subprocess.run(['adb', 'push', '--sync', './proj/qnn_libs', f"{target_path}/proj/"], check=True, stdout=DEVNULL, stderr=DEVNULL)
    subprocess.run(['adb', 'push', '--sync', './proj/hparams.json', f"{target_path}/proj/"], check=True)
    subprocess.run(['adb', 'push', '--sync', './proj/workspace.json', f"{target_path}/proj/"], check=True)
    subprocess.run(['adb', 'shell', f'chmod +x {target_path}/proj/bin/*'], check=True)
    
    command = f'adb shell "{target_path}/proj/bin/powerserve-run -d {target_path}/proj -n 1500 -p \\\"{args.prompt}\\\"'
    if args.speculation:
        command += ' --use-spec'
    if args.cpu_only:
        command += ' --cpu-only'
    command += '"'
    
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("\033[31mError occurred while running the model on the phone.\033[0m")
        print("\033[31mThe process might have aborted. Please check the phone's logcat for more details.\033[0m")
        print("\033[31mExiting gracefully...\033[0m")
        sys.exit(0)
    

def main():
    # check whether tools as child directory of now directory
    if not os.path.exists("tools"):
        print("\033[31mPlease run this script from the root directory of the PowerServe project.\033[0m")
        sys.exit(1)
    
    # Create help text for models
    model_help = f"Supported models: {', '.join(SUPPORTED_MODELS)}\n"
    model_help += f"Models supporting speculation: {', '.join(MODELS_WITH_SPECULATION)}"
    
    parser = argparse.ArgumentParser(description='PowerServe End-to-End Script', formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest='command')
    
    # Compile subcommand
    compile_parser = subparsers.add_parser('compile', help='Compile the binary using Docker')
    
    # Run subcommand
    default_prompt = "Which one is larger? 9.11 or 9.8"
    run_parser = subparsers.add_parser('run', help=f'Run the model on the phone\n{model_help}')
    run_parser.add_argument('-p', '--prompt', default=default_prompt, help='Prompt text')
    run_parser.add_argument('-f', '--prompt-file', help='File to read prompt from')
    run_parser.add_argument('-s', '--speculation', action='store_true', default=False, help='Enable speculation')
    run_parser.add_argument('-c', '--cpu-only', action='store_true', default=False, help='Use CPU only')
    run_parser.add_argument('model_name', help='Name of the model to run')
    
    args = parser.parse_args()
    
    if args.command == 'compile':
        compile_binary()
    elif args.command == 'run':
        run_model(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()