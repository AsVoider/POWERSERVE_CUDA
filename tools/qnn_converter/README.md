# Convert the model in safetensors format to a QNN executable binary format
1. Set up the QNN environment
    - Complete the configuration of the QNN environment following the instructions at https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/linux_setup.html?product=1601111740009302.
    - After completing the configuration, navigate to the current directory and set the environment variables as follows:
        ```sh
        export QNN_SDK=/path/to/aforementioned/QNN/installation/directory 
        source $QNN_SDK/bin/envsetup.sh
        #If successful, it will display:
        #[INFO] AISW SDK environment set
        #[INFO] QNN_SDK_ROOT: /path/to/aforementioned/QNN/installation/directory
        #[INFO] SNPE_ROOT: /path/to/aforementioned/QNN/installation/directory
        ```
2. Run the one-click conversion script to complete the conversion
    ```sh
    python converter.py \
    --model-folder Llama-3.2-1B-Instruct \
    --model-name llama3_2_1b \
    --system-prompt-file system_prompt_llama.txt \ 
    --prompt-file lab_intro_llama.md \
    --batch-size 1 128 \
    --artifact-name llama3_2_1b
    # Note:system-prompt-file and lab_intro_llama.md should be adjusted based on whether it is an Instruct model and the model template
    ```
    After the conversion is completed, copy the resulting output folder to the correct location  for the QNN model on the phone to run