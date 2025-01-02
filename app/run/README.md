# CLI

## Quick Start

To get started right away, run the following command, making sure to use the correct path for the model you have:
- [hugging-face](https://huggingface.co/PowerInfer/Llama-3.1-8B-PowerServe-QNN/tree/main)


```bash
./run -m models/llama3.1-8b-instruct --prompt "Once upon a time"
```

## Common Options

In this section, we cover the most commonly used options for running the `run` program with models:

-  `--work-folder [-d] DIRECTORY` The directory containing GGUF or QNN models
- `--n_predicts [-n] N` Set the number of tokens to predict when generating text. Adjusting this value can influence the length of the generated text.
- `--no-qnn` Set this flag to diable QNN backend (if compiled with SMART_WITH_QNN=ON)


## Input Prompts

The `run` program provides several ways to interact with the models using input prompts:

- `--prompt [-p] PROMPT`: Provide a prompt directly as a command-line option.
- `--prompt-file [-f] FNAME`: Provide a file containing a prompt or multiple prompts.

## Additional Options

These options provide extra functionality and customization when running the LLaMA models:

-   `-h, --help`: Display a help message showing all available options and their default values. This is particularly useful for checking the latest options and default values, as they can change frequently, and the information in this document may become outdated.
