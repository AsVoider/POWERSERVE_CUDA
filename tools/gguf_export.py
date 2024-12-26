#!/usr/bin/python3

import argparse
from pathlib import Path
import platform
import shutil
from typing import List, Dict, Literal, Set
import subprocess
import json
from datetime import datetime
import logging

today = datetime.today().strftime("%Y_%m_%d")

logging.basicConfig(
    filename=f"smartserving_{today}.log",
    level=logging.INFO,
    format="[%(asctime)s] - [%(levelname)s] - %(funcName)s - %(message)s",
    datefmt="%Y-%m-%d-%H:%M:%S",
)

root_folder = Path(".").absolute()
current_platform = platform.machine()

support_type = ["f32", "q8_0"]
support_type_l = Literal["f32", "q8_0"]
support_plat = ["x86_64", "aarch64"]
support_plat_l = Literal["x86_64", "aarch64"]


def execute_command(cmd_args):
    cmd = " ".join(map(str, cmd_args))
    print(f"> {cmd}")
    p = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE, encoding="utf-8")
    p.wait()
    if p.returncode != 0:
        error_info = p.stderr.read()
        logging.error(error_info)
        print(error_info)
    assert p.returncode == 0


def read_json(path: Path) -> Dict:
    config = {}
    with open(path, "r") as fp:
        config = json.load(fp)
    return config


def write_json(path: Path, config: Dict):
    with open(path, "w") as fp:
        json.dump(config, fp, indent=4)


def export_executable(out_path: Path, plats: Set[support_plat_l]) -> Path:
    file_dir = Path("bin")
    generate_model_path = out_path / file_dir
    base_targets = []
    # base_targets = ["run", "server", "config_generator", "perplexity_test"]

    if not generate_model_path.exists():
        generate_model_path.mkdir(parents=True, exist_ok=True)

    for plat in plats:
        targets = base_targets.copy()
        build_dir = f"build_{plat}"

        if not (bin_dir := generate_model_path / plat).exists():
            bin_dir.mkdir(parents=True, exist_ok=True)

        print(f">>>>>>>>>> prepare executables files [{plat}] <<<<<<<<<<")
        if plat == "aarch64":
            execute_command([
                "cmake",
                "-DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake",
                "-DANDROID_ABI=arm64-v8a",
                "-DANDROID_PLATFORM=android-34",
                "-DBUILD_SHARED_LIBS=OFF",
                "-DGGML_OPENMP=OFF",
                "-DSMART_WITH_QNN=ON",
                "-S",
                root_folder,
                "-B",
                root_folder / build_dir,
            ])
        elif plat == "x86_64":
            execute_command([
                "cmake",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DSMART_WITH_QNN=OFF",
                "-S",
                root_folder,
                "-B",
                root_folder / build_dir,
            ])
        else:
            assert False, f"{plat} not support"

        compiled = "" if not targets else "--target " + " ".join(targets)
        execute_command(["cmake", "--build", build_dir, "-j12", compiled])
        execute_command(["cp", "-r", root_folder / build_dir / "out/*", generate_model_path / f"{plat}/"])

    return file_dir


def export_gguf(
    model_path: Path,
    model_id: str,
    out_path: Path,
    out_type: support_type_l,
    qnn_path: Path = None,
    only_embed: bool = False,
) -> Path:
    file_dir = Path("")
    generate_model_path = out_path / file_dir

    if not generate_model_path.exists():
        generate_model_path.mkdir(parents=True, exist_ok=True)

    # generate vocab.gguf
    print(">>>>>>>>>> generate vocab file <<<<<<<<<<")
    tool_path = root_folder / "tools/convert_hf_to_gguf/convert_hf_to_gguf.py"
    vocab_file_name = "vocab.gguf"
    execute_command(
        ["python", tool_path, model_path, "--outfile", generate_model_path / vocab_file_name, "--vocab-only"]
    )

    # generate weights file
    print(f">>>>>>>>>> generate weight files: {out_type} <<<<<<<<<<")
    tool_path = root_folder / "tools/convert_hf_to_gguf/convert_hf_to_gguf.py"
    execute_command(
        ["python", tool_path, model_path, "--outfile", generate_model_path / "weights.gguf", "--outtype", out_type]
    )

    print(f">>>>>>>>>> generate config file <<<<<<<<<<")
    # TODO: use python tools replace cpp tools
    tool_path = out_path / f"bin/{current_platform}/smart-config_generator"
    params_file_name = "model.json"
    execute_command([
        tool_path,
        "--file-path",
        generate_model_path / f"weights.gguf",
        "--target-path",
        generate_model_path / params_file_name,
    ])

    model_config = {"model_id": "", "model_arch": "", "version": 0, "llm_config": {}, "vision": {}}
    src: Dict = read_json(generate_model_path / params_file_name)
    model_config["model_id"] = model_id
    model_config["model_arch"] = src["model_arch"]
    model_config["version"] = src["version"]
    src.pop("model_arch")
    src.pop("version")
    model_config["llm_config"] = src
    rope_config = {}
    for k, v in model_config["llm_config"].items():
        if "rope" in k:
            rope_config[k] = v
    model_config["llm_config"] = {k: v for k, v in model_config["llm_config"].items() if k not in rope_config.keys()}
    model_config["llm_config"]["rope_config"] = rope_config
    write_json(generate_model_path / params_file_name, model_config)

    if only_embed:
        print(f">>>>>>>>>> generate embed weight file to replace raw weight file <<<<<<<<<<")
        tool_path = root_folder / "tools/extract_embd_from_vl/main.py"
        execute_command([
            "python",
            tool_path,
            "--model-path",
            model_path,
            "--out-path",
            generate_model_path / "weights.gguf",
            "--out-type",
            "f32",
        ])

    print(f">>>>>>>>>> copy qnn files <<<<<<<<<<")
    qnn_workspace = generate_model_path / "qnn-workspace"
    if not qnn_workspace.exists():
        qnn_workspace.mkdir(parents=True, exist_ok=True)
    if qnn_path:
        execute_command(["cp", "-r", qnn_path / "*", qnn_workspace])

    return file_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="SmartServing", description="SmartServing CommandLine Tool (GGUF Exporter)")
    parser.add_argument("-m", "--model-path", type=Path, required=True, help="Model path")
    parser.add_argument("-o", "--out-path", type=Path, default=Path(f"./model-{today}/"), help="Output path")
    parser.add_argument("-t", "--out-type", type=str, choices=support_type, default="q8_0")
    parser.add_argument("--model-id", type=str, default=f"model-{today}", help="Model ID")
    parser.add_argument("--qnn-path", type=Path, help="Qnn model path", default=None)
    parser.add_argument("--only-embed", action="store_true")

    args = parser.parse_args()

    export_executable(args.out_path, {"aarch64", "x86_64"})
    export_gguf(args.model_path, args.model_id, args.out_path, args.out_type, args.qnn_path, args.only_embed)
