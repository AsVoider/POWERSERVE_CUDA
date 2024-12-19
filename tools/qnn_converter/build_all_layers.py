import argparse
import multiprocessing
import os
import shutil
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--build-folder", type=Path, required=True)
parser.add_argument("--batch-sizes", type=int, nargs="+", required=True)
parser.add_argument("--n-model-chunks", type=int, required=True)
parser.add_argument("--artifact-name", type=str, required=True)
parser.add_argument("--graph-names", type=str, nargs="+", required=True)
parser.add_argument("--embedding", action="store_true")
args = parser.parse_args()


def run(cmd_args: list):
    cmd = " ".join(map(str, cmd_args))
    print(f"{cmd}")
    ret = subprocess.Popen(cmd, shell=True).wait()
    assert ret == 0


root_folder = Path(".").absolute()


def main(chunk_id: int):
    batch_sizes = args.batch_sizes  # 128
    n_calibration_tokens = 128 * 4

    pid = multiprocessing.current_process().pid

    build_folder = args.build_folder  # Path('/disk3/cfy/qnn/build1_32')
    output_folder = root_folder / "output"

    # build_folder.mkdir(parents=True, exist_ok=True)
    # output_folder.mkdir(parents=True, exist_ok=True)

    os.chdir(build_folder)

    if args.embedding:
        sdir = "output_embedding"
    else:
        sdir = f"model_chunk_{chunk_id}"
    for batch_size in batch_sizes:
        run([
            "python",
            root_folder / "build_context_binary.py",
            "--model",
            build_folder / sdir / f"batch_{batch_size}" / "onnx_model" / f"batch_{batch_size}.onnx",
            "--encoding",
            build_folder / sdir / f"batch_{batch_size}" / f"batch_{batch_size}.encodings",
            "--io-spec",
            build_folder / sdir / f"batch_{batch_size}" / f"batch_{batch_size}.io.json",
            "--input-list",
            build_folder / sdir / f"batch_{batch_size}" / "input_list.txt",
            "--output-folder",
            build_folder / sdir,
            "--artifact-name",
            f"{args.artifact_name}_{chunk_id}",
            "--graph-names",
            " ".join(args.graph_names),
            # '--generate-bin', 1 if batch_size==batch_sizes[-1] else 0
        ])


multiprocessing.Process()
# pool = multiprocessing.Pool(args.n_model_chunks if args.n_model_chunks<24 else 24)
pool = multiprocessing.Pool(1)
pool.map(main, range(args.n_model_chunks))
