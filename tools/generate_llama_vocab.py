import argparse
import sys


parser = argparse.ArgumentParser(description="Generate a vocab GGUF file from a HuggingFace model")
parser.add_argument("model_folder", type=str, help="HuggingFace model folder")
parser.add_argument("-o", "--output", type=str, required=True, help="Output vocab file")
args = parser.parse_args()

sys.argv = [sys.argv[0], args.model_folder, "--vocab-only", "--outfile", args.output]

from convert_hf_to_gguf import convert_hf_to_gguf


convert_hf_to_gguf.main()
