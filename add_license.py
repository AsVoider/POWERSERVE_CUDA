import argparse
import os
from pathlib import Path


license = """
// Copyright 2024-2025 PowerServe Authors
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
"""

def insert_text_at_beginning(file_path, text_to_insert):
    with open(file_path, 'r+') as file:
        content = file.read()
        file.seek(0, 0)
        file.write(text_to_insert + '\n' + content)

def find_files(directory: Path):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".cpp") or file.endswith(".hpp") or file.endswith(".c*"):
                target = os.path.join(root, file)
                print(target)
                insert_text_at_beginning(target, license)

def main():
    parser = argparse.ArgumentParser(prog="SmartServing", description="SmartServing License Add Tool")
    parser.add_argument("-d", "--dir", type=Path, required=True)
    parser.add_argument("-e", "--exclude", type=str)

    args = parser.parse_args()

    find_files(args.dir)


if __name__ == "__main__":
    main()
