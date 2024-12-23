import os
import sys
from pathlib import Path
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('files', type=Path, nargs='+')
args = parser.parse_args()


def is_empty(text: str) -> bool:
    return text.strip() == ''


def read(path: Path) -> str:
    with open(path, 'r') as f:
        return f.read()


def write(path: Path, text: str):
    with open(path, 'w') as f:
        f.write(text)


def remove_head_empty_lines(text: str) -> str:
    lines = text.split('\n')
    while len(lines) >= 1 and is_empty(lines[0]):
        lines = lines[1:]
    return '\n'.join(lines)


def ensure_tail_empty_line(text: str) -> str:
    lines = text.split('\n')
    while len(lines) >= 2 and is_empty(lines[-1]) and is_empty(lines[-2]):
        lines = lines[:-1]
    if len(lines) == 0 or not is_empty(lines[-1]):
        lines.append('')
    return '\n'.join(lines)


def remove_trailing_spaces(text: str) -> str:
    lines = text.split('\n')
    lines = [line.rstrip() for line in lines]
    return '\n'.join(lines)


def compact_consecutive_empty_lines(text: str) -> str:
    lines = text.split('\n')
    new_lines = []
    for i in range(len(lines)):
        if i + 3 <= len(lines) and is_empty(lines[i]) and is_empty(lines[i + 1]) and is_empty(lines[i + 2]):
            continue
        else:
            new_lines.append(lines[i])
    return '\n'.join(new_lines)


PROTECT_COMMENT = '//<Protected by format.py!!!>'


def protect_pragma(text: str) -> str:
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if line.lstrip().startswith('#pragma'):
            lines[i] = line.replace('#pragma', PROTECT_COMMENT + '#pragma')
    return '\n'.join(lines)


def clang_format(path: str):
    clang_format_path = os.getenv('CLANG_FORMAT', 'clang-format')
    assert os.system(f'{clang_format_path} -i {path}') == 0


def unprotect_pragma(text: str) -> str:
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if line.lstrip().startswith(PROTECT_COMMENT + '#pragma'):
            lines[i] = line.replace(PROTECT_COMMENT + '#pragma', '#pragma')
    return '\n'.join(lines)


def handle_normal_text_files(path: Path):
    text = read(path)
    text = remove_head_empty_lines(text)
    text = remove_head_empty_lines(text)
    text = compact_consecutive_empty_lines(text)
    text = ensure_tail_empty_line(text)
    write(path, text)


def handle_cpp_sources(path: Path):
    text = read(path)
    text = remove_head_empty_lines(text)
    text = remove_head_empty_lines(text)
    text = ensure_tail_empty_line(text)
    text = protect_pragma(text)
    write(path, text)
    clang_format(path)
    text = read(path)
    text = unprotect_pragma(text)
    write(path, text)

for path in args.files:
    path = Path(path)
    if not path.exists():
        print(f'{path} does not exist')
        continue

    if (path.suffix in ['.py', 'md', '.sh', '.yml'] or
        path.name in ['CMakeLists.txt', '.gitignore', '.gitmodules', 'requirements.txt']):
        handle_normal_text_files(path)
    elif path.suffix in ['.c', '.h', '.cpp', '.hpp']:
        handle_cpp_sources(path)

ret = os.system('git -c color.ui=always diff --exit-code --ignore-submodules=dirty')
print(f'git diff returns {ret}')
exit(0 if ret == 0 else 1)
