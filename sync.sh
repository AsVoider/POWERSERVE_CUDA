#!/bin/bash

set -umex

rsync -avzP ipads:/ssd/smallthinker_3b/ ~/Downloads/smallthinker_3b/

./smartserving create \
    -m ~/Downloads/smallthinker_3b \
    -o ~/Downloads/smallthinker \
    --exe-path ~/Downloads/smallthinker_3b/bin/aarch64

rsync -avzP ~/Downloads/smallthinker/ 8gen4:~/smallthinker/

rsync -avzP prompt_*.txt 8gen4:~/
