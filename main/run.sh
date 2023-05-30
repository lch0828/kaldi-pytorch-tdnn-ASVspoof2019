#!/bin/sh

python3 train.py 40 600 tdnn

python3 eval.py 40 600 tdnn
