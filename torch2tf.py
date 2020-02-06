#!/usr/bin/env python

import argparse
import os
from os import path
import shutil
import subprocess
import sys
import tempfile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        required=True
    )
    parser.add_argument(
        '--output',
        required=True
    )
    parser.add_argument(
        '--model-class',
        help='path to model class. Example --model-class package1.module:ClassName',
        required=True
    )
    parser.add_argument(
        '--input-shape',
        required=True,
        nargs='+',
        help='comma-separated tensor dimensions. Example: --input-shape 1,1,28,28'
    )
    return parser.parse_args()


def find_script(name):
    script_dir = os.path.dirname(os.path.realpath(__file__))

    try:
        exec_path = subprocess.check_output(['which', name])
        return exec_path.decode().strip('\n')
    except subprocess.CalledProcessError:
        if path.exists(path.join(script_dir, name + '.py')):
            return path.join(script_dir, name + '.py')
        else:
            print("Error: script %s not found" % name)
            sys.exit(1)


def main():
    args = parse_args()
    torch2onnx_path = find_script('torch2onnx')
    onnx2tf_path = find_script('onnx2tf')

    intermediate_output = tempfile.mktemp()

    args1 = [
        torch2onnx_path,
        '--input',
        args.input,
        '--output',
        intermediate_output,
        '--model-class',
        args.model_class,
        '--input-shape',
    ]

    for shape in args.input_shape:
        args1.append(shape)

    subprocess.call(args1)

    args2 = [
        onnx2tf_path,
        '--input',
        intermediate_output,
        '--output',
        args.output,
    ]

    try:
        subprocess.call(args2)
    finally:
        os.remove(intermediate_output)


if __name__ == '__main__':
    main()
