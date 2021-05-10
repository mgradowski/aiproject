#! /usr/bin/env bash

set -e

apt install python3.8
curl 'https://bootstrap.pypa.io/get-pip.py' -o '/tmp/get-pip.py'
python3.8 '/tmp/get-pip.py'
rm '/tmp/get-pip.py'
python3.7 -m pip install -r './requirements.txt'