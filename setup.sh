#!/bin/bash

mkdir output

# setup python environments and dependencies
python3 -m venv env
source env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# setup YoloV4
git clone https://github.com/theAIGuysCode/yolov4-deepsort.git
mv ./yolov4-tiny.weights ./yolov4-deepsort/data
mv ./yolov4.weights ./yolov4-deepsort/data
mv checkpoints/ ./yolov4-deepsort
cp ./configs/yolov4-config.py ./yolov4-deepsort/core/config.py

# setup YoloV5
yolov5_dir="yolov5-deepsort"
git clone --recurse-submodules git@github.com:mikel-brostrom/Yolov5_DeepSort_Pytorch.git "${yolov5_dir}"
pushd "${yolov5_dir}"
git checkout 8aec0b5
popd
cp ./configs/yolov5-deepsort-config.yaml ./yolov5-deepsort/deep_sort_pytorch/configs/deep_sort.yaml
pushd yolov5-deepsort
python3 -m pip install -r requirements.txt
popd
