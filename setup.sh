#!/bin/bash


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
cp ./config.py ./yolov4-deepsort/core
mkdir output

# setup YoloV5
git clone --recurse-submodules https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch.git yolov5_deepsort
pushd yolov5_deepsort
python3 -m pip install -r requirements.txt
popd
