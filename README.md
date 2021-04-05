# Human Detection in Aerial Video (NRT)

This project is submmitted as part of the evaluation for Senior Machine Learning Engineer Test for NRT.
The goal of the project is to process a video file and detect humans using a CNN/DNN model.

The code is written in python (version 3.7.9) and PyTorch (version 1.7.1) is utilized. Although a lot of open-source projects are available for object detection, [Detectron2](https://github.com/facebookresearch/detectron2) (from Facebook Research) and [YOLOv5](https://github.com/ultralytics/yolov5) are chosen in this project for fast inference and the ready availability of a model-zoo. Three models (Mask-RCNN, RetinatNet, Keypoint-RCNN) are included by name from detectron2 and yolov5s in the `detect.py` code file. This code was run in Ubuntu 20.04 system with Nvidia GeForce RTX2070 card and cuda version 11.0.


## Installation and Inference:

1. Choose and download MiniConda for Python 3.7 from [here](https://docs.conda.io/en/latest/miniconda.html)

`bash Miniconda3-latest-Linux-x86_64.sh`

2. Install the requirements:

`conda env create -n nsrt -f environment.yml`

3. Activate the environment:

`conda activate nsrt`

4. Install detectron2 from source:

`python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'`

5. \[Optional, for tracking\] Clone deep_sort_pytorch repo and download weights for tracking:

```
git clone https://github.com/ZQPei/deep_sort_pytorch.git
cd deep_sort_pytorch/deep_sort/deep/checkpoint
```
download ckpt.t7 from
https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6 to this folder

```
cd ../../../../
```

6. Run inference with options:

```
python3 detect.py \
        --device 0 \
        --library detectron2 \
        --model_name COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
        --source https://drive.google.com/open?id=1L0ee-kdtwayN-tlCzXyWVUCqOGwmLj_A
```
or
```
python3 detect.py \
        --device cpu \
        --library yolov5 \
        --model_name yolov5s \
        --source input.mp4
```
For tracking,
```
python3 track.py \
        --library yolov5 \
        --model_name yolov5s \
        --source input.mp4 \
        --config_deepsort deep_sort_pytorch/configs/deep_sort.yaml
```

This will save the output video with humans detected/tracked in every frame.

## Observations and future developments:

Most of these models for object detection trade accuracy for speed. In this repository some of the major models for object detection are explored. In addition to detection in separate frames, an interesting direction is human tracking/persistence. This could be useful in several scenarios where monitoring each individual separately is required. Future projects can explore the [models](https://blog.netcetera.com/object-detection-and-tracking-in-2020-f10fb6ff9af3) in this domain such as Deep SORT, ROLO and Tracktor++.

## TO-DO:
1. Multi-threading and buffers for parallel processing
2. Batch inference? 
3. Check for any downloadable link and download video