# Human Detection in Aerial Video (NRT)

This project is submmitted as part of the evaluation for Senior Machine Learning Engineer Test for NRT.
The goal of the project is to process a video file, detect humans using a CNN/DNN model.

The code is written in python and PyTorch 1.7 is utilized. Although a lot of open-source projects are available for object detection,
[Detectron2](https://github.com/facebookresearch/detectron2) (from Facebook Research) is chosen in this project for fast inference and the availability of model-zoo. Three models (Mask-RCNN, RetinatNet, Keypoint-RCNN) are included by name in the `detect.py` code file. 


## Installation and Inference:

1. Choose and download MiniConda for Python 3.7 from [here](https://docs.conda.io/en/latest/miniconda.html)

`bash Miniconda3-latest-Linux-x86_64.sh`

2. Install the requirements:

`conda env create -n nsrt -f environment.yml`

3. Activate the environment:

`conda activate nsrt`

4. Install detectron2 and gdown:

`python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'`
`pip install gdown`

5. Run inference with options:

```
python3 detect.py \
        --device cuda:0 \
        --model_name COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
        --source_file TopDown_AerialVideo_1080.mp4
```

This will output the processed video with humans detected in every frame.

## Observations and future developments:

Most of these models for object detection trade accuracy for speed. In this repository only major models except YOLO versions can be explored. In addition to detection in separate frames, what could be interesting is human tracking/persistence. This could be useful in several cases when you have to monitor each individual separately. Future projects can explore the [models](https://blog.netcetera.com/object-detection-and-tracking-in-2020-f10fb6ff9af3) in this domain such as Deep SORT, ROLO and Tracktor++