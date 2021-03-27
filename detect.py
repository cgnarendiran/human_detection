import argparse
import time
from tqdm import tqdm
import os
import requests
import cv2
import torch
import gdown
# import matplotlib.pyplot as plt


# import some common detectron2 utilities
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog



def process_video(source):
    """
    Function to extract the attributes of the source video file
    Returns: 
        cap: 'VideoCapture' object
        w,h: size
        fps: frame per sec
        nframes: total frames in the video
    """
    if os.path.isfile(source):
        cap = cv2.VideoCapture(source)
    else:
        try:
            requests.get(source, stream=True)
            file_name = "input.mp4"
            print( "Downloading video file:%s\n"%file_name)
            gdown.download(source, file_name,  quiet=False)
            print( "%s downloaded!\n"%file_name )
            cap = cv2.VideoCapture(file_name)
        except:
            print("Invalid source input")


    assert cap.isOpened(), "Failed to open %s" % source

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) 
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return cap, (w,h) ,fps, nframes



def build_predictor(model_name: str):
    """
    Build a predictor function with models available in detectron2
    To learn more: https://github.com/facebookresearch/detectron2/blob/e49c7882468229b98135a9ecc57aad6c38fea0a0/MODEL_ZOO.md
    In this code we try out Mask-RCNN and Keypoint-RCNN
    Returns:
        predictor instance
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.DEVICE = args.device

    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
    predictor = DefaultPredictor(cfg)
    
    return predictor




def draw_person_boxes(frame, boxes, probs):
    """
    Draw the bounding boxes for all the predicted instances whose confidence scores are more than 0.7
    Returns:
        frame: frame with the bounding boxes predicting people
    """
    # convert color space for numpy
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # for all the boxes:
    for (box, prob) in zip(boxes, probs):
        
        # extract the properties of the box and text:
        (startX, startY, endX, endY) = box.astype("int")
        label = "{}: {:.2f}%".format("Person", prob * 100)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 1
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_w, text_h = text_size
        
        text_color_bg = (0,0,0) # black bg for text
        text_color = (255,255,255) # white text
        box_color = (255,0,0) # red box
        
        # draw the bb prediction on the frame
        cv2.rectangle(frame, (startX, startY), (endX, endY), box_color , 1)
        
        # include text:
        y = startY - text_h if startY - text_h > text_h else startY + text_h
        cv2.rectangle(frame, (startX, y - text_h), (startX + text_w, startY-1), text_color_bg, -1)
        cv2.putText(frame, label, (startX, y), font, font_scale, text_color, thickness)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame




def detect():
    """
    Detect people in the incoming video and export it after drawing bounding boxes
    Returns:
        None
    """

    # set the device used for inference:
    torch.device(args.device)

    # read and store coco-labels as dict as all models are trained on coco dataset
    with open(args.label_file) as lf:
        label_list = lf.read().strip("\n").split("\n")
        label_dict = {label: i for i, label in enumerate(label_list)}

    # process the input video and get the attributes:
    cap, (w, h), fps, nframes = process_video(args.source)

    # build a mask-cnn predictor:
    predictor = build_predictor(args.model_name)

    # codec format to store the video: mp4v usually works for mp4:
    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    # assert not os.path.isfile(args.output_file), "File with the name %s already exists"%args.output_file
    # build the writer with same attributes:
    vid_writer = cv2.VideoWriter(args.output_file, fourcc, fps, (w, h))

    # progress bar using tqdm:
    pbar = tqdm(total=nframes)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break    # when the last frame is read  
        
        # predict and bring the outputs to cpu:
        outputs = predictor(frame)
        predictions = outputs["instances"].to("cpu")

        # find the instance indices with person:
        person_idx = [predictions.pred_classes==label_dict["person"]]

        # extract the corresponding boxes and scores:
        boxes = predictions.pred_boxes[person_idx].tensor.numpy()
        probs = predictions.scores[person_idx].numpy()

        # draw boxes and write the frame to the video:
        box_frame = draw_person_boxes(frame, boxes, probs)
        vid_writer.write(box_frame)

        pbar.update(1)
    pbar.close()

    # release the video capture object and write object:
    cap.release()
    vid_writer.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name', type=str, default='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml', help='name of the mdoel')
    # parser.add_argument('--model_name', type=str, default='"COCO-Detection/retinanet_R_50_FPN_3x.yaml"', help='name of the mdoel')
    parser.add_argument('--model_name', type=str, default='COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml', help='name of the mdoel')
    
    # For a list of all models available in detectron2:
    # https://github.com/facebookresearch/detectron2/blob/e49c7882468229b98135a9ecc57aad6c38fea0a0/detectron2/model_zoo/model_zoo.py

    parser.add_argument('--source', type=str, default='https://drive.google.com/uc?id=1L0ee-kdtwayN-tlCzXyWVUCqOGwmLj_A', help='source file path or url link')  # input file/folder, 0 for webcam
    parser.add_argument('--output_file', type=str, default='output.mp4', help='output file path')  # output folder
    parser.add_argument('--label_file', type=str, default='coco-labels-paper.txt', help='output folder')  # output folder
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='cuda:0', help='cuda:device id (i.e. 0 or 0,1) or cpu')
    args = parser.parse_args()
    print(args)

    detect()
