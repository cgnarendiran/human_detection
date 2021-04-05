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

class HumanDetection:
    def __init__(self, args):
        self.source = args.source
        self.output = args.output_file
        self.model_name = args.model_name
        self.device = args.device
        self.nms_thresh = args.nms_thresh
        self.library = args.library # yolov5 or detectron
        self.select_device()

        # codec format to store the video: mp4v usually works for mp4:
        self.fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
        # read and store coco-labels as dict as all models are trained on coco dataset
        with open(args.label_file) as lf:
            label_list = lf.read().strip("\n").split("\n")
            self.label_dict = {label: i for i, label in enumerate(label_list)}

    def select_device(self):

        if self.device == "cpu":
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False

        elif self.device:  # non-cpu device requested
            os.environ['CUDA_VISIBLE_DEVICES'] = self.device  # set environment variable
            assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {self.device} requested'  # check availability

        cuda = not self.device == "cpu" and torch.cuda.is_available()
        if cuda:
            for d in self.device.split(','):
                p = torch.cuda.get_device_properties(int(d))
                print("Using cuda device", p.name, "\n")
        else: 
            print("Using CPU \n")
        return torch.device('cuda:0' if cuda else 'cpu')

    def process_video(self):
        """
        Function to extract the attributes of the source video file
        Calculates attributes: 
            cap: 'VideoCapture' object
            w,h: size
            fps: frame per sec
            nframes: total frames in the video
        """
        if os.path.isfile(self.source):
            self.cap = cv2.VideoCapture(self.source)
        else:
            try:
                file_name = "input.mp4"
                self.source = self.source.replace('open', 'uc')
                print( "\nDownloading video file from drive link to %s\n"%file_name)
                gdown.download(self.source, file_name,  quiet=False)
                print( "%s downloaded!\n"%file_name )
                self.cap = cv2.VideoCapture(file_name)
            except Exception:
                raise RuntimeError("Invalid source input, please specify a Google drive link or a downloaded local file as input \n")


        assert self.cap.isOpened(), "Failed to open %s" % self.source

        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) 
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return



    def build_predictor(self):
        """
        Build a predictor function with models available in detectron2
        To learn more: https://github.com/facebookresearch/detectron2/blob/e49c7882468229b98135a9ecc57aad6c38fea0a0/MODEL_ZOO.md
        In this code we try out Mask-RCNN and Keypoint-RCNN
        Calculates attributes: 
            predictor: model instance
        """
        if self.library == "yolov5":
            self.predictor = torch.hub.load('ultralytics/yolov5', self.model_name)
            self.predictor.iou = self.nms_thresh  # NMS IoU threshold (0-1)
        if self.library == "detectron2":
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(self.model_name))
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
            cfg.MODEL.NMS_THRESH = self.nms_thresh  # NMS IoU threshold
            if self.device == "cpu": cfg.MODEL.DEVICE = self.device
            # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.model_name)
            self.predictor = DefaultPredictor(cfg)



    def draw_person_boxes(self, frame, boxes, probs):
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




    def detect(self):
        """
        Detect people in the incoming video and export it after drawing bounding boxes
        Returns:
            None
        """
        # process the input video and get the attributes:
        self.process_video()

        # build a rcnn/ yolov5 predictor:
        self.build_predictor()

        
        # assert not os.path.isfile(args.output_file), "File with the name %s already exists"%args.output_file
        # build the writer with same attributes:
        self.vid_writer = cv2.VideoWriter(self.output, self.fourcc, self.fps, (self.w, self.h))

        # inference time:
        start = time.time()
        print("Started inference\n")
        
        # progress bar using tqdm:
        pbar = tqdm(total=self.nframes)

        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret == False:
                break    # when the last frame is read  

            # different formats of results:
            if self.library == "yolov5":
                # predict and bring the outputs to cpu:
                results = self.predictor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # convert to RGB
                predictions = results.xyxy[0].cpu()
                # find the instance indices with person:
                person_idx = predictions[:,5] == self.label_dict["person"]
                # extract the corresponding boxes and scores:
                boxes = predictions[person_idx,:4].numpy()
                probs = predictions[person_idx,4].numpy()

            if self.library == "detectron2":
                # predict and bring the outputs to cpu:
                results = self.predictor(frame) # RGB conversion done automatically in detectron
                predictions = results["instances"].to("cpu")
                # find the instance indices with person:
                person_idx = [predictions.pred_classes == self.label_dict["person"]]
                # extract the corresponding boxes and scores:
                boxes = predictions.pred_boxes[person_idx].tensor.numpy()
                probs = predictions.scores[person_idx].numpy()

            # draw boxes and write the frame to the video:
            if len(boxes): # check whether there are predictions
                box_frame = self.draw_person_boxes(frame, boxes, probs)
            else:
                box_frame = frame
            self.vid_writer.write(box_frame)

            pbar.update(1)
        pbar.close()

        # release the video capture object and write object:
        self.cap.release()
        self.vid_writer.release()

        print("Inferene on the video file took %0.3f seconds"%(time.time()-start))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dectecron2 models:
    parser.add_argument('--library', type=str, default='detectron2', help='name of the library to use for detection: (yolov5 or detectron2')
    parser.add_argument('--model_name', type=str, default='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml', help='name of the mdoel')
    # parser.add_argument('--model_name', type=str, default='"COCO-Detection/retinanet_R_50_FPN_3x.yaml"', help='name of the mdoel')
    # parser.add_argument('--model_name', type=str, default='COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml', help='name of the mdoel')
    # For a list of all models available in detectron2:
    # https://github.com/facebookresearch/detectron2/blob/e49c7882468229b98135a9ecc57aad6c38fea0a0/detectron2/model_zoo/model_zoo.py

    # yolov5 models:
    # parser.add_argument('--library', type=str, default='yolov5', help='name of the library to use for detection: (yolov5 or detectron2')
    # parser.add_argument('--model_name', type=str, default='yolov5s', help='name of the mdoel')
    # For a list of all models available in yolov5:
    # https://github.com/ultralytics/yolov5

    parser.add_argument('--source', type=str, default='https://drive.google.com/open?id=1L0ee-kdtwayN-tlCzXyWVUCqOGwmLj_A', help='source file path or url link')  # input file/folder, 0 for webcam
    parser.add_argument('--output_file', type=str, default='output.mp4', help='output file path')  # output folder
    parser.add_argument('--label_file', type=str, default='coco-labels-paper.txt', help='output folder')  # output folder
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='0', help='cuda device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='Non-maximum supression IoU threshold')
    args = parser.parse_args()
    print("Arguments:", args)

    pipeline = HumanDetection(args)
    pipeline.detect()
