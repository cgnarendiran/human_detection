import time
import argparse
from tqdm import tqdm
import os
import requests
import cv2

import torch
from detect import HumanDetection # import the class for detection

# deep sort libraries
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

class HumanTracking(HumanDetection):
    def __init__(self, args):
        super().__init__(args)
        self.tracker_model = args.tracker_model_name
        self.config_deepsort = args.config_deepsort

    def build_tracker(self):
        """
        Build the deep sort tracker from default config
        To change config, tweak from the yaml file here: deep_sort_pytorch/configs/deep_sort.yaml
        """
        cfg = get_config()
        cfg.merge_from_file(self.config_deepsort)
        use_cuda = not self.device == "cpu" and torch.cuda.is_available()

        self.tracker = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, 
                        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, 
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, 
                        n_init=cfg.DEEPSORT.N_INIT, 
                        nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=use_cuda)
    

    def bbox_cwh(self, *xyxy):
        """" 
        Convert corner pixel values of the bbox (x1, y1, x2, y2) to the center, width and height (xc, yc, w, h)
        Calculates:
            [xc, yc, w, h]
        """
        bbox_left = min([xyxy[0], xyxy[2]])
        bbox_top = min([xyxy[1], xyxy[3]])
        bbox_w = abs(xyxy[0] - xyxy[2])
        bbox_h = abs(xyxy[1] - xyxy[3])
        x_c = (bbox_left + bbox_w / 2)
        y_c = (bbox_top + bbox_h / 2)
        return [x_c, y_c, bbox_w, bbox_h]


    def compute_color_for_labels(self, label):
        """
        Simple function that adds fixed color depending on the ids
        Calculates 
            color: tuple of size 3
        """
        palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
        return tuple(color)


    def draw_id_boxes(self, frame, boxes, identities):
        """
        Draw boxes containing tracked objects
        Returns:
            frame: frame with the bounding boxes of the tracked people
        """
        for box, identity in zip(boxes, identities):
            x1, y1, x2, y2 = box.astype('int')
            # box text and bar
            id = int(identity) if identity is not None else 0

            label = '{}: {:d}'.format("Person", id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 1
            text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
            text_w, text_h = text_size

            text_color = (255,255,255) # white text
            id_color = self.compute_color_for_labels(id) # consistent colors for each id

            # draw bbox:
            cv2.rectangle(frame, (x1, y1), (x2, y2), id_color, 1)

            # insert text and background:
            y = y1 - text_h if y1 - text_h > text_h else y1 + text_h
            cv2.rectangle(frame, (x1, y - text_h - 1), (x1 + text_w + 1, y1 - 1), id_color, -1)
            cv2.putText(frame, label, (x1, y), font, font_scale, text_color, thickness)

        return frame


    def track(self):
        """
        Tracks people in the incoming video and export it after drawing bounding boxes
        Returns:
            None
        """
        # process the input video and get the attributes:
        self.process_video()

        # build a rcnn/ yolov5 predictor (only yolov5 is used for now)
        self.build_predictor()

        # build a deep sort tracker:
        self.build_tracker()

        # assert not os.path.isfile(args.output_file), "File with the name %s already exists"%args.output_file
        # build the writer with same attributes:
        self.vid_writer = cv2.VideoWriter(self.output, self.fourcc, self.fps, (self.w, self.h))

        # inference time:
        start = time.time()
        print("Started tracking\n")
        
        # progress bar using tqdm:
        pbar = tqdm(total=self.nframes)

        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret == False:
                break    # when the last frame is read  
            
            # predict and bring the outputs to cpu:
            results = self.predictor(frame)
            predictions = results.xyxy[0].cpu()
            # find the instance indices with person:
            person_idx = predictions[:,5] == self.label_dict["person"]
            predictions = predictions[person_idx].numpy() 

            if len(predictions): # check whether there are predictions
                bbox_xywh = []
                confs = []

                # Adapt detections to deep sort input format
                for *xyxy, conf, _ in predictions: # x1, y1, x2, y2, conf, cls
                    # print(xyxy, conf)
                    bbox = self.bbox_cwh(*xyxy) # returns a list [x_c, y_c, w, h]
                    bbox_xywh.append(bbox)
                    confs.append([conf])

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # Pass detections to deepsort
                outputs = self.tracker.update(xywhs, confss, frame)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    box_frame = self.draw_id_boxes(frame, bbox_xyxy, identities)
                    self.vid_writer.write(box_frame)
            else:
                deepsort.increment_ages()

            pbar.update(1)
        pbar.close()

        # release the video capture object and write object:
        self.cap.release()
        self.vid_writer.release()

        print("Inferene on the video file took %0.3f seconds"%(time.time()-start))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dectecron2 models:
    # parser.add_argument('--library', type=str, default='detectron2', help='name of the library to use for detection: (yolov5 or detectron2')
    # parser.add_argument('--model_name', type=str, default='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml', help='name of the mdoel')
    # parser.add_argument('--model_name', type=str, default='"COCO-Detection/retinanet_R_50_FPN_3x.yaml"', help='name of the mdoel')
    # parser.add_argument('--model_name', type=str, default='COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml', help='name of the mdoel')
    # For a list of all models available in detectron2:
    # https://github.com/facebookresearch/detectron2/blob/e49c7882468229b98135a9ecc57aad6c38fea0a0/detectron2/model_zoo/model_zoo.py

    # yolov5 models:
    parser.add_argument('--library', type=str, default='yolov5', help='name of the library to use for detection: (yolov5 or detectron2')
    parser.add_argument('--model_name', type=str, default='yolov5s', help='name of the mdoel')
    # For a list of all models available in yolov5:
    # https://github.com/ultralytics/yolov5

    parser.add_argument('--tracker_model_name', type=str, default='deep_sort', help='name of the tracker model')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")

    parser.add_argument('--source', type=str, default='https://drive.google.com/open?id=1L0ee-kdtwayN-tlCzXyWVUCqOGwmLj_A', help='source file path or url link')  # input file/folder, 0 for webcam
    parser.add_argument('--output_file', type=str, default='tracker_output.mp4', help='output file path')  # output folder
    parser.add_argument('--label_file', type=str, default='coco-labels-paper.txt', help='output folder')  # output folder
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='0', help='cuda device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='Non-maximum supression IoU threshold')
    args = parser.parse_args()
    print("Arguments:", args)

    pipeline = HumanTracking(args)
    pipeline.track()
