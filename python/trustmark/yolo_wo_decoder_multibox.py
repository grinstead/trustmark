# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import torch
from torch import nn
from .bbox_detector.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.resnet import resnet50

class fasterRCNN(nn.Module):

    def __init__(
        self,
        # transform parameters
        ckpt_path = '__none__',
        num_classes = 3,
        min_size = 100,
        max_size = 2048
    ):
        super().__init__()
        
        # in this we initialize the fasterrcnn 
        norm_layer = nn.BatchNorm2d
        backbone = resnet50(norm_layer=norm_layer)
        trainable_backbone_layers = _validate_trainable_layers(True, None, 5, 5)
        backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)#, returned_layers=[1])
        self.yolo_model = FasterRCNN(backbone, num_classes=num_classes, 
                                     image_mean=[0.485, 0.456, 0.406], 
                                     image_std=[0.229, 0.224, 0.225],
                                     min_size=min_size, max_size=max_size, 
                                     _skip_resize=True, box_detections_per_img=4)

        if ckpt_path != '__none__':
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path):

        sd = torch.load(path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]

        keys = list(sd.keys())
        for k in keys:
            if k.startswith("yolo."):
                sd[k[5:]] = sd.pop(k)
        misses, ignores = self.load_state_dict(sd, strict=False)
        #print(f"Yolo Restored from {path}")
        #print(misses, ignores)
        
    def forward(self, in_images, boxes=None, labels=None):

        #The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
        #image, and should be in 0-1 range. Different images can have different sizes.
        images = [im*0.5+0.5 for im in in_images] # convert image to list and normalize to [0,1]
        targets = None
        if boxes is not None:
            targets = []
            for i in range(len(images)):
                targets.append({})
                targets[i]['boxes'] = boxes[i]
                targets[i]['labels'] = labels[i]
                targets[i]['image_id'] = torch.tensor([i])

        detections, losses = self.yolo_model(images, targets)

        #print('targets', targets)
        # normalize the boxes in detections to [0,1] again
        boxes = []
        labels = []
        scores = []
        for i in range(len(detections)):

            if len(detections[i]['boxes'])==0:
                boxes.append(torch.tensor([[0, 0, images[i].shape[2], images[i].shape[1]]]).to(images[i].device))
                labels.append(torch.tensor([0]).to(images[i].device))
                scores.append(torch.tensor([0]).to(images[i].device))
            else:
                in_boxes_int = (detections[i]['boxes']).type(torch.int32)
                in_boxes_int[:, [1,3]] = in_boxes_int[:, [1,3]].clamp(0, images[i].shape[1])
                in_boxes_int[:, [0,2]] = in_boxes_int[:, [0,2]].clamp(0, images[i].shape[2])

                bad_idx = ((in_boxes_int[:, 3]-in_boxes_int[:, 1])<1) | ((in_boxes_int[:, 2]-in_boxes_int[:,0])<1)
                detections[i]['boxes'][bad_idx, 0] = 0
                detections[i]['boxes'][bad_idx, 1] = 0
                detections[i]['boxes'][bad_idx, 2] = images[i].shape[2]
                detections[i]['boxes'][bad_idx, 3] = images[i].shape[1]
                detections[i]['labels'][bad_idx] = 0

                boxes.append(detections[i]['boxes'])
                labels.append(detections[i]['labels'])
                scores.append(detections[i]['scores'])

         
        # return the secret and the boxes, and the losses
        out = {}
        out['boxes'] = boxes
        out['labels'] = labels
        out['scores'] = scores
        
        return out, losses
