import os
import pandas as pd
from torchvision.io import read_image
import utils
from torch.utils.data import Dataset
import utils
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
from skimage.measure import label, regionprops, regionprops_table


class GripperDataset(Dataset):
    def __init__(self, annotations_file, mode, transform=None, target_transform=None):

        self.all_img_idx = self.get_idx(annotations_file)
        self.valid_indexes = self.get_valid_indexes(mode)
        self.color_fps, self.depth_fps, self.label_fps = self.get_fps(mode)
        self.transform = transform
        self.target_transform = target_transform

    def get_idx(self, annotations_file):
        with open(annotations_file) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        img_ids = []
        for line in lines:
            img_ids.append(line.split(".")[0].split("/")[-1])

        return img_ids

    def get_fps(self, mode):
        color_fp = "C:/Users/Adnan/Desktop/work/challenge/data/{}/color/{}.png"
        depth_fp = "C:/Users/Adnan/Desktop/work/challenge/data/{}/depth/{}.png"
        label_fp = "C:/Users/Adnan/Desktop/work/challenge/data/{}/label/{}.png"

        all_color_fps = []
        all_depth_fps = []
        all_label_fps = []

        for idx in self.valid_indexes:
            all_color_fps.append(color_fp.format(mode, idx))
            all_depth_fps.append(depth_fp.format(mode, idx))
            all_label_fps.append(label_fp.format(mode, idx))

        return all_color_fps, all_depth_fps, all_label_fps

    def __len__(self):
        return len(self.valid_indexes)

    def get_valid_indexes(self, mode):

        valid_idx = []
        label_fp = "C:/Users/Adnan/Desktop/work/challenge/data/{}/label/{}.png"

        for img_idx in self.all_img_idx:
            f = label_fp.format(mode, img_idx)
            l = utils.label2array(f)
            l = np.array(l, dtype=np.uint8)
            if np.sum(l) > 0:
                valid_idx.append(img_idx)

        return valid_idx

    def __getitem__(self, idx):

        im = utils.rgb2array(self.color_fps[idx])
        im = im.astype(np.int32) / 255

        d = utils.depth2array(self.depth_fps[idx])
        d = (d - 0) / (66 - 0)

        l = utils.label2array(self.label_fps[idx])
        l = np.array(l, dtype=np.uint8)

        rgbd = np.concatenate((im, d), axis=-1)
        rgbd = torch.as_tensor(rgbd, dtype=torch.float32)

        ret, labels = cv2.connectedComponents(l)
        # breakpoint()
        masks = []
        for lab in range(1, ret):
            mask = np.array(labels, dtype=np.uint8)
            mask[labels == lab] = 1
            mask = np.clip(mask, 0, 1)
            masks.append(mask)
        #     plt.imshow('component',mask)
        masks = np.array(masks)
        masks = np.expand_dims(masks, axis=-1)

        label_im = label(l[:, :, 0])
        regions = regionprops(label_im)
        bbox = []
        width, height = im.shape[:2]
        for num, x in enumerate(regions):

            box = regions[num].bbox
            xmin, ymin, xmax, ymax = box[1], box[0], box[3], box[2]
            # assert xmin >= 0
            # assert xmax <= width
            # assert xmin <= xmax

            # assert ymin >= 0
            # assert ymax <= height
            # assert ymin <= ymax
            bbox.append([box[1], box[0], box[3], box[2]])

        boxes = torch.tensor(bbox, dtype=torch.float32)

        # num_objs = masks.shape[0]
        # # boxes = torch.zeros([masks.shape[0], 4], dtype=torch.float32)

        # boxes = []
        # for i in range(num_objs):
        #     breakpoint()
        #     pos = np.where(masks[i])
        #     xmin = np.min(pos[1])
        #     xmax = np.max(pos[1])
        #     ymin = np.min(pos[0])
        #     ymax = np.max(pos[0])
        #     # Check if area is larger than a threshold
        #     # A = abs((xmax - xmin) * (ymax - ymin))
        #     # print(A)
        #     # if A < 5:
        #     #     print("Nr before deletion:", num_objs)
        #     #     obj_ids = np.delete(obj_ids, [i])
        #     #     # print('Area smaller than 5! Box coordinates:', [xmin, ymin, xmax, ymax])
        #     #     print("Nr after deletion:", len(obj_ids))
        #     #     continue
        #     #     # xmax=xmax+5
        #     #     # ymax=ymax+5

        #     boxes.append([xmin, ymin, xmax, ymax])
        #     print(boxes)

        # boxes = torch.tensor(boxes, dtype=torch.float32)

        labels = np.ones((masks.shape[0]))
        labels = torch.tensor(labels, dtype=torch.int64)
        masks = np.squeeze(masks, axis=-1)
        masks = torch.tensor(masks, dtype=torch.uint8)

        annots = {}

        annots["masks"] = masks
        annots["boxes"] = boxes
        annots["labels"] = labels
        # annots["file"] = self.color_fps[idx]

        return rgbd, annots


# train_dataset = GripperDataset("./data/train/train_data.txt", "train")

# train_dataset[0]
