import dataloader
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import argparse
import wandb
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import utils
import os
import pathlib


args = None
parser = argparse.ArgumentParser(description="MaskRCNN Inference for RGBD images")

parser.add_argument("--wb", help="WandB", type=int, default=1)
parser.add_argument(
    "--load_chkpt", help="path to load checkpoint weights", default=None
)

parser.add_argument(
    "--output_viz", help="path to saving validation set output", default="../outs/"
)


parser = parser.parse_args(args)

if parser.wb == 1:
    wandb.init(project="maskrcnn", entity="cranxter")


def viz(i, a, val_idx):

    fig2, ax2 = plt.subplots(figsize=(12, 12), dpi=40)
    ax2.imshow(i[:, :, :3])

    filter_idx = (a["scores"] > 0.5).cpu().detach().numpy()
    a["masks"] = a["masks"][filter_idx]
    a["boxes"] = a["boxes"][filter_idx]

    print(a["masks"].shape)
    all_masks = np.zeros(i.shape[:2])

    for idx in range(len(a["masks"])):
        coords = a["boxes"][idx].cpu().detach().numpy()
        m = a["masks"][idx].cpu().detach().numpy()
        # m[m < 0.5] = 0
        # m[m > 0.5] = 1
        all_masks += m[0]

        # ax2.imshow(m[0], alpha=0.3)
        rect = patches.Rectangle(
            (coords[0], coords[1]),
            abs(coords[0] - coords[2]),
            abs(coords[1] - coords[3]),
            linewidth=8,
            edgecolor="g",
            facecolor="none",
        )

        # ax2.add_patch(rect)
    ax2.imshow(all_masks, alpha=0.5, cmap="jet")

    pathlib.Path(f"{parser.output_viz}").mkdir(parents=True, exist_ok=True)
    fig2.savefig(f"{parser.output_viz}/{val_idx}" + ".png")


num_classes = 2
# load an instance segmentation model pre-trained pre-trained on COCO

model = torchvision.models.detection.maskrcnn_resnet50_fpn(
    pretrained=True,
    image_mean=[0.485, 0.456, 0.406, 0.0138],
    image_std=[0.229, 0.224, 0.225, 0.040],
)


model.backbone.body.conv1 = nn.Conv2d(
    4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
)
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# now get the number of input features for the mask classifier
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
# and replace the mask predictor with a new one
model.roi_heads.mask_predictor = MaskRCNNPredictor(
    in_features_mask, hidden_layer, num_classes
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)


checkpoint = torch.load(parser.load_chkpt)
model.load_state_dict(checkpoint["model_state_dict"])


print("Model Loaded")

color_fp = "./data/test/color/"
depth_fp = "./data/test/depth/"

model.eval()

for f in os.listdir(color_fp):

    cl_f = color_fp + f
    dp_f = depth_fp + f

    im = utils.rgb2array(cl_f)
    im = im.astype(np.int32) / 255

    d = utils.depth2array(dp_f)
    d = (d - 0) / (66 - 0)

    rgbd = np.concatenate((im, d), axis=-1)
    rgbd = torch.as_tensor(rgbd, dtype=torch.float32)

    inp = torch.permute(rgbd, (2, 0, 1)).to(device)
    inp = torch.unsqueeze(inp, dim=0)
    outs = model(inp)
    viz(rgbd, outs[0], f.split(".")[0])
    print("Done ", f)
