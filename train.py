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
import pathlib

args = None
parser = argparse.ArgumentParser(description="MaskRCNN for gripping surface estimation")

parser.add_argument("--wb", help="WandB", type=int, default=1)
parser.add_argument(
    "--load_chkpt", help="path to load checkpoint weights", default=None
)
parser.add_argument(
    "--save_chkpt", help="path to save checkpoint weights", default="../weights"
)
parser.add_argument(
    "--output_viz", help="path to saving validation set output", default="../outs/"
)

parser.add_argument(
    "--batch_size", help="path to saving validation set output", default=3, type=int
)

parser.add_argument("--epochs", help="num of epochs", default=100, type=int)


parser = parser.parse_args(args)

if parser.wb == 1:
    wandb.init(project="maskrcnn", entity="cranxter")


def viz(i, a, val_idx, iter):

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
    pathlib.Path(f"{parser.output_viz}/{val_idx}").mkdir(parents=True, exist_ok=True)
    fig2.savefig(f"{parser.output_viz}/{val_idx}/{iter}" + ".png")


train_dataset = dataloader.GripperDataset("./data/train/train_data.txt", "train")
val_dataset = dataloader.GripperDataset("./data/train/val_data.txt", "train")


train_dataloader = DataLoader(
    train_dataset,
    batch_size=parser.batch_size,
    shuffle=True,
    num_workers=0,
    collate_fn=lambda x: list(zip(*x)),
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=parser.batch_size,
    shuffle=True,
    num_workers=0,
    collate_fn=lambda x: list(zip(*x)),
)


num_classes = 2

model = torchvision.models.detection.maskrcnn_resnet50_fpn(
    pretrained=True,
    image_mean=[0.485, 0.456, 0.406, 0.0138],
    image_std=[0.229, 0.224, 0.225, 0.040],
)

model.backbone.body.conv1 = nn.Conv2d(
    4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(
    in_features_mask, hidden_layer, num_classes
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.0005)

num_epochs = parser.epochs
loss_list = []

print(device)
iteration = 1
base_epoch = 0

if parser.load_chkpt:
    checkpoint = torch.load(f"{parser.load_chkpt}")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    base_epoch = checkpoint["epoch"]
    iteration = checkpoint["iteration"]
    # iteration = 1780

    print("MODEL LOADED")

for epoch in range(base_epoch, num_epochs):
    loss_epoch = []

    for images, targets in train_dataloader:

        images = list(torch.permute(image, (2, 0, 1)).to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        model = model.float()
        print("sending to model")
        loss_dict = model(images, targets)
        print("model done")
        print(loss_dict)
        # print [loss for loss in loss_dict.values()]
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        print(
            "Epoch {} | Iteration {}| Loss = {:.4f} ".format(
                epoch, iteration, losses.item()
            )
        )

        loss_epoch.append(losses.item())

        if parser.wb == 1:
            wandb.log({"loss": losses.item()})

        if iteration % 10 == 0:

            val_idx = [2, 10]
            model.eval()

            for idx in val_idx:
                i, a = val_dataset[idx]
                inp = torch.permute(i, (2, 0, 1)).to(device)
                inp = torch.unsqueeze(inp, dim=0)
                outs = model(inp)
                viz(i, outs[0], idx, iteration)
            model.train()

        iteration += 1
    loss_epoch_mean = np.mean(loss_epoch)
    loss_list.append(loss_epoch_mean)
    # loss_list.append(loss_epoch_mean)
    print("Average loss for epoch = {:.4f} ".format(loss_epoch_mean))

    pathlib.Path(parser.save_chkpt).mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration,
        },
        "{}/maskrcnn_{}.pt".format(parser.save_chkpt, epoch),
    )

