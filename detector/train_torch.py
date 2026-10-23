"""
TACO Mask R-CNN Training (PyTorch)

Replaces the old TF/Keras detector.py.
Works on Python 3.13, optimized for low-VRAM GPUs (4GB).

Usage:
    python train_torch.py --dataset ../data --class_map ./taco_config/map_10.csv --round 0
    python train_torch.py --dataset ../data --class_map ./taco_config/map_10.csv --round 0 --resume models/taco_maskrcnn.pth
"""

import os
import json
import csv
import argparse
import copy
import numpy as np
import torch
import torchvision
from torchvision import transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pycocotools.coco import COCO
import gc

# ---------------------------------------------------------------------------
# Config — memory-optimized for 4 GB VRAM
# ---------------------------------------------------------------------------
IMAGE_MIN_DIM = 480
IMAGE_MAX_DIM = 640
BATCH_SIZE = 1
NUM_WORKERS = 0          # 0 avoids Windows DataLoader issues
LR = 0.005
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 20
STEPS_PER_EPOCH = 200
VALIDATION_STEPS = 50
PRINT_EVERY = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Parse class map CSV  (e.g. map_10.csv)
# ---------------------------------------------------------------------------
def parse_class_map(class_map_path):
    """Returns {original_class_name: merged_class_name} and sorted unique merged names."""
    class_map = {}
    with open(class_map_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                orig, merged = row[0].strip(), row[1].strip()
                class_map[orig] = merged
    merged_names = sorted(set(class_map.values()))
    return class_map, merged_names


def remap_annotations(dataset_dict, class_map, merged_names):
    """Remap category_ids in a COCO dict based on class_map."""
    # Build mapping: original_cat_id -> new_cat_id (1-based, 0 reserved for bg)
    orig_cats = {c["id"]: c["name"] for c in dataset_dict["categories"]}
    name_to_new = {name: i + 1 for i, name in enumerate(merged_names)}

    id_map = {}
    for orig_id, orig_name in orig_cats.items():
        merged_name = class_map.get(orig_name, orig_name)
        if merged_name in name_to_new:
            id_map[orig_id] = name_to_new[merged_name]
        else:
            id_map[orig_id] = 0  # map to background / ignore

    # Remap annotations
    for ann in dataset_dict["annotations"]:
        old_cat = ann["category_id"]
        ann["category_id"] = id_map.get(old_cat, 0)

    # Update categories
    dataset_dict["categories"] = [
        {"id": i + 1, "name": name, "supercategory": ""}
        for i, name in enumerate(merged_names)
    ]

    # Filter out annotations with category_id 0
    dataset_dict["annotations"] = [
        a for a in dataset_dict["annotations"] if a["category_id"] > 0
    ]
    return dataset_dict


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TACODataset(Dataset):
    def __init__(self, dataset_dir, ann_file, class_map_path, image_size_min=IMAGE_MIN_DIM, image_size_max=IMAGE_MAX_DIM):
        self.dataset_dir = dataset_dir
        self.image_size_min = image_size_min
        self.image_size_max = image_size_max

        # Load and remap annotations
        with open(ann_file, "r") as f:
            coco_dict = json.load(f)

        class_map, merged_names = parse_class_map(class_map_path)
        coco_dict = remap_annotations(coco_dict, class_map, merged_names)

        self.num_classes = len(merged_names) + 1  # +1 for background

        # Write temp remapped file for COCO loader
        self._tmp_ann = ann_file + ".remapped.json"
        with open(self._tmp_ann, "w") as f:
            json.dump(coco_dict, f)

        self.coco = COCO(self._tmp_ann)
        self.image_ids = sorted(self.coco.getImgIds())

        # Pre-filter: only keep images that have at least one annotation
        self.image_ids = [
            img_id for img_id in self.image_ids
            if len(self.coco.getAnnIds(imgIds=[img_id])) > 0
        ]

        print(f"Dataset loaded: {len(self.image_ids)} images, {self.num_classes} classes (incl. background)")

    def __len__(self):
        return len(self.image_ids)

    def _resize(self, image, target):
        """Resize image and scale bounding boxes / masks accordingly."""
        w, h = image.size
        scale = self.image_size_max / max(w, h)
        scale = min(scale, 1.0)
        new_w = min(int(w * scale), self.image_size_max)
        new_h = min(int(h * scale), self.image_size_max)

        image = image.resize((new_w, new_h), Image.BILINEAR)

        # Pad to square
        padded = Image.new("RGB", (self.image_size_max, self.image_size_max), (0, 0, 0))
        padded.paste(image, (0, 0))

        S = self.image_size_max
        has_boxes = "boxes" in target and len(target["boxes"]) > 0
        has_masks = "masks" in target and len(target["masks"]) > 0

        if has_boxes:
            # Scale boxes
            target["boxes"] = target["boxes"] * scale
            target["boxes"][:, 0::2].clamp_(0, new_w)
            target["boxes"][:, 1::2].clamp_(0, new_h)
            # Filter degenerate boxes
            widths = target["boxes"][:, 2] - target["boxes"][:, 0]
            heights = target["boxes"][:, 3] - target["boxes"][:, 1]
            keep = (widths > 1) & (heights > 1)
            target["boxes"] = target["boxes"][keep]
            target["labels"] = target["labels"][keep]
            if has_masks:
                target["masks"] = target["masks"][keep]
            has_masks = has_masks and len(target["masks"]) > 0

        if has_masks:
            # Resize masks from original image size to (new_h, new_w), then pad to (S, S)
            masks = target["masks"].float().unsqueeze(1)
            masks = torch.nn.functional.interpolate(masks, size=(new_h, new_w), mode="nearest").squeeze(1)
            padded_masks = torch.zeros(masks.shape[0], S, S, dtype=torch.bool)
            padded_masks[:, :new_h, :new_w] = masks > 0.5
            target["masks"] = padded_masks

        # Always ensure consistent empty targets
        if not has_boxes or len(target["boxes"]) == 0:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
        if not has_masks:
            target["masks"] = torch.zeros((0, S, S), dtype=torch.bool)

        return padded, target

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.dataset_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        img_h, img_w = img_info["height"], img_info["width"]
        boxes, labels, masks = [], [], []
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]
            if w < 1 or h < 1:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            mask = self.coco.annToMask(ann)
            # Ensure all masks are (img_h, img_w)
            if mask.shape != (img_h, img_w):
                mask = np.array(Image.fromarray(mask).resize((img_w, img_h), Image.NEAREST))
            masks.append(mask)

        target = {}
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.from_numpy(np.stack(masks).astype(np.uint8))
            # Filter degenerate boxes before resize
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            target["boxes"] = boxes[keep]
            target["labels"] = labels[keep]
            target["masks"] = masks[keep]
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["masks"] = torch.zeros((0, img_h, img_w), dtype=torch.uint8)

        target["image_id"] = torch.tensor([img_id])

        image, target = self._resize(image, target)
        image = T.ToTensor()(image)

        return image, target

    def __del__(self):
        if hasattr(self, "_tmp_ann") and os.path.exists(self._tmp_ann):
            os.remove(self._tmp_ann)


def collate_fn(batch):
    return tuple(zip(*batch))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def get_model(num_classes, pretrained=True):
    try:
        model = maskrcnn_resnet50_fpn_v2(
            weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None,
            trainable_backbone_layers=3,
        )
    except (AttributeError, TypeError):
        from torchvision.models.detection import maskrcnn_resnet50_fpn
        model = maskrcnn_resnet50_fpn(
            weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None,
            trainable_backbone_layers=3,
        )
    # Replace box predictor
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)
    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_every=PRINT_EVERY):
    model.train()
    total_loss = 0.0
    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += losses.item()
        if (i + 1) % print_every == 0:
            avg = total_loss / print_every
            print(f"  Epoch [{epoch}] step [{i+1}/{len(data_loader)}] loss: {avg:.4f}")
            total_loss = 0.0
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return total_loss / max(len(data_loader), 1)


@torch.no_grad()
def validate(model, data_loader, device, steps=VALIDATION_STEPS):
    model.train()  # compute loss in train mode
    total_loss = 0.0
    for i, (images, targets) in enumerate(data_loader):
        if i >= steps:
            break
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        total_loss += sum(loss for loss in loss_dict.values()).item()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    model.train()
    return total_loss / max(steps, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TACO Mask R-CNN Training (PyTorch)")
    parser.add_argument("--dataset", required=True, help="Path to TACO dataset dir")
    parser.add_argument("--class_map", required=True, help="Path to class map CSV")
    parser.add_argument("--round", type=int, default=0, help="Dataset split round")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--output", type=str, default="models", help="Output directory")
    parser.add_argument("--image_min", type=int, default=IMAGE_MIN_DIM)
    parser.add_argument("--image_max", type=int, default=IMAGE_MAX_DIM)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Image size: {args.image_min}-{args.image_max}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")

    # Datasets
    train_ann = os.path.join(args.dataset, f"annotations_{args.round}_train.json")
    val_ann = os.path.join(args.dataset, f"annotations_{args.round}_val.json")

    train_ds = TACODataset(args.dataset, train_ann, args.class_map, args.image_min, args.image_max)
    val_ds = TACODataset(args.dataset, val_ann, args.class_map, args.image_min, args.image_max)

    num_classes = train_ds.num_classes
    print(f"Num classes: {num_classes}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )

    # Model
    model = get_model(num_classes, pretrained=True)
    model.to(DEVICE)

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, momentum=0.9, weight_decay=WEIGHT_DECAY,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Output dir
    os.makedirs(args.output, exist_ok=True)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(start_epoch + 1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        train_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)
        val_loss = validate(model, val_loader, DEVICE)
        lr_scheduler.step()

        print(f"  Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(args.output, f"taco_epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "num_classes": num_classes,
        }, ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.output, "taco_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "num_classes": num_classes,
            }, best_path)
            print(f"  Saved best model (val_loss={val_loss:.4f})")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Models saved to: {args.output}/")


if __name__ == "__main__":
    main()
