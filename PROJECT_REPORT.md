# TACO: Trash Annotations in Context — Instance Segmentation

## Computer Vision Lab Project Report

---

## 1. Project Definition

**Title:** Automated Detection and Segmentation of Street Litter using Mask R-CNN on the TACO Dataset

**Objective:** To build an instance segmentation model that can detect, classify, and segment different types of trash/litter found in real-world street images using deep learning and computer vision techniques.

**Problem Statement:** Manual identification and sorting of street litter is labour-intensive and error-prone. This project automates the process by training a Mask R-CNN model on the TACO (Trash Annotations in Context) dataset to recognize 10 categories of waste items (Can, Bottle, Bottle cap, Cup, Lid, Plastic bag + wrapper, Pop tab, Straw, Cigarette, and Other) from images captured in real-world outdoor environments.

**Relevance to Computer Vision:** This project applies core computer vision concepts — image classification, object detection, instance segmentation, feature extraction using Convolutional Neural Networks (CNNs), Region Proposal Networks (RPN), and Feature Pyramid Networks (FPN) — to solve a real-world environmental problem.

---

## 2. Project Abstract

Littering is a growing environmental concern worldwide. Detecting and categorizing trash in outdoor environments manually is impractical at scale. This project presents an automated instance segmentation pipeline using Mask R-CNN with a ResNet-50-FPN backbone, trained on the TACO dataset — a real-world dataset of street litter containing 1,500 images across 60 categories of waste. The 60 original classes are mapped into 10 merged categories for practical usability. The model is trained for 20 epochs on a 4 GB NVIDIA RTX 3050 GPU using PyTorch, achieving a Mean Average Precision (mAP@0.5) of 19.0% and a Mean Recall of 28.2% on the test set. The system can detect and segment individual trash items from images, providing both bounding boxes and pixel-level masks. This demonstrates the application of deep learning-based computer vision to environmental monitoring and waste management.

---

## 3. Tech Stack & Tools Used

| Component | Technology |
|-----------|-----------|
| Language | Python 3.13 |
| Deep Learning Framework | PyTorch 2.9 |
| Computer Vision Library | torchvision (Mask R-CNN, ResNet50-FPN) |
| Image Processing | Pillow, OpenCV, scikit-image |
| Dataset Format | COCO (Common Objects in Context) |
| Annotation Handling | pycocotools |
| GPU | NVIDIA RTX 3050 (4 GB VRAM) |
| IDE | VS Code / Kaggle Notebook |

---

## 4. Dataset — TACO (Trash Annotations in Context)

**Source:** [TACO Dataset](http://tacodataset.org/)

- **Total Images:** 1,500 real-world street photos
- **Original Classes:** 60 categories of litter (bottles, cans, wrappers, cigarette butts, etc.)
- **Merged Classes (map_10.csv):** 60 classes mapped to 10 practical categories:
  1. Can
  2. Bottle
  3. Bottle cap
  4. Cup
  5. Lid
  6. Plastic bag + wrapper
  7. Pop tab
  8. Straw
  9. Cigarette
  10. Other

**Format:** COCO JSON — each annotation contains image metadata, bounding boxes, segmentation masks, and category IDs.

---

## 5. Model Architecture — Mask R-CNN

Mask R-CNN extends Faster R-CNN by adding a parallel mask prediction branch. The architecture consists of:

```
Input Image
    │
    ▼
ResNet-50 Backbone (Feature Extraction)
    │
    ▼
Feature Pyramid Network (FPN) — multi-scale features
    │
    ├──► Region Proposal Network (RPN) — proposes candidate regions
    │
    ├──► ROI Align — extracts fixed-size features per proposal
    │
    ├──► Box Head (FC layers) — classifies objects + refines bounding boxes
    │
    └──► Mask Head (Conv layers) — predicts pixel-level segmentation mask
```

**Key Components:**
- **ResNet-50:** Pretrained CNN backbone that extracts hierarchical features from images
- **FPN:** Creates multi-scale feature maps to detect objects of various sizes
- **RPN:** Proposes regions likely to contain objects
- **ROI Align:** Extracts equal-sized feature maps from each proposed region (avoids quantization errors)
- **Box Predictor:** Classifies each region and refines bounding box coordinates
- **Mask Predictor:** Generates a binary segmentation mask for each detected object

---

## 6. Step-by-Step Implementation with Commands

### Step 1: Environment Setup

```bash
# Clone/download the TACO dataset repository
cd C:\Users\pkg15\OneDrive\Desktop\CV\TACO

# Create a virtual environment (Python 3.13)
py -3.13 -m venv taco_venv

# Activate the virtual environment
.\taco_venv\Scripts\activate

# Install PyTorch with CUDA support (for RTX 3050)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install pycocotools pillow numpy matplotlib
```

### Step 2: Dataset Preparation

```bash
# Navigate to detector directory
cd detector

# Download TACO images from Flickr (only if images are not already present)
cd /kaggle/working/TACO
python download.py

# Split dataset into train/val/test (80/10/10 split, 10 random trials)
cd detector
python split_dataset.py --dataset_dir ../data --nr_trials 10 --test_percentage 10 --val_percentage 10
```

**What `split_dataset.py` does:**
- Reads `annotations.json` (the full TACO dataset)
- Randomly shuffles images
- Splits into 80% train, 10% validation, 10% test
- Creates separate JSON files: `annotations_0_train.json`, `annotations_0_val.json`, `annotations_0_test.json`

### Step 3: Training

```bash
# Train Mask R-CNN with class map (10 merged classes)
python train_torch.py --dataset ../data --class_map ./taco_config/map_10.csv --round 0

# Train with custom settings (optional)
python train_torch.py --dataset ../data --class_map ./taco_config/map_10.csv --round 0 --epochs 40 --image_min 600 --image_max 800

# Resume training from checkpoint
python train_torch.py --dataset ../data --class_map ./taco_config/map_10.csv --round 0 --resume models/taco_best.pth
```

### Step 4: Inference & Evaluation

```bash
# Evaluate on test set (COCO mAP metrics)
python inference_torch.py --model models/taco_best.pth --class_map ./taco_config/map_10.csv --dataset ../data --round 0 --evaluate

# Predict on a single image
python inference_torch.py --model models/taco_best.pth --class_map ./taco_config/map_10.csv --image ../data/batch_1/000001.jpg --visualize

# Predict with custom threshold
python inference_torch.py --model models/taco_best.pth --class_map ./taco_config/map_10.csv --image ../data/batch_3/IMG_4852.JPG --visualize --threshold 0.3
```

---

## 7. Code Walkthrough — Line by Line

### 7.1 Training Script (`train_torch.py`)

#### Imports & Configuration

```python
import os
import json
import csv
import argparse
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
```

- `torch`: Core PyTorch library for tensor operations and GPU computation
- `torchvision`: Provides pretrained models (Mask R-CNN), transforms, and datasets
- `maskrcnn_resnet50_fpn_v2`: The Mask R-CNN model with ResNet-50 backbone and Feature Pyramid Network
- `FastRCNNPredictor`: Replaces the default box predictor to match our number of classes
- `MaskRCNNPredictor`: Replaces the default mask predictor for our custom classes
- `COCO`: From pycocotools, handles COCO-format annotation loading and evaluation
- `PIL.Image`: Image loading and preprocessing

```python
IMAGE_MIN_DIM = 480
IMAGE_MAX_DIM = 640
BATCH_SIZE = 1
NUM_WORKERS = 0
LR = 0.005
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 20
STEPS_PER_EPOCH = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

- **IMAGE_MIN_DIM / IMAGE_MAX_DIM:** Images are resized to fit within 480–640 pixels (reduces VRAM usage on 4 GB GPU)
- **BATCH_SIZE = 1:** One image at a time (critical for low VRAM)
- **LR = 0.005:** Initial learning rate for SGD optimizer
- **NUM_EPOCHS = 20:** Number of complete passes through the training set
- **DEVICE:** Automatically selects GPU (CUDA) if available, otherwise CPU

#### Class Map Parsing

```python
def parse_class_map(class_map_path):
    class_map = {}
    with open(class_map_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                orig, merged = row[0].strip(), row[1].strip()
                class_map[orig] = merged
    merged_names = sorted(set(class_map.values()))
    return class_map, merged_names
```

**Computer Vision Concept — Class Mapping:** The TACO dataset has 60 fine-grained classes. For practical detection, we merge them into 10 broader categories using `map_10.csv`. For example, "Clear plastic bottle", "Glass bottle", "Other plastic bottle" all map to "Bottle". This reduces class imbalance and improves detection of common categories.

```python
def remap_annotations(dataset_dict, class_map, merged_names):
    orig_cats = {c["id"]: c["name"] for c in dataset_dict["categories"]}
    name_to_new = {name: i + 1 for i, name in enumerate(merged_names)}
    id_map = {}
    for orig_id, orig_name in orig_cats.items():
        merged_name = class_map.get(orig_name, orig_name)
        id_map[orig_id] = name_to_new.get(merged_name, 0)
    for ann in dataset_dict["annotations"]:
        ann["category_id"] = id_map.get(ann["category_id"], 0)
    dataset_dict["categories"] = [
        {"id": i + 1, "name": name, "supercategory": ""}
        for i, name in enumerate(merged_names)
    ]
    dataset_dict["annotations"] = [
        a for a in dataset_dict["annotations"] if a["category_id"] > 0
    ]
    return dataset_dict
```

**What it does:** Remaps all annotation category IDs from the original 60-class system to the new 10-class system. Class ID 0 is reserved for background (what the model learns as "not an object").

#### Dataset Class

```python
class TACODataset(Dataset):
    def __init__(self, dataset_dir, ann_file, class_map_path, image_size_min=IMAGE_MIN_DIM, image_size_max=IMAGE_MAX_DIM):
        self.dataset_dir = dataset_dir
        self.image_size_min = image_size_min
        self.image_size_max = image_size_max

        with open(ann_file, "r") as f:
            coco_dict = json.load(f)

        class_map, merged_names = parse_class_map(class_map_path)
        coco_dict = remap_annotations(coco_dict, class_map, merged_names)
        self.num_classes = len(merged_names) + 1  # +1 for background

        self._tmp_ann = ann_file + ".remapped.json"
        with open(self._tmp_ann, "w") as f:
            json.dump(coco_dict, f)

        self.coco = COCO(self._tmp_ann)
        self.image_ids = sorted(self.coco.getImgIds())
        self.image_ids = [
            img_id for img_id in self.image_ids
            if len(self.coco.getAnnIds(imgIds=[img_id])) > 0
        ]
```

- **Inherits from `torch.utils.data.Dataset`:** Makes it compatible with PyTorch DataLoader for batch loading
- **`COCO()`:** Initializes the COCO API for annotation parsing
- **Filters images:** Only keeps images that have at least one annotation (empty images are useless for training)

#### Image Loading & Mask Generation (`__getitem__`)

```python
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
        if mask.shape != (img_h, img_w):
            mask = np.array(Image.fromarray(mask).resize((img_w, img_h), Image.NEAREST))
        masks.append(mask)
```

**Computer Vision Concepts:**
- **Bounding Boxes:** [x, y, x+w, y+w] format — the rectangular region around each object
- **Instance Masks:** Binary pixel-level segmentation — each pixel is 1 (object) or 0 (background)
- **`coco.annToMask()`:** Converts polygon annotations to binary masks using rasterization
- **EXIF Orientation:** Images from phones may be rotated; the dataset loader handles this

#### Resize & Pad

```python
def _resize(self, image, target):
    w, h = image.size
    scale = self.image_size_max / max(w, h)
    scale = min(scale, 1.0)  # Never upscale
    new_w = min(int(w * scale), self.image_size_max)
    new_h = min(int(h * scale), self.image_size_max)

    image = image.resize((new_w, new_h), Image.BILINEAR)
    padded = Image.new("RGB", (self.image_size_max, self.image_size_max), (0, 0, 0))
    padded.paste(image, (0, 0))
```

**Computer Vision Concept — Image Preprocessing:**
- **Aspect-ratio-preserving resize:** Scales the image so the longest side fits within IMAGE_MAX_DIM, maintaining the original aspect ratio
- **Zero-padding:** Pads the shorter dimension with black (0,0,0) pixels to create a square image — required because PyTorch batches require same-sized tensors
- **`Image.BILINEAR`:** Bilinear interpolation for smooth resizing

```python
    target["boxes"] = target["boxes"] * scale
    target["boxes"][:, 0::2].clamp_(0, new_w)
    target["boxes"][:, 1::2].clamp_(0, new_h)
    widths = target["boxes"][:, 2] - target["boxes"][:, 0]
    heights = target["boxes"][:, 3] - target["boxes"][:, 1]
    keep = (widths > 1) & (heights > 1)
    target["boxes"] = target["boxes"][keep]
    target["labels"] = target["labels"][keep]
```

- **Box scaling:** Bounding boxes must be scaled by the same factor as the image
- **Clamping:** Ensures boxes stay within image bounds after scaling
- **Filtering:** Removes degenerate boxes (zero or negative width/height) that result from aggressive clipping

#### Mask Resize

```python
    if has_masks:
        masks = target["masks"].float().unsqueeze(1)
        masks = torch.nn.functional.interpolate(
            masks, size=(new_h, new_w), mode="nearest"
        ).squeeze(1)
        padded_masks = torch.zeros(masks.shape[0], S, S, dtype=torch.bool)
        padded_masks[:, :new_h, :new_w] = masks > 0.5
        target["masks"] = padded_masks
```

**Computer Vision Concept — Mask Interpolation:**
- **`nearest` interpolation:** Nearest-neighbour downsampling preserves binary mask values (0 or 1) without introducing artefacts
- **Threshold at 0.5:** Converts float mask back to binary after interpolation

#### Model Construction

```python
def get_model(num_classes, pretrained=True):
    model = maskrcnn_resnet50_fpn_v2(
        weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None,
        trainable_backbone_layers=3,
    )
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model
```

**Computer Vision Concepts:**
- **Transfer Learning:** Starts with weights pretrained on COCO (80 classes, 330K images) — the backbone already knows how to extract edges, textures, shapes
- **`trainable_backbone_layers=3`:** Freezes only the first 2 layers of ResNet-50; fine-tunes the last 3 layers + FPN + heads
- **`FastRCNNPredictor`:** Replaces the final classification layer to output 11 classes (10 trash + background) instead of COCO's 81
- **`MaskRCNNPredictor`:** Replaces the mask head to predict 11-channel masks instead of 81

#### Training Loop

```python
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
```

**Computer Vision / Deep Learning Concepts:**
- **`model.train()`:** Enables dropout and batch normalisation training mode
- **Forward pass:** `model(images, targets)` computes 5 losses:
  1. `rpn_class_loss` — RPN objectness classification error
  2. `rpn_bbox_loss` — RPN bounding box regression error
  3. `mrcnn_class_loss` — Mask R-CNN classification error
  4. `mrcnn_bbox_loss` — Mask R-CNN box refinement error
  5. `mrcnn_mask_loss` — Mask prediction binary cross-entropy error
- **Backpropagation:** `losses.backward()` computes gradients
- **Gradient Clipping:** `clip_grad_norm_(5.0)` prevents exploding gradients
- **SGD with Momentum:** Standard optimiser for object detection (better generalisation than Adam for this task)

#### Validation

```python
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
    model.train()
    return total_loss / max(steps, 1)
```

- **`@torch.no_grad()`:** Disables gradient computation (saves memory and speeds up)
- **`model.train()`:** Kept in train mode to compute loss (same as training)
- **Checks overfitting:** If train loss decreases but val loss increases, the model is overfitting

#### Main Training Orchestration

```python
model = get_model(num_classes, pretrained=True)
model.to(DEVICE)

optimizer = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=args.lr, momentum=0.9, weight_decay=WEIGHT_DECAY,
)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
```

- **`model.to(DEVICE)`:** Moves model to GPU
- **Weight Decay:** L2 regularisation (1e-4) to prevent overfitting
- **CosineAnnealingLR:** Learning rate schedule — starts high, gradually decreases following a cosine curve, then resets

---

### 7.2 Inference Script (`inference_torch.py`)

#### Model Loading

```python
def load_model(checkpoint_path, num_classes):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    is_v2 = "backbone.fpn.inner_blocks.0.1.weight" in state_dict
    model = get_model(saved_num_classes, use_v2=is_v2)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model
```

- **Architecture detection:** Checks whether the checkpoint was saved with FPN v1 or v2 by looking for BatchNorm layer keys
- **`model.eval()`:** Disables dropout and sets batch norm to eval mode (uses running statistics)

#### Prediction

```python
@torch.no_grad()
def predict(model, image, threshold=0.5, max_dim=IMAGE_MAX_DIM):
    tensor, scale = preprocess(image, max_dim)
    outputs = model([tensor.to(DEVICE)])[0]

    keep = outputs["scores"] > threshold
    boxes = outputs["boxes"][keep].cpu().numpy() / scale
    labels = outputs["labels"][keep].cpu().numpy()
    scores = outputs["scores"][keep].cpu().numpy()
    masks = outputs["masks"][keep].cpu().numpy() > 0.5

    return {"boxes": boxes, "labels": labels, "scores": scores, "masks": masks}
```

**Computer Vision Concept — Non-Maximum Suppression (NMS):**
- The model outputs multiple overlapping predictions per object
- **Score thresholding:** Removes predictions with confidence < 0.5
- **Scale recovery:** Boxes are divided by the resize scale to map back to original image coordinates
- **Mask thresholding:** Float probability masks are binarised at 0.5

#### COCO Evaluation

```python
def evaluate_coco(model, dataset_dir, ann_file, class_map_path, max_dim=IMAGE_MAX_DIM):
    coco_gt = COCO(remapped)  # Ground truth annotations
    coco_dt = coco_gt.loadRes(coco_results)  # Model predictions
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
```

**Computer Vision Metrics:**
- **AP (Average Precision):** Area under the precision-recall curve at IoU = 0.50:0.95
- **AP@0.5:** AP at IoU threshold 0.5 (lenient — box overlaps 50% with ground truth)
- **AP@0.75:** AP at IoU threshold 0.75 (strict — requires 75% overlap)
- **AR (Average Recall):** Maximum recall at different detection counts

---

### 7.3 Class Map (`map_10.csv`)

```csv
Aerosol,Can
Aluminium foil,Other
Battery,Other
Clear plastic bottle,Bottle
Glass bottle,Bottle
Drink can,Can
Food Can,Can
Plastic bottle cap,Bottle cap
Metal bottle cap,Bottle cap
Paper cup,Cup
Disposable plastic cup,Cup
Plastic straw,Straw
Cigarette,Cigarette
```

**60 → 10 mapping:** Reduces rare classes (e.g., "Aluminium blister pack" → "Other") and groups similar items (all bottle types → "Bottle"). This improves model performance on the most common litter categories.

---

### 7.4 Dataset Split (`split_dataset.py`)

```bash
python split_dataset.py --dataset_dir ../data --nr_trials 10 --test_percentage 10 --val_percentage 10
```

**What it does:**
1. Reads `annotations.json`
2. Shuffles images randomly
3. Splits: 80% train, 10% val, 10% test
4. Creates 10 different random splits (for cross-validation)
5. Outputs: `annotations_0_train.json`, `annotations_0_val.json`, `annotations_0_test.json`, etc.

---

## 8. Results

| Metric | Value |
|--------|-------|
| AP@0.5 (mAP) | **19.0%** |
| AP@[0.5:0.95] | **11.9%** |
| AP@0.75 | **12.3%** |
| AR@100 | **28.2%** |
| AR@10 | **28.0%** |
| Training Images | 1,200 |
| Validation Images | 150 |
| Test Images | 150 |
| Classes | 10 + background |
| Epochs | 20 |

**Analysis:**
- The model learns to detect larger objects better (AP_large = 15.6%) than small ones (AP_small = 3.1%)
- Recall improves significantly with more detections (AR@1 = 21.2% → AR@100 = 28.2%)
- Results are reasonable given the small dataset size (1,200 training images) and 4 GB GPU constraint

---

## 9. Computer Vision Concepts Summary

| Concept | Where Used |
|---------|-----------|
| **Image Classification** | ResNet-50 backbone classifies image regions |
| **Object Detection** | RPN proposes bounding boxes, box head classifies them |
| **Instance Segmentation** | Mask head predicts pixel-level masks for each instance |
| **Transfer Learning** | Pretrained COCO weights fine-tuned on TACO |
| **Feature Pyramid Network** | Multi-scale feature extraction for small/large objects |
| **Region Proposal Network** | Generates candidate object regions |
| **ROI Align** | Extracts fixed-size features from variable-size regions |
| **Non-Maximum Suppression** | Removes duplicate detections |
| **Data Augmentation** | Image resizing, padding, normalisation |
| **Data Augmentation** | Class mapping to handle class imbalance |
| **Loss Functions** | Cross-entropy (classification), Smooth L1 (regression), Binary CE (mask) |
| **COCO Evaluation** | Standardised mAP/mAR metrics |

---

## 10. File Structure

```
TACO/
├── data/
│   ├── annotations.json              # Full dataset annotations
│   ├── annotations_0_train.json      # Training split
│   ├── annotations_0_val.json        # Validation split
│   ├── annotations_0_test.json       # Test split
│   ├── batch_1/ ... batch_15/        # Image directories
├── detector/
│   ├── train_torch.py                # Training script (PyTorch)
│   ├── inference_torch.py            # Inference & evaluation script
│   ├── split_dataset.py              # Dataset splitting utility
│   ├── download.py                   # Image download script
│   ├── taco_config/
│   │   └── map_10.csv                # Class mapping (60 → 10 classes)
│   ├── models/
│   │   ├── taco_best.pth             # Best model checkpoint
│   │   └── taco_epoch_*.pth          # Per-epoch checkpoints
│   └── results/
│       └── prediction.jpg            # Visualised predictions
├── requirements.txt
└── README.md
```

---

## 11. Commands Reference (Quick Copy-Paste)

### Full Pipeline

```bash
# 1. Setup environment
cd C:\Users\pkg15\OneDrive\Desktop\CV\TACO
py -3.13 -m venv taco_venv
.\taco_venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install pycocotools pillow numpy matplotlib

# 2. Split dataset
cd detector
python split_dataset.py --dataset_dir ../data --nr_trials 10 --test_percentage 10 --val_percentage 10

# 3. Train model
python train_torch.py --dataset ../data --class_map ./taco_config/map_10.csv --round 0

# 4. Evaluate
python inference_torch.py --model models/taco_best.pth --class_map ./taco_config/map_10.csv --dataset ../data --round 0 --evaluate

# 5. Visualise predictions
python inference_torch.py --model models/taco_best.pth --class_map ./taco_config/map_10.csv --image ../data/batch_1/000001.jpg --visualize
```

---

*Prepared for Computer Vision Lab — [Your Name] — [Date]*
