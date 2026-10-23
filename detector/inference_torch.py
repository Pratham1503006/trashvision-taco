"""
TACO Mask R-CNN Inference & Evaluation (PyTorch)

Usage:
    # Test on a single image
    python inference_torch.py --model models/taco_best.pth --class_map ./taco_config/map_10.csv --image path/to/image.jpg --visualize

    # Evaluate on test set
    python inference_torch.py --model models/taco_best.pth --class_map ./taco_config/map_10.csv --dataset ../data --round 0 --evaluate
"""

import os
import json
import csv
import argparse
import numpy as np
import torch
import torchvision
from torchvision import transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import gc
import warnings
warnings.filterwarnings("ignore")

IMAGE_MAX_DIM = 640
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_class_map(class_map_path):
    class_map = {}
    with open(class_map_path, "r") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                class_map[row[0].strip()] = row[1].strip()
    merged_names = sorted(set(class_map.values()))
    return class_map, merged_names


def get_model(num_classes, pretrained=False, use_v2=None):
    if use_v2:
        try:
            model = maskrcnn_resnet50_fpn_v2(weights=None, trainable_backbone_layers=3)
            print("Using MaskRCNN-ResNet50-FPNv2")
        except Exception as e:
            print(f"v2 failed ({e}), using v1")
            model = maskrcnn_resnet50_fpn(weights=None, trainable_backbone_layers=3)
            print("Using MaskRCNN-ResNet50-FPN")
    else:
        model = maskrcnn_resnet50_fpn(weights=None, trainable_backbone_layers=3)
        print("Using MaskRCNN-ResNet50-FPN")
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


def load_model(checkpoint_path, num_classes):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    saved_num_classes = ckpt.get("num_classes", num_classes)

    # Detect architecture: v2 has "backbone.fpn.inner_blocks.0.1.weight" (BatchNorm)
    is_v2 = "backbone.fpn.inner_blocks.0.1.weight" in state_dict
    model = get_model(saved_num_classes, use_v2=is_v2)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def preprocess(image, max_dim=IMAGE_MAX_DIM):
    w, h = image.size
    scale = min(max_dim / max(w, h), 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.BILINEAR)
    padded = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
    padded.paste(image, (0, 0))
    return T.ToTensor()(padded), scale


@torch.no_grad()
def predict(model, image, threshold=0.5, max_dim=IMAGE_MAX_DIM):
    tensor, scale = preprocess(image, max_dim)
    outputs = model([tensor.to(DEVICE)])[0]

    keep = outputs["scores"] > threshold
    boxes = outputs["boxes"][keep].cpu().numpy() / scale
    labels = outputs["labels"][keep].cpu().numpy()
    scores = outputs["scores"][keep].cpu().numpy()
    masks = outputs["masks"][keep].cpu().numpy() > 0.5

    return {
        "boxes": boxes,
        "labels": labels,
        "scores": scores,
        "masks": masks,
    }


def visualize(image, predictions, class_names, save_path=None):
    vis = image.copy()
    draw = ImageDraw.Draw(vis)

    colors_rgb = [
        (255, 107, 107), (78, 205, 196), (69, 183, 209), (150, 206, 180), (255, 234, 167),
        (221, 160, 221), (152, 216, 200), (247, 220, 111), (187, 143, 206), (133, 193, 233),
    ]

    for i, (box, label, score, mask) in enumerate(
        zip(predictions["boxes"], predictions["labels"], predictions["scores"], predictions["masks"])
    ):
        rgb = colors_rgb[label % len(colors_rgb)]
        x1, y1, x2, y2 = box.astype(int)
        draw.rectangle([x1, y1, x2, y2], outline=rgb, width=3)

        name = class_names.get(label, f"class_{label}")
        text = f"{name} {score:.2f}"
        draw.text((x1, max(y1 - 15, 0)), text, fill=rgb)

        # Overlay mask
        mask_arr = mask[:, :, 0] if mask.ndim == 3 else mask
        mask_img = Image.fromarray(mask_arr.astype(np.uint8) * 255).resize(vis.size)
        overlay = Image.new("RGBA", vis.size, (*rgb, 60))
        mask_layer = Image.new("L", vis.size, 0)
        mask_layer.paste(mask_img, (0, 0))
        vis = Image.composite(
            Image.new("RGB", vis.size, rgb), vis, mask_layer
        )

    if save_path:
        vis.save(save_path)
        print(f"Saved visualization to {save_path}")
    return vis


@torch.no_grad()
def evaluate_coco(model, dataset_dir, ann_file, class_map_path, max_dim=IMAGE_MAX_DIM):
    """Run COCO-style evaluation on the test set."""
    class_map, merged_names = parse_class_map(class_map_path)

    # Load and remap annotations
    with open(ann_file, "r") as f:
        coco_dict = json.load(f)

    orig_cats = {c["id"]: c["name"] for c in coco_dict["categories"]}
    name_to_new = {name: i + 1 for i, name in enumerate(merged_names)}
    id_map = {}
    for orig_id, orig_name in orig_cats.items():
        merged = class_map.get(orig_name, orig_name)
        id_map[orig_id] = name_to_new.get(merged, 0)

    # Build reverse map for prediction -> original category_id
    reverse_map = {}
    for orig_id, new_id in id_map.items():
        if new_id > 0:
            reverse_map[new_id] = orig_id

    # Load remapped annotations for COCO
    remapped = ann_file + ".remapped_eval.json"
    coco_dict["annotations"] = [a for a in coco_dict["annotations"] if id_map.get(a["category_id"], 0) > 0]
    for ann in coco_dict["annotations"]:
        ann["category_id"] = id_map.get(ann["category_id"], 0)
    coco_dict["categories"] = [
        {"id": i + 1, "name": n, "supercategory": ""} for i, n in enumerate(merged_names)
    ]

    with open(remapped, "w") as f:
        json.dump(coco_dict, f)

    coco_gt = COCO(remapped)
    img_ids = sorted(coco_gt.getImgIds())

    num_classes = len(merged_names) + 1
    model = model
    model.eval()

    coco_results = []
    print(f"Evaluating {len(img_ids)} images...")

    for i, img_id in enumerate(img_ids):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(dataset_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        tensor, scale = preprocess(image, max_dim)
        with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
            outputs = model([tensor.to(DEVICE)])[0]

        boxes = outputs["boxes"].cpu().numpy() / scale
        scores = outputs["scores"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            coco_results.append({
                "image_id": img_id,
                "category_id": int(label),
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(score),
            })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(img_ids)}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Run COCO evaluation
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Cleanup
    os.remove(remapped)
    return coco_eval.stats


def main():
    parser = argparse.ArgumentParser(description="TACO Mask R-CNN Inference")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--class_map", required=True, help="Path to class map CSV")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset directory (for evaluation)")
    parser.add_argument("--round", type=int, default=0, help="Dataset split round")
    parser.add_argument("--image", type=str, default=None, help="Single image path for inference")
    parser.add_argument("--visualize", action="store_true", help="Save visualization")
    parser.add_argument("--evaluate", action="store_true", help="Run COCO evaluation on test set")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--save_dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    _, merged_names = parse_class_map(args.class_map)
    num_classes = len(merged_names) + 1
    class_names = {i + 1: name for i, name in enumerate(merged_names)}

    print(f"Device: {DEVICE}")
    print(f"Loading model from {args.model}...")
    model = load_model(args.model, num_classes)
    print(f"Model loaded. Classes: {num_classes}")

    if args.evaluate:
        if not args.dataset:
            print("Error: --dataset required for evaluation")
            return
        os.makedirs(args.save_dir, exist_ok=True)
        test_ann = os.path.join(args.dataset, f"annotations_{args.round}_test.json")
        if not os.path.exists(test_ann):
            test_ann = os.path.join(args.dataset, "annotations.json")
        evaluate_coco(model, args.dataset, test_ann, args.class_map)

    elif args.image:
        image = Image.open(args.image).convert("RGB")
        preds = predict(model, image, threshold=args.threshold)
        print(f"\nDetections: {len(preds['boxes'])}")
        for i, (box, label, score) in enumerate(
            zip(preds["boxes"], preds["labels"], preds["scores"])
        ):
            name = class_names.get(label, f"class_{label}")
            print(f"  {i+1}. {name}: {score:.3f}  bbox={[int(x) for x in box]}")

        if args.visualize:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, "prediction.jpg")
            visualize(image, preds, class_names, save_path=save_path)
    else:
        print("Specify --image for single inference or --evaluate for evaluation")


if __name__ == "__main__":
    main()
