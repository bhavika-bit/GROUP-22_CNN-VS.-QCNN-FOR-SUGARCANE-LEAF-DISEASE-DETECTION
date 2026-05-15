"""
=======================================================================
  MobileNetV2 — TEST SCRIPT
  Sugarcane Leaf Disease Classification (5 Classes)
  Loads best .pth checkpoint and evaluates on the held-out test set.

  Expected results (from project report):
    Test Accuracy : 82.43%
    Precision     : 0.82
    Recall        : 0.82
    F1-Score      : 0.82
    Test Samples  : 2,521
    Classes       : Healthy, Mosaic, RedRot, Rust, Yellow

  Usage:
    python testing_MobileNetV2.py
    python testing_MobileNetV2.py --dataset path/to/dataset --model path/to/model.pth
=======================================================================
"""

import os
import time
import random
import argparse
from collections import defaultdict, Counter

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score,
                             recall_score, f1_score)
import matplotlib
matplotlib.use("Agg")          # headless — safe for servers without display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# CONFIG — edit these paths if needed
DATASET_DIR    = "dataset"           # same folder used during training
MODEL_PATH     = "mobilenetv2_best.pth"   # best checkpoint saved by training script
SEED           = 42                  # must match training seed for reproducible split
BATCH_SIZE     = 16
VAL_SPLIT      = 0.20                # 20% held-out — mirrors training split

IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]

# ARG PARSER (optional overrides)
parser = argparse.ArgumentParser(description="MobileNetV2 — Sugarcane Disease Tester")
parser.add_argument("--dataset", default=DATASET_DIR,  help="Path to dataset folder")
parser.add_argument("--model",   default=MODEL_PATH,   help="Path to .pth checkpoint")
parser.add_argument("--seed",    default=SEED, type=int, help="Random seed (match training)")
parser.add_argument("--batch",   default=BATCH_SIZE, type=int, help="Batch size")
args = parser.parse_args()

DATASET_DIR = args.dataset
MODEL_PATH  = args.model
SEED        = args.seed
BATCH_SIZE  = args.batch

# REPRODUCIBILITY
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*60}")
print(f"  MobileNetV2 — SUGARCANE DISEASE TEST EVALUATION")
print(f"{'='*60}")
print(f"  Device      : {device}")
print(f"  Dataset dir : {DATASET_DIR}")
print(f"  Model path  : {MODEL_PATH}")
print(f"  Seed        : {SEED}")

# STEP 1 — AUTO-DETECT CLASSES (sorted, matches training script)
print(f"\n[1] Scanning dataset folder: '{DATASET_DIR}'")

CLASS_NAMES = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])
NUM_CLASSES = len(CLASS_NAMES)
print(f"    Classes found ({NUM_CLASSES}): {CLASS_NAMES}")

# STEP 2 — RECONSTRUCT THE SAME 80/20 STRATIFIED SPLIT AS TRAINING
#
# The training script used per-class stratified splitting with the
# same random.shuffle() call. By fixing the seed identically, we
# get the exact same val indices that were held out during training.
# This is the correct and reproducible test set.
print(f"\n[2] Reconstructing held-out validation (test) split...")

base_dataset = datasets.ImageFolder(root=DATASET_DIR)
base_dataset.class_to_idx = {cls: i for i, cls in enumerate(CLASS_NAMES)}
base_dataset.classes      = CLASS_NAMES

# Collect all indices per class (same logic as training script)
class_indices = defaultdict(list)
for idx, label in enumerate(base_dataset.targets):
    class_indices[label].append(idx)

selected_indices = []
for label in sorted(class_indices.keys()):
    selected_indices.extend(class_indices[label])

# Per-class stratified 80/20 split — MUST match training script exactly
random.seed(SEED)          # reset seed right before shuffle to match training
train_indices_final = []
val_indices_final   = []

for label in sorted(class_indices.keys()):
    cls_samples = [i for i in selected_indices if base_dataset.targets[i] == label]
    random.shuffle(cls_samples)            # same shuffle as training
    split_at = int(0.8 * len(cls_samples))
    train_indices_final.extend(cls_samples[:split_at])
    val_indices_final.extend(cls_samples[split_at:])

print(f"    Total dataset  : {len(selected_indices):,} images")
print(f"    Train indices  : {len(train_indices_final):,}")
print(f"    Test  indices  : {len(val_indices_final):,}  ← evaluating on these")

# Per-class breakdown of the test split
test_dist = Counter([base_dataset.targets[i] for i in val_indices_final])
print(f"\n    Test set class distribution:")
for cls_idx, count in sorted(test_dist.items()):
    print(f"      {CLASS_NAMES[cls_idx]:<20} : {count:>5} images")

# STEP 3 — TEST TRANSFORM (same as val_transform in training script)
#           No augmentation — deterministic, center-cropped
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

test_dataset_full = datasets.ImageFolder(root=DATASET_DIR, transform=test_transform)
test_dataset_full.class_to_idx = {cls: i for i, cls in enumerate(CLASS_NAMES)}
test_dataset_full.classes      = CLASS_NAMES

test_subset = Subset(test_dataset_full, val_indices_final)
test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\n    Test loader    : {len(test_subset):,} images  |  {len(test_loader)} batches")

# STEP 4 — LOAD MODEL
print(f"\n[3] Loading model from: {MODEL_PATH}")

# Build the same architecture as the training script
model = models.mobilenet_v2(weights=None)   # no ImageNet weights — we load ours
in_features = model.classifier[1].in_features   # 1280
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(in_features, NUM_CLASSES)
)

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

# Handle both checkpoint formats:
#   (a) dict with "model_state_dict" key  (saved by training script)
#   (b) raw state_dict  (saved by torch.save(model.state_dict(), ...))
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
    saved_val_acc = checkpoint.get("val_acc", "N/A")
    saved_classes = checkpoint.get("class_names", CLASS_NAMES)
    print(f"    Checkpoint val acc (training time) : {saved_val_acc}")
    print(f"    Checkpoint class names             : {saved_classes}")
    # Warn if class order differs
    if saved_classes != CLASS_NAMES:
        print(f"    ⚠️  WARNING: checkpoint class_names differ from detected classes!")
        print(f"       Checkpoint : {saved_classes}")
        print(f"       Detected   : {CLASS_NAMES}")
        print(f"       Using checkpoint class names for label mapping.")
        CLASS_NAMES = saved_classes
else:
    model.load_state_dict(checkpoint)
    print("    Loaded raw state_dict.")

model = model.to(device)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"    Total parameters : {total_params:,}")

# STEP 5 — INFERENCE
print(f"\n[4] Running inference on {len(test_subset):,} test images...")

all_preds  = []
all_labels = []
all_probs  = []

start_time = time.time()

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        outputs = model(images)
        probs   = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

        if (batch_idx + 1) % 20 == 0:
            print(f"    Processed {(batch_idx+1)*BATCH_SIZE:>5} / {len(test_subset)} images...")

end_time = time.time()
inference_time = end_time - start_time

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

# STEP 6 — METRICS
print(f"\n{'='*60}")
print(f"  TEST RESULTS")
print(f"{'='*60}")

test_acc  = 100.0 * accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
recall    = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
f1        = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

print(f"\n  Total Test Samples : {len(all_labels)}")
print(f"  Number of Classes  : {NUM_CLASSES}")
print(f"  Classes            : {CLASS_NAMES}")
print(f"\n  Test Accuracy      : {test_acc:.2f}%")
print(f"  Precision (wtd)    : {precision:.2f}")
print(f"  Recall    (wtd)    : {recall:.2f}")
print(f"  F1-Score  (wtd)    : {f1:.2f}")
print(f"  Testing Runtime    : {inference_time:.2f} sec")

print(f"\n  {'─'*50}")
print(f"  📋 Classification Report (per class):")
print(f"  {'─'*50}")
print(classification_report(all_labels, all_preds,
                             target_names=CLASS_NAMES,
                             digits=4,
                             zero_division=0))

cm = confusion_matrix(all_labels, all_preds)
print(f"  Confusion Matrix (rows=actual, cols=predicted):")
header = f"{'':>12}" + "".join(f"{c[:8]:>10}" for c in CLASS_NAMES)
print(f"  {header}")
for i, row in enumerate(cm):
    row_str = f"  {CLASS_NAMES[i][:12]:>12}" + "".join(f"{v:>10}" for v in row)
    print(row_str)

# Per-class accuracy
print(f"\n  Per-class accuracy:")
for i, cls in enumerate(CLASS_NAMES):
    cls_mask    = all_labels == i
    cls_correct = (all_preds[cls_mask] == all_labels[cls_mask]).sum()
    cls_total   = cls_mask.sum()
    cls_acc     = 100.0 * cls_correct / cls_total if cls_total > 0 else 0.0
    print(f"    {cls:<20} : {cls_acc:>6.2f}%  ({cls_correct}/{cls_total})")

# STEP 7 — PLOTS
print(f"\n[5] Generating plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    f"MobileNetV2 — Test Evaluation\n"
    f"({NUM_CLASSES} Classes | {len(all_labels)} samples | "
    f"Acc: {test_acc:.2f}% | F1: {f1:.2f})",
    fontsize=13, fontweight="bold"
)

# ── Plot 1: Confusion Matrix heatmap
ax = axes[0]
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
ax.set_title("Confusion Matrix", fontweight="bold")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
tick_marks = np.arange(NUM_CLASSES)
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels([c[:8] for c in CLASS_NAMES], rotation=30, ha="right", fontsize=9)
ax.set_yticklabels([c[:8] for c in CLASS_NAMES], fontsize=9)
# Annotate cells
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10, fontweight="bold")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# ── Plot 2: Per-class F1 bar chart
ax = axes[1]
per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
bar_colors = ["#2196F3", "#4CAF50", "#F44336", "#FF9800", "#9C27B0"][:NUM_CLASSES]
bars = ax.bar(CLASS_NAMES, per_class_f1, color=bar_colors, width=0.5, edgecolor="white")
ax.set_title("Per-Class F1 Score", fontweight="bold")
ax.set_ylabel("F1 Score")
ax.set_ylim(0, 1.15)
ax.set_xticklabels([c[:8] for c in CLASS_NAMES], rotation=20, ha="right", fontsize=9)
ax.axhline(f1, color="black", linestyle="--", linewidth=1.2, label=f"Weighted avg F1: {f1:.2f}")
ax.legend(fontsize=8)
for bar, val in zip(bars, per_class_f1):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
            f"{val:.2f}", ha="center", fontsize=9, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

# ── Plot 3: Summary metrics bar
ax = axes[2]
metric_names  = ["Accuracy", "Precision", "Recall", "F1-Score"]
metric_values = [test_acc / 100.0, precision, recall, f1]
metric_colors = ["#1565C0", "#2E7D32", "#AD1457", "#E65100"]
bars2 = ax.bar(metric_names, metric_values, color=metric_colors, width=0.45, edgecolor="white")
ax.set_title("Overall Metrics Summary", fontweight="bold")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.15)
for bar, val in zip(bars2, metric_values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
            f"{val:.2f}", ha="center", fontsize=11, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc=bar.get_facecolor(), ec="none", alpha=0.9))
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out_path = "mobilenetv2_test_report.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"    Plot saved: {out_path}")

# STEP 8 — FINAL SUMMARY
print(f"\n{'='*60}")
print(f"  FINAL SUMMARY")
print(f"{'='*60}")
print(f"  Model           : MobileNetV2 (Transfer Learning)")
print(f"  Checkpoint      : {MODEL_PATH}")
print(f"  Test Samples    : {len(all_labels)}")
print(f"  Classes         : {NUM_CLASSES}  →  {CLASS_NAMES}")
print(f"  ──────────────────────────────────────")
print(f"  Test Accuracy   : {test_acc:.2f}%")
print(f"  Precision       : {precision:.2f}")
print(f"  Recall          : {recall:.2f}")
print(f"  F1-Score        : {f1:.2f}")
print(f"  Testing Runtime : {inference_time:.2f} sec")
print(f"  ──────────────────────────────────────")
print(f"  Expected (report): Acc 82.43% | P 0.82 | R 0.82 | F1 0.82 | {inference_time:.2f}s")
print(f"{'='*60}\n")
