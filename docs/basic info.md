# Basic Infor: EBHI-SEG Current Dataset Snapshot

## 1) Model Type To Use (Current Plan)
- Task type: Multiclass semantic segmentation (pixel-wise classification)
- Framework: PyTorch
- Suggested starter model: U-Net++ with EfficientNet-B3 encoder
- Output classes: 7 channels total (background + 6 pathology classes)

Why this setup:
- Strong baseline for medical segmentation
- Good accuracy/effort tradeoff for Kaggle free GPU
- Easy to improve later with better augmentation and tuning

## 2) Classes In This Dataset
The folder structure indicates 6 pathology classes:
1. Adenocarcinoma
2. High-grade IN
3. Low-grade IN
4. Normal
5. Polyp
6. Serrated adenoma

For training labels, use:
- 0 = background
- 1..6 = class IDs from the list above

## 3) Current Dataset Size (Found Locally)
- Total image files: 2254
- Total label files: 2234
- Valid image-label pairs (same filename in image/label): 2226

Note:
- Dataset readme mentions 5170 images, but the current local folder contains 2254 images.

## 4) Class Imbalance (Using Valid Paired Samples)

| Class | Paired Samples | Share of Dataset |
|---|---:|---:|
| Adenocarcinoma | 795 | 35.71% |
| Low-grade IN | 637 | 28.62% |
| Polyp | 474 | 21.29% |
| High-grade IN | 186 | 8.36% |
| Normal | 76 | 3.41% |
| Serrated adenoma | 58 | 2.61% |

Imbalance summary:
- Largest class: Adenocarcinoma (795)
- Smallest class: Serrated adenoma (58)
- Largest/smallest ratio: about 13.71x

## 5) Mask Characteristics
- Mask mode: grayscale (L)
- Mask resolution: 224 x 224
- Pixel value range observed: 0..255 (all values present globally)

Interpretation:
- Masks are not direct class-index masks (not only 0..6).
- Masks behave like intensity masks; use foreground extraction (mask > 0) before assigning class ID from folder label.

Foreground ratio behavior (paired data):
- Adenocarcinoma: mean 0.6884, median 0.7006
- High-grade IN: mean 0.7087, median 0.7190
- Low-grade IN: mean 0.6811, median 0.6926
- Normal: mean 0.6365, median 0.6412
- Polyp: mean 0.5626, median 0.5661
- Serrated adenoma: mean 0.6579, median 0.6624

Extra check:
- No empty masks found
- 37 Adenocarcinoma masks have foreground > 99%

## 6) Abnormalities To Fix Before Training
Filename pairing issues were found:
- Images without matching labels: 28
- Labels without matching images: 8

Per class mismatches:
- Adenocarcinoma: images=814, labels=803, paired=795, missing-label-for-image=19, missing-image-for-label=8
- High-grade IN: images=186, labels=186, paired=186, no mismatch
- Low-grade IN: images=639, labels=637, paired=637, missing-label-for-image=2
- Normal: images=76, labels=76, paired=76, no mismatch
- Polyp: images=481, labels=474, paired=474, missing-label-for-image=7
- Serrated adenoma: images=58, labels=58, paired=58, no mismatch

Observed naming anomaly example:
- In Low-grade IN, one missing filename pattern contains an apparent typo like "GTGT...".

## 7) Minimum Pre-Training Cleanup Checklist
1. Use only valid paired filenames.
2. Drop unpaired files from training.
3. Convert each mask to binary foreground with (mask > 0).
4. Map foreground pixels to class ID from the class folder.
5. Use stratified train/val/test split by class.
6. Track macro Dice and mIoU (especially because classes are imbalanced).

## 8) Suggested First Baseline Target
- Train a single multiclass model with background + 6 classes.
- Keep this as baseline before trying heavier models.
- Compare per-class Dice to ensure minority classes are not ignored.
