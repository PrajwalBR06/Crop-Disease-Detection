# Crop Disease Diagnosis Assistant

An image-classification system that takes a **leaf photo** and outputs:

1. **Predicted disease name** + confidence score
2. A **severity estimate** (Early / Moderate / Advanced) based on confidence
3. A **one-line suggested remedy** for that disease

Built as a clean, laptop-friendly capstone project using **transfer learning**
on a pretrained **ResNet-18**. Runs on CPU or a small local GPU — no cloud
GPU required.

---

## How it works (in one paragraph)

We fine-tune a ResNet-18 that was already pretrained on ImageNet. All layers
are frozen except the last residual block (`layer4`) and a new final
classification layer sized to our number of leaf classes. Freezing most of the
network keeps the number of trainable weights small, so training is fast even
on a CPU. At inference time, the app converts the model's top-class confidence
into a severity label and looks up a remedy string from a small dictionary.

---

## Project structure

```
crop-disease-assistant/
├── data/                 # train/val/test folders (created by data_prep.py)
│   ├── train/<class>/...
│   ├── val/<class>/...
│   └── test/<class>/...
├── models/               # saved checkpoints + confusion_matrix.png
│   └── best_model.pth
├── data_prep.py          # splits raw dataset into train/val/test (80/10/10)
├── train.py              # transfer-learning training + evaluation
├── diagnosis_map.py      # remedy dictionary + severity logic
├── app.py                # Gradio demo web app
├── requirements.txt
└── README.md
```

**Classes used (6):** Tomato Early Blight, Tomato Late Blight, Tomato Leaf
Mold, Tomato Healthy, Potato Early Blight, Potato Healthy.

---

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it
#   Windows (PowerShell):
venv\Scripts\Activate.ps1
#   Windows (cmd):
venv\Scripts\activate.bat
#   macOS / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> The pinned `torch`/`torchvision` versions run on CPU everywhere and will use
> your GPU automatically if CUDA is available. For a specific CUDA build, use
> the selector at https://pytorch.org/get-started/locally/.

---

## Step 1 — Prepare the data

1. Download the **PlantVillage** dataset from Kaggle
   (e.g. https://www.kaggle.com/datasets/emmarex/plantdisease) or any mirror.
2. Keep only the 6 class folders listed above (delete the rest to save space).
   The folder should look like:
   ```
   raw_dataset/
     Tomato___Early_blight/
     Tomato___Late_blight/
     Tomato___Leaf_Mold/
     Tomato___healthy/
     Potato___Early_blight/
     Potato___healthy/
   ```
3. Run the split script (80% train / 10% val / 10% test):

```bash
python data_prep.py --raw_dir path/to/raw_dataset --out_dir data --max_per_class 500
```

This copies images into `data/train`, `data/val`, and `data/test` and prints a
summary table of the split.

---

## Step 2 — Train the model

```bash
python train.py --data_dir data --epochs 8 --batch_size 32
```

This will:
- Train ResNet-18 with data augmentation (flip, rotation, color jitter)
- Track train/val loss + accuracy each epoch
- Save the best checkpoint to `models/best_model.pth`
- Print a classification report (precision / recall / F1) on the test set
- Save the confusion matrix to `models/confusion_matrix.png`

**Optional (CUDA GPU only):**
```bash
# Train with mixed precision for speed/memory savings:
python train.py --mixed_precision

# Benchmark training time with vs. without mixed precision:
python train.py --benchmark_amp
```
> Mixed precision requires a CUDA GPU. On a CPU-only laptop these flags are
> ignored automatically and training proceeds normally.

---

## Step 3 — Run the demo app

```bash
python app.py
```

It loads `models/best_model.pth`, opens in your browser (usually
`http://127.0.0.1:7860`), and lets you upload a leaf photo. You'll see the
predicted disease with confidence bars plus a formatted diagnosis card showing
confidence %, severity, and the suggested remedy.

---

## Results

_Fill this in after training._

- **Test accuracy:** 98.50%
- **Confusion matrix:**

  ![Confusion Matrix]![Error Screenshot](https://github.com)



---

## Notes

- Severity thresholds: `>90%` → Advanced/Confirmed, `70-90%` → Moderate,
  `<70%` → Early/Uncertain (recheck with a clearer photo). See `diagnosis_map.py`.
- The remedy strings are general guidance, not a substitute for advice from a
  qualified agronomist.
