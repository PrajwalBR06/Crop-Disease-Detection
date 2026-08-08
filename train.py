"""
train.py
--------
Trains a leaf-disease image classifier using TRANSFER LEARNING on a
pretrained ResNet-18.

KEY IDEAS (easy to explain to evaluators):
  * Transfer learning: we start from a ResNet-18 already trained on ImageNet.
    Its early layers already know generic features (edges, textures, colors),
    so we only need to "fine-tune" the last part for our leaf classes.
  * Freezing: we FREEZE every layer except the last residual block (layer4)
    and the new final fully-connected (fc) layer. Fewer trainable weights
    => much faster training on a CPU or small GPU.
  * Device handling: we auto-detect CUDA. If it's present we can optionally
    use mixed precision (--mixed_precision) for a speed/memory win.

WHAT IT PRODUCES:
  * models/best_model.pth  -> checkpoint of the best (highest val-accuracy)
                              model, including the class-name list.
  * models/confusion_matrix.png -> confusion matrix on the test set.
  * A printed classification report (precision / recall / F1) on the test set.

USAGE:
    python train.py --data_dir data --epochs 8 --batch_size 32
    # on a CUDA machine you can also add:  --mixed_precision
    # to benchmark AMP vs no-AMP:           --benchmark_amp   (CUDA only)
"""

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# scikit-learn is used only for the final evaluation report + confusion matrix.
from sklearn.metrics import classification_report, confusion_matrix

# matplotlib is used to save the confusion matrix as a PNG.
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (works on headless machines)
import matplotlib.pyplot as plt


# ImageNet normalization stats (required because ResNet-18 was trained on them).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224  # ResNet-18 expects 224x224 inputs


def build_transforms():
    """Create the image transforms for train vs. eval.

    Training uses augmentation (flip, rotation, color jitter) so the model
    generalizes better. Val/test use only resize + normalize (no randomness).
    """
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),           # left-right flip
        transforms.RandomRotation(20),               # small rotations
        transforms.ColorJitter(brightness=0.2,       # lighting variation
                              contrast=0.2,
                              saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def build_dataloaders(data_dir, batch_size):
    """Create train/val/test DataLoaders from data/<split>/<class>/ folders."""
    train_tf, eval_tf = build_transforms()
    data_dir = Path(data_dir)

    # ImageFolder infers class labels from the subfolder names alphabetically.
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tf)

    # num_workers=0 is the safest default on Windows/laptops. Bump it up on
    # Linux if you want faster data loading.
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    # All three splits share the same class ordering (from train_ds).
    return train_loader, val_loader, test_loader, train_ds.classes


def build_model(num_classes):
    """Load pretrained ResNet-18 and adapt it for our number of classes.

    We freeze everything, then UNfreeze:
      * layer4 (the last residual block)
      * fc     (the new final classifier)
    """
    # `weights=` is the modern torchvision API; it downloads ImageNet weights.
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except AttributeError:
        # Fallback for older torchvision versions that use pretrained=True.
        model = models.resnet18(pretrained=True)

    # 1) Freeze ALL parameters first.
    for param in model.parameters():
        param.requires_grad = False

    # 2) Unfreeze the last residual block (layer4) so it can fine-tune.
    for param in model.layer4.parameters():
        param.requires_grad = True

    # 3) Replace the final FC layer to output `num_classes` scores.
    #    A newly created layer has requires_grad=True by default, so it trains.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def run_one_epoch(model, loader, criterion, optimizer, device,
                  train_mode, scaler=None):
    """Run a single pass over `loader`.

    If train_mode is True, it updates weights (optimizer.step). Otherwise it
    only evaluates. Returns (avg_loss, accuracy).
    `scaler` is a GradScaler when mixed precision is enabled, else None.
    """
    model.train() if train_mode else model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    # torch.set_grad_enabled: gradients on for training, off for eval.
    with torch.set_grad_enabled(train_mode):
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            if train_mode:
                optimizer.zero_grad()

            # ----- Mixed precision path (CUDA only) -----
            if scaler is not None and train_mode:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # ----- Standard full-precision path -----
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if train_mode:
                    loss.backward()
                    optimizer.step()

            # Track running statistics.
            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * inputs.size(0)
            running_correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = running_correct / total
    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, device, epochs, lr,
                use_amp):
    """Full train/val loop. Returns the best model weights + total time."""
    criterion = nn.CrossEntropyLoss()

    # Only optimize parameters that require gradients (layer4 + fc).
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=lr)

    # GradScaler is only meaningful on CUDA with mixed precision enabled.
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    start = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_one_epoch(
            model, train_loader, criterion, optimizer, device,
            train_mode=True, scaler=scaler)
        val_loss, val_acc = run_one_epoch(
            model, val_loader, criterion, optimizer, device,
            train_mode=False, scaler=None)

        print(f"Epoch {epoch:2d}/{epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        # Keep the best model based on validation accuracy.
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

    total_time = time.time() - start
    print(f"\nBest validation accuracy: {best_acc:.3f}")
    print(f"Training time: {total_time:.1f} seconds "
          f"({'mixed precision' if use_amp else 'full precision'})")

    # Load the best weights back into the model before returning.
    model.load_state_dict(best_weights)
    return model, total_time


def evaluate_on_test(model, test_loader, device, class_names, out_dir):
    """Run the test set, print a classification report, save confusion matrix."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    # ----- Classification report (precision / recall / F1 per class) -----
    print("\n================ TEST CLASSIFICATION REPORT ================")
    print(classification_report(
        all_labels, all_preds,
        target_names=class_names, digits=3, zero_division=0))

    # ----- Confusion matrix -----
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Test Set)")

    # Write the count inside each cell for readability.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="black", fontsize=8)

    fig.tight_layout()
    cm_path = Path(out_dir) / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to: {cm_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Train leaf-disease classifier.")
    parser.add_argument("--data_dir", default="data",
                        help="Folder with train/val/test subfolders (default: data).")
    parser.add_argument("--models_dir", default="models",
                        help="Where to save checkpoints (default: models).")
    parser.add_argument("--epochs", type=int, default=8,
                        help="Number of training epochs (default: 8).")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (default: 32). Lower it if you run out "
                            "of memory on CPU/small GPU.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 0.001).")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Enable CUDA mixed precision (autocast + GradScaler). "
                            "Ignored automatically if no CUDA GPU is present.")
    parser.add_argument("--benchmark_amp", action="store_true",
                        help="CUDA only: train once WITHOUT and once WITH mixed "
                            "precision and print a training-time comparison.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # DEVICE AUTO-DETECTION
    # Use CUDA if available, otherwise fall back to CPU. No manual config.
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    print(f"Using device: {device}")

    # NOTE: Mixed precision (torch.cuda.amp) requires a CUDA GPU. On a CPU-only
    # laptop it has no effect, so we disable it and print a friendly note.
    if (args.mixed_precision or args.benchmark_amp) and not cuda_available:
        print("[INFO] Mixed precision requested but no CUDA GPU found. "
              "Falling back to normal CPU training (mixed precision requires CUDA).")

    Path(args.models_dir).mkdir(parents=True, exist_ok=True)

    # Build data + discover class names.
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        args.data_dir, args.batch_size)
    print(f"Classes ({len(class_names)}): {class_names}")

    # ------------------------------------------------------------------
    # OPTIONAL AMP BENCHMARK (CUDA only): train twice and compare times.
    # ------------------------------------------------------------------
    if args.benchmark_amp and cuda_available:
        print("\n--- Benchmark run 1: WITHOUT mixed precision ---")
        model_a = build_model(len(class_names)).to(device)
        _, time_no_amp = train_model(model_a, train_loader, val_loader,
                                     device, args.epochs, args.lr, use_amp=False)

        print("\n--- Benchmark run 2: WITH mixed precision ---")
        model_b = build_model(len(class_names)).to(device)
        best_model, time_amp = train_model(model_b, train_loader, val_loader,
                                            device, args.epochs, args.lr, use_amp=True)

        print("\n================ AMP TIMING COMPARISON ================")
        print(f"Full precision : {time_no_amp:.1f} s")
        print(f"Mixed precision: {time_amp:.1f} s")
        if time_amp > 0:
            print(f"Speedup        : {time_no_amp / time_amp:.2f}x")
    else:
        # ------------------------------------------------------------------
        # SINGLE TRAINING RUN (the normal path).
        # ------------------------------------------------------------------
        use_amp = bool(args.mixed_precision and cuda_available)
        model = build_model(len(class_names)).to(device)
        best_model, _ = train_model(model, train_loader, val_loader,
                                    device, args.epochs, args.lr, use_amp=use_amp)

    # ------------------------------------------------------------------
    # SAVE CHECKPOINT: include class names so the app can map indices -> names.
    # ------------------------------------------------------------------
    ckpt_path = Path(args.models_dir) / "best_model.pth"
    torch.save({
        "model_state": best_model.state_dict(),
        "class_names": class_names,
        "img_size": IMG_SIZE,
        "arch": "resnet18",
    }, ckpt_path)
    print(f"\nSaved best model checkpoint to: {ckpt_path.resolve()}")

    # ------------------------------------------------------------------
    # FINAL EVALUATION on the untouched test set.
    # ------------------------------------------------------------------
    evaluate_on_test(best_model, test_loader, device, class_names, args.models_dir)


if __name__ == "__main__":
    main()
