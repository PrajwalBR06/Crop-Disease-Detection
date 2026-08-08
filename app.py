"""
app.py
------
A simple Gradio web app for the Crop Disease Diagnosis Assistant.

WHAT IT DOES:
  * Loads the trained checkpoint (models/best_model.pth).
  * Lets the user upload a leaf photo.
  * Runs the model and shows:
      - predicted disease (with per-class confidence via a Label component)
      - a nicely formatted Markdown block with confidence %, severity,
        and the suggested remedy.

RUN IT:
    python app.py
  Then open the local URL it prints (usually http://127.0.0.1:7860).

NOTE:
  This app auto-detects CUDA vs CPU, exactly like train.py. Inference on a
  single image is fast even on CPU.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import models, transforms
import gradio as gr

# Reuse the remedy lookup + severity logic from our knowledge-base module.
from diagnosis_map import get_remedy, get_severity

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CKPT_PATH = Path("models") / "best_model.pth"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Auto-detect device (same policy as training).
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    """Load the checkpoint and rebuild the ResNet-18 with the right head.

    Returns (model, class_names, img_size).
    """
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CKPT_PATH}\n"
            "Train the model first with:  python train.py"
        )

    # weights_only is safe here because we saved a plain dict ourselves.
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    class_names = ckpt["class_names"]
    img_size = ckpt.get("img_size", 224)

    # Rebuild architecture (no need for pretrained weights now; we load ours).
    try:
        model = models.resnet18(weights=None)
    except TypeError:
        model = models.resnet18(pretrained=False)

    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, len(class_names))
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

    return model, class_names, img_size


def build_transform(img_size):
    """Transform a single uploaded image the same way as eval/test."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# Load everything ONCE at startup (not per request) so the app is responsive.
MODEL, CLASS_NAMES, IMG_SIZE = load_model()
TRANSFORM = build_transform(IMG_SIZE)


def pretty_class_name(name: str) -> str:
    """Turn 'Tomato___Early_blight' into 'Tomato - Early blight' for display."""
    return name.replace("___", " - ").replace("_", " ")


def predict(image):
    """Gradio callback: takes a PIL image, returns (label_dict, markdown_str).

    label_dict feeds Gradio's Label component (shows a confidence bar chart).
    markdown_str feeds a Markdown component with the formatted diagnosis.
    """
    if image is None:
        return {}, "Please upload a leaf photo to get a diagnosis."

    # Ensure 3-channel RGB (handles grayscale / RGBA uploads gracefully).
    image = image.convert("RGB")

    # Preprocess -> add batch dimension -> move to device.
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(tensor)
        probs = F.softmax(logits, dim=1)[0]  # per-class probabilities

    # Build the {class_name: probability} dict for the Label component.
    confidences = {
        pretty_class_name(CLASS_NAMES[i]): float(probs[i])
        for i in range(len(CLASS_NAMES))
    }

    # Identify the top prediction.
    top_idx = int(torch.argmax(probs).item())
    top_class = CLASS_NAMES[top_idx]
    top_conf = float(probs[top_idx])

    # Look up severity + remedy using our knowledge-base module.
    severity = get_severity(top_conf)
    remedy = get_remedy(top_class)

    # Compose a clean Markdown report.
    markdown = f"""
### Diagnosis Result

- **Predicted disease:** {pretty_class_name(top_class)}
- **Confidence:** {top_conf * 100:.1f}%
- **Severity:** {severity}

**Suggested action:**
> {remedy}

_This is an AI estimate for guidance only. For critical decisions, confirm
with a local agricultural expert._
"""
    return confidences, markdown


# ---------------------------------------------------------------------------
# BUILD THE GRADIO INTERFACE
# ---------------------------------------------------------------------------
def build_ui():
    with gr.Blocks(title="Crop Disease Diagnosis Assistant") as demo:
        gr.Markdown(
            "# Crop Disease Diagnosis Assistant\n"
            "Upload a photo of a **tomato or potato leaf** and get a predicted "
            "disease, a confidence-based severity estimate, and a suggested remedy."
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload leaf photo")
                submit_btn = gr.Button("Diagnose", variant="primary")
            with gr.Column():
                # Label shows the top classes with confidence bars.
                label_output = gr.Label(num_top_classes=3,
                                        label="Class confidences")
                # Markdown shows the formatted human-readable diagnosis.
                markdown_output = gr.Markdown()

        submit_btn.click(
            fn=predict,
            inputs=image_input,
            outputs=[label_output, markdown_output],
        )

    return demo


if __name__ == "__main__":
    print(f"Loaded model on {DEVICE}. Classes: {CLASS_NAMES}")
    app = build_ui()
    # inbrowser=True opens the default browser automatically.
    app.launch(inbrowser=True)
