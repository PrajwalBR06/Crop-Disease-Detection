"""
diagnosis_map.py
----------------
Central "knowledge base" for the Crop Disease Diagnosis Assistant.

It contains two things:
  1. REMEDY_MAP  -> a dictionary mapping each disease class name to a short,
                    human-readable suggested action / remedy.
  2. get_severity() -> a helper that turns a model confidence score into a
                    severity label, following the rules in the project spec.

Keeping this logic in one small, well-commented file makes it easy to explain
and easy to extend (just add more classes/remedies to the dictionary).
"""

# ---------------------------------------------------------------------------
# 1. REMEDY LOOKUP
# ---------------------------------------------------------------------------
# The KEYS here MUST exactly match the class-folder names produced by
# data_prep.py (which come from the dataset folder names). If you add or rename
# a class, update this dictionary too.
#
# The values are intentionally short (one line) so they display cleanly in the
# Gradio app. They are general guidance, not a substitute for an agronomist.
REMEDY_MAP = {
    "Tomato___Early_blight":
        "Apply a copper-based or chlorothalonil fungicide within 3-5 days; "
        "remove and destroy affected lower leaves.",
    "Tomato___Late_blight":
        "Act fast: remove infected plants, apply a fungicide with mancozeb, "
        "and avoid overhead watering to reduce leaf wetness.",
    "Tomato___Leaf_Mold":
        "Improve airflow and lower humidity in the greenhouse; apply a "
        "labelled fungicide and avoid wetting the foliage.",
    "Tomato___healthy":
        "No disease detected. Keep monitoring, water at the base, and maintain "
        "good spacing for airflow.",
    "Potato___Early_blight":
        "Apply a preventive fungicide (chlorothalonil/mancozeb) and ensure "
        "balanced nitrogen; remove infected debris.",
    "Potato___healthy":
        "No disease detected. Continue regular monitoring and maintain proper "
        "hilling and irrigation practices.",
}

# A safe fallback message used if the predicted class is somehow missing from
# the dictionary above (should not normally happen).
DEFAULT_REMEDY = (
    "No specific remedy found for this class. Please consult a local "
    "agricultural extension officer."
)


def get_remedy(class_name: str) -> str:
    """Return the one-line remedy string for a given class name.

    Args:
        class_name: The predicted disease class (must match a REMEDY_MAP key).

    Returns:
        The remedy string, or DEFAULT_REMEDY if the class is unknown.
    """
    return REMEDY_MAP.get(class_name, DEFAULT_REMEDY)


# ---------------------------------------------------------------------------
# 2. SEVERITY LOGIC
# ---------------------------------------------------------------------------
# The severity label is derived purely from the model's confidence in its top
# prediction. This is a simple heuristic (NOT a medical grading of the plant):
# a very confident prediction is treated as a "confirmed" diagnosis, while a
# low-confidence prediction is flagged as uncertain so the user rechecks.
#
#   confidence > 90%   -> "Advanced / Confirmed"
#   70% - 90%          -> "Moderate"
#   below 70%          -> "Early / Uncertain"
def get_severity(confidence: float) -> str:
    """Map a confidence score (0.0-1.0) to a severity label.

    Args:
        confidence: Model confidence for the top class, in the range [0, 1].

    Returns:
        A short severity label string.
    """
    # Convert to a percentage for readability against the spec thresholds.
    confidence_pct = confidence * 100.0

    if confidence_pct > 90.0:
        return "Advanced / Confirmed"
    elif confidence_pct >= 70.0:
        return "Moderate"
    else:
        return "Early / Uncertain - recommend rechecking with a clearer photo"


# ---------------------------------------------------------------------------
# Quick self-test: run `python diagnosis_map.py` to sanity-check the logic.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Remedy lookup demo:")
    for cls in REMEDY_MAP:
        print(f"  {cls:28s} -> {get_remedy(cls)}")

    print("\nSeverity logic demo:")
    for c in [0.95, 0.80, 0.55]:
        print(f"  confidence={c*100:.0f}%  ->  {get_severity(c)}")
