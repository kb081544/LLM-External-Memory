"""Draw ground-truth/predicted highlight boxes on a REAL Multimodal-Mind2Web screenshot
(as opposed to render.py, which re-renders the unstyled cleaned_html in a headless
browser). Real screenshots have no live interactivity either (still offline replay --
see run.py's module docstring) but look like an actual page instead of a broken one.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

GREEN = (46, 160, 67)
RED = (229, 72, 77)


def highlight_and_save(
    screenshot_bytes: bytes,
    out_path: Path,
    gt_bbox: tuple[float, float, float, float] | None,
    pred_bbox: tuple[float, float, float, float] | None,
    pred_correct: bool,
) -> bool:
    """Draws green outline on the ground-truth box, and a green/red outline (correct/
    wrong) on the predicted box if different from ground truth. Returns True if at
    least one box was actually drawn."""
    img = Image.open(BytesIO(screenshot_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    drew_any = False

    if gt_bbox:
        x, y, w, h = gt_bbox
        draw.rectangle([x, y, x + w, y + h], outline=GREEN, width=4)
        drew_any = True

    if pred_bbox and pred_bbox != gt_bbox:
        x, y, w, h = pred_bbox
        draw.rectangle([x, y, x + w, y + h], outline=GREEN if pred_correct else RED, width=4)
        drew_any = True

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return drew_any
