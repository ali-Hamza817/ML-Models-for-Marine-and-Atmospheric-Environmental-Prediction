#!/usr/bin/env python3
"""
Format a pasted flowchart image for paper submission.

This script:
- loads flowchart_pipeline.jpg from the repo root
- upscales with Lanczos to a submission-friendly size
- applies mild contrast enhancement and sharpening
- exports PNG/PDF/TIFF with 600 DPI metadata
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[2]
# Accept .png or .jpg, whichever is present
_SRC_PNG = REPO_ROOT / "flowchart_pipeline.png"
_SRC_JPG = REPO_ROOT / "flowchart_pipeline.jpg"
SRC = _SRC_PNG if _SRC_PNG.exists() else _SRC_JPG
OUT_DIR = REPO_ROOT / "outputs" / "figures"

# 8 inches at 600 dpi ~= 4800 px. Keep aspect ratio.
TARGET_WIDTH = 4800
DPI = (600, 600)


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"Missing source image: {SRC}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    img = Image.open(SRC).convert("RGB")
    src_w, src_h = img.size

    scale = TARGET_WIDTH / float(src_w)
    target_size = (int(src_w * scale), int(src_h * scale))

    # Upscale, then gently enhance readability.
    img_up = img.resize(target_size, resample=Image.Resampling.LANCZOS)
    img_up = ImageOps.autocontrast(img_up, cutoff=1)
    img_up = ImageEnhance.Contrast(img_up).enhance(1.06)
    img_up = ImageEnhance.Sharpness(img_up).enhance(1.15)
    img_up = img_up.filter(ImageFilter.UnsharpMask(radius=1.6, percent=110, threshold=2))

    base = OUT_DIR / "flowchart_pipeline_formatted"

    # PNG for quick inspection.
    img_up.save(base.with_suffix(".png"), dpi=DPI, optimize=True)
    # TIFF for PLOS submission pipelines.
    img_up.save(base.with_suffix(".tiff"), dpi=DPI, compression="tiff_lzw")
    # PDF (raster embedded) for manuscript assembly if needed.
    img_up.save(base.with_suffix(".pdf"), dpi=DPI)

    print(f"Source: {SRC} ({src_w}x{src_h})")
    print(f"Saved: {base.with_suffix('.png')} ({target_size[0]}x{target_size[1]})")
    print(f"Saved: {base.with_suffix('.tiff')} at {DPI[0]} DPI")
    print(f"Saved: {base.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

