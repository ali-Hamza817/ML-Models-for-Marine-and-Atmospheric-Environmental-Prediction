"""
PLOS Figure Batch Fix Script
============================
Automatically fixes all TIFF figures in plos_submission/ to meet PLOS requirements:
  - Remove alpha channel (RGBA → RGB)
  - Scale to max 2250 × 2625 pixels (at 300 dpi) while preserving aspect ratio
  - Set DPI to 300
  - Apply LZW compression
  - Flatten image (single background layer)
  - File size target: < 10 MB

Output saved to: plos_submission/plos_compliant/
Original files are NOT modified.
"""

import os
import sys
import math
from pathlib import Path

# Handle very large images (S2_Fig etc.) safely
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # disable decompression bomb check – we trust our own files

# ── PLOS limits ──────────────────────────────────────────────────────────────
TARGET_DPI    = 300          # output dpi
MAX_W_PX      = 2250         # pixels at 300 dpi  (=7.5 inch)
MAX_H_PX      = 2625         # pixels at 300 dpi  (=8.75 inch)
MAX_FILE_MB   = 10.0
# ─────────────────────────────────────────────────────────────────────────────

INPUT_DIR  = Path(r"D:\SCI\数据集\marine-ml-benchmark\outputs\plos_submission")
OUTPUT_DIR = INPUT_DIR / "plos_compliant"
OUTPUT_DIR.mkdir(exist_ok=True)

TIFF_FILES = sorted(INPUT_DIR.glob("*.tif")) + sorted(INPUT_DIR.glob("*.tiff"))
# Exclude files that are already inside the output folder
TIFF_FILES = [f for f in TIFF_FILES if f.parent == INPUT_DIR]

print(f"{'='*60}")
print(f"  PLOS Figure Batch Fix")
print(f"  Input:  {INPUT_DIR}")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Files to process: {len(TIFF_FILES)}")
print(f"{'='*60}\n")


def scale_to_fit(w, h, max_w, max_h):
    """Return new (w, h) scaled down to fit within max_w × max_h, aspect-ratio preserved."""
    if w <= max_w and h <= max_h:
        return w, h
    scale = min(max_w / w, max_h / h)
    return math.floor(w * scale), math.floor(h * scale)


def remove_alpha(img: Image.Image) -> Image.Image:
    """Convert RGBA/LA → RGB/L by pasting onto a white background."""
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])   # use alpha as mask
        return bg
    if img.mode == "LA":
        bg = Image.new("L", img.size, 255)
        bg.paste(img, mask=img.split()[1])
        return bg
    if img.mode == "P":
        return img.convert("RGB")
    if img.mode == "CMYK":
        return img.convert("RGB")
    return img


results = []

for fpath in TIFF_FILES:
    fname = fpath.name
    out_path = OUTPUT_DIR / fname
    fsize_in = fpath.stat().st_size / (1024 * 1024)

    print(f"Processing: {fname}  ({fsize_in:.1f} MB)")

    try:
        img = Image.open(fpath)
        img.load()

        orig_w, orig_h = img.size
        orig_dpi = img.info.get("dpi", (TARGET_DPI, TARGET_DPI))
        orig_mode = img.mode
        print(f"  Original : {orig_w}×{orig_h} px  |  mode={orig_mode}  |  dpi={orig_dpi}")

        # Step 1 – Remove alpha channel
        img = remove_alpha(img)

        # Step 2 – Scale to PLOS limits (preserve aspect ratio, high-quality Lanczos)
        new_w, new_h = scale_to_fit(orig_w, orig_h, MAX_W_PX, MAX_H_PX)
        if (new_w, new_h) != (orig_w, orig_h):
            print(f"  Scaling  : {orig_w}×{orig_h} → {new_w}×{new_h}")
            img = img.resize((new_w, new_h), Image.LANCZOS)
        else:
            print(f"  Scaling  : no change needed")

        # Step 3 – Save as TIFF with LZW compression and correct DPI
        img.save(
            out_path,
            format="TIFF",
            compression="tiff_lzw",
            dpi=(TARGET_DPI, TARGET_DPI),
        )

        fsize_out = out_path.stat().st_size / (1024 * 1024)
        size_ok   = fsize_out < MAX_FILE_MB
        dim_ok    = new_w <= MAX_W_PX and new_h <= MAX_H_PX

        status = "✅ OK" if (size_ok and dim_ok) else "⚠️  CHECK"
        if not size_ok:
            status += f"  – still {fsize_out:.1f} MB"

        print(f"  Output   : {new_w}×{new_h} px  |  {fsize_out:.2f} MB  |  {status}")
        results.append((fname, fsize_in, fsize_out, new_w, new_h, size_ok and dim_ok, None))

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        results.append((fname, fsize_in, None, None, None, False, str(e)))

    print()

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Summary")
print(f"{'='*60}")
print(f"{'File':<20} {'In(MB)':>8} {'Out(MB)':>9} {'Pixels':>15} {'Pass?':>6}")
print(f"{'-'*60}")
for fname, fi, fo, w, h, ok, err in results:
    if err:
        print(f"{fname:<20} {fi:>7.1f}   ERROR: {err[:30]}")
    else:
        dims = f"{w}×{h}"
        flag = "✅" if ok else "❌"
        print(f"{fname:<20} {fi:>7.1f}  {fo:>8.2f}  {dims:>15}  {flag}")

n_ok  = sum(1 for *_, ok, err in results if ok and not err)
n_err = sum(1 for *_, ok, err in results if err)
print(f"\n  ✅ Passed: {n_ok} / {len(results)}    ❌ Errors: {n_err}")
print(f"\nCompliant files saved to:\n  {OUTPUT_DIR}")
