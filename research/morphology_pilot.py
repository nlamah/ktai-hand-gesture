"""Paired pilot on a fixed, letter-balanced sample from the fresh baseline."""
import csv
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from research.clean_training import ROOT, write_csv
from research.data import Detector, decode_image
from research.kaggle import LETTERS


def variants(image):
    rgb = np.asarray(image)
    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    overlay = cv2.addWeighted(rgb, 1.0, cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB), 0.25, 0)
    return {"erosie": cv2.erode(rgb, kernel), "dilatatie": cv2.dilate(rgb, kernel),
            "canny_overlay": overlay}


def main():
    base = ROOT / "runs/clean_training"
    with (base / "all_images.csv").open(encoding="utf-8-sig", newline="") as handle:
        baseline = list(csv.DictReader(handle))
    rng = random.Random(42)
    out = base / "morphology_pilot"
    out.mkdir(exist_ok=True)
    detector = Detector()
    results, panels = [], []
    try:
        for letter in LETTERS:
            for status, size in [("no_hand", 20), ("ok", 10)]:
                pool = [r for r in baseline if r["letter"] == letter and r["status"] == status]
                for row in rng.sample(pool, size):
                    image = decode_image(Path(row["original_path"]).read_bytes())
                    if detector.analyze(image)["status"] != status:
                        raise ValueError(f"Baseline mismatch for {row['filename']}")
                    for name, pixels in variants(image).items():
                        processed = Image.fromarray(pixels)
                        result = detector.analyze(processed)
                        results.append(dict(id=row["id"], filename=row["filename"], letter=letter,
                                            original_status=status, variant=name, status=result["status"]))
                        if status == "no_hand" and result["status"] == "ok":
                            panel = Image.new("RGB", (300, 170), "white")
                            marked = image.copy()
                            draw = ImageDraw.Draw(marked)
                            for x, y, _ in result["image_landmarks"]:
                                px, py = x * image.width, y * image.height
                                draw.ellipse((px-2, py-2, px+2, py+2), fill="#00ff88")
                            panel.paste(processed.resize((150, 150)), (0, 0))
                            panel.paste(marked.resize((150, 150)), (150, 0))
                            ImageDraw.Draw(panel).text((4, 152), f"{row['filename']} | {name}", fill="black")
                            panels.append(panel)
    finally:
        detector.close()
    write_csv(out / "per_foto.csv", results)
    summary = []
    for variant in sorted({r["variant"] for r in results}):
        group = [r for r in results if r["variant"] == variant]
        missed = [r for r in group if r["original_status"] == "no_hand"]
        controls = [r for r in group if r["original_status"] == "ok"]
        summary.append(dict(variant=variant, missed=len(missed),
                            recovered=sum(r["status"] == "ok" for r in missed),
                            controls=len(controls), lost=sum(r["status"] != "ok" for r in controls)))
    write_csv(out / "summary.csv", summary)
    for start in range(0, len(panels), 24):
        page = panels[start:start+24]
        sheet = Image.new("RGB", (900, ((len(page)+2)//3)*170), "white")
        for i, panel in enumerate(page):
            sheet.paste(panel, (300*(i%3), 170*(i//3)))
        sheet.save(out / f"controle_{start//24+1:02d}.jpg", quality=93)


if __name__ == "__main__":
    main()
