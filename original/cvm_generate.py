#!/usr/bin/env python3
"""
CVM generation -- one self-contained, shareable script that produces the DP deliverable.
========================================================================================
Runs the Douglas-Peucker CVM pipeline over the annotated study and writes `FINAL/`:

  FINAL/Batch1 .. Batch15/<scan>.png   overlay per scan (thin lines): grey mask, green
                                       Douglas-Peucker quad, magenta posterior-height line,
                                       yellow inferior chord + perpendicular to the dome apex
  FINAL/measurements.csv               scan, vertebra, posterior_height_px, dome_height_px,
                                       doming_ratio, review_status
  FINAL/all_scans_montage.png          every overlay on one page, labelled with the file name
  FINAL/LEGEND.txt                     colour key + ratio convention

Ratio convention (denominator is always a POSTERIOR height):
    C2 doming ratio = C2 dome / C3 posterior      (C2's own height includes the dens)
    C3 doming ratio = C3 dome / C3 posterior
    C4 doming ratio = C4 dome / C4 posterior

All the fitting / measuring / drawing is the deployable engine `Deploy/run_cvm.py`
(`process_image`), unchanged -- this file is only the batch driver + montage. Deterministic.

Requirements: numpy, opencv-python.   Usage:  python3 cvm_generate.py
"""
import os, sys, csv, glob, json, shutil, tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------------------------------------------------------------ configuration
ROOT   = os.path.dirname(os.path.abspath(__file__))
DEPLOY = os.path.join(ROOT, "Deploy")                 # holds run_cvm.py (the engine)
TEMP   = os.path.join(ROOT, "temp")                   # Batch*/output.json + images
FINAL  = os.path.join(ROOT, "FINAL")                  # output deliverable
WORKERS = max(1, (os.cpu_count() or 2) - 1)
# ----------------------------------------------------------------------------

sys.path.insert(0, DEPLOY)
import run_cvm as rc
import numpy as np
import cv2

LEGEND = """CVM measured scans -- line legend
=================================
Each PNG is a lateral cervical-spine radiograph with the C2/C3/C4 landmark geometry drawn
on top as thin lines. Every number is in measurements.csv; nothing is written on the images.
Folder layout mirrors the original batches.

Line colours
  grey     vertebral-body mask outline (the annotation)
  green    landmark quadrilateral -- the 4 dominant body corners from Douglas-Peucker
           simplification of the convex hull to 4 vertices (for C2 the two superior corners
           use the inscribed/circumscribed average, to skip the dens)
  magenta  posterior height line = posterior-superior corner to posterior-inferior corner
           (ratio denominator)
  yellow   inferior-border chord (between the two inferior quad corners) + the perpendicular
           up to the endplate dome apex = concavity depth (ratio numerator)

measurements.csv columns
  scan                 scan file name (matches the PNG, without extension)
  vertebra             C2 / C3 / C4
  posterior_height_px  this vertebra's own posterior-wall height, pixels
  dome_height_px       inferior-endplate concavity depth, pixels (0 = flat/convex)
  doming_ratio         dome_height_px / posterior_height_px   (>= 0.10  ->  domed)
  review_status        BLANK -- manual review column (ok / curved-body / bad-mask / other)

Ratio convention: the denominator is always a POSTERIOR height.
  C2 doming ratio = C2 dome / C3 posterior   (C2's own height includes the dens)
  C3 doming ratio = C3 dome / C3 posterior
  C4 doming ratio = C4 dome / C4 posterior
"""


def build_overlays_and_csv():
    if os.path.isdir(FINAL):
        for b in glob.glob(os.path.join(FINAL, "Batch*")):
            shutil.rmtree(b, ignore_errors=True)          # clear old overlays only
    os.makedirs(FINAL, exist_ok=True)
    junk = tempfile.mkdtemp(prefix="cvm_quads_")          # per-vertebra JSON -> discarded
    idx = rc.index_images(TEMP, exclude=[FINAL, os.path.join(TEMP, "cvm_output")])

    tasks, missing = [], []
    for bj in sorted(glob.glob(os.path.join(TEMP, "Batch*", "output.json")),
                     key=lambda p: int(os.path.basename(os.path.dirname(p))[5:])):
        batch = os.path.basename(os.path.dirname(bj))
        ov_dir = os.path.join(FINAL, batch); os.makedirs(ov_dir, exist_ok=True)
        d = json.load(open(bj))
        cats = {c["id"]: c["name"] for c in d.get("categories", [])}
        by_img = {}
        for a in d.get("annotations", []):
            by_img.setdefault(a["image_id"], []).append(a)
        for im in d.get("images", []):
            path, exact = rc.resolve_image(im["file_name"], idx)
            stem = os.path.splitext(os.path.basename(im["file_name"]))[0]
            anns = by_img.get(im["id"], [])
            if path is None:
                for a in anns:
                    missing.append([stem, cats.get(a.get("category_id"), str(a.get("category_id"))),
                                    "", "", "", "image-missing"])
                continue
            tasks.append({"path": path, "exact": exact, "W": im["width"], "H": im["height"],
                          "anns": anns, "cats": cats, "file_name": im["file_name"],
                          "stem": stem, "ov_dir": ov_dir, "q_dir": junk})
    print(f"{len(tasks)} scans queued, {len(missing)} missing rows")

    results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(rc.process_image, t) for t in tasks]
        for n, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if n % 25 == 0 or n == len(tasks):
                print(f"  [{n}/{len(tasks)}] done")

    rows, n_ok, n_err = list(missing), 0, 0
    for stem, r, tag, warn, status in results:
        rows.extend(r)
        n_ok += int(status == "ok")
        if status.startswith("error"):
            n_err += 1; print("  ERROR", stem, status)
    rows.sort(key=lambda x: (x[0], x[1]))

    with open(os.path.join(FINAL, "measurements.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scan", "vertebra", "posterior_height_px", "dome_height_px",
                    "doming_ratio", "review_status"])
        w.writerows([r[:5] + [""] for r in rows])            # drop internal note; review_status blank
    with open(os.path.join(FINAL, "LEGEND.txt"), "w") as f:
        f.write(LEGEND)
    shutil.rmtree(junk, ignore_errors=True)

    clean = sum(1 for x in rows if x[2] != "")
    n_png = len(glob.glob(os.path.join(FINAL, "Batch*", "*.png")))
    print(f"FINAL/ -> {n_png} overlay PNGs, {len(rows)} csv rows, {clean} clean | "
          f"images ok {n_ok} | errored {n_err}")


def build_montage(cols=12, cell_w=260):
    def wrap(name, n=30):
        return [name[i:i + n] for i in range(0, len(name), n)] or [name]
    pngs = []
    for b in sorted(glob.glob(os.path.join(FINAL, "Batch*")),
                    key=lambda x: int(os.path.basename(x)[5:])):
        pngs += sorted(glob.glob(os.path.join(b, "*.png")))
    crops = []
    for p in pngs:
        img = cv2.imread(p)
        if img is None:
            continue
        b_, g, r = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
        color = (np.abs(r - g) > 40) | (np.abs(g - b_) > 40) | (np.abs(r - b_) > 40)
        ys, xs = np.where(color)
        if len(xs) == 0:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        pad = int((x1 - x0) * 0.28) + 10
        c = img[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
        c = cv2.resize(c, (cell_w, int(cell_w * c.shape[0] / c.shape[1])))
        lines = wrap(os.path.splitext(os.path.basename(p))[0]); bar = 13 * len(lines) + 6
        cv2.rectangle(c, (0, 0), (cell_w, bar), (0, 0, 0), -1)
        for j, ln in enumerate(lines):
            cv2.putText(c, ln, (3, 12 + 13 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1, cv2.LINE_AA)
        crops.append(c)
    H = max(c.shape[0] for c in crops)
    crops = [cv2.copyMakeBorder(c, 0, H - c.shape[0], 0, 3, cv2.BORDER_CONSTANT) for c in crops]
    grid = []
    for i in range(0, len(crops), cols):
        row = crops[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(crops[0]))
        grid.append(np.hstack(row))
    out = os.path.join(FINAL, "all_scans_montage.png")
    cv2.imwrite(out, np.vstack(grid))
    print(f"montage -> {out} ({len(crops)} cells, {cols} cols)")


def main():
    build_overlays_and_csv()
    build_montage()
    print("Done.")


if __name__ == "__main__":
    main()
