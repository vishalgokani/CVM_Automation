#!/usr/bin/env python3
"""
CVM Douglas-Peucker automation
==============================

Processes a directory containing batch folders such as:

    semiautomated/
        batch001/
            image1.bmp
            image2.bmp
            masks/
                image1_C2.bmp
                image1_C3.bmp
                image1_C4.bmp
            ...

For every original BMP, the script:
  1. Finds the C2/C3/C4 binary masks.
  2. Fits a 4-corner Douglas-Peucker quadrilateral to each mask.
  3. Measures the anterior, posterior, superior and inferior quadrilateral
     edges in pixels.
  4. Measures inferior-endplate dome depth from the inferior chord.
  5. Calculates CVM doming ratios:
       C2 = C2 dome / C3 posterior height
       C3 = C3 dome / C3 posterior height
       C4 = C4 dome / C4 posterior height
  6. Writes an overlay into:
       <batch>/overlaid_Douglas-Peucker/
  7. Writes ONE CSV containing all scans into:
       <target directory>/cvm_measurements.csv

Only Douglas-Peucker is used. No quadrilateral-fitter or min-area rectangle.

IMPORTANT:
This is a standalone reconstruction of the geometry from the supplied CVM
examples/specification. It does not depend on the previous Deploy/run_cvm.py.

RUNNING IN ANACONDA PROMPT
--------------------------

Create the environment:

    conda create -n cvm_automation python=3.11
    conda activate cvm_automation

Install dependencies:

    pip install -r requirements.txt

Run on the target directory:

    python cvm_automation.py --directory "\\wnresearch\Drobo\Vishal_Graham\ML Review\Spine\CVM_annotation\semiautomated"

Optional:

    python cvm_automation.py --directory "D:\\some\\folder"

Optional tuning:

    python cvm_automation.py --directory "D:\\some\\folder" --epsilon 0.02

The default epsilon is adaptive. The script searches for an epsilon that
produces exactly four Douglas-Peucker corners. --epsilon can be used to
override the starting fraction of the contour perimeter.

MASK NAMING
-----------

The script accepts several common layouts, including:

    masks/image_C2.bmp
    masks/image_C3.bmp
    masks/image_C4.bmp

or:

    masks/C2/image.bmp
    masks/C3/image.bmp
    masks/C4/image.bmp

It also recognizes c2/c3/c4 case-insensitively.

If the C2/C3/C4 labels are absent but exactly three masks can be associated
with an image, the script orders them vertically (upper=C2, middle=C3,
lower=C4), which is appropriate for a lateral cervical spine.

OUTPUT CSV
----------

One row per original radiograph. Columns include:
  - batch
  - filename
  - C2/C3/C4 four edge lengths
  - C2/C3/C4 dome heights
  - C2/C3/C4 dome ratios
  - additional width/height, aspect, diagonal and area measurements
  - processing/QC status

All distances are pixel distances.

The overlay shows:
  - Douglas-Peucker quadrilateral
  - posterior/anterior/superior/inferior edge labels
  - inferior chord
  - dome-depth line and apex
  - vertebral labels
"""

import argparse
import csv
import math
import os
import re
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_EPSILON = 0.02
MIN_MASK_AREA = 20

EDGE_NAMES = ("anterior", "posterior", "superior", "inferior")


# ---------------------------------------------------------------------------
# Basic geometry
# ---------------------------------------------------------------------------

def distance(a, b):
    return float(np.linalg.norm(np.asarray(b, dtype=float) -
                                np.asarray(a, dtype=float)))


def cross2(a, b):
    """2-D scalar cross product."""
    return float(a[0] * b[1] - a[1] * b[0])


def point_line_distance(point, a, b):
    v = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    w = np.asarray(point, dtype=float) - np.asarray(a, dtype=float)
    L = np.linalg.norm(v)
    if L == 0:
        return 0.0
    return abs(cross2(v, w)) / L


def polygon_area(points):
    p = np.asarray(points, dtype=float)
    return float(abs(np.cross(p, np.roll(p, -1, axis=0)).sum()) / 2.0)


def order_quad(points):
    """
    Return four vertices in clockwise image order:
        top-left, top-right, bottom-right, bottom-left

    This is deliberately based on the image coordinate system rather than
    assuming the vertebral body is perfectly horizontal.
    """
    p = np.asarray(points, dtype=float)

    s = p[:, 0] + p[:, 1]
    d = p[:, 0] - p[:, 1]

    tl = p[np.argmin(s)]
    br = p[np.argmax(s)]
    tr = p[np.argmax(d)]
    bl = p[np.argmin(d)]

    q = np.array([tl, tr, br, bl], dtype=float)

    # The sum/difference method can theoretically select duplicates for very
    # unusual quadrilaterals. Fall back to angular ordering if necessary.
    if len({tuple(x) for x in q}) != 4:
        c = p.mean(axis=0)
        ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
        q = p[np.argsort(ang)]

        # rotate to top-left-ish first
        idx = np.argmin(q[:, 0] + q[:, 1])
        q = np.roll(q, -idx, axis=0)

    # Ensure clockwise order in image coordinates.
    area_signed = np.cross(q, np.roll(q, -1, axis=0)).sum()
    if area_signed < 0:
        q = q[[0, 3, 2, 1]]

    return q


def classify_edges(q):
    """
    Determine anatomical edge names from the ordered quadrilateral.

    In a lateral cervical radiograph:
        posterior = left
        anterior  = right
        superior  = upper
        inferior  = lower

    We choose the left/right pair by x-coordinate and the upper/lower pair
    by y-coordinate. This remains stable for the oblique vertebral bodies
    in the supplied examples.
    """
    q = np.asarray(q, dtype=float)

    # Four geometric edges.
    edges = [
        (q[0], q[1]),  # top
        (q[1], q[2]),  # right
        (q[2], q[3]),  # bottom
        (q[3], q[0]),  # left
    ]

    # top/bottom: compare mean y
    top_idx = min(range(4), key=lambda i: np.mean(edges[i], axis=0)[1])
    bottom_idx = max(range(4), key=lambda i: np.mean(edges[i], axis=0)[1])

    # remaining two are anterior/posterior.
    remaining = [i for i in range(4) if i not in (top_idx, bottom_idx)]

    # The edge with larger mean x is anterior.
    anterior_idx = max(
        remaining,
        key=lambda i: np.mean(edges[i], axis=0)[0]
    )
    posterior_idx = remaining[0] if remaining[1] == anterior_idx else remaining[1]

    return {
        "superior": edges[top_idx],
        "inferior": edges[bottom_idx],
        "anterior": edges[anterior_idx],
        "posterior": edges[posterior_idx],
    }


# ---------------------------------------------------------------------------
# Mask / Douglas-Peucker processing
# ---------------------------------------------------------------------------

def read_binary_mask(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read mask: {path}")

    mask = (img > 0).astype(np.uint8)

    # Fill small holes/noise without materially changing the annotated shape.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8)

    if int(mask.sum()) < MIN_MASK_AREA:
        raise ValueError("Mask contains too few foreground pixels")

    return mask


def find_dp_quad(mask, epsilon_start=DEFAULT_EPSILON):
    """
    Fit the convex hull with Douglas-Peucker and search for exactly four
    vertices.

    The original CVM description specifies:
      "Douglas-Peucker simplification of the convex hull to 4 vertices."

    We therefore simplify the convex hull, not the raw noisy contour.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        raise ValueError("No contour found")

    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) < MIN_MASK_AREA:
        raise ValueError("Contour area too small")

    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)

    if perimeter <= 0:
        raise ValueError("Invalid contour perimeter")

    # First search a broad range. Starting around 2% of perimeter reproduces
    # the usual DP scale for these masks while the search makes the procedure
    # robust to differences in mask resolution.
    epsilons = []

    for frac in np.linspace(
        max(0.0005, epsilon_start * 0.20),
        max(0.01, epsilon_start * 5.0),
        300
    ):
        epsilons.append(frac * perimeter)

    # Also search a larger range if needed.
    for frac in np.linspace(0.001, 0.20, 500):
        epsilons.append(frac * perimeter)

    candidates = []
    seen = set()

    for eps in epsilons:
        approx = cv2.approxPolyDP(hull, eps, True)
        n = len(approx)

        if n == 4:
            q = approx[:, 0, :].astype(float)
            key = tuple(np.round(q.flatten(), 3))
            if key not in seen:
                seen.add(key)
                candidates.append(q)

    if not candidates:
        # If a perfect 4-corner solution is impossible, select the
        # approximation closest to four vertices and report that condition.
        best = None
        best_score = float("inf")

        for frac in np.linspace(0.001, 0.20, 500):
            approx = cv2.approxPolyDP(hull, frac * perimeter, True)
            score = abs(len(approx) - 4) + 0.001 * frac
            if score < best_score:
                best_score = score
                best = approx

        if best is None or len(best) != 4:
            raise ValueError(
                f"Douglas-Peucker could not produce 4 corners "
                f"(best had {len(best) if best is not None else 0})"
            )

        q = best[:, 0, :].astype(float)
    else:
        # Prefer the candidate closest to the requested epsilon. If several
        # candidates have the same number of vertices, this gives deterministic
        # behavior.
        target = epsilon_start * perimeter
        q = min(
            candidates,
            key=lambda x: abs(
                cv2.arcLength(x.astype(np.float32).reshape(-1, 1, 2), True)
                - target
            )
        )

    return order_quad(q), contour


# ---------------------------------------------------------------------------
# Dome measurement
# ---------------------------------------------------------------------------

def measure_dome(mask, contour, inferior_edge, quad):
    """
    Estimate inferior-endplate dome depth.

    The inferior quadrilateral edge is treated as the chord. Foreground
    contour points near that chord are examined, and the maximum perpendicular
    displacement toward the vertebral-body interior is the dome depth.

    Restricting the search to a band around the inferior chord prevents the
    opposite/superior endplate from being incorrectly interpreted as the dome.
    """
    a, b = np.asarray(inferior_edge[0], float), np.asarray(inferior_edge[1], float)
    v = b - a
    L = np.linalg.norm(v)

    if L == 0:
        return 0.0, None

    centroid = np.asarray(quad, float).mean(axis=0)

    # Sign pointing toward the body interior.
    centroid_cross = cross2(v, centroid - a)

    if abs(centroid_cross) < 1e-8:
        return 0.0, None

    points = contour[:, 0, :].astype(float)

    cross_values = np.array(
        [cross2(v, p - a) for p in points],
        dtype=float
    )

    distances = np.abs(cross_values) / L
    t = np.dot(points - a, v) / (L * L)

    # Interior-side points.
    interior = (
        np.sign(cross_values) == np.sign(centroid_cross)
    )

    # Only examine points whose projection lies on the inferior chord.
    on_chord_projection = (t >= -0.05) & (t <= 1.05)

    # Restrict to a neighborhood of the inferior endplate. This is important:
    # otherwise the superior contour can have a much larger distance to the
    # inferior chord and would falsely become the "dome."
    quad_height = max(
        distance(quad[0], quad[3]),
        distance(quad[1], quad[2]),
        1.0
    )

    # 35% of vertebral height is intentionally generous for tilted bodies,
    # but excludes the opposite endplate in the supplied examples.
    near_inferior = distances <= 0.35 * quad_height

    valid = interior & on_chord_projection & near_inferior

    if not np.any(valid):
        return 0.0, None

    idxs = np.where(valid)[0]
    best_local = idxs[np.argmax(distances[valid])]

    dome = float(distances[best_local])
    apex = points[best_local]

    return dome, apex


# ---------------------------------------------------------------------------
# Mask identification
# ---------------------------------------------------------------------------

def normalize_name(name):
    s = Path(name).stem.lower()

    # Remove common mask/vertebra tokens while preserving the original stem.
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"(^|_)(mask|masks)(?=_|$)", "_", s)
    s = re.sub(r"(^|_)(c2|c3|c4)(?=_|$)", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")

    return s


def vertebra_from_path(path):
    """
    Identify C2/C3/C4 from filename or parent folder.
    """
    parts = [p.lower() for p in Path(path).parts]

    for p in reversed(parts):
        if re.search(r"(^|[_\-\s])c2([_\-\s.]|$)", p) or p == "c2":
            return "C2"
        if re.search(r"(^|[_\-\s])c3([_\-\s.]|$)", p) or p == "c3":
            return "C3"
        if re.search(r"(^|[_\-\s])c4([_\-\s.]|$)", p) or p == "c4":
            return "C4"

    stem = Path(path).stem.lower()

    # Allow c2/c3/c4 anywhere as a token, including forms such as image_C2.
    m = re.search(r"(?:^|[_\-\s])(c[234])(?:[_\-\s.]|$)", stem)
    if m:
        return m.group(1).upper()

    return None


def all_mask_files(batch_dir):
    mask_root = batch_dir / "masks"

    if not mask_root.is_dir():
        return []

    return [
        p for p in mask_root.rglob("*")
        if p.is_file() and p.suffix.lower() == ".bmp"
    ]


def group_masks_for_image(image_path, mask_files):
    """
    Return {C2: path, C3: path, C4: path}.

    Matching is based primarily on normalized filename. Explicit C2/C3/C4
    labels are preferred. If three unlabeled masks are associated with an
    image, they are vertically ordered.
    """
    image_key = normalize_name(image_path.name)

    explicit = {}
    unlabeled = []

    for p in mask_files:
        key = normalize_name(p.name)

        # Require the mask to resemble the image filename after removing
        # mask/vertebra tokens.
        if key != image_key:
            continue

        v = vertebra_from_path(p)

        if v:
            explicit[v] = p
        else:
            unlabeled.append(p)

    # Also allow a looser filename containment match.
    if len(explicit) + len(unlabeled) < 3:
        for p in mask_files:
            if p in explicit.values() or p in unlabeled:
                continue

            key = normalize_name(p.name)

            if image_key in key or key in image_key:
                v = vertebra_from_path(p)
                if v:
                    explicit[v] = p
                else:
                    unlabeled.append(p)

    result = dict(explicit)

    # If unlabeled masks remain, infer C2/C3/C4 by vertical position.
    if unlabeled:
        centers = []

        for p in unlabeled:
            try:
                m = read_binary_mask(p)
                ys, xs = np.where(m > 0)
                if len(xs):
                    centers.append((float(np.mean(ys)), p))
            except Exception:
                pass

        centers.sort(key=lambda x: x[0])

        for v, (_, p) in zip(
            [x for x in ("C2", "C3", "C4") if x not in result],
            centers
        ):
            result[v] = p

    return result


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

def draw_text(img, text, point, scale=0.65, thickness=2):
    x, y = int(round(point[0])), int(round(point[1]))

    # Black outline followed by white text for visibility on radiographs.
    cv2.putText(
        img, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, scale,
        (0, 0, 0), thickness + 3, cv2.LINE_AA
    )
    cv2.putText(
        img, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, scale,
        (255, 255, 255), thickness, cv2.LINE_AA
    )


def draw_quad_overlay(image, results):
    """
    Draw all successful vertebral measurements on the original image.
    """
    if len(image.shape) == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = image.copy()

    # Colors in BGR.
    quad_color = (0, 255, 0)
    posterior_color = (255, 0, 255)
    dome_color = (0, 255, 255)

    for v in ("C2", "C3", "C4"):
        r = results.get(v)

        if not r or r.get("quad") is None:
            continue

        q = np.asarray(r["quad"], dtype=np.int32)

        cv2.polylines(
            overlay,
            [q.reshape(-1, 1, 2)],
            True,
            quad_color,
            2,
            cv2.LINE_AA
        )

        # Label at centroid.
        center = q.mean(axis=0)
        draw_text(overlay, v, center, scale=0.65, thickness=2)

        edges = r["edges"]

        # Posterior.
        p1, p2 = np.asarray(edges["posterior"], dtype=int)
        cv2.line(
            overlay, tuple(p1), tuple(p2),
            posterior_color, 3, cv2.LINE_AA
        )

        # Inferior chord.
        i1, i2 = np.asarray(edges["inferior"], dtype=int)
        cv2.line(
            overlay, tuple(i1), tuple(i2),
            dome_color, 2, cv2.LINE_AA
        )

        # Dome perpendicular.
        apex = r.get("dome_apex")

        if apex is not None:
            apex = np.asarray(apex, dtype=float)
            a = np.asarray(edges["inferior"][0], dtype=float)
            b = np.asarray(edges["inferior"][1], dtype=float)

            v = b - a
            denom = np.dot(v, v)

            if denom > 0:
                t = np.dot(apex - a, v) / denom
                foot = a + t * v

                cv2.line(
                    overlay,
                    tuple(np.round(foot).astype(int)),
                    tuple(np.round(apex).astype(int)),
                    dome_color,
                    2,
                    cv2.LINE_AA
                )
                cv2.circle(
                    overlay,
                    tuple(np.round(apex).astype(int)),
                    4,
                    dome_color,
                    -1
                )

    return overlay


# ---------------------------------------------------------------------------
# Measurement row
# ---------------------------------------------------------------------------

def measurement_for_mask(mask_path, vertebra, epsilon):
    mask = read_binary_mask(mask_path)

    quad, contour = find_dp_quad(mask, epsilon)
    edges = classify_edges(quad)

    dome, apex = measure_dome(
        mask,
        contour,
        edges["inferior"],
        quad
    )

    anterior = distance(*edges["anterior"])
    posterior = distance(*edges["posterior"])
    superior = distance(*edges["superior"])
    inferior = distance(*edges["inferior"])

    width = (anterior + posterior) / 2.0
    height = (superior + inferior) / 2.0

    diagonal1 = distance(quad[0], quad[2])
    diagonal2 = distance(quad[1], quad[3])

    return {
        "mask_path": str(mask_path),
        "vertebra": vertebra,
        "quad": quad,
        "edges": edges,
        "dome_apex": apex,
        "anterior_px": anterior,
        "posterior_px": posterior,
        "superior_px": superior,
        "inferior_px": inferior,
        "dome_px": dome,
        "width_mean_px": width,
        "height_mean_px": height,
        "width_height_ratio": width / height if height else np.nan,
        "dome_posterior_ratio": dome / posterior if posterior else np.nan,
        "dome_height_ratio": dome / height if height else np.nan,
        "area_px2": polygon_area(quad),
        "diagonal_1_px": diagonal1,
        "diagonal_2_px": diagonal2,
        "mask_area_px2": float(mask.sum()),
    }


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "batch",
    "filename",
    "status",
    "message",

    "C2_anterior_px",
    "C2_posterior_px",
    "C2_superior_px",
    "C2_inferior_px",
    "C2_dome_px",
    "C2_width_mean_px",
    "C2_height_mean_px",
    "C2_width_height_ratio",
    "C2_dome_height_ratio",
    "C2_area_px2",
    "C2_diagonal_1_px",
    "C2_diagonal_2_px",
    "C2_mask_area_px2",

    "C3_anterior_px",
    "C3_posterior_px",
    "C3_superior_px",
    "C3_inferior_px",
    "C3_dome_px",
    "C3_width_mean_px",
    "C3_height_mean_px",
    "C3_width_height_ratio",
    "C3_dome_height_ratio",
    "C3_area_px2",
    "C3_diagonal_1_px",
    "C3_diagonal_2_px",
    "C3_mask_area_px2",

    "C4_anterior_px",
    "C4_posterior_px",
    "C4_superior_px",
    "C4_inferior_px",
    "C4_dome_px",
    "C4_width_mean_px",
    "C4_height_mean_px",
    "C4_width_height_ratio",
    "C4_dome_height_ratio",
    "C4_area_px2",
    "C4_diagonal_1_px",
    "C4_diagonal_2_px",
    "C4_mask_area_px2",

    # Official CVM doming ratios.
    "C2_doming_ratio_C2_dome_over_C3_posterior",
    "C3_doming_ratio_C3_dome_over_C3_posterior",
    "C4_doming_ratio_C4_dome_over_C4_posterior",
]


def fmt(x):
    if x is None:
        return ""
    try:
        if np.isnan(x):
            return ""
    except Exception:
        pass
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.6f}"
    return str(x)


def make_csv_row(batch, filename, results, status="ok", message=""):
    row = {
        "batch": batch,
        "filename": filename,
        "status": status,
        "message": message,
    }

    for v in ("C2", "C3", "C4"):
        r = results.get(v)

        if r:
            row[f"{v}_anterior_px"] = fmt(r["anterior_px"])
            row[f"{v}_posterior_px"] = fmt(r["posterior_px"])
            row[f"{v}_superior_px"] = fmt(r["superior_px"])
            row[f"{v}_inferior_px"] = fmt(r["inferior_px"])
            row[f"{v}_dome_px"] = fmt(r["dome_px"])
            row[f"{v}_width_mean_px"] = fmt(r["width_mean_px"])
            row[f"{v}_height_mean_px"] = fmt(r["height_mean_px"])
            row[f"{v}_width_height_ratio"] = fmt(r["width_height_ratio"])
            row[f"{v}_dome_height_ratio"] = fmt(r["dome_height_ratio"])
            row[f"{v}_area_px2"] = fmt(r["area_px2"])
            row[f"{v}_diagonal_1_px"] = fmt(r["diagonal_1_px"])
            row[f"{v}_diagonal_2_px"] = fmt(r["diagonal_2_px"])
            row[f"{v}_mask_area_px2"] = fmt(r["mask_area_px2"])
        else:
            for col in CSV_COLUMNS:
                if col.startswith(v + "_"):
                    row[col] = ""

    c2 = results.get("C2")
    c3 = results.get("C3")
    c4 = results.get("C4")

    # The official CVM convention:
    # C2 dome / C3 posterior
    # C3 dome / C3 posterior
    # C4 dome / C4 posterior
    if c2 and c3 and c3["posterior_px"] > 0:
        row["C2_doming_ratio_C2_dome_over_C3_posterior"] = fmt(
            c2["dome_px"] / c3["posterior_px"]
        )
    else:
        row["C2_doming_ratio_C2_dome_over_C3_posterior"] = ""

    if c3 and c3["posterior_px"] > 0:
        row["C3_doming_ratio_C3_dome_over_C3_posterior"] = fmt(
            c3["dome_px"] / c3["posterior_px"]
        )
    else:
        row["C3_doming_ratio_C3_dome_over_C3_posterior"] = ""

    if c4 and c4["posterior_px"] > 0:
        row["C4_doming_ratio_C4_dome_over_C4_posterior"] = fmt(
            c4["dome_px"] / c4["posterior_px"]
        )
    else:
        row["C4_doming_ratio_C4_dome_over_C4_posterior"] = ""

    return row


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_image(image_path, batch_dir, epsilon):
    batch = batch_dir.name
    filename = image_path.name

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if image is None:
        return make_csv_row(
            batch, filename, {},
            status="error",
            message="Could not read original BMP"
        )

    mask_files = all_mask_files(batch_dir)
    mask_map = group_masks_for_image(image_path, mask_files)

    results = {}
    errors = []

    for v in ("C2", "C3", "C4"):
        mask_path = mask_map.get(v)

        if mask_path is None:
            errors.append(f"{v} mask not found")
            continue

        try:
            results[v] = measurement_for_mask(
                mask_path,
                v,
                epsilon
            )
        except Exception as e:
            errors.append(f"{v}: {e}")

    # Create overlay even when some vertebrae are missing.
    overlay = draw_quad_overlay(image, results)

    output_dir = batch_dir / "overlaid_Douglas-Peucker"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / image_path.name

    if not cv2.imwrite(str(output_path), overlay):
        errors.append("Could not write overlay")

    if errors:
        status = "partial" if results else "error"
        message = "; ".join(errors)
    else:
        status = "ok"
        message = ""

    return make_csv_row(
        batch,
        filename,
        results,
        status=status,
        message=message
    )


def find_batches(root):
    """
    Find directories named batch001, batch002, ... at any depth immediately
    below the supplied target, while avoiding generated output directories.
    """
    batches = []

    for p in root.rglob("*"):
        if not p.is_dir():
            continue

        if re.fullmatch(r"batch\d+", p.name, re.IGNORECASE):
            batches.append(p)

    return sorted(
        set(batches),
        key=lambda p: (
            int(re.search(r"\d+", p.name).group()),
            str(p).lower()
        )
    )


def find_original_images(batch):
    """
    Original BMPs are BMP files directly inside the batch folder.

    Anything under masks/ or the generated overlay directory is excluded.
    """
    return sorted(
        [
            p for p in batch.glob("*.bmp")
            if p.is_file()
        ],
        key=lambda p: p.name.lower()
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run standalone Douglas-Peucker CVM measurements."
    )

    parser.add_argument(
        "--directory",
        required=True,
        help="Parent directory containing batch001, batch002, etc."
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help=f"Initial Douglas-Peucker epsilon as a fraction of perimeter "
             f"(default {DEFAULT_EPSILON})."
    )

    args = parser.parse_args()

    root = Path(args.directory).expanduser()

    if not root.exists():
        raise SystemExit(f"Directory does not exist: {root}")

    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    batches = find_batches(root)

    if not batches:
        raise SystemExit(
            f"No batch001/batch002/... folders found beneath:\n{root}"
        )

    print("=" * 72)
    print("CVM Douglas-Peucker automation")
    print("=" * 72)
    print(f"Target:  {root}")
    print(f"Batches: {len(batches)}")
    print(f"Epsilon: {args.epsilon}")
    print()

    rows = []
    total_images = 0

    for batch in batches:
        images = find_original_images(batch)

        print(f"{batch.name}: {len(images)} original BMPs")

        for i, image_path in enumerate(images, 1):
            print(
                f"  [{i}/{len(images)}] {image_path.name}",
                end=" ... ",
                flush=True
            )

            row = process_image(
                image_path,
                batch,
                args.epsilon
            )

            rows.append(row)
            total_images += 1

            print(row["status"])

    # Parent-level CSV: the only CSV produced by the production pipeline.
    csv_path = root / "cvm_measurements.csv"

    rows.sort(
        key=lambda r: (
            r["batch"].lower(),
            r["filename"].lower()
        )
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_COLUMNS,
            extrasaction="ignore"
        )
        writer.writeheader()

        for row in rows:
            writer.writerow({
                col: row.get(col, "")
                for col in CSV_COLUMNS
            })

    ok = sum(r["status"] == "ok" for r in rows)
    partial = sum(r["status"] == "partial" for r in rows)
    errors = sum(r["status"] == "error" for r in rows)

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Images processed : {total_images}")
    print(f"Complete         : {ok}")
    print(f"Partial          : {partial}")
    print(f"Errors           : {errors}")
    print(f"CSV              : {csv_path}")
    print()
    print("Overlays are in each batch's:")
    print("    overlaid_Douglas-Peucker/")
    print()


if __name__ == "__main__":
    main()
