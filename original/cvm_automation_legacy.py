#!/usr/bin/env python3
"""
CVM AUTOMATION — DOUGLAS-PEUCKER ONLY
=====================================

This is a standalone replacement for the original CVM batch driver. It
processes the current flat final_dataset structure:

    final_dataset/
        *.bmp                         original radiographs
        masks/
            *.bmp                     vertebral masks
            c2/*.bmp, c3/*.bmp, c4/*.bmp   (also supported)

It writes:

    final_dataset/
        overlaid_Douglas-Peucker/
            *.bmp

    <parent of final_dataset>/
        cvm_measurements.csv

IMPORTANT — DOME GEOMETRY
-------------------------
The original cvm_generate.py documentation describes the yellow measurement
as:

    inferior-border chord (between the two inferior Douglas-Peucker corners)
    + the perpendicular up to the endplate dome apex

and defines dome_height_px as the inferior-endplate concavity depth.

Therefore the dome is NOT constrained to the midpoint of the inferior edge.

This implementation searches the ENTIRE inferior Douglas-Peucker chord. For
each point of the chord, it examines the actual annotated mask contour and
finds the perpendicular displacement from the chord to the inferior
endplate. The maximum perpendicular displacement is the dome height.

The overlay draws:
    grey dashed = original vertebral-body mask
    green       = Douglas-Peucker quadrilateral
    magenta     = posterior vertebral-body height
    yellow      = entire inferior quadrilateral chord
    cyan        = perpendicular at the DEEPEST dome location

This follows the geometry documented in the original cvm_generate.py:
the inferior chord plus the perpendicular to the dome apex. The supplied
original documentation also states that C2/C3/C4 are measured with a
posterior-height denominator and that >= 0.10 was used by the original
driver; this script uses the user's requested strict rule: > 0.10.

RATIO CONVENTION
----------------
    C2 = C2 dome / C3 posterior vertebral-body height
    C3 = C3 dome / C3 posterior vertebral-body height
    C4 = C4 dome / C4 posterior vertebral-body height

DOMING
------
    ratio > 0.10 -> doming
    ratio <= 0.10 -> no doming

AUTOMATED CVM
-------------
    no doming C2/C3/C4 -> CVM1
    doming C2 only      -> CVM2
    doming C2 + C3     -> CVM3
    doming C2 + C3 + C4 -> CVM4-6

Any other complete pattern is reported as "atypical". Missing/failed
measurements are also reported as "atypical" rather than being forced into
a CVM class.

ANACONDA PROMPT
---------------
Create the environment:

    conda create -n cvm_automation python=3.11
    conda activate cvm_automation

Install dependencies:

    pip install -r requirements.txt

Run on the final dataset:

    python cvm_automation.py --directory "\\wnresearch\Drobo\Vishal_Graham\ML Review\Spine\CVM_annotation\semiautomated\final_dataset"

Optional Douglas-Peucker starting epsilon:

    python cvm_automation.py --directory "PATH_TO_FINAL_DATASET" --epsilon 0.02

The script automatically searches a range of epsilon values to obtain a
four-corner Douglas-Peucker polygon. --epsilon controls which four-corner
solution is preferred.

DEPENDENCIES
------------
numpy
opencv-python
"""

import argparse
import csv
import math
import re
from pathlib import Path

import cv2
import numpy as np


VERTEBRAE = ("C2", "C3", "C4")
DEFAULT_EPSILON = 0.02
MIN_MASK_AREA = 20
DOMING_THRESHOLD = 0.10


# ============================================================================
# BASIC GEOMETRY
# ============================================================================

def distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(b - a))


def polygon_area(points):
    p = np.asarray(points, dtype=float)
    return float(
        abs(np.cross(p[:, 0], np.roll(p[:, 1], -1)).sum()) / 2.0
    )


def cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def order_quad(points):
    """
    Order four image-space vertices as:

        top-left, top-right, bottom-right, bottom-left

    This ordering is subsequently used to define anatomical edges.
    """
    p = np.asarray(points, dtype=float)

    if p.shape != (4, 2):
        raise ValueError("Quadrilateral must contain exactly four vertices")

    s = p[:, 0] + p[:, 1]
    d = p[:, 0] - p[:, 1]

    q = np.array([
        p[np.argmin(s)],
        p[np.argmax(d)],
        p[np.argmax(s)],
        p[np.argmin(d)],
    ], dtype=float)

    if len({tuple(x) for x in q}) != 4:
        center = p.mean(axis=0)
        angles = np.arctan2(
            p[:, 1] - center[1],
            p[:, 0] - center[0]
        )
        q = p[np.argsort(angles)]

        # Rotate so the first point is the upper-left-most point.
        q = np.roll(
            q,
            -int(np.argmin(q[:, 0] + q[:, 1])),
            axis=0
        )

    return q


def classify_edges(q):
    """
    Anatomical edge assignment used by the original CVM visualization:

        superior = upper edge
        anterior = right edge
        inferior = lower edge
        posterior = left edge

    Image coordinates have x increasing to the right and y increasing
    downward.
    """
    q = np.asarray(q, dtype=float)

    edges = {
        "superior": (q[0], q[1]),
        "anterior": (q[1], q[2]),
        "inferior": (q[2], q[3]),
        "posterior": (q[3], q[0]),
    }

    # Correct obvious global flips in strongly tilted images.
    superior_y = np.mean(edges["superior"], axis=0)[1]
    inferior_y = np.mean(edges["inferior"], axis=0)[1]

    if superior_y > inferior_y:
        edges["superior"], edges["inferior"] = (
            edges["inferior"],
            edges["superior"],
        )

    posterior_x = np.mean(edges["posterior"], axis=0)[0]
    anterior_x = np.mean(edges["anterior"], axis=0)[0]

    if posterior_x > anterior_x:
        edges["posterior"], edges["anterior"] = (
            edges["anterior"],
            edges["posterior"],
        )

    return edges


# ============================================================================
# MASKS
# ============================================================================

def read_binary_mask(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Cannot read mask: {path}")

    mask = (img > 0).astype(np.uint8)

    # Retain the largest connected foreground component.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    if n > 1:
        largest = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )
        mask = (labels == largest).astype(np.uint8)

    if int(mask.sum()) < MIN_MASK_AREA:
        raise ValueError("Mask contains too few foreground pixels")

    return mask


def normalized_name(path_or_name):
    """
    Normalize an image/mask stem for matching.

    C2/C3/C4 and mask tokens are removed because masks may be named either:

        scan_C2.bmp
        scan_mask_C2.bmp
        masks/C2/scan.bmp

    while the original is:

        scan.bmp
    """
    s = Path(path_or_name).stem.lower()
    s = re.sub(r"[\s\-]+", "_", s)

    s = re.sub(
        r"(^|_)(mask|masks)(?=_|$)",
        "_",
        s
    )

    s = re.sub(
        r"(^|_)(c2|c3|c4)(?=_|$)",
        "_",
        s
    )

    s = re.sub(r"_+", "_", s).strip("_")

    return s


def vertebra_from_path(path):
    """
    Determine C2/C3/C4 from either:
        filename
    or:
        parent folder
    """
    stem = Path(path).stem.lower()

    match = re.search(
        r"(?:^|[_\-\s])(c[234])(?:[_\-\s.]|$)",
        stem
    )

    if match:
        return match.group(1).upper()

    for part in reversed(Path(path).parts):
        if part.lower() in ("c2", "c3", "c4"):
            return part.upper()

    return None


def build_mask_index(mask_root):
    """
    Index all BMP masks once rather than recursively searching the mask
    directory for every radiograph.
    """
    index = {}

    for path in sorted(mask_root.rglob("*.bmp")):
        if not path.is_file():
            continue

        v = vertebra_from_path(path)

        if v not in VERTEBRAE:
            continue

        key = normalized_name(path)

        index.setdefault((key, v), []).append(path)

    return index


def find_masks_for_image(image_path, mask_index):
    """
    Match the original radiograph with its C2/C3/C4 masks.

    Exact normalized filename matching is preferred. If there are duplicate
    candidates, a mask in an explicit C2/C3/C4 directory is preferred.
    """
    image_key = normalized_name(image_path)
    result = {}

    for v in VERTEBRAE:
        candidates = mask_index.get((image_key, v), [])

        if candidates:
            # Prefer the candidate whose parent directory is literally C2/C3/C4.
            explicit = [
                p for p in candidates
                if p.parent.name.lower() == v.lower()
            ]

            result[v] = explicit[0] if explicit else candidates[0]

    # Conservative fallback for naming variants.
    if len(result) < 3:
        for (key, v), candidates in mask_index.items():
            if v in result:
                continue

            if image_key in key or key in image_key:
                explicit = [
                    p for p in candidates
                    if p.parent.name.lower() == v.lower()
                ]
                result[v] = explicit[0] if explicit else candidates[0]

    return result


# ============================================================================
# DOUGLAS-PEUCKER QUADRILATERAL
# ============================================================================

def find_dp_quad(mask, epsilon_start=DEFAULT_EPSILON):
    """
    Douglas-Peucker simplification of the mask convex hull to exactly four
    vertices.

    Only Douglas-Peucker is used. No quadrilateral-fitter and no
    min-area-rectangle alternative is used.
    """
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        raise ValueError("No contour found")

    contour = max(
        contours,
        key=cv2.contourArea
    )

    if cv2.contourArea(contour) < MIN_MASK_AREA:
        raise ValueError("Contour area too small")

    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)

    if perimeter <= 0:
        raise ValueError("Invalid contour perimeter")

    candidates = []
    seen = set()

    # Search for exact four-vertex DP solutions.
    fractions = np.unique(np.concatenate([
        np.linspace(
            max(0.0005, epsilon_start * 0.15),
            max(0.01, epsilon_start * 6.0),
            300
        ),
        np.linspace(0.001, 0.20, 500),
    ]))

    for fraction in fractions:
        approx = cv2.approxPolyDP(
            hull,
            float(fraction * perimeter),
            True
        )

        if len(approx) != 4:
            continue

        q = approx[:, 0, :].astype(float)
        key = tuple(np.round(q.flatten(), 3))

        if key not in seen:
            seen.add(key)
            candidates.append(q)

    if not candidates:
        raise ValueError(
            "Douglas-Peucker could not produce exactly four corners"
        )

    target_epsilon = epsilon_start * perimeter

    q = min(
        candidates,
        key=lambda x: abs(
            cv2.arcLength(
                x.astype(np.float32).reshape(-1, 1, 2),
                True
            ) - target_epsilon
        )
    )

    return order_quad(q), contour


# ============================================================================
# ORIGINAL-STYLE DOME MEASUREMENT
# ============================================================================

def measure_dome(mask, contour, inferior_edge, quad):
    """
    Measure the deepest inferior-endplate concavity over the ENTIRE inferior
    Douglas-Peucker edge.

    This implements the geometry documented in the original cvm_generate.py:

        inferior-border chord + perpendicular to the dome apex

    Algorithm:
        1. Treat the inferior DP edge as the chord.
        2. For every actual mask contour point, calculate its perpendicular
           distance to that chord.
        3. Keep points whose perpendicular projection falls on the chord.
        4. Keep only points on the vertebral-body interior side.
        5. Restrict the search to a band near the inferior edge so the
           superior endplate cannot become the dome.
        6. Select the contour point with MAXIMUM perpendicular distance.
        7. The perpendicular foot on the inferior DP edge is the exact dome
           base; the contour point is the exact dome apex.

    This means the dome location is determined by the deepest point anywhere
    along the inferior chord — NOT the midpoint.
    """
    a = np.asarray(inferior_edge[0], dtype=float)
    b = np.asarray(inferior_edge[1], dtype=float)

    v = b - a
    L = np.linalg.norm(v)

    if L <= 0:
        return 0.0, None, None

    centroid = np.asarray(quad, dtype=float).mean(axis=0)

    # Determine which side of the chord contains the vertebral body.
    centroid_cross = cross2(v, centroid - a)

    if abs(centroid_cross) < 1e-8:
        return 0.0, None, None

    points = contour[:, 0, :].astype(float)

    # Signed perpendicular displacement from the chord.
    cross_values = np.array(
        [cross2(v, p - a) for p in points],
        dtype=float
    )

    perpendicular_distance = (
        np.abs(cross_values) / L
    )

    # Position of each contour point's perpendicular projection along the
    # inferior chord, expressed as t=0..1.
    t = np.dot(
        points - a,
        v
    ) / (L * L)

    # Only contour points on the vertebral-body side of the chord.
    interior = (
        np.sign(cross_values) == np.sign(centroid_cross)
    )

    # Only contour points whose perpendicular foot lies on the chord.
    on_chord = (
        (t >= 0.0) &
        (t <= 1.0)
    )

    # Limit search to the inferior neighborhood. This prevents the superior
    # endplate from being selected when the vertebral body is tilted.
    body_height = max(
        distance(quad[0], quad[3]),
        distance(quad[1], quad[2]),
        1.0
    )

    near_inferior = (
        perpendicular_distance <= 0.35 * body_height
    )

    valid = (
        interior &
        on_chord &
        near_inferior
    )

    if not np.any(valid):
        return 0.0, None, None

    valid_indices = np.where(valid)[0]

    # THE key operation: choose the deepest perpendicular contour point over
    # the complete inferior edge.
    best_idx = valid_indices[
        np.argmax(perpendicular_distance[valid])
    ]

    dome_height = float(
        perpendicular_distance[best_idx]
    )

    apex = points[best_idx]

    # Exact perpendicular foot on the inferior DP chord.
    t_best = float(t[best_idx])
    foot = a + t_best * v

    return dome_height, foot, apex


# ============================================================================
# MEASUREMENT
# ============================================================================

def measure_mask(mask_path, vertebra, epsilon):
    mask = read_binary_mask(mask_path)

    quad, contour = find_dp_quad(
        mask,
        epsilon
    )

    edges = classify_edges(quad)

    dome, dome_base, dome_apex = measure_dome(
        mask,
        contour,
        edges["inferior"],
        quad
    )

    anterior = distance(
        *edges["anterior"]
    )

    posterior = distance(
        *edges["posterior"]
    )

    superior = distance(
        *edges["superior"]
    )

    inferior = distance(
        *edges["inferior"]
    )

    mean_width = (
        superior + inferior
    ) / 2.0

    mean_height = (
        anterior + posterior
    ) / 2.0

    return {
        "mask": mask,
        "contour": contour,
        "quad": quad,
        "edges": edges,

        "dome_px": dome,
        "dome_base": dome_base,
        "dome_apex": dome_apex,

        "anterior_px": anterior,
        "posterior_px": posterior,
        "superior_px": superior,
        "inferior_px": inferior,

        "width_mean_px": mean_width,
        "height_mean_px": mean_height,

        "width_height_ratio": (
            mean_width / mean_height
            if mean_height > 0 else np.nan
        ),

        "dome_height_ratio": (
            dome / mean_height
            if mean_height > 0 else np.nan
        ),

        "area_px2": polygon_area(quad),

        "diagonal_1_px": distance(
            quad[0], quad[2]
        ),

        "diagonal_2_px": distance(
            quad[1], quad[3]
        ),

        "mask_area_px2": float(
            mask.sum()
        ),
    }


# ============================================================================
# OVERLAY
# ============================================================================

def draw_dashed_polyline(
    image,
    points,
    color=(150, 150, 150),
    thickness=1,
    dash=8,
    gap=7
):
    """
    Draw a faint dashed closed contour.
    """
    out = image.copy()
    pts = np.asarray(points, dtype=float)

    for i in range(len(pts)):
        p0 = pts[i]
        p1 = pts[(i + 1) % len(pts)]

        vec = p1 - p0
        length = np.linalg.norm(vec)

        if length <= 0:
            continue

        unit = vec / length
        position = 0.0

        while position < length:
            end = min(
                position + dash,
                length
            )

            a = p0 + unit * position
            b = p0 + unit * end

            cv2.line(
                out,
                tuple(np.round(a).astype(int)),
                tuple(np.round(b).astype(int)),
                color,
                thickness,
                cv2.LINE_AA
            )

            position += dash + gap

    return out


def draw_text(image, text, point, scale=0.5):
    x, y = np.round(point).astype(int)

    cv2.putText(
        image,
        text,
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        3,
        cv2.LINE_AA
    )

    cv2.putText(
        image,
        text,
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


def make_overlay(original, results):
    if original.ndim == 2:
        overlay = cv2.cvtColor(
            original,
            cv2.COLOR_GRAY2BGR
        )
    else:
        overlay = original.copy()

    # BGR.
    MASK_GREY = (150, 150, 150)
    QUAD_GREEN = (0, 255, 0)
    POSTERIOR_MAGENTA = (255, 0, 255)
    INFERIOR_YELLOW = (0, 255, 255)
    DOME_CYAN = (255, 255, 0)

    for v in VERTEBRAE:
        r = results.get(v)

        if r is None:
            continue

        # ---------------------------------------------------------------
        # Faint dashed original mask outline.
        # ---------------------------------------------------------------
        contours, _ = cv2.findContours(
            r["mask"].astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        for contour in contours:
            if cv2.contourArea(contour) >= MIN_MASK_AREA:
                overlay = draw_dashed_polyline(
                    overlay,
                    contour[:, 0, :],
                    MASK_GREY,
                    thickness=1,
                    dash=7,
                    gap=7
                )

        # ---------------------------------------------------------------
        # Douglas-Peucker quadrilateral.
        # ---------------------------------------------------------------
        q = np.round(
            r["quad"]
        ).astype(int)

        cv2.polylines(
            overlay,
            [q.reshape(-1, 1, 2)],
            True,
            QUAD_GREEN,
            2,
            cv2.LINE_AA
        )

        # ---------------------------------------------------------------
        # Posterior height.
        # ---------------------------------------------------------------
        p0, p1 = [
            np.round(x).astype(int)
            for x in r["edges"]["posterior"]
        ]

        cv2.line(
            overlay,
            tuple(p0),
            tuple(p1),
            POSTERIOR_MAGENTA,
            2,
            cv2.LINE_AA
        )

        # ---------------------------------------------------------------
        # Inferior DP chord.
        # ---------------------------------------------------------------
        i0, i1 = [
            np.round(x).astype(int)
            for x in r["edges"]["inferior"]
        ]

        cv2.line(
            overlay,
            tuple(i0),
            tuple(i1),
            INFERIOR_YELLOW,
            2,
            cv2.LINE_AA
        )

        # ---------------------------------------------------------------
        # Deepest dome.
        #
        # The cyan line is drawn from the exact perpendicular foot on the
        # inferior DP edge to the exact contour apex selected by measure_dome.
        # ---------------------------------------------------------------
        if (
            r["dome_base"] is not None and
            r["dome_apex"] is not None
        ):
            base = np.asarray(
                r["dome_base"],
                dtype=float
            )

            apex = np.asarray(
                r["dome_apex"],
                dtype=float
            )

            cv2.line(
                overlay,
                tuple(np.round(base).astype(int)),
                tuple(np.round(apex).astype(int)),
                DOME_CYAN,
                2,
                cv2.LINE_AA
            )

            cv2.circle(
                overlay,
                tuple(np.round(base).astype(int)),
                3,
                INFERIOR_YELLOW,
                -1
            )

            cv2.circle(
                overlay,
                tuple(np.round(apex).astype(int)),
                4,
                DOME_CYAN,
                -1
            )

        # Vertebral label.
        center = r["quad"].mean(axis=0)

        draw_text(
            overlay,
            v,
            center,
            scale=0.60
        )

    return overlay


# ============================================================================
# CSV
# ============================================================================

CSV_COLUMNS = [
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
    "C2_is_doming",

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
    "C3_is_doming",

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
    "C4_is_doming",

    "C2_doming_ratio_C2_dome_over_C3_posterior",
    "C3_doming_ratio_C3_dome_over_C3_posterior",
    "C4_doming_ratio_C4_dome_over_C4_posterior",

    "doming_pattern",
    "predicted_CVM",
]


def fmt(value):
    if value is None:
        return ""

    try:
        if np.isnan(value):
            return ""
    except (TypeError, ValueError):
        pass

    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"

    return str(value)


def make_row(filename, results, status, message):
    row = {
        "filename": filename,
        "status": status,
        "message": message,
    }

    for v in VERTEBRAE:
        r = results.get(v)

        if r is None:
            for col in CSV_COLUMNS:
                if col.startswith(v + "_"):
                    row[col] = ""
            continue

        row[f"{v}_anterior_px"] = fmt(
            r["anterior_px"]
        )

        row[f"{v}_posterior_px"] = fmt(
            r["posterior_px"]
        )

        row[f"{v}_superior_px"] = fmt(
            r["superior_px"]
        )

        row[f"{v}_inferior_px"] = fmt(
            r["inferior_px"]
        )

        row[f"{v}_dome_px"] = fmt(
            r["dome_px"]
        )

        row[f"{v}_width_mean_px"] = fmt(
            r["width_mean_px"]
        )

        row[f"{v}_height_mean_px"] = fmt(
            r["height_mean_px"]
        )

        row[f"{v}_width_height_ratio"] = fmt(
            r["width_height_ratio"]
        )

        row[f"{v}_dome_height_ratio"] = fmt(
            r["dome_height_ratio"]
        )

        row[f"{v}_area_px2"] = fmt(
            r["area_px2"]
        )

        row[f"{v}_diagonal_1_px"] = fmt(
            r["diagonal_1_px"]
        )

        row[f"{v}_diagonal_2_px"] = fmt(
            r["diagonal_2_px"]
        )

        row[f"{v}_mask_area_px2"] = fmt(
            r["mask_area_px2"]
        )

    c2 = results.get("C2")
    c3 = results.get("C3")
    c4 = results.get("C4")

    # ---------------------------------------------------------------
    # Official ratios.
    # ---------------------------------------------------------------
    c2_ratio = None
    c3_ratio = None
    c4_ratio = None

    if (
        c2 is not None and
        c3 is not None and
        c3["posterior_px"] > 0
    ):
        c2_ratio = (
            c2["dome_px"] /
            c3["posterior_px"]
        )

    if (
        c3 is not None and
        c3["posterior_px"] > 0
    ):
        c3_ratio = (
            c3["dome_px"] /
            c3["posterior_px"]
        )

    if (
        c4 is not None and
        c4["posterior_px"] > 0
    ):
        c4_ratio = (
            c4["dome_px"] /
            c4["posterior_px"]
        )

    row[
        "C2_doming_ratio_C2_dome_over_C3_posterior"
    ] = fmt(c2_ratio)

    row[
        "C3_doming_ratio_C3_dome_over_C3_posterior"
    ] = fmt(c3_ratio)

    row[
        "C4_doming_ratio_C4_dome_over_C4_posterior"
    ] = fmt(c4_ratio)

    # ---------------------------------------------------------------
    # Doming threshold: STRICTLY > 0.10.
    # ---------------------------------------------------------------
    states = {}

    states["C2"] = (
        None
        if c2_ratio is None
        else c2_ratio > DOMING_THRESHOLD
    )

    states["C3"] = (
        None
        if c3_ratio is None
        else c3_ratio > DOMING_THRESHOLD
    )

    states["C4"] = (
        None
        if c4_ratio is None
        else c4_ratio > DOMING_THRESHOLD
    )

    for v in VERTEBRAE:
        state = states[v]

        if state is None:
            row[f"{v}_is_doming"] = ""
        else:
            row[f"{v}_is_doming"] = (
                "yes" if state else "no"
            )

    # ---------------------------------------------------------------
    # Predicted CVM.
    #
    # Valid developmental sequence:
    #   000 = CVM1
    #   100 = CVM2
    #   110 = CVM3
    #   111 = CVM4-6
    #
    # Everything else = atypical.
    # ---------------------------------------------------------------
    if all(
        states[v] is not None
        for v in VERTEBRAE
    ):
        pattern = (
            int(states["C2"]),
            int(states["C3"]),
            int(states["C4"]),
        )

        pattern_string = "".join(
            str(x) for x in pattern
        )

        mapping = {
            (0, 0, 0): "CVM1",
            (1, 0, 0): "CVM2",
            (1, 1, 0): "CVM3",
            (1, 1, 1): "CVM4-6",
        }

        predicted = mapping.get(
            pattern,
            "atypical"
        )
    else:
        pattern_string = ""
        predicted = "atypical"

    row["doming_pattern"] = pattern_string
    row["predicted_CVM"] = predicted

    return row


# ============================================================================
# IMAGE PROCESSING
# ============================================================================

def process_image(
    image_path,
    mask_index,
    output_dir,
    epsilon
):
    original = cv2.imread(
        str(image_path),
        cv2.IMREAD_UNCHANGED
    )

    if original is None:
        return make_row(
            image_path.name,
            {},
            "error",
            "Could not read original BMP"
        )

    mask_map = find_masks_for_image(
        image_path,
        mask_index
    )

    results = {}
    errors = []

    for v in VERTEBRAE:
        mask_path = mask_map.get(v)

        if mask_path is None:
            errors.append(
                f"{v} mask not found"
            )
            continue

        try:
            results[v] = measure_mask(
                mask_path,
                v,
                epsilon
            )
        except Exception as exc:
            errors.append(
                f"{v}: {exc}"
            )

    overlay = make_overlay(
        original,
        results
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir /
        image_path.name
    )

    if not cv2.imwrite(
        str(output_path),
        overlay
    ):
        errors.append(
            "Could not write overlay"
        )

    if errors:
        status = (
            "partial"
            if results
            else "error"
        )
        message = "; ".join(errors)
    else:
        status = "ok"
        message = ""

    return make_row(
        image_path.name,
        results,
        status,
        message
    )


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "CVM Douglas-Peucker automation for a flat final_dataset."
        )
    )

    parser.add_argument(
        "--directory",
        required=True,
        help=(
            "final_dataset directory containing original BMPs and "
            "the masks folder"
        )
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help=(
            "Preferred Douglas-Peucker epsilon as a fraction of "
            "convex-hull perimeter. Default: 0.02"
        )
    )

    args = parser.parse_args()

    dataset = Path(
        args.directory
    ).expanduser()

    if not dataset.is_dir():
        raise SystemExit(
            f"ERROR: directory does not exist:\n{dataset}"
        )

    mask_root = (
        dataset /
        "masks"
    )

    if not mask_root.is_dir():
        raise SystemExit(
            f"ERROR: masks folder does not exist:\n{mask_root}"
        )

    output_dir = (
        dataset /
        "overlaid_Douglas-Peucker"
    )

    # CSV belongs in the parent directory.
    csv_path = (
        dataset.parent /
        "cvm_measurements.csv"
    )

    # Only original BMPs directly inside final_dataset.
    # The masks and overlay directories are therefore never interpreted as
    # radiograph input.
    image_files = sorted(
        [
            p for p in dataset.glob("*.bmp")
            if p.is_file()
        ],
        key=lambda p: p.name.lower()
    )

    if not image_files:
        raise SystemExit(
            "ERROR: no BMP files found directly inside:\n"
            f"{dataset}"
        )

    print("=" * 78)
    print("CVM DOUGLAS-PEUCKER AUTOMATION")
    print("=" * 78)
    print(f"Dataset       : {dataset}")
    print(f"Original BMPs : {len(image_files)}")
    print(f"Masks         : {mask_root}")
    print(f"Overlays      : {output_dir}")
    print(f"CSV           : {csv_path}")
    print(f"DP epsilon    : {args.epsilon}")
    print(f"Doming rule   : ratio > {DOMING_THRESHOLD:.2f}")
    print()
    print(
        "Dome geometry : deepest perpendicular concavity over the "
        "ENTIRE inferior DP edge"
    )
    print()
    print(
        "CVM mapping   : 000=CVM1, 100=CVM2, 110=CVM3, "
        "111=CVM4-6; all other patterns=atypical"
    )
    print()

    print("Indexing masks...")
    mask_index = build_mask_index(
        mask_root
    )

    print(
        f"Indexed {sum(len(x) for x in mask_index.values())} "
        f"vertebral masks."
    )
    print()

    rows = []

    for i, image_path in enumerate(
        image_files,
        start=1
    ):
        print(
            f"[{i}/{len(image_files)}] "
            f"{image_path.name}",
            end=" ... ",
            flush=True
        )

        row = process_image(
            image_path,
            mask_index,
            output_dir,
            args.epsilon
        )

        rows.append(row)

        print(
            row["status"]
        )

        if row["message"]:
            print(
                f"    {row['message']}"
            )

        if row["predicted_CVM"]:
            print(
                f"    predicted CVM: "
                f"{row['predicted_CVM']} "
                f"(doming pattern {row['doming_pattern']})"
            )

    rows.sort(
        key=lambda r: r["filename"].lower()
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

    n_ok = sum(
        r["status"] == "ok"
        for r in rows
    )

    n_partial = sum(
        r["status"] == "partial"
        for r in rows
    )

    n_error = sum(
        r["status"] == "error"
        for r in rows
    )

    print()
    print("=" * 78)
    print("DONE")
    print("=" * 78)
    print(f"Total images : {len(rows)}")
    print(f"Complete     : {n_ok}")
    print(f"Partial      : {n_partial}")
    print(f"Errors       : {n_error}")
    print(f"CSV          : {csv_path}")
    print(f"Overlays     : {output_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
