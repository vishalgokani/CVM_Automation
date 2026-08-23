#!/usr/bin/env python3
"""
Post-process C2-C4 segmentation masks into CVM doming measurements.

This script assumes upstream segmentation has already produced one binary mask
per vertebra. It fits a Douglas-Peucker quadrilateral to each mask, measures the
deepest inferior-endplate concavity, draws review overlays, and writes a CSV
with automated CVM class predictions.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np


VERTEBRAE = ("C2", "C3", "C4")
DEFAULT_EPSILON = 0.02
DEFAULT_MIN_MASK_AREA = 20
DEFAULT_INFERIOR_ENDPOINT_TRIM = 0.04
DEFAULT_ZOOM_SCALE = 3.0
DEFAULT_ZOOM_MARGIN = 35
DOMING_THRESHOLD = 0.10


CSV_COLUMNS = [
    "filename",
    "status",
    "message",
    "C2_mask_path",
    "C2_posterior_height_px",
    "C2_inferior_chord_px",
    "C2_dome_height_px",
    "C2_dome_base_x",
    "C2_dome_base_y",
    "C2_dome_apex_x",
    "C2_dome_apex_y",
    "C2_doming_ratio_C2_dome_over_C3_posterior",
    "C2_is_doming",
    "C3_mask_path",
    "C3_posterior_height_px",
    "C3_inferior_chord_px",
    "C3_dome_height_px",
    "C3_dome_base_x",
    "C3_dome_base_y",
    "C3_dome_apex_x",
    "C3_dome_apex_y",
    "C3_doming_ratio_C3_dome_over_C3_posterior",
    "C3_is_doming",
    "C4_mask_path",
    "C4_posterior_height_px",
    "C4_inferior_chord_px",
    "C4_dome_height_px",
    "C4_dome_base_x",
    "C4_dome_base_y",
    "C4_dome_apex_x",
    "C4_dome_apex_y",
    "C4_doming_ratio_C4_dome_over_C4_posterior",
    "C4_is_doming",
    "doming_pattern",
    "predicted_CVM",
]


def fmt(value: Any) -> str:
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


def as_points(points: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    arr = np.squeeze(arr)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2), got {arr.shape}")
    return arr


def distance(a: Any, b: Any) -> float:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    return float(np.linalg.norm(b_arr - a_arr))


def cross2(a: Any, b: Any) -> float:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != (2,) or b_arr.shape != (2,):
        raise ValueError(
            f"cross2 expects two 2D vectors, got {a_arr.shape} and {b_arr.shape}"
        )
    return float(a_arr[0] * b_arr[1] - a_arr[1] * b_arr[0])


def polygon_area(points: Any) -> float:
    p = as_points(points, name="polygon")
    if len(p) < 3:
        return 0.0
    x = p[:, 0]
    y = p[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def order_quad(points: Any) -> np.ndarray:
    p = as_points(points, name="quadrilateral")
    if p.shape != (4, 2):
        raise ValueError(f"Quadrilateral must contain exactly four vertices, got {p.shape}")

    center = p.mean(axis=0)
    angles = np.arctan2(p[:, 1] - center[1], p[:, 0] - center[0])
    q = p[np.argsort(angles)]

    signed_area = 0.5 * (
        np.dot(q[:, 0], np.roll(q[:, 1], -1))
        - np.dot(q[:, 1], np.roll(q[:, 0], -1))
    )
    if signed_area < 0:
        q = q[::-1]

    q = np.roll(q, -int(np.argmin(q[:, 0] + q[:, 1])), axis=0)
    return q.astype(float)


def classify_edges(quad: Any) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Assign anatomical edges from image-space position.

    The inferior border is the edge between the two bottom-most quadrilateral
    corners. This is intentionally stricter than choosing the edge with the
    lowest midpoint, because a long anterior/posterior sidewall can have a low
    midpoint in tilted vertebrae and should not become the inferior chord.
    """
    q = order_quad(quad)
    edges = [(q[i], q[(i + 1) % 4]) for i in range(4)]

    bottom_vertices = set(np.argsort(q[:, 1])[-2:].tolist())
    top_vertices = set(np.argsort(q[:, 1])[:2].tolist())

    inferior_i = None
    superior_i = None
    for i in range(4):
        pair = {i, (i + 1) % 4}
        if pair == bottom_vertices:
            inferior_i = i
        if pair == top_vertices:
            superior_i = i

    centers = np.array([(a + b) / 2.0 for a, b in edges])
    if inferior_i is None:
        inferior_i = int(np.argmax(np.minimum(q[:, 1], np.roll(q[:, 1], -1))))
    if superior_i is None:
        superior_i = int(np.argmin(np.maximum(q[:, 1], np.roll(q[:, 1], -1))))

    remaining = [i for i in range(4) if i not in (superior_i, inferior_i)]
    posterior_i = min(remaining, key=lambda i: centers[i, 0])
    anterior_i = max(remaining, key=lambda i: centers[i, 0])

    return {
        "superior": edges[superior_i],
        "anterior": edges[anterior_i],
        "inferior": edges[inferior_i],
        "posterior": edges[posterior_i],
    }


def read_binary_mask(path: Path, min_area: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read mask: {path}")

    mask = (img > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest).astype(np.uint8)

    if int(mask.sum()) < min_area:
        raise ValueError("Mask contains too few foreground pixels")
    return mask


def normalized_name(path_or_name: Path | str) -> str:
    stem = Path(path_or_name).stem.lower()
    stem = re.sub(r"[\s\-]+", "_", stem)
    stem = re.sub(r"(^|_)(mask|masks|seg|segmentation)(?=_|$)", "_", stem)
    stem = re.sub(r"(^|_)(c2|c3|c4)(?=_|$)", "_", stem)
    return re.sub(r"_+", "_", stem).strip("_")


def vertebra_from_path(path: Path) -> str | None:
    stem = path.stem.lower()
    match = re.search(r"(?:^|[_\-\s])(c[234])(?:[_\-\s.]|$)", stem)
    if match:
        return match.group(1).upper()

    for part in reversed(path.parts):
        if part.lower() in ("c2", "c3", "c4"):
            return part.upper()
    return None


def build_mask_index(mask_root: Path) -> dict[tuple[str, str], list[Path]]:
    index: dict[tuple[str, str], list[Path]] = {}
    for path in sorted(mask_root.rglob("*.bmp")):
        if not path.is_file():
            continue
        vertebra = vertebra_from_path(path)
        if vertebra not in VERTEBRAE:
            continue
        index.setdefault((normalized_name(path), vertebra), []).append(path)
    return index


def find_masks_for_image(
    image_path: Path,
    mask_index: dict[tuple[str, str], list[Path]],
) -> dict[str, Path]:
    image_key = normalized_name(image_path)
    result: dict[str, Path] = {}

    for vertebra in VERTEBRAE:
        candidates = mask_index.get((image_key, vertebra), [])
        if candidates:
            explicit = [p for p in candidates if p.parent.name.lower() == vertebra.lower()]
            result[vertebra] = explicit[0] if explicit else candidates[0]

    if len(result) == len(VERTEBRAE):
        return result

    for (key, vertebra), candidates in mask_index.items():
        if vertebra in result:
            continue
        if image_key in key or key in image_key:
            explicit = [p for p in candidates if p.parent.name.lower() == vertebra.lower()]
            result[vertebra] = explicit[0] if explicit else candidates[0]

    return result


def find_dp_quad(
    mask: np.ndarray,
    epsilon_start: float,
    min_area: int,
) -> tuple[np.ndarray, np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found")

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        raise ValueError("Contour area too small")

    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    if perimeter <= 0:
        raise ValueError("Invalid contour perimeter")

    fractions = np.unique(
        np.concatenate(
            [
                np.linspace(max(0.0005, epsilon_start * 0.15), max(0.01, epsilon_start * 6.0), 300),
                np.linspace(0.001, 0.20, 500),
            ]
        )
    )

    candidates: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for fraction in fractions:
        approx = cv2.approxPolyDP(hull, float(fraction * perimeter), True)
        if len(approx) != 4:
            continue
        q = approx[:, 0, :].astype(float)
        key = tuple(np.round(q.flatten(), 3))
        if key not in seen:
            seen.add(key)
            candidates.append(q)

    if not candidates:
        raise ValueError("Douglas-Peucker could not produce exactly four corners")

    target = epsilon_start * perimeter
    best = min(
        candidates,
        key=lambda q: abs(
            cv2.arcLength(q.astype(np.float32).reshape(-1, 1, 2), True) - target
        ),
    )
    return order_quad(best), contour


def cyclic_arc(points: np.ndarray, start: int, end: int) -> np.ndarray:
    if start <= end:
        return points[start : end + 1]
    return np.vstack([points[start:], points[: end + 1]])


def inferior_contour_arc(
    contour_points: np.ndarray,
    inferior_edge: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """
    Return the actual mask-contour arc between the two inferior quadrilateral
    corners.

    The full contour also contains anterior and posterior sidewalls. Searching
    the full contour lets those sidewall points masquerade as a dome apex. The
    inferior endplate is instead the contour path between the two inferior
    chord endpoints that stays closest to the chord.
    """
    points = as_points(contour_points, name="contour")
    a = np.asarray(inferior_edge[0], dtype=float)
    b = np.asarray(inferior_edge[1], dtype=float)
    v = b - a
    length = float(np.linalg.norm(v))
    if length <= 0:
        raise ValueError("Invalid inferior edge length")

    start = int(np.argmin(np.linalg.norm(points - a, axis=1)))
    end = int(np.argmin(np.linalg.norm(points - b, axis=1)))
    arc1 = cyclic_arc(points, start, end)
    arc2 = cyclic_arc(points, end, start)

    def arc_score(arc: np.ndarray) -> float:
        distances = np.array([abs(cross2(v, p - a)) / length for p in arc], dtype=float)
        t = np.dot(arc - a, v) / (length * length)
        on_chord = (t >= -0.10) & (t <= 1.10)
        if np.any(on_chord):
            distances = distances[on_chord]
        return float(np.percentile(distances, 75) + 0.002 * len(arc))

    return arc1 if arc_score(arc1) <= arc_score(arc2) else arc2


def measure_dome(
    contour: np.ndarray,
    inferior_edge: tuple[np.ndarray, np.ndarray],
    quad: np.ndarray,
    endpoint_trim: float,
) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    a = np.asarray(inferior_edge[0], dtype=float)
    b = np.asarray(inferior_edge[1], dtype=float)
    v = b - a
    length = float(np.linalg.norm(v))
    if length <= 0:
        raise ValueError("Invalid inferior edge length")

    centroid = as_points(quad, name="quadrilateral").mean(axis=0)
    centroid_cross = cross2(v, centroid - a)
    if abs(centroid_cross) < 1e-8:
        return 0.0, None, None

    points = inferior_contour_arc(contour[:, 0, :], inferior_edge)
    cross_values = np.array([cross2(v, p - a) for p in points], dtype=float)
    distances = np.abs(cross_values) / length
    t = np.dot(points - a, v) / (length * length)

    body_height = max(
        distance(*classify_edges(quad)["posterior"]),
        distance(*classify_edges(quad)["anterior"]),
        1.0,
    )
    valid = (
        (np.sign(cross_values) == np.sign(centroid_cross))
        & (t >= endpoint_trim)
        & (t <= 1.0 - endpoint_trim)
        & (distances <= 0.45 * body_height)
    )
    if not np.any(valid):
        return 0.0, None, None

    valid_indices = np.where(valid)[0]
    best_idx = int(valid_indices[np.argmax(distances[valid])])
    apex = points[best_idx]
    foot = a + float(t[best_idx]) * v
    return float(distances[best_idx]), foot, apex


def measure_mask(
    mask_path: Path,
    vertebra: str,
    epsilon: float,
    min_area: int,
    endpoint_trim: float,
) -> dict[str, Any]:
    mask = read_binary_mask(mask_path, min_area)
    quad, contour = find_dp_quad(mask, epsilon, min_area)
    edges = classify_edges(quad)
    dome_height, dome_base, dome_apex = measure_dome(
        contour,
        edges["inferior"],
        quad,
        endpoint_trim,
    )

    return {
        "vertebra": vertebra,
        "mask_path": mask_path,
        "mask": mask,
        "contour": contour,
        "quad": quad,
        "edges": edges,
        "posterior_height_px": distance(*edges["posterior"]),
        "inferior_chord_px": distance(*edges["inferior"]),
        "dome_height_px": dome_height,
        "dome_base": dome_base,
        "dome_apex": dome_apex,
        "area_px2": polygon_area(quad),
        "mask_area_px2": int(mask.sum()),
    }


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def transform_points(points: Any, *, scale: float, offset: np.ndarray) -> np.ndarray:
    pts = as_points(points, name="points")
    return (pts - offset) * scale


def draw_dashed_polyline(
    image: np.ndarray,
    points: Any,
    color: tuple[int, int, int],
    *,
    scale: float = 1.0,
    offset: np.ndarray | None = None,
    thickness: int = 1,
    dash: int = 6,
    gap: int = 4,
) -> None:
    if offset is None:
        offset = np.array([0.0, 0.0], dtype=float)
    pts = np.round(transform_points(points, scale=scale, offset=offset)).astype(int)
    if len(pts) < 2:
        return

    period = dash + gap
    phase = 0.0
    for i in range(len(pts)):
        p0 = pts[i].astype(float)
        p1 = pts[(i + 1) % len(pts)].astype(float)
        segment = p1 - p0
        length = float(np.linalg.norm(segment))
        if length <= 0:
            continue
        direction = segment / length
        position = 0.0
        while position < length:
            in_dash = phase < dash
            step = min(length - position, (dash if in_dash else period) - phase)
            if in_dash and step > 0:
                start = p0 + direction * position
                end = p0 + direction * (position + step)
                cv2.line(
                    image,
                    tuple(np.round(start).astype(int)),
                    tuple(np.round(end).astype(int)),
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
            position += step
            phase = (phase + step) % period


def transformed_point(point: Any, *, scale: float, offset: np.ndarray) -> tuple[int, int]:
    p = (np.asarray(point, dtype=float) - offset) * scale
    return tuple(np.round(p).astype(int))


def draw_overlay(
    original: np.ndarray,
    results: dict[str, dict[str, Any]],
    *,
    scale: float = 1.0,
    offset: tuple[int, int] = (0, 0),
) -> np.ndarray:
    overlay = ensure_bgr(original)
    if scale != 1.0:
        overlay = cv2.resize(
            overlay,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    offset_arr = np.array(offset, dtype=float)
    mask_grey = (230, 230, 230)
    quad_green = (80, 145, 85)
    posterior_magenta = (255, 0, 255)
    inferior_yellow = (0, 255, 255)
    dome_cyan = (255, 255, 0)
    line_thickness = max(1, int(round(0.40 * scale)))
    mask_thickness = max(1, int(round(0.35 * scale)))
    marker_radius = max(1, int(round(0.85 * scale)))
    dash = max(4, int(round(5 * scale)))
    gap = max(3, int(round(4 * scale)))

    for vertebra in VERTEBRAE:
        result = results.get(vertebra)
        if result is None:
            continue

        contours, _ = cv2.findContours(
            result["mask"].astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        for contour in contours:
            if cv2.contourArea(contour) >= DEFAULT_MIN_MASK_AREA:
                draw_dashed_polyline(
                    overlay,
                    contour[:, 0, :],
                    mask_grey,
                    scale=scale,
                    offset=offset_arr,
                    thickness=mask_thickness,
                    dash=dash,
                    gap=gap,
                )

        quad = np.round(
            transform_points(result["quad"], scale=scale, offset=offset_arr)
        ).astype(int)
        cv2.polylines(
            overlay,
            [quad.reshape(-1, 1, 2)],
            True,
            quad_green,
            line_thickness,
            cv2.LINE_AA,
        )

        p0, p1 = [
            transformed_point(p, scale=scale, offset=offset_arr)
            for p in result["edges"]["posterior"]
        ]
        cv2.line(overlay, p0, p1, posterior_magenta, line_thickness, cv2.LINE_AA)

        i0, i1 = [
            transformed_point(p, scale=scale, offset=offset_arr)
            for p in result["edges"]["inferior"]
        ]
        cv2.line(overlay, i0, i1, inferior_yellow, line_thickness, cv2.LINE_AA)

        if result["dome_base"] is not None and result["dome_apex"] is not None:
            base = transformed_point(result["dome_base"], scale=scale, offset=offset_arr)
            apex = transformed_point(result["dome_apex"], scale=scale, offset=offset_arr)
            cv2.line(overlay, base, apex, dome_cyan, line_thickness, cv2.LINE_AA)
            cv2.circle(overlay, base, marker_radius, inferior_yellow, -1, cv2.LINE_AA)
            cv2.circle(overlay, apex, marker_radius, dome_cyan, -1, cv2.LINE_AA)

    return overlay


def crop_bounds_for_results(
    image_shape: tuple[int, ...],
    results: dict[str, dict[str, Any]],
    margin: int,
) -> tuple[int, int, int, int] | None:
    points = [
        result["quad"]
        for vertebra in VERTEBRAE
        if (result := results.get(vertebra)) is not None
    ]
    if not points:
        return None

    all_points = np.vstack(points)
    height, width = image_shape[:2]
    x0 = max(0, int(np.floor(all_points[:, 0].min())) - margin)
    y0 = max(0, int(np.floor(all_points[:, 1].min())) - margin)
    x1 = min(width, int(np.ceil(all_points[:, 0].max())) + margin)
    y1 = min(height, int(np.ceil(all_points[:, 1].max())) + margin)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def add_measurements_to_row(
    row: dict[str, str],
    results: dict[str, dict[str, Any]],
    target_folder: Path,
) -> None:
    for vertebra in VERTEBRAE:
        result = results.get(vertebra)
        if result is None:
            continue

        base = result["dome_base"]
        apex = result["dome_apex"]
        try:
            mask_path = result["mask_path"].relative_to(target_folder)
        except ValueError:
            mask_path = result["mask_path"]
        row[f"{vertebra}_mask_path"] = str(mask_path)
        row[f"{vertebra}_posterior_height_px"] = fmt(result["posterior_height_px"])
        row[f"{vertebra}_inferior_chord_px"] = fmt(result["inferior_chord_px"])
        row[f"{vertebra}_dome_height_px"] = fmt(result["dome_height_px"])
        row[f"{vertebra}_dome_base_x"] = fmt(base[0] if base is not None else None)
        row[f"{vertebra}_dome_base_y"] = fmt(base[1] if base is not None else None)
        row[f"{vertebra}_dome_apex_x"] = fmt(apex[0] if apex is not None else None)
        row[f"{vertebra}_dome_apex_y"] = fmt(apex[1] if apex is not None else None)


def classify_cvm(row: dict[str, str], results: dict[str, dict[str, Any]]) -> None:
    c2 = results.get("C2")
    c3 = results.get("C3")
    c4 = results.get("C4")

    c2_ratio = None
    c3_ratio = None
    c4_ratio = None
    if c2 is not None and c3 is not None and c3["posterior_height_px"] > 0:
        c2_ratio = c2["dome_height_px"] / c3["posterior_height_px"]
    if c3 is not None and c3["posterior_height_px"] > 0:
        c3_ratio = c3["dome_height_px"] / c3["posterior_height_px"]
    if c4 is not None and c4["posterior_height_px"] > 0:
        c4_ratio = c4["dome_height_px"] / c4["posterior_height_px"]

    ratios = {"C2": c2_ratio, "C3": c3_ratio, "C4": c4_ratio}
    row["C2_doming_ratio_C2_dome_over_C3_posterior"] = fmt(c2_ratio)
    row["C3_doming_ratio_C3_dome_over_C3_posterior"] = fmt(c3_ratio)
    row["C4_doming_ratio_C4_dome_over_C4_posterior"] = fmt(c4_ratio)

    states: dict[str, bool | None] = {
        vertebra: None if ratio is None else ratio > DOMING_THRESHOLD
        for vertebra, ratio in ratios.items()
    }
    for vertebra, state in states.items():
        row[f"{vertebra}_is_doming"] = "" if state is None else ("yes" if state else "no")

    if any(states[vertebra] is None for vertebra in VERTEBRAE):
        row["doming_pattern"] = ""
        row["predicted_CVM"] = "atypical"
        return

    pattern = tuple(int(bool(states[vertebra])) for vertebra in VERTEBRAE)
    row["doming_pattern"] = "".join(str(value) for value in pattern)
    row["predicted_CVM"] = {
        (0, 0, 0): "CVM1",
        (1, 0, 0): "CVM2",
        (1, 1, 0): "CVM3",
        (1, 1, 1): "CVM4-6",
    }.get(pattern, "atypical")


def process_image(
    image_path: Path,
    mask_index: dict[tuple[str, str], list[Path]],
    target_folder: Path,
    output_folder: Path,
    zoom_output_folder: Path,
    epsilon: float,
    min_area: int,
    endpoint_trim: float,
    zoom_scale: float,
    zoom_margin: int,
) -> dict[str, str]:
    row = {column: "" for column in CSV_COLUMNS}
    row["filename"] = image_path.name

    original = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if original is None:
        row["status"] = "error"
        row["message"] = "Could not read original image"
        return row

    mask_map = find_masks_for_image(image_path, mask_index)
    results: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    for vertebra in VERTEBRAE:
        mask_path = mask_map.get(vertebra)
        if mask_path is None:
            errors.append(f"{vertebra}: mask not found")
            continue
        try:
            results[vertebra] = measure_mask(
                mask_path,
                vertebra,
                epsilon,
                min_area,
                endpoint_trim,
            )
        except Exception as exc:
            errors.append(f"{vertebra}: {exc}")

    overlay = draw_overlay(original, results)
    output_folder.mkdir(parents=True, exist_ok=True)
    overlay_path = output_folder / image_path.name
    if not cv2.imwrite(str(overlay_path), overlay):
        errors.append("Could not write overlay")

    crop_bounds = crop_bounds_for_results(original.shape, results, zoom_margin)
    if crop_bounds is not None:
        x0, y0, x1, y1 = crop_bounds
        crop = original[y0:y1, x0:x1]
        zoom_overlay = draw_overlay(
            crop,
            results,
            scale=zoom_scale,
            offset=(x0, y0),
        )
        zoom_output_folder.mkdir(parents=True, exist_ok=True)
        zoom_path = zoom_output_folder / f"{image_path.stem}_c2_c4_zoom.png"
        if not cv2.imwrite(str(zoom_path), zoom_overlay):
            errors.append("Could not write zoom overlay")

    add_measurements_to_row(row, results, target_folder)
    classify_cvm(row, results)

    row["status"] = "ok" if not errors else ("partial" if results else "error")
    row["message"] = "; ".join(errors)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure CVM doming from existing C2-C4 segmentation masks."
    )
    parser.add_argument(
        "--target-folder",
        required=True,
        type=Path,
        help="Folder containing original BMP images and a masks subfolder.",
    )
    parser.add_argument(
        "--masks-folder",
        type=Path,
        default=None,
        help="Optional mask folder. Defaults to <target-folder>/masks.",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=None,
        help="Optional overlay output folder. Defaults to <target-folder>/postprocessing_overlays.",
    )
    parser.add_argument(
        "--zoom-output-folder",
        type=Path,
        default=None,
        help=(
            "Optional high-resolution C2-C4 zoom overlay folder. Defaults to "
            "<target-folder>/postprocessing_zoom_overlays."
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path. Defaults to <target-folder>/cvm_postprocessing_measurements.csv.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help="Preferred Douglas-Peucker epsilon fraction of hull perimeter.",
    )
    parser.add_argument(
        "--min-mask-area",
        type=int,
        default=DEFAULT_MIN_MASK_AREA,
        help="Minimum foreground pixels required in a vertebral mask.",
    )
    parser.add_argument(
        "--inferior-endpoint-trim",
        type=float,
        default=DEFAULT_INFERIOR_ENDPOINT_TRIM,
        help=(
            "Fraction of each inferior chord end excluded from dome-apex search "
            "to avoid anterior/posterior sidewall points. Default: 0.04."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of images to process for quick review.",
    )
    parser.add_argument(
        "--zoom-scale",
        type=float,
        default=DEFAULT_ZOOM_SCALE,
        help="Upsampling scale for C2-C4 zoom overlays. Default: 3.0.",
    )
    parser.add_argument(
        "--zoom-margin",
        type=int,
        default=DEFAULT_ZOOM_MARGIN,
        help="Pixel margin around C2-C4 for zoom overlays before upsampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_folder = args.target_folder.expanduser().resolve()
    masks_folder = (
        args.masks_folder.expanduser().resolve()
        if args.masks_folder is not None
        else target_folder / "masks"
    )
    output_folder = (
        args.output_folder.expanduser().resolve()
        if args.output_folder is not None
        else target_folder / "postprocessing_overlays"
    )
    zoom_output_folder = (
        args.zoom_output_folder.expanduser().resolve()
        if args.zoom_output_folder is not None
        else target_folder / "postprocessing_zoom_overlays"
    )
    csv_path = (
        args.csv.expanduser().resolve()
        if args.csv is not None
        else target_folder / "cvm_postprocessing_measurements.csv"
    )

    if not target_folder.is_dir():
        raise SystemExit(f"ERROR: target folder does not exist: {target_folder}")
    if not masks_folder.is_dir():
        raise SystemExit(f"ERROR: masks folder does not exist: {masks_folder}")
    if not 0.0 <= args.inferior_endpoint_trim < 0.5:
        raise SystemExit("ERROR: --inferior-endpoint-trim must be >= 0 and < 0.5")
    if args.zoom_scale < 1.0:
        raise SystemExit("ERROR: --zoom-scale must be >= 1.0")
    if args.zoom_margin < 0:
        raise SystemExit("ERROR: --zoom-margin must be >= 0")

    image_files = sorted(
        [p for p in target_folder.glob("*.bmp") if p.is_file()],
        key=lambda p: p.name.lower(),
    )
    if args.limit is not None:
        image_files = image_files[: args.limit]
    if not image_files:
        raise SystemExit(f"ERROR: no BMP images found directly inside: {target_folder}")

    print("Indexing masks...")
    mask_index = build_mask_index(masks_folder)
    print(f"Indexed {sum(len(paths) for paths in mask_index.values())} vertebral masks.")

    rows: list[dict[str, str]] = []
    for index, image_path in enumerate(image_files, start=1):
        print(f"[{index}/{len(image_files)}] {image_path.name} ... ", end="", flush=True)
        row = process_image(
            image_path,
            mask_index,
            target_folder,
            output_folder,
            zoom_output_folder,
            args.epsilon,
            args.min_mask_area,
            args.inferior_endpoint_trim,
            args.zoom_scale,
            args.zoom_margin,
        )
        rows.append(row)
        print(row["status"])
        if row["message"]:
            print(f"  {row['message']}")
        print(
            f"  predicted CVM: {row['predicted_CVM']} "
            f"(doming pattern {row['doming_pattern']})"
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Overlays: {output_folder}")
    print(f"Zooms   : {zoom_output_folder}")
    print(f"CSV     : {csv_path}")


if __name__ == "__main__":
    main()
