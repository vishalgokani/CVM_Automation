#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# CVM 4-column publication figure generator
#
# Input one original lateral radiograph per row. The script automatically finds:
#
#   <original parent>/masks/c2/<same filename>
#   <original parent>/masks/c3/<same filename>
#   <original parent>/masks/c4/<same filename>
#   <original parent>/postprocessing_overlays/<same filename>
#
# The union of the c2/c3/c4 masks defines the region of interest. The ROI is
# expanded by 10%, adjusted toward the figure slot aspect ratio, and reused for
# the ROI, CV mask, and postprocessed columns.
#
# Example:
#
#   python make_cvm_figure.py ^
#     --cvm1 "C:\data\test\Patient-1066_13male_lateral_spine_20240329.bmp" ^
#     --cvm2 "C:\data\test\Patient-1070_8female_lateral_spine_20251007.bmp" ^
#     --cvm3 "C:\data\test\Patient-1066_13male_lateral_spine_20251014.bmp" ^
#     --cvm46 "C:\data\test\Patient-1095_13female_lateral_spine_20231016_02.bmp" ^
#     --output "cvm_figure.svg" ^
#     --jpg-output "cvm_figure.jpg"
#
# Requirements:
#   Pillow. cairosvg is optional and is only needed when --jpg-output is used.
# -----------------------------------------------------------------------------

import argparse
import base64
import io
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw


# ----------------------------- Layout constants ------------------------------

SVG_WIDTH = 1340
SVG_HEIGHT = 2680

LEFT_MARGIN = 140
RIGHT_MARGIN = 40
TOP_MARGIN = 155
BOTTOM_MARGIN = 45

SLOT_W = 260
SLOT_H = 590
COL_GAP = 26
ROW_GAP = 30

HEADER_Y = 55
TOP_RULE_Y = 118

FONT_FAMILY = "Times New Roman, Times, serif"
HEADER_FONT = 30
ROW_FONT = 30

PLACEHOLDER_STROKE = "#B8B8B8"
RULE_STROKE = "#000000"

C2_COLOR = (0, 170, 80)
C3_COLOR = (230, 30, 35)
C4_COLOR = (135, 60, 190)
ROI_BOX_COLOR = (255, 0, 0)

COLUMN_HEADERS = [
    ("Original Lateral", "Radiograph"),
    ("Region of", "Interest"),
    ("CV", "Mask"),
    ("Postprocessed",),
]

ROW_SPECS = [
    ("cvm1", "CVM 1"),
    ("cvm2", "CVM 2"),
    ("cvm3", "CVM 3"),
    ("cvm46", "CVM 4-6"),
]

MASK_NAMES = ("c2", "c3", "c4")
MASK_COLORS = {
    "c2": C2_COLOR,
    "c3": C3_COLOR,
    "c4": C4_COLOR,
}


# ------------------------------- Image helpers -------------------------------

def load_rgb(path):
    with Image.open(path) as im:
        return im.convert("RGB")


def load_mask(path, target_size):
    with Image.open(path) as im:
        mask = im.convert("L")
        if mask.size != target_size:
            mask = mask.resize(target_size, Image.Resampling.NEAREST)
        return mask.point(lambda px: 255 if px > 0 else 0)


def pil_to_png_data_uri(im):
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def bbox_from_masks(masks):
    union = Image.new("L", masks[0].size, 0)
    for mask in masks:
        union = Image.composite(Image.new("L", mask.size, 255), union, mask)
    bbox = union.getbbox()
    if bbox is None:
        raise ValueError("combined c2/c3/c4 mask is empty")
    return bbox


def expanded_aspect_crop_box(bbox, image_size, padding_fraction=0.10):
    left, top, right, bottom = bbox
    image_w, image_h = image_size

    box_w = right - left
    box_h = bottom - top
    pad_x = box_w * padding_fraction
    pad_y = box_h * padding_fraction

    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    target_ratio = SLOT_H / SLOT_W

    crop_w = box_w + 2 * pad_x
    crop_h = box_h + 2 * pad_y

    if crop_h / crop_w < target_ratio:
        crop_h = crop_w * target_ratio
    else:
        crop_w = crop_h / target_ratio

    crop_w = min(crop_w, image_w)
    crop_h = min(crop_h, image_h)

    left = center_x - crop_w / 2
    right = center_x + crop_w / 2
    top = center_y - crop_h / 2
    bottom = center_y + crop_h / 2

    if left < 0:
        right -= left
        left = 0
    if right > image_w:
        left -= right - image_w
        right = image_w
    if top < 0:
        bottom -= top
        top = 0
    if bottom > image_h:
        top -= bottom - image_h
        bottom = image_h

    left = max(0, int(round(left)))
    top = max(0, int(round(top)))
    right = min(image_w, int(round(right)))
    bottom = min(image_h, int(round(bottom)))
    return (left, top, right, bottom)


def draw_roi_box(original, crop_box):
    boxed = original.copy()
    line_width = max(1, round(min(original.size) / 700))
    draw = ImageDraw.Draw(boxed)
    draw.rectangle(crop_box, outline=ROI_BOX_COLOR, width=line_width)
    return boxed


def overlay_masks(original_crop, masks, crop_box):
    out = original_crop.convert("RGBA")
    empty = Image.new("RGBA", original_crop.size)
    for name, mask in masks.items():
        mask_crop = mask.crop(crop_box)
        color_layer = Image.new("RGBA", original_crop.size, MASK_COLORS[name] + (125,))
        out = Image.alpha_composite(out, Image.composite(color_layer, empty, mask_crop))
    return out.convert("RGB")


def find_required_row_files(original_path):
    filename = original_path.name
    parent = original_path.parent

    mask_paths = {
        name: parent / "masks" / name / filename
        for name in MASK_NAMES
    }

    post_candidates = [
        parent / "postprocessing_overlays" / filename,
        parent / "postprocessing overlays" / filename,
        parent / "postprocessing" / filename,
    ]
    post_path = next((path for path in post_candidates if path.exists()), post_candidates[0])

    missing = []
    for label, path in mask_paths.items():
        if not path.exists():
            missing.append(f"{label} mask: {path}")
    if not post_path.exists():
        tried = "; ".join(str(path) for path in post_candidates)
        missing.append(f"postprocessing overlay, tried: {tried}")

    if missing:
        details = "\n  ".join(missing)
        raise FileNotFoundError(f"Missing derived files for {original_path}:\n  {details}")

    return mask_paths, post_path


def build_row_images(original_path):
    if not original_path.exists():
        raise FileNotFoundError(f"original radiograph not found: {original_path}")

    mask_paths, post_path = find_required_row_files(original_path)

    original = load_rgb(original_path)
    masks = {
        name: load_mask(path, original.size)
        for name, path in mask_paths.items()
    }
    bbox = bbox_from_masks(list(masks.values()))
    crop_box = expanded_aspect_crop_box(bbox, original.size)

    post = load_rgb(post_path)
    if post.size != original.size:
        post = post.resize(original.size, Image.Resampling.BILINEAR)

    original_crop = original.crop(crop_box)
    return [
        draw_roi_box(original, crop_box),
        original_crop,
        overlay_masks(original_crop, masks, crop_box),
        post.crop(crop_box),
    ]


# -------------------------------- SVG helpers --------------------------------

def text(x, y, value, size, anchor="middle", weight="normal"):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="{FONT_FAMILY}" font-size="{size}" font-weight="{weight}" '
        f'fill="#000000">{escape(value)}</text>'
    )


def multiline_text(x, y, lines, size, anchor="middle", weight="normal", line_gap=34):
    parts = [
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="{FONT_FAMILY}" font-size="{size}" font-weight="{weight}" '
        f'fill="#000000">'
    ]
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else line_gap
        parts.append(f'<tspan x="{x}" dy="{dy}">{escape(line)}</tspan>')
    parts.append("</text>")
    return "".join(parts)


def image_element(im, x, y, w, h):
    uri = pil_to_png_data_uri(im)
    return (
        f'<image x="{x}" y="{y}" width="{w}" height="{h}" '
        f'preserveAspectRatio="xMidYMid meet" href="{uri}" />'
    )


def placeholder(x, y, w, h):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
        f'fill="#FFFFFF" stroke="{PLACEHOLDER_STROKE}" stroke-width="2"/>'
    )


def build_svg(row_images):
    x_positions = [LEFT_MARGIN + c * (SLOT_W + COL_GAP) for c in range(4)]
    y_positions = [TOP_MARGIN + r * (SLOT_H + ROW_GAP) for r in range(4)]

    out = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{SVG_WIDTH}" height="{SVG_HEIGHT}" '
        f'viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">'
    )
    out.append('<rect width="100%" height="100%" fill="#FFFFFF"/>')

    for c, header_lines in enumerate(COLUMN_HEADERS):
        cx = x_positions[c] + SLOT_W / 2
        out.append(multiline_text(cx, HEADER_Y, header_lines, HEADER_FONT, weight="bold"))

    out.append(multiline_text(72, HEADER_Y, ("CVM", "Class"), HEADER_FONT, weight="bold"))
    out.append(
        f'<line x1="35" y1="{TOP_RULE_Y}" x2="{SVG_WIDTH - RIGHT_MARGIN}" y2="{TOP_RULE_Y}" '
        f'stroke="{RULE_STROKE}" stroke-width="3"/>'
    )

    for r, (_, row_label) in enumerate(ROW_SPECS):
        cy = y_positions[r] + SLOT_H / 2 + ROW_FONT / 3
        out.append(text(72, cy, row_label, ROW_FONT))

        images = row_images[r]
        for c in range(4):
            x = x_positions[c]
            y = y_positions[r]
            if images is None:
                out.append(placeholder(x, y, SLOT_W, SLOT_H))
            else:
                out.append(image_element(images[c], x, y, SLOT_W, SLOT_H))

    out.append("</svg>")
    return "\n".join(out)


def write_jpg_from_svg(svg_path, jpg_path):
    try:
        import cairosvg
    except ImportError as exc:
        raise RuntimeError("Saving JPG requires cairosvg in this environment.") from exc

    png_bytes = cairosvg.svg2png(
        url=str(svg_path),
        output_width=SVG_WIDTH,
        output_height=SVG_HEIGHT,
    )
    with Image.open(io.BytesIO(png_bytes)) as im:
        im.convert("RGB").save(jpg_path, "JPEG", quality=95, subsampling=0)


# ---------------------------------- CLI --------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Generate the CVM publication SVG. Provide one original radiograph "
            "for each row; masks and postprocessing overlays are found automatically."
        ),
        allow_abbrev=False,
    )

    for key, row_label in ROW_SPECS:
        p.add_argument(f"--{key}", type=Path, default=None, help=f"Original radiograph for {row_label}.")

    p.add_argument(
        "--output",
        type=Path,
        default=Path("cvm_figure.svg"),
        help="Output SVG path (default: cvm_figure.svg).",
    )
    p.add_argument(
        "--jpg-output",
        type=Path,
        default=None,
        help="Optional output JPG path rendered from the SVG.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    row_images = []
    for key, row_label in ROW_SPECS:
        original_path = getattr(args, key)
        if original_path is None:
            row_images.append(None)
            continue

        print(f"Building {row_label}: {original_path}")
        row_images.append(build_row_images(original_path))

    svg = build_svg(row_images)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    print(f"Wrote SVG: {args.output.resolve()}")

    if args.jpg_output is not None:
        args.jpg_output.parent.mkdir(parents=True, exist_ok=True)
        write_jpg_from_svg(args.output, args.jpg_output)
        print(f"Wrote JPG: {args.jpg_output.resolve()}")


if __name__ == "__main__":
    main()
