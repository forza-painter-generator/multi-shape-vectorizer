"""
FH6 JSON output writer.

Converts optimized shape parameters into FH6-compatible JSON format
that can be imported into Forza Horizon 6 via forza-painter-fh6.

Reference:
  - IMPLEMENTATION_PLAN.md §6.4 Import JSON Format
  - IMPLEMENTATION_PLAN.md §8.4 Step 4 JSON Output
  - forza-painter-fh6/src/fh6_typecode_import.py — decode() + import logic
"""

import json
from pathlib import Path
from typing import Optional

import torch


# FH6 coordinate divisor — maps canvas pixels to FH6 coordinate units
# This needs calibration against known FH6 data.
# Default: 1 canvas pixel = DIVISOR FH6 units (tentative)
DEFAULT_COORD_DIVISOR = 1.0


def shape_params_to_fh6(
    cx: torch.Tensor,
    cy: torch.Tensor,
    rx: torch.Tensor,
    ry: torch.Tensor,
    angle: torch.Tensor,
    colors: torch.Tensor,
    opacity: torch.Tensor,
    type_indices: torch.Tensor,
    type_map: dict,
    coord_divisor: float = DEFAULT_COORD_DIVISOR,
    canvas_height: int = 512,
) -> list[dict]:
    """
    Convert optimized shape parameters to FH6 JSON format.

    Key conversions:
      - FH6 Y-axis is flipped: fh6_y = -canvas_y  (or canvas_height - canvas_y)
      - FH6 angle direction is reversed: (360 - angle) % 360
      - Scale: rx / DIVISOR, ry / DIVISOR
      - Color: [R, G, B, A] uint8 (0-255)
      - Skew: always 0

    Args:
        cx, cy: [N] center positions in canvas pixels
        rx, ry: [N] scale radii in canvas pixels
        angle: [N] rotation in degrees (counterclockwise)
        colors: [N, 3] RGB in [0, 1]
        opacity: [N] opacity in [0, 1]
        type_indices: [N] long, template index for each shape
        type_map: dict {template_index: type_code}
        coord_divisor: scale divisor (canvas_px → FH6 units)
        canvas_height: for Y-axis flip

    Returns:
        list of shape dicts ready for JSON serialization
    """
    N = cx.shape[0]
    shapes = []

    # Convert to numpy for iteration
    cx_np = cx.detach().cpu().numpy()
    cy_np = cy.detach().cpu().numpy()
    rx_np = rx.detach().cpu().numpy()
    ry_np = ry.detach().cpu().numpy()
    angle_np = angle.detach().cpu().numpy()
    colors_np = colors.detach().cpu().numpy()
    opacity_np = opacity.detach().cpu().numpy()
    type_np = type_indices.detach().cpu().numpy()

    for i in range(N):
        # Get type_code from template index
        tidx = int(type_np[i])
        type_code = type_map.get(tidx, 1048677)  # Default to square if unknown

        # FH6 coordinate conversions
        fh6_cx = cx_np[i] / coord_divisor
        fh6_cy = -(cy_np[i] - canvas_height / 2) / coord_divisor  # Y flip + center
        fh6_sx = rx_np[i] / coord_divisor
        fh6_sy = ry_np[i] / coord_divisor
        fh6_angle = (360.0 - angle_np[i]) % 360.0  # Reverse angle direction

        # Color: [0,1] → [0,255] uint8
        r = int(round(colors_np[i][0] * 255))
        g = int(round(colors_np[i][1] * 255))
        b = int(round(colors_np[i][2] * 255))
        a = int(round(opacity_np[i] * 255))

        shape = {
            "type": int(type_code),
            "data": [
                float(fh6_cx),
                float(fh6_cy),
                float(fh6_sx),
                float(fh6_sy),
                float(fh6_angle),
                0.0,  # skew
            ],
            "color": [r, g, b, a],
            "mask": 0,
        }
        shapes.append(shape)

    return shapes


def write_fh6_json(
    shapes: list[dict],
    output_path: Path,
    indent: int = 2,
) -> None:
    """
    Write shapes to FH6-compatible JSON file.

    Args:
        shapes: list of shape dicts from shape_params_to_fh6()
        output_path: output JSON path
        indent: JSON indentation
    """
    output = {"shapes": shapes}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=indent)
    print(f"FH6 JSON written to {output_path} ({len(shapes)} shapes)")


def generate_fh6_json(
    renderer,
    type_map: dict,
    output_path: Path,
    coord_divisor: float = DEFAULT_COORD_DIVISOR,
) -> None:
    """
    Convenience function: extract params from renderer and write FH6 JSON.

    Args:
        renderer: STEVectorRenderer instance
        type_map: dict {template_index: type_code}
        output_path: output JSON path
        coord_divisor: scale divisor
    """
    shapes = shape_params_to_fh6(
        cx=renderer.cx,
        cy=renderer.cy,
        rx=renderer.rx,
        ry=renderer.ry,
        angle=renderer.angle,
        colors=renderer.colors,
        opacity=renderer.opacity,
        type_indices=renderer.type_indices,
        type_map=type_map,
        coord_divisor=coord_divisor,
        canvas_height=renderer.canvas_height,
    )
    write_fh6_json(shapes, output_path)
