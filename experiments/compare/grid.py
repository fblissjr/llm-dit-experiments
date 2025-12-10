"""PIL-based grid generation for experiment comparison."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .models import ComparisonSpec, ExperimentRun

# Constants
DEFAULT_THUMBNAIL_SIZE = (256, 256)
LABEL_HEIGHT = 24
PADDING = 8
HEADER_HEIGHT = 32
ROW_HEADER_WIDTH = 150
FONT_SIZE = 12
HEADER_FONT_SIZE = 14
BG_COLOR = (30, 30, 30)
TEXT_COLOR = (200, 200, 200)
METRIC_COLOR = (150, 150, 150)
BORDER_COLOR = (60, 60, 60)


def _get_font(size: int = FONT_SIZE, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a TrueType font, fall back to default."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def generate_grid(
    spec: ComparisonSpec,
    output_path: Path | None = None,
    thumbnail_size: tuple[int, int] = DEFAULT_THUMBNAIL_SIZE,
    show_metrics: bool = True,
) -> Image.Image:
    """
    Generate NxM grid image (prompts x variable values).

    Rows: prompt_ids
    Columns: variable_values

    Args:
        spec: ComparisonSpec defining what to compare
        output_path: Optional path to save the grid
        thumbnail_size: Size of each thumbnail (width, height)
        show_metrics: Whether to show metric scores below thumbnails

    Returns:
        PIL Image of the grid
    """
    exp = spec.experiment

    # Filter by spec
    prompts = spec.prompts or exp.prompt_ids
    variables = spec.variables or [str(v) for v in exp.variable_values]
    seeds = spec.seeds or ([exp.seeds[0]] if exp.seeds else [42])
    seed = seeds[0]  # Use first seed for grid

    # Build image lookup: (prompt_id, variable_value) -> ExperimentImage
    image_map = {}
    for img in exp.images:
        if img.seed == seed:
            key = (img.prompt_id, str(img.variable_value))
            image_map[key] = img

    # Calculate grid dimensions
    n_rows = len(prompts)
    n_cols = len(variables)

    if n_rows == 0 or n_cols == 0:
        # Return empty placeholder
        return Image.new("RGB", (400, 100), color=BG_COLOR)

    cell_w = thumbnail_size[0] + PADDING * 2
    cell_h = thumbnail_size[1] + (LABEL_HEIGHT if show_metrics else 0) + PADDING * 2

    grid_w = ROW_HEADER_WIDTH + n_cols * cell_w
    grid_h = HEADER_HEIGHT + n_rows * cell_h

    # Create canvas
    canvas = Image.new("RGB", (grid_w, grid_h), color=BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    font = _get_font(FONT_SIZE)
    header_font = _get_font(HEADER_FONT_SIZE, bold=True)

    # Draw column headers (variable values)
    for col_idx, var_val in enumerate(variables):
        x = ROW_HEADER_WIDTH + col_idx * cell_w + cell_w // 2
        label = str(var_val)[:25]  # Truncate long values
        if label == "":
            label = '""'  # Show empty string explicitly
        draw.text(
            (x, HEADER_HEIGHT // 2),
            label,
            fill=TEXT_COLOR,
            font=header_font,
            anchor="mm",
        )

    # Draw row headers (prompt IDs) and images
    for row_idx, prompt_id in enumerate(prompts):
        y_base = HEADER_HEIGHT + row_idx * cell_h

        # Row header (truncate if needed)
        label = prompt_id[:18]
        draw.text(
            (PADDING, y_base + cell_h // 2),
            label,
            fill=TEXT_COLOR,
            font=font,
            anchor="lm",
        )

        # Images in this row
        for col_idx, var_val in enumerate(variables):
            x = ROW_HEADER_WIDTH + col_idx * cell_w + PADDING
            y = y_base + PADDING

            key = (prompt_id, var_val)
            if key in image_map:
                img_data = image_map[key]
                try:
                    thumb = Image.open(img_data.path)
                    thumb.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
                    # Center the thumbnail if it's smaller than cell
                    paste_x = x + (thumbnail_size[0] - thumb.width) // 2
                    paste_y = y + (thumbnail_size[1] - thumb.height) // 2
                    canvas.paste(thumb, (paste_x, paste_y))

                    # Draw metric label if available
                    if show_metrics and img_data.siglip_score is not None:
                        score_text = f"SL:{img_data.siglip_score:.3f}"
                        draw.text(
                            (x + thumbnail_size[0] // 2, y + thumbnail_size[1] + 4),
                            score_text,
                            fill=METRIC_COLOR,
                            font=font,
                            anchor="mt",
                        )
                except Exception:
                    # Draw placeholder for missing/broken images
                    draw.rectangle(
                        [x, y, x + thumbnail_size[0], y + thumbnail_size[1]],
                        outline=BORDER_COLOR,
                    )
                    draw.text(
                        (x + thumbnail_size[0] // 2, y + thumbnail_size[1] // 2),
                        "Error",
                        fill=METRIC_COLOR,
                        font=font,
                        anchor="mm",
                    )
            else:
                # No image for this combination
                draw.rectangle(
                    [x, y, x + thumbnail_size[0], y + thumbnail_size[1]],
                    outline=BORDER_COLOR,
                )

    if output_path:
        canvas.save(output_path)

    return canvas


def generate_side_by_side(
    spec: ComparisonSpec,
    value_a: str,
    value_b: str,
    prompt_id: str,
    seed: int = 42,
    output_path: Path | None = None,
) -> Image.Image:
    """
    Generate side-by-side comparison of two variable values.

    Args:
        spec: ComparisonSpec with experiment data
        value_a: First variable value to compare
        value_b: Second variable value to compare
        prompt_id: Which prompt to compare
        seed: Which seed to use
        output_path: Optional path to save the image

    Returns:
        PIL Image with both images side by side
    """
    exp = spec.experiment

    # Find images
    img_a = None
    img_b = None
    for img in exp.images:
        if img.prompt_id == prompt_id and img.seed == seed:
            if str(img.variable_value) == value_a:
                img_a = img
            elif str(img.variable_value) == value_b:
                img_b = img

    if not img_a or not img_b:
        raise ValueError(f"Could not find images for values '{value_a}' and '{value_b}'")

    # Load images
    pil_a = Image.open(img_a.path)
    pil_b = Image.open(img_b.path)

    # Create combined canvas
    gap = 20
    combined_w = pil_a.width + pil_b.width + gap
    header_h = 50
    combined_h = max(pil_a.height, pil_b.height) + header_h

    canvas = Image.new("RGB", (combined_w, combined_h), color=BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    header_font = _get_font(16, bold=True)
    font = _get_font(12)

    # Paste images
    canvas.paste(pil_a, (0, header_h))
    canvas.paste(pil_b, (pil_a.width + gap, header_h))

    # Labels
    label_a = value_a if value_a else '""'
    label_b = value_b if value_b else '""'
    draw.text(
        (pil_a.width // 2, 25),
        label_a,
        fill=TEXT_COLOR,
        font=header_font,
        anchor="mm",
    )
    draw.text(
        (pil_a.width + gap + pil_b.width // 2, 25),
        label_b,
        fill=TEXT_COLOR,
        font=header_font,
        anchor="mm",
    )

    # Show metrics if available
    if img_a.siglip_score is not None:
        draw.text(
            (pil_a.width // 2, 42),
            f"SL: {img_a.siglip_score:.3f}",
            fill=METRIC_COLOR,
            font=font,
            anchor="mt",
        )
    if img_b.siglip_score is not None:
        draw.text(
            (pil_a.width + gap + pil_b.width // 2, 42),
            f"SL: {img_b.siglip_score:.3f}",
            fill=METRIC_COLOR,
            font=font,
            anchor="mt",
        )

    if output_path:
        canvas.save(output_path)

    return canvas
