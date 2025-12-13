"""grid generation utilities for vl experiments."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def make_grid(
    images: list,
    labels: list[str],
    cols: int,
    output_path: Path | str,
    cell_size: int = 256,
    label_height: int = 25,
) -> Path:
    """create a labeled grid from a list of images.

    args:
        images: list of image paths or PIL Image objects
        labels: list of labels (same length as images)
        cols: number of columns
        output_path: where to save the grid
        cell_size: size of each cell in pixels
        label_height: height of label area above each cell

    returns:
        path to saved grid
    """
    output_path = Path(output_path)

    rows = (len(images) + cols - 1) // cols
    grid_w = cols * cell_size
    grid_h = rows * (cell_size + label_height)

    grid = Image.new('RGB', (grid_w, grid_h), 'white')
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    except:
        font = ImageFont.load_default()

    for i, (img_item, label) in enumerate(zip(images, labels)):
        row, col = i // cols, i % cols
        x = col * cell_size
        y = row * (cell_size + label_height)

        # Handle both paths and PIL Image objects
        if isinstance(img_item, Image.Image):
            img = img_item.convert('RGB').resize((cell_size, cell_size))
            grid.paste(img, (x, y + label_height))
        else:
            img_path = Path(img_item)
            if img_path.exists():
                img = Image.open(img_path).convert('RGB').resize((cell_size, cell_size))
                grid.paste(img, (x, y + label_height))
            else:
                # draw placeholder for missing image
                draw.rectangle([x, y + label_height, x + cell_size, y + label_height + cell_size], fill='gray')
                draw.text((x + 10, y + label_height + cell_size // 2), 'missing', fill='white', font=font)

        draw.text((x + 5, y + 5), label, fill='black', font=font)

    grid.save(output_path)
    return output_path


def make_alpha_grid(
    output_dir: Path | str,
    prefix: str,
    alphas: list[float],
    output_name: str = 'grid.png',
    cell_size: int = 256,
) -> Path:
    """create grid for alpha sweep experiment.

    expects files named: {prefix}_a{alpha*10}.png
    """
    output_dir = Path(output_dir)
    images = [output_dir / f'{prefix}_a{int(a*10)}.png' for a in alphas]
    labels = [f'alpha={a}' for a in alphas]
    return make_grid(images, labels, len(alphas), output_dir / output_name, cell_size)


def make_alpha_strength_grid(
    output_dir: Path | str,
    prefix: str,
    alphas: list[float],
    strengths: list[float],
    output_name: str = 'grid.png',
    cell_size: int = 256,
) -> Path:
    """create grid for alpha x strength sweep.

    expects files named: {prefix}_a{alpha*10}_s{strength*10}.png
    rows = alphas, cols = strengths
    """
    output_dir = Path(output_dir)
    images = []
    labels = []

    for a in alphas:
        for s in strengths:
            images.append(output_dir / f'{prefix}_a{int(a*10)}_s{int(s*10)}.png')
            labels.append(f'a={a} s={s}')

    return make_grid(images, labels, len(strengths), output_dir / output_name, cell_size)
