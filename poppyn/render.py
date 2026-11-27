"""Render pixel-exact maps, tiles and optional captions."""

import math
from textwrap import wrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin


WATER = (10, 16, 18)
LAND = (25, 42, 45)
MARK = (255, 246, 228)
OUTSIDE = (33, 33, 39)
MAP_FILENAME = "poppyn-map.png"
HEAT_COLOURS = np.array([LAND, (23, 101, 103), (69, 169, 143), (225, 155, 83), MARK])


def binary_image(land, marks, *, outside=None):
    pixels = np.where(marks, 2, land).astype(np.uint8)
    palette = (*WATER, *LAND, *MARK)
    if outside is not None:
        pixels[outside] = 3
        palette += OUTSIDE
    image = Image.fromarray(pixels)
    image.putpalette(palette)
    return image


def heatmap_image(counts, land, *, outside=None):
    maximum = float(counts.max())
    intensities = np.log1p(counts)
    steps = 255 if outside is None else 254
    if maximum:
        intensities /= math.log1p(maximum)
        intensities *= steps - 1
    pixels = np.where(land, 1 + np.rint(intensities), 0).astype(np.uint8)
    colours = np.linspace(0, 1, len(HEAT_COLOURS))
    samples = np.linspace(0, 1, steps)
    gradient = np.column_stack([np.interp(samples, colours, channel) for channel in HEAT_COLOURS.T])
    palette = [*WATER, *np.rint(gradient).astype(np.uint8).ravel()]
    if outside is not None:
        pixels[outside] = 255
        palette.extend(OUTSIDE)
    image = Image.fromarray(pixels)
    image.putpalette(palette)
    return image, maximum


def write_tiles(directory, image, *, tile_width, bits):
    directory.mkdir(parents=True)
    tiles = 0
    byte_count = 0
    for x, left in enumerate(range(0, image.width, tile_width)):
        column = directory / str(x)
        column.mkdir()
        for y, top in enumerate(range(0, image.height, tile_width)):
            tile = image.crop((left, top, min(left + tile_width, image.width), min(top + tile_width, image.height)))
            path = column / f"{y}.png"
            tile.save(path, bits=bits)
            tiles += 1
            byte_count += path.stat().st_size
    return tiles, byte_count


def caption_font(size, bold=False):
    for name, index in (
        ("/System/Library/Fonts/Avenir Next.ttc", 2 if bold else 7),
        ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", 0),
    ):
        try:
            return ImageFont.truetype(name, size, index=index)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def annotate(image, title, subtitle, details, bounds, *, projected_bounds=None,
             projection_name="Cylindrical equal-area", source_credit=None):
    if projected_bounds is None:
        west, south, east, north = bounds
        latitude = (south + north) / 2
        half_pixel = (north - south) / (2 * image.height)
        area = 6371**2 * math.radians((east - west) / image.width) * (
            math.sin(math.radians(latitude + half_pixel)) - math.sin(math.radians(latitude - half_pixel))
        )
        area_caption = f"Approx. {area:,.2f} km² per pixel at the map centre"
    else:
        west, south, east, north = projected_bounds
        area = (east - west) * (north - south) / (image.width * image.height * 1e6)
        area_caption = f"{area:,.2f} km² per pixel · {projection_name}"
    heading = [
        (title, 1, True, 150),
        (subtitle, 0.58, True, 140),
        *((text, 0.40, False, 125) for text in details),
    ]
    credits = (
        f"{image.width:,} × {image.height:,} px · {area_caption}",
        source_credit,
        "Land boundary from Natural Earth",
        "Poppyn by naspli.io · github.com/naspli/poppyn",
    )
    footer = [(line, 0.34, False, 115) for line in credits if line]
    padding = max(2, round(min(image.width / 60, image.height / 22)))
    size = max(1, round(min(image.width / 72, image.height / 12)))
    overlay = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(overlay)
    while size >= 1:
        rows = []
        for text, scale, bold, alpha in heading + footer:
            font = caption_font(max(1, round(size * scale)), bold)
            box = draw.textbbox((0, 0), text, font=font, anchor="lt")
            rows.append((text, font, alpha, box[2] - min(0, box[0]), max(box[3], round(font.size * 1.35))))
        if (max(row[3] for row in rows) <= image.width - 2 * padding
                and sum(row[4] for row in rows) + size <= image.height - 2 * padding):
            break
        size -= 1
    else:
        raise ValueError("map is too small for an on-map caption. Use a larger width or omit --annotated")

    for block, y in ((rows[:len(heading)], padding),
                     (rows[len(heading):], image.height - padding - sum(row[4] for row in rows[len(heading):]))):
        for text, font, alpha, _, height in block:
            draw.text((padding, y), text, font=font, fill=(*MARK, alpha), anchor="lt",
                      stroke_width=max(1, font.size // 18), stroke_fill=(*WATER, 190))
            y += height
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def write_map(path, image, *, raw):
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("poppyn-kind", "raw" if raw else "binary")
    image.save(path, bits=8 if raw else 2, pnginfo=metadata)
