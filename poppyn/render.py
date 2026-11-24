"""Render pixel-exact maps, tiles and optional captions."""

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont


WATER = (10, 16, 18)
LAND = (25, 42, 45)
MARK = (255, 246, 228)
HEAT_COLOURS = np.array([LAND, (23, 101, 103), (69, 169, 143), (225, 155, 83), MARK])


def binary_image(land, marks):
    image = Image.fromarray(np.where(marks, 2, land).astype(np.uint8))
    image.putpalette((*WATER, *LAND, *MARK))
    return image


def heatmap_image(counts, land):
    maximum = float(counts.max())
    intensities = np.log1p(counts)
    if maximum:
        intensities /= math.log1p(maximum)
        intensities *= 254
    pixels = np.where(land, 1 + np.rint(intensities), 0).astype(np.uint8)
    colours = np.linspace(0, 1, len(HEAT_COLOURS))
    samples = np.linspace(0, 1, 255)
    gradient = np.column_stack([np.interp(samples, colours, channel) for channel in HEAT_COLOURS.T])
    image = Image.fromarray(pixels)
    image.putpalette((*WATER, *np.rint(gradient).astype(np.uint8).ravel().tolist()))
    return image, maximum


def write_level(output, level, image, *, tile_width, bits):
    directory = output / str(level.width)
    directory.mkdir(parents=True)
    image.save(directory / "map.png", bits=bits)
    tiles = 0
    byte_count = 0
    for x, left in enumerate(range(0, level.width, tile_width)):
        column = directory / str(x)
        column.mkdir()
        for y, top in enumerate(range(0, level.height, tile_width)):
            tile = image.crop((left, top, min(left + tile_width, level.width), min(top + tile_width, level.height)))
            path = column / f"{y}.png"
            tile.save(path, bits=bits)
            tiles += 1
            byte_count += path.stat().st_size
    return tiles, byte_count


def write_annotation(path, image, title, lines, bounds):
    west, south, east, north = bounds
    latitude = (south + north) / 2
    half_pixel = (north - south) / (2 * image.height)
    area = 6371**2 * math.radians((east - west) / image.width) * (
        math.sin(math.radians(latitude + half_pixel)) - math.sin(math.radians(latitude - half_pixel))
    )
    size = max(16, round(image.width / 75))
    heading = ImageFont.load_default(size=size + size // 3)
    body = ImageFont.load_default(size=size)
    caption = [
        (title, heading, MARK),
        *((line, body, MARK) for line in lines),
        (f"{image.width:,} by {image.height:,} pixels · about {area:,.2f} sq km per pixel at the map centre", body, MARK),
        ("WorldPop 2020 · DOI 10.5258/SOTON/WP00647 · CC BY 4.0 · Poppyn by naspli", body, (182, 195, 190)),
    ]
    padding = size
    width = max(image.width, math.ceil(max(font.getlength(text) for text, font, _ in caption)) + 2 * padding)
    row_heights = [math.ceil(font.size * 1.4) for _, font, _ in caption]
    canvas = Image.new("RGB", (width, image.height + sum(row_heights) + 2 * padding), WATER)
    canvas.paste(image.convert("RGB"), ((width - image.width) // 2, 0))
    draw = ImageDraw.Draw(canvas)
    y = image.height + padding
    for (text, font, colour), height in zip(caption, row_heights):
        draw.text((padding, y), text, font=font, fill=colour, anchor="lt")
        y += height
    canvas.save(path)
