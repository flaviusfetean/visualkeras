from typing import Any
from PIL import ImageColor, ImageDraw, Image, ImageFont
from collections import defaultdict
import numpy as np
import aggdraw


class RectShape:
    x1: int
    x2: int
    y1: int
    y2: int
    _fill: Any
    _outline: Any

    @property
    def fill(self):
        return self._fill

    @property
    def outline(self):
        return self._outline

    @fill.setter
    def fill(self, v):
        self._fill = get_rgba_tuple(v)

    @outline.setter
    def outline(self, v):
        self._outline = get_rgba_tuple(v)

    def _get_pen_brush(self):
        pen = aggdraw.Pen(self._outline)
        brush = aggdraw.Brush(self._fill)
        return pen, brush


class Box(RectShape):
    de: int
    shade: int

    def draw(self, draw: ImageDraw):
        pen, brush = self._get_pen_brush()

        if hasattr(self, 'de') and self.de > 0:
            brush_s1 = aggdraw.Brush(fade_color(self.fill, self.shade))
            brush_s2 = aggdraw.Brush(fade_color(self.fill, 2 * self.shade))

            draw.line([self.x1 + self.de, self.y1 - self.de, self.x1 + self.de, self.y2 - self.de], pen)
            draw.line([self.x1 + self.de, self.y2 - self.de, self.x1, self.y2], pen)
            draw.line([self.x1 + self.de, self.y2 - self.de, self.x2 + self.de, self.y2 - self.de], pen)

            draw.polygon([self.x1, self.y1,
                          self.x1 + self.de, self.y1 - self.de,
                          self.x2 + self.de, self.y1 - self.de,
                          self.x2, self.y1
                          ], pen, brush_s1)

            draw.polygon([self.x2 + self.de, self.y1 - self.de,
                          self.x2, self.y1,
                          self.x2, self.y2,
                          self.x2 + self.de, self.y2 - self.de
                          ], pen, brush_s2)

        draw.rectangle([self.x1, self.y1, self.x2, self.y2], pen, brush)


class Circle(RectShape):

    def draw(self, draw: ImageDraw):
        pen, brush = self._get_pen_brush()
        draw.ellipse([self.x1, self.y1, self.x2, self.y2], pen, brush)


class Ellipses(RectShape):

    def draw(self, draw: ImageDraw):
        pen, brush = self._get_pen_brush()
        w = self.x2 - self.x1
        d = int(w / 7)
        draw.ellipse([self.x1 + (w - d) / 2, self.y1 + 1 * d, self.x1 + (w + d) / 2, self.y1 + 2 * d], pen, brush)
        draw.ellipse([self.x1 + (w - d) / 2, self.y1 + 3 * d, self.x1 + (w + d) / 2, self.y1 + 4 * d], pen, brush)
        draw.ellipse([self.x1 + (w - d) / 2, self.y1 + 5 * d, self.x1 + (w + d) / 2, self.y1 + 6 * d], pen, brush)


class ColorWheel:

    def __init__(self, colors: list = None):
        self._cache = dict()
        self.colors = colors if colors is not None else ["#ffd166", "#ef476f", "#06d6a0", "#118ab2", "#073b4c"]

    def get_color(self, class_type: type):
        if class_type not in self._cache.keys():
            index = len(self._cache.keys()) % len(self.colors)
            self._cache[class_type] = self.colors[index]
        return self._cache.get(class_type)


def fade_color(color: tuple, fade_amount: int) -> tuple:
    r = max(0, color[0] - fade_amount)
    g = max(0, color[1] - fade_amount)
    b = max(0, color[2] - fade_amount)
    return r, g, b, color[3]


def get_rgba_tuple(color: Any) -> tuple:
    """

    :param color:
    :return: (R, G, B, A) tuple
    """
    if isinstance(color, tuple):
        rgba = color
    elif isinstance(color, int):
        rgba = (color >> 16 & 0xff, color >> 8 & 0xff, color & 0xff, color >> 24 & 0xff)
    else:
        rgba = ImageColor.getrgb(color)

    if len(rgba) == 3:
        rgba = (rgba[0], rgba[1], rgba[2], 255)
    return rgba


def get_keys_by_value(d, v):
    for key in d.keys():  # reverse search the dict for the value
        if d[key] == v:
            yield key


def self_multiply(tensor_tuple: tuple):
    """

    :param tensor_tuple:
    :return:
    """
    tensor_list = list(tensor_tuple)
    if None in tensor_list:
        tensor_list.remove(None)
    if len(tensor_list) == 0:
        return 0
    s = tensor_list[0]
    for i in range(1, len(tensor_list)):
        s *= tensor_list[i]
    return s


def vertical_image_concat(im1: Image, im2: Image, background_fill: Any = 'white'):
    """
    Vertical concatenation of two PIL images.

    :param im1: top image
    :param im2: bottom image
    :param background_fill: Color for the image background. Can be str or (R,G,B,A).
    :return: concatenated image
    """
    dst = Image.new('RGBA', (max(im1.width, im2.width), im1.height + im2.height), background_fill)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def linear_layout(images: list, max_width: int = -1, max_height: int = -1, horizontal: bool = True, padding: int = 0,
                  spacing: int = 0, background_fill: Any = 'white'):
    """
    Creates a linear layout of a passed list of images in horizontal or vertical orientation. The layout will wrap in x
    or y dimension if a maximum value is exceeded.

    :param images: List of PIL images
    :param max_width: Maximum width of the image. Only enforced in horizontal orientation.
    :param max_height: Maximum height of the image. Only enforced in vertical orientation.
    :param horizontal: If True, will draw images horizontally, else vertically.
    :param padding: Top, bottom, left, right border distance in pixels.
    :param spacing: Spacing in pixels between elements.
    :param background_fill: Color for the image background. Can be str or (R,G,B,A).
    :return:
    """
    coords = list()
    width = 0
    height = 0

    x, y = padding, padding

    for img in images:
        if horizontal:
            if max_width != -1 and x + img.width > max_width:
                # make a new row
                x = padding
                y = height - padding + spacing
            coords.append((x, y))

            width = max(x + img.width + padding, width)
            height = max(y + img.height + padding, height)

            x += img.width + spacing
        else:
            if max_height != -1 and y + img.height > max_height:
                # make a new column
                x = width - padding + spacing
                y = padding
            coords.append((x, y))

            width = max(x + img.width + padding, width)
            height = max(y + img.height + padding, height)

            y += img.height + spacing

    layout = Image.new('RGBA', (width, height), background_fill)
    for img, coord in zip(images, coords):
        layout.paste(img, coord)

    return layout

############################## Here is the code that I, Flavius, have added ##################################
#
#
##############################################################################################################

class ColorScheme():
    def __init__(self, color_map):
        self.color_map = color_map
        self.keys = []
        self.keys_names = []

    def decode_color_map(self):
        self.keys = self.color_map.keys()

        for key in self.keys:
            if isinstance(key, str):
                self.keys_names.append(key)

    def get_color_scheme(self, layer):

        referenced_key = ''

        if len(self.keys_names) and (layer.name is not None): # means we also have names included in the keys
                                     #and our layer has a name so it s worth searching
            name = layer.name
            for key_name in self.keys_names:
                if name.find(key_name) >= 0:
                    referenced_key = key_name
                    break

        final_key = referenced_key if referenced_key != '' else type(layer)

        fill= self.color_map.get(final_key, {}).get('fill', 'orange')
        outline = self.color_map.get(final_key, {}).get('outline', 'black')

        return fill, outline

def get_legend_total_width(font, num_types, types_names):
    legend_height = font.getsize("M")[1]
    tot_width = legend_height * num_types
    txt_lens = [font.getsize(txt)[0] for txt in types_names]
    txt_len = np.sum(txt_lens, axis=0)

    return tot_width + txt_len

def size_search(img_width, color_scheme):
    layer_types = color_scheme.keys
    num_types = len(layer_types)
    types_names = [layer_type if isinstance(layer_type, str) else layer_type.__name__ for layer_type in layer_types]

    fontsize = 0
    legend_width = 0
    img_width_dummy = img_width * 0.25

    # this 2 whiles are for a binary search in the fontsize space (we don t have a font height or width, but only a
    # kind of parameter, "scale", but it s very abstract, and we can check the font's height, so we perform a binary
    # search for the height
    while legend_width < img_width * 0.25:
        chgsize = 1
        font = ImageFont.truetype("arial.ttf", size=chgsize)
        legend_width = get_legend_total_width(font, num_types, types_names)

        while legend_width < img_width_dummy:
            chgsize *= 2
            font = ImageFont.truetype("arial.ttf", size=chgsize)
            legend_width = get_legend_total_width(font, num_types, types_names)

        font = ImageFont.truetype("arial.ttf", size=chgsize - 1)
        fontsize += chgsize - 1
        img_width_dummy -= get_legend_total_width(font, num_types, types_names)

    legend_height = font.getsize("M")[1]
    return legend_height, fontsize
