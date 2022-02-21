from PIL import ImageFont
from math import ceil
from .utils import *
from .layer_utils import *
from .graph import _DummyLayer


def layered_view(model, to_file: str = None, min_z: int = 20, min_xy: int = 20, max_z: int = 400,
                 max_xy: int = 2000,
                 scale_z: float = 0.1, scale_xy: float = 4, type_ignore: list = None, index_ignore: list = None,
                 color_map: dict = None, one_dim_orientation: str = 'z',
                 background_fill: Any = 'white', draw_volume: bool = True, padding: int = 10,
                 spacing: int = 10, draw_funnel: bool = True, shade_step=10, legend: bool = False,
                 font: ImageFont = None, font_color: Any = 'black') -> Image:
    """
    Generates a architecture visualization for a given linear keras model (i.e. one input and output tensor for each
    layer) in layered style (great for CNN).

    :param model: A keras model that will be visualized.
    :param to_file: Path to the file to write the created image to. If the image does not exist yet it will be created, else overwritten. Image type is inferred from the file ending. Providing None will disable writing.
    :param min_z: Minimum z size in pixel a layer will have.
    :param min_xy: Minimum x and y size in pixel a layer will have.
    :param max_z: Maximum z size in pixel a layer will have.
    :param max_xy: Maximum x and y size in pixel a layer will have.
    :param scale_z: Scalar multiplier for the z size of each layer.
    :param scale_xy: Scalar multiplier for the x and y size of each layer.
    :param type_ignore: List of layer types in the keras model to ignore during drawing.
    :param index_ignore: List of layer indexes in the keras model to ignore during drawing.
    :param color_map: Dict defining fill and outline for each layer by class type. Will fallback to default values for not specified classes.
    :param one_dim_orientation: Axis on which one dimensional layers should be drawn. Can  be 'x', 'y' or 'z'.
    :param background_fill: Color for the image background. Can be str or (R,G,B,A).
    :param draw_volume: Flag to switch between 3D volumetric view and 2D box view.
    :param padding: Distance in pixel before the first and after the last layer.
    :param spacing: Spacing in pixel between two layers
    :param draw_funnel: If set to True, a funnel will be drawn between consecutive layers
    :param shade_step: Deviation in lightness for drawing shades (only in volumetric view)
    :param legend: Add a legend of the layers to the image
    :param font: Font that will be used for the legend. Leaving this set to None, will use the default font.
    :param font_color: Color for the font if used. Can be str or (R,G,B,A).

    :return: Generated architecture image.
    """

    # Iterate over the model to compute bounds and generate boxes

    boxes = list()
    layer_y = list()
    color_wheel = ColorWheel()
    current_z = padding
    x_off = -1

    layer_types = list()

    img_height = 0
    max_right = 0

    if type_ignore is None:
        type_ignore = list()

    if index_ignore is None:
        index_ignore = list()

    if color_map is None:
        color_map = dict()

    color_scheme = ColorScheme(color_map)
    color_scheme.decode_color_map()

    id_to_num_mapping, adj_matrix = model_to_adj_matrix(model)
    model_layers = model_to_hierarchy_lists(model, id_to_num_mapping, adj_matrix)

    model_layout = ComplexVolume(id_to_num_mapping, adj_matrix, model_layers)

    layer_types = model_layout.compute_layers_3d_sizes(layer_types, type_ignore, one_dim_orientation,
                                         min_xy, min_z, scale_xy, scale_z, max_xy, max_z)

    # add fake output layers
    model_layers.append([_DummyLayer(model.output_names[i], None if True else self_multiply(model.output_shape[i])) for i in range(len(model.outputs))])
    id_to_num_mapping, adj_matrix = augment_output_layers(model, model_layers[-1], id_to_num_mapping, adj_matrix)

    boxes, img_height, img_width = model_layout.compute_boxes_coordinates(spacing, padding, shade_step, draw_volume,
                                                                          color_scheme, color_wheel)


    # Generate image
    img = Image.new('RGBA', (int(ceil(img_width)), int(ceil(img_height))), background_fill)
    draw = aggdraw.Draw(img)


    # Draw created boxes


    model_layout.draw_boxes_and_funnels(boxes, draw, draw_funnel)


    # Create layer color legend
    if legend:
        if font is None:
            font = ImageFont.load_default()

        layer_types = color_scheme.keys
        num_types = len(layer_types)

        patches = list()

        legend_height, fontsize = size_search(img_width, color_scheme)
        cube_size = legend_height

        font = ImageFont.truetype("arial.ttf", size=fontsize)

        de = cube_size // 3

        spacing = font.getsize("M")[1]

        for layer_type in layer_types:
            if not isinstance(layer_type, str):
                label = layer_type.__name__
            else:
                label = layer_type

            text_size = font.getsize(label)
            label_patch_size = (cube_size + spacing + text_size[0], cube_size + de)
            # this only works if cube_size is bigger than text height

            img_box = Image.new('RGBA', label_patch_size, background_fill)
            img_text = Image.new('RGBA', label_patch_size, (0, 0, 0, 0))
            draw_box = aggdraw.Draw(img_box)
            draw_text = ImageDraw.Draw(img_text)

            box = Box()
            box.x1 = 0
            box.x2 = box.x1 + cube_size
            box.y1 = de
            box.y2 = box.y1 + cube_size
            box.de = de
            box.shade = 10
            box.fill = color_scheme.color_map.get(layer_type, {}).get('fill', "#000000")
            box.outline = color_scheme.color_map.get(layer_type, {}).get('outline', "#000000")
            box.draw(draw_box)

            text_x = box.x2 + box.de
            text_y = (label_patch_size[1] - text_size[1]) / 2  # 2D center; use text_height and not the current label!
            draw_text.text((text_x, text_y), label, font=font, fill='black')

            draw_box.flush()
            img_box.paste(img_text, mask=img_text)
            patches.append(img_box)

        legend_image = linear_layout(patches, max_width=img.width, max_height=img.height, padding=padding,
                                     spacing=10,
                                     background_fill=background_fill, horizontal=True)
        img = vertical_image_concat(img, legend_image, background_fill=background_fill)


    if to_file is not None:
        img.save(to_file)

    return img
