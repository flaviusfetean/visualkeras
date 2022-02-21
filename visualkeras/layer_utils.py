import numpy as np
from .utils import *
from collections.abc import Iterable
import aggdraw

try:
    from tensorflow.keras.layers import Layer
except ModuleNotFoundError:
    try:
        from keras.layers import Layer
    except ModuleNotFoundError:
        pass


class SpacingDummyLayer(Layer):

    def __init__(self, spacing: int = 50):
        super().__init__()
        self.spacing = spacing


def get_incoming_layers(layer):
    for i, node in enumerate(layer._inbound_nodes):
        if isinstance(node.inbound_layers, Iterable):
            for inbound_layer in node.inbound_layers:
                yield inbound_layer
        else:  # for tf 2.3
            yield node.inbound_layers


def get_outgoing_layers(layer):
    for i, node in enumerate(layer._outbound_nodes):
        yield node.outbound_layer


def model_to_adj_matrix(model):
    if hasattr(model, 'built'):
        if not model.built:
            model.build()
    layers = model.layers

    adj_matrix = np.zeros((len(layers), len(layers)))
    id_to_num_mapping = dict()

    for layer in layers:
        layer_id = id(layer)

        if layer_id not in id_to_num_mapping:
            id_to_num_mapping[layer_id] = len(id_to_num_mapping.keys())

        for inbound_layer in get_incoming_layers(layer):
            inbound_layer_id = id(inbound_layer)

            if inbound_layer not in layers:  #correction brought
                """it is entirely possible that the layer is an input layer
                not present in the architecture but representable => it will not
                figure in the model s layers thus it is misrepresented in the 
                array, leaving us with an array with a shape too small => we have 
                to make space in the adj matrix for the input layer"""
                #adj_matrix = np.column_stack((adj_matrix, np.zeros((adj_matrix.shape[0]))))
                #adj_matrix = np.row_stack((adj_matrix, np.zeros((adj_matrix.shape[1]))))
                #model.layers.insert(model.layers.index(layer), inbound_layer)
                """does not work because it does not figure in the model layers where we look
                for it later and we get errors, so we skip this layer altogether, fuck it. Let the
                first conv layer to be the input layer
                """
                continue
            if inbound_layer_id not in id_to_num_mapping:
                id_to_num_mapping[inbound_layer_id] = len(id_to_num_mapping.keys())

            src = id_to_num_mapping[inbound_layer_id]
            tgt = id_to_num_mapping[layer_id]
            adj_matrix[src, tgt] += 1

    return id_to_num_mapping, adj_matrix


def find_layer_by_id(model, _id):
    for layer in model.layers:  # manually because get_layer does not access model.layers
        if id(layer) == _id:
            return layer
    return None


def find_layer_by_name(model, name):
    for layer in model.layers:  # manually because get_layer does not access model.layers
        if layer.name == name:
            return layer
    return None


def find_input_layers(model, id_to_num_mapping=None, adj_matrix=None):
    if adj_matrix is None:
        id_to_num_mapping, adj_matrix = model_to_adj_matrix(model)
    for i in np.where(np.sum(adj_matrix, axis=0) == 0)[0]:  # find all nodes with 0 inputs
        for key in get_keys_by_value(id_to_num_mapping, i):
            yield find_layer_by_id(model, key)


def find_output_layers(model):
    for name in model.output_names:
        yield model.get_layer(name=name)


def model_to_hierarchy_lists(model, id_to_num_mapping=None, adj_matrix=None):
    if adj_matrix is None:
        id_to_num_mapping, adj_matrix = model_to_adj_matrix(model)
    hierarchy = [set(find_input_layers(model, id_to_num_mapping, adj_matrix))]
    prev_layers = set(hierarchy[0])
    finished = False

    while not finished:
        layer = list()
        finished = True
        for start_layer in hierarchy[-1]:
            start_layer_idx = id_to_num_mapping[id(start_layer)]
            for end_layer_idx in np.where(adj_matrix[start_layer_idx] > 0)[0]:
                finished = False
                for end_layer_id in get_keys_by_value(id_to_num_mapping, end_layer_idx):
                    end_layer = find_layer_by_id(model, end_layer_id)
                    incoming_to_end_layer = set(get_incoming_layers(end_layer))
                    intersection = set(incoming_to_end_layer).intersection(prev_layers)
                    if len(intersection) == len(incoming_to_end_layer):
                        if end_layer not in layer:
                            layer.append(end_layer)
                            prev_layers.add(end_layer)
        if not finished:
            hierarchy.append(layer)

    print(hierarchy)

    return hierarchy


def augment_output_layers(model, output_layers, id_to_num_mapping, adj_matrix):

    adj_matrix = np.pad(adj_matrix, ((0, len(output_layers)), (0, len(output_layers))), mode='constant', constant_values=0)

    for dummy_output in output_layers:
        id_to_num_mapping[id(dummy_output)] = len(id_to_num_mapping.keys())

    for i, output_layer in enumerate(find_output_layers(model)):
        output_layer_idx = id_to_num_mapping[id(output_layer)]
        dummy_layer_idx = id_to_num_mapping[id(output_layers[i])]

        adj_matrix[output_layer_idx, dummy_layer_idx] += 1

    return id_to_num_mapping, adj_matrix


def is_internal_input(layer):
    try:
        import tensorflow.python.keras.engine.input_layer.InputLayer
        if isinstance(layer, tensorflow.python.keras.engine.input_layer.InputLayer):
            return True
    except ModuleNotFoundError:
        pass

    try:
        import keras
        if isinstance(layer, keras.engine.input_layer.InputLayer):
            return True
    except ModuleNotFoundError:
        pass

    return False


######################### My addition - for creating complex volumetric visualizations ################################
#######################################################################################################################


class ComplexVolume():
    def __init__(self, id_to_num_mapping, adj_matrix, hierarchy):
        self.id_to_num_mapping = id_to_num_mapping
        self.box_idx_to_num_mapping = np.zeros(shape=(len(id_to_num_mapping)))
        self.adj_matrix = adj_matrix
        self.layer_tree = hierarchy
        self.layers_3d_sizes = []
        self.layers_3d_max_xys = []
        self.boxes_sizes = []

    def compute_layers_3d_sizes(self,layer_types, type_ignore, one_dim_orientation,
                                min_xy, min_z, scale_xy, scale_z, max_xy, max_z):
        for lvl in self.layer_tree:
            max_x, max_zet, max_y = 0, 0, 0
            crt_boxes = []
            for layer in lvl:
                """Boxes sizes part"""
                # Ignore layers that the use has opted out to
                if type(layer) in type_ignore:
                    continue

                # Do no render the SpacingDummyLayer, just increase the pointer
                if type(layer) == SpacingDummyLayer:
                    current_z += layer.spacing
                    continue

                layer_type = type(layer)

                if layer_type not in layer_types:
                    layer_types.append(layer_type)

                x = min_xy
                y = min_xy
                z = min_z

                if isinstance(layer.output_shape, tuple):
                    shape = layer.output_shape
                elif isinstance(layer.output_shape, list) and len(
                        layer.output_shape) == 1:  # drop dimension for non seq. models
                    shape = layer.output_shape[0]
                else:
                    raise RuntimeError(f"not supported tensor shape {layer.output_shape}")

                if len(shape) >= 4:
                    x = min(max(shape[1] * scale_xy, x), max_xy)
                    y = min(max(shape[2] * scale_xy, y), max_xy)
                    z = min(max(self_multiply(shape[3:]) * scale_z, z), max_z)
                elif len(shape) == 3:
                    x = min(max(shape[1] * scale_xy, x), max_xy)
                    y = min(max(shape[2] * scale_xy, y), max_xy)
                    z = min(max(z, z), max_z)
                elif len(shape) == 2:
                    if one_dim_orientation == 'x':
                        x = min(max(shape[1] * scale_xy, x), max_xy)
                    elif one_dim_orientation == 'y':
                        y = min(max(shape[1] * scale_xy, y), max_xy)
                    elif one_dim_orientation == 'z':
                        z = min(max(shape[1] * scale_z, z), max_z)
                    else:
                        raise ValueError(f"unsupported orientation {one_dim_orientation}")
                else:
                    raise RuntimeError(f"not supported tensor shape {layer.output_shape}")

                crt_boxes.append([x, y, z])
                max_x, max_y, max_zet = max(x, max_x), max(y, max_y), max(z, max_zet)

                """End boxes sizes part"""

            self.boxes_sizes.append(crt_boxes)
            self.layers_3d_max_xys.append((max_y, max_x))

            y_3d_layer = max_y * len(lvl) + (max_y // 3) * (3 * len(lvl))
                        #^height * ^ nr of layers + ^depth * ^ 2*(nr of layers=nr of depths + nr of spaces between)
            x_3d_layer = max_x
            z_3d_layer = max_zet
            self.layers_3d_sizes.append([x_3d_layer, y_3d_layer, z_3d_layer])
        return layer_types

    def compute_boxes_coordinates(self, spacing, padding, shade_step, draw_volume, color_scheme, color_wheel):

        img_height = max([shape[1] for shape in self.layers_3d_sizes])
        img_width = np.sum([spacing + shape[2] for shape in self.layers_3d_sizes], axis=0)

        current_z = padding
        x_off = -1

        boxes_final = []

        crt_idx = 0

        for i, (boxes, lvl) in enumerate(zip(self.boxes_sizes, self.layer_tree)):
            local_y_off = (img_height - self.layers_3d_sizes[i][1]) / 2
            de_off = 0
            for j, (shape, layer) in enumerate(zip(boxes, lvl)):

                self.box_idx_to_num_mapping[crt_idx] = self.id_to_num_mapping[id(layer)]
                crt_idx += 1

                [x, y, z] = shape
                y_off = local_y_off + j * (self.layers_3d_max_xys[i][1] + self.layers_3d_max_xys[i][0] // 3)

                box = Box()
                box.de = 0
                if draw_volume:
                    box.de = x / 3
                if x_off == -1: # the first layer goes outside si we have to introduce an offset on x_axis
                    x_off = box.de / 2

                # top left coordinate
                box.x1 = x_off + current_z - box.de / 2
                box.y1 = box.de + y_off + de_off

                de_off += box.de

                # bottom right coordinate
                box.x2 = box.x1 + z
                box.y2 = box.y1 + y

                box.fill, box.outline = color_scheme.get_color_scheme(layer)

                box.shade = shade_step
                #layer_y.append(box.y2 - (box.y1 - box.de))
                boxes_final.append(box)
            current_z += (self.layers_3d_sizes[i])[2] + spacing

        return boxes_final, img_height, img_width

    def draw_boxes_and_funnels(self, boxes, draw, draw_funnel):

        for i, box in enumerate(boxes):
            pen = aggdraw.Pen(get_rgba_tuple(box.fill))
            if box.fill == get_rgba_tuple('white'):
                pen = aggdraw.Pen(get_rgba_tuple(get_rgba_tuple('black')))

            box.draw(draw)

            crt_layer_num = int(self.box_idx_to_num_mapping[i])
            next_layers_nums = np.where(self.adj_matrix[crt_layer_num] > 0)[0]

            if draw_funnel:
                for layer_num in next_layers_nums:

                    nxt_idx_array = np.where(self.box_idx_to_num_mapping == layer_num)[0]
                    next_box_idx = nxt_idx_array[0] if len(nxt_idx_array) else None

                    if next_box_idx == None:
                        continue

                    next_box = boxes[next_box_idx]

                    draw.line([box.x2 + box.de, box.y1 - box.de,
                               next_box.x1 + next_box.de, next_box.y1 - next_box.de], pen)

                    draw.line([box.x2 + box.de, box.y2 - box.de,
                               next_box.x1 + next_box.de, next_box.y2 - next_box.de], pen)

                    draw.line([box.x2, box.y2,
                               next_box.x1, next_box.y2], pen)

                    draw.line([box.x2, box.y1,
                               next_box.x1, next_box.y1], pen)

        draw.flush()