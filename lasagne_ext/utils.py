import pickle, gzip, warnings, pydot, numpy as np
from lasagne.layers import get_all_layers, get_all_param_values, set_all_param_values

class model_visual(object):
    @staticmethod
    def _get_hex_color(layer_type):
        """
        Determines the hex color for a layer. Some classes are given
        default values, all others are calculated pseudorandomly
        from their name.
        :parameters:
            - layer_type : string
                Class name of the layer

        :returns:
            - color : string containing a hex color.

        :usage:
            >>> color = get_hex_color('MaxPool2DDNN')
            '#9D9DD2'
        """

        if 'Input' in layer_type:
            return '#A2CECE'
        if 'Conv' in layer_type:
            return '#7C9ABB'
        if 'Dense' in layer_type:
            return '#6CCF8D'
        if 'Pool' in layer_type:
            return '#9D9DD2'
        else:
            return '#{0:x}'.format(hash(layer_type) % 2 ** 24)

    @classmethod
    def _get_pydot_graph(self, layers, output_shape=True, fill=False, verbose=False):
        """
        Creates a PyDot graph of the network defined by the given layers.
        :parameters:
            - layers : list
                List of the layers, as obtained from lasange.layers.get_all_layers
            - output_shape: (default `True`)
                If `True`, the output shape of each layer will be displayed.
            - verbose: (default `False`)
                If `True`, layer attributes like filter shape, stride, etc.
                will be displayed.
            - verbose:
        :returns:
            - pydot_graph : PyDot object containing the graph

        """
        pydot_graph = pydot.Dot('Network', graph_type='digraph')
        pydot_nodes = {}
        pydot_edges = []
        for i, layer in enumerate(layers):
            layer_type = '{0}'.format(layer.__class__.__name__)
            key = repr(layer)
            label = layer_type
            color = '#FFFFFF'
            if fill:
                color = self._get_hex_color(layer_type)
            if verbose:
                for attr in ['num_filters', 'num_units', 'ds',
                             'filter_shape', 'stride', 'strides', 'p', 'name']:
                    if hasattr(layer, attr):
                        label += '\n' + \
                                 '{0}: {1}'.format(attr, getattr(layer, attr))
                if hasattr(layer, 'nonlinearity'):
                    try:
                        nonlinearity = layer.nonlinearity.__name__
                    except AttributeError:
                        nonlinearity = layer.nonlinearity.__class__.__name__
                    label += '\n' + 'nonlinearity: {0}'.format(nonlinearity)

            if output_shape:
                label += '\n' + \
                         'Output shape: {0}'.format(layer.output_shape)
                # 'Output shape: {0}'.format(layer.get_output_shape())
            pydot_nodes[key] = pydot.Node(key,
                                          label=label,
                                          shape='record',
                                          fillcolor=color,
                                          style='filled',
                                          )

            if hasattr(layer, 'input_layers'):
                for input_layer in layer.input_layers:
                    pydot_edges.append([repr(input_layer), key])

            if hasattr(layer, 'input_layer'):
                pydot_edges.append([repr(layer.input_layer), key])

        for node in pydot_nodes.values():
            pydot_graph.add_node(node)
        for edge in pydot_edges:
            pydot_graph.add_edge(
                pydot.Edge(pydot_nodes[edge[0]], pydot_nodes[edge[1]]))
        return pydot_graph

    @classmethod
    def draw_to_file(self, layers, filename, **kwargs):
        """
        Draws a network diagram to a file
        :parameters:
            - layers : list
                List of the layers, as obtained from lasange.layers.get_all_layers
            - filename: string
                The filename to save output to.
            - **kwargs: see docstring of get_pydot_graph for other options
        """
        dot = self._get_pydot_graph(layers, **kwargs)

        ext = filename[filename.rfind('.') + 1:]
        with open(filename, 'wb') as fid:
            fid.write(dot.create(format=ext))

    @classmethod
    def draw_to_notebook(self, layers, **kwargs):
        """
        Draws a network diagram in an IPython notebook
        :parameters:
            - layers : list
                List of the layers, as obtained from lasange.layers.get_all_layers
            - **kwargs: see docstring of get_pydot_graph for other options
        """
        from IPython.display import Image  # needed to render in notebook
        dot = self._get_pydot_graph(layers, **kwargs)
        return Image(dot.create_png())

def plot_model(model, filename, **kwargs):
    """
    Plot model structure and save into file
    :param model:
    :param filename:
    :param kwargs:
    :return:
    """
    model_layers = get_all_layers(model)
    model_visual.draw_to_file(layers=model_layers, filename=filename, **kwargs)

def count_layer_params(layer, unwrap_shared=True, **tags):
    """
    Return parameter numbers for a layer, if you need to count all parameters of a model, use Lasagne's count_params() instead
    :param layer:
    :param unwrap_shared:
    :param tags:
    :return:
    """
    params = layer.get_params(unwrap_shared=unwrap_shared, **tags)
    shapes = [p.get_value().shape for p in params]
    counts = [np.prod(shape) for shape in shapes]
    return sum(counts)

def _print_row(fields, positions):
    line = ''
    for i in range(len(fields)):
        line += str(fields[i])
        line = line[:positions[i]]
        line += ' ' * (positions[i] - len(line))
    print(line)

def print_layer_summary(layer, positions=[.33, .55, .67, 1.]):
    output_shape = layer.output_shape
    parent_layer_name = ''
    if hasattr(layer, 'input_layer'):
        parent_layer_name = layer.input_layer.name
    elif hasattr(layer, 'input_layers'):
        for l in layer.input_layers:
            parent_layer_name += l.name + ', '
        parent_layer_name = parent_layer_name[:-2]
    else:
        pass

    fields = [layer.name + ' (' + layer.__class__.__name__ + ')', output_shape, count_layer_params(layer), parent_layer_name]
    _print_row(fields, positions)

def print_model_summary(model, line_length=120, positions=[.3, .55, .65, 1.]):
    """
    Print model summary, like in Keras model.summary()
    :param model:
    :param line_length: total length of printed lines
    :param positions: relative or absolute positions of log elements in each line
    :return:
    """
    layers = get_all_layers(model)

    if positions[-1] <= 1:
        positions = [int(line_length * p) for p in positions]
    to_display = ['Layer (type)', 'Output Shape', 'Param #', 'Connected to']

    print('_' * line_length)
    _print_row(to_display, positions)
    print('=' * line_length)

    total_params = 0
    for i in range(len(layers)):
        print_layer_summary(layers[i], positions=positions)
        if i == len(layers) - 1:
            print('=' * line_length)
        else:
            print('_' * line_length)
        total_params += count_layer_params(layers[i])

    print('Total params: %s' % total_params)
    print('_' * line_length)

class chunked_byte_writer(object):
    """
    This class is used for by-passing the bug in gzip/zlib library: when data length exceeds unsigned int limit, gzip/zlib will break
    file: a file object
    """
    def __init__(self, file, chunksize=4294967295):
        self.file = file
        self.chunksize = chunksize
    def write(self, data):
        for i in range(0, len(data), self.chunksize):
            self.file.write(data[i:i+self.chunksize])

class gpickle(object):
    """
    A pickle class with gzip enabled
    """
    @staticmethod
    def dump(data, filename):
        with gzip.open(filename, mode='wb') as f:
            pickle.dump(data, chunked_byte_writer(f))
            f.close()
    @staticmethod
    def load(filename):
        """
        The chunked read mechanism here is for by-passing the bug in gzip/zlib library: when
        data length exceeds unsigned int limit, gzip/zlib will break
        :param filename:
        :return:
        """
        buf = b''
        chunk = b'NULL'
        with gzip.open(filename, mode='rb') as f:
            while len(chunk) > 0:
                chunk = f.read(429496729)
                buf += chunk
        data = pickle.loads(buf)
        return data

    @staticmethod
    def loads(buf):
        return pickle.loads(buf)

def save_model_raw(model, filename, userdata=None):
    """
    Save model weights and optional userdata into .gpkl file
    If layer names are not needed, use 'save_model_raw()' and 'load_model_raw()' may save same space if there're layers sharing weights
    :param model:
    :param filename:
    :param userdata:
    :return:
    """
    model_para_values = get_all_param_values(model)
    gpickle.dump((model_para_values, userdata), filename)

def load_model_raw(model, filename):
    """
    Load weights from file into model, return userdata
    :param model:
    :param filename:
    :return:
    """
    model_para_values, userdata = gpickle.load(filename)
    set_all_param_values(model, model_para_values)
    return userdata

def save_model(filename, model, userdata=None, unwrap_shared=True, **tags):
    """
    Save model weights plus layer names and optional userdata into .gpkl file
    :param filename:
    :param model:
    :param userdata:
    :param unwrap_shared:
    :param tags:
    :return:
    """
    layers = get_all_layers(model)
    model_para_values = []
    layer_idx = 0
    for layer in layers:
        name = layer.name
        if name is None:
            name = '~layer_%d' % layer_idx    # default name starts with '~'
        layer_params = layer.get_params(unwrap_shared=unwrap_shared, **tags)
        layer_values = [p.get_value() for p in layer_params]
        model_para_values.append((name, layer_values))
        layer_idx += 1
    gpickle.dump((model_para_values, userdata), filename)

def load_model(filename, model=None, check_layername=True, unwrap_shared=True, **tags):
    """
    Load model weights and userdata from file which was saved by 'save_model()'
    :param filename:
    :param model: optional, if given, the model weights will be set with 'model_para_values' from the file
    :param check_layername: whether check layer name consistency when setting model weights
    :param unwrap_shared:
    :param tags:
    :return: (model_para_values, userdata) if 'model=None', otherwise only userdata is returned
    """
    model_para_values, userdata = gpickle.load(filename)
    if model is None:
        return model_para_values, userdata
    else:
        layers = get_all_layers(model)
        layer_idx = 0
        for layer in layers:
            layer_params = layer.get_params(unwrap_shared=unwrap_shared, **tags)
            name, layer_values = model_para_values[layer_idx]
            if check_layername and name != layer.name and not name.startswith('~layer_'):
                warnings.warn('Inconsistent layer name: %s <-> %s at %dth layer' % (name, layer.name, layer_idx))
            for p, v in zip(layer_params, layer_values):
                p.set_value(v)
            layer_idx += 1
        return userdata

def set_weights(model, model_para_values, mapping_dict='auto', unwrap_shared=True, **tags):
    """
    Set model layers' weights by 'mapping_dict' or natural order.
    When 'mapping_dict' is 'auto', then a mapping dict will be built automatically by common layer names between model and
    model_para_values
    :param model:
    :param model_para_values: list of tuples (name, layer_values)
    :param mapping_dict: {None, 'auto', or dict with format of {target_layer_name: source_layer_name}}
    :param unwrap_shared:
    :param tags:
    :return:
    """
    layers = get_all_layers(model)
    #-- if mapping_dict is not given, then the model weights will be set by natural order, from beginning layer to ending layer
    #-- if len(model_para_values) != len(layers), the iteration will stop at either one the shortest
    if mapping_dict is None:
        for layer, layer_values_with_name in zip(layers, model_para_values):
            name, layer_values = layer_values_with_name
            layer_params = layer.get_params(unwrap_shared=unwrap_shared, **tags)
            for p, v in zip(layer_params, layer_values):
                p.set_value(v)
    else:
        #--- build a mapping dict automatically ---#
        if mapping_dict == 'auto':
            target_layer_names = set()
            source_layer_names = set()
            for layer in layers:
                target_layer_names.add(layer.name)
            for name, _ in model_para_values:
                source_layer_names.add(name)
            mapping_dict = dict()
            for name in target_layer_names & source_layer_names:
                mapping_dict[name] = name
        #--- do the mapping ---#
        for target_name in mapping_dict:
            source_name = mapping_dict[target_name]
            for layer in layers:
                if layer.name == target_name:
                    target_layer = layer
                    break
            for name, layer_values in model_para_values:
                if name == source_name:
                    break
            layer_params = target_layer.get_params(unwrap_shared=unwrap_shared, **tags)
            for p, v in zip(layer_params, layer_values):
                p.set_value(v)

def freeze_layer(layer):
    """
    Freeze a layer, so its weights won't be updated during training
    :param layer:
    :return:
    """
    for param in layer.params:
        layer.params[param].discard('trainable')
    return layer  # optional, if you want to use it in-line

def get_layer_by_name(model, name):
    layers = get_all_layers(model)
    for layer in layers:
        if layer.name == name:
            return layer








