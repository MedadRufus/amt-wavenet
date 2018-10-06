import numpy as np
import tensorflow as tf

from .ops import causal_conv

HKEY = 'histograms'


def create_variable(name, shape):
    '''Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialition.'''
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable

def create_bias_variable(name, shape):
    '''Create a bias variable with the specified name and shape and initialize
    it to zero.'''
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name)


class WaveNetModel(object):
    '''Implements the WaveNet4Transcription (WN4T) network.

    Usage (with the architecture as in the DeepMind paper):
        dilations = [2**i for i in range(N)] * M
        filter_width = 2  # Convolutions just use 2 samples.
        residual_channels = 16  # Not specified in the paper.
        dilation_channels = 32  # Not specified in the paper.
        skip_channels = 16      # Not specified in the paper.
        net = WaveNetModel(batch_size, dilations, filter_width,
                           residual_channels, dilation_channels,
                           skip_channels)
        loss = net.loss(input_batch)
    '''

    def __init__(self,
                 batch_size,
                 dilations,
                 filter_width,
                 residual_channels,
                 dilation_channels,
                 skip_channels,
                 output_channels=2**7,
                 use_biases=False,
                 initial_filter_width=32):
        '''Initializes the WaveNet model.

        Args:
            batch_size: How many audio files are supplied per batch
                (recommended: 1).
            dilations: A list with the dilation factor for each layer.
            filter_width: The samples that are included in each convolution,
                after dilating.
            residual_channels: How many filters to learn for the residual.
            dilation_channels: How many filters to learn for the dilated
                convolution.
            skip_channels: How many filters to learn that contribute to the
                sigmoid output note probabilities.
            output_channels: How many notes are predicted on the output.
                Default: 128 (as given by MIDI standard).
            use_biases: Whether to add a bias layer to each convolution.
                Default: False.
            initial_filter_width: The width of the initial filter of the
                convolution applied to the scalar input.
        '''

        self.batch_size = batch_size
        self.dilations = dilations
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.output_channels = output_channels
        self.use_biases = use_biases
        self.skip_channels = skip_channels
        self.initial_filter_width = initial_filter_width

        self.receptive_field = WaveNetModel.calculate_receptive_field(
            self.filter_width, self.dilations, self.initial_filter_width)
        self.variables = self._create_variables()

    @staticmethod
    def calculate_receptive_field(filter_width, dilations,
                                  initial_filter_width):
        receptive_field = (filter_width - 1) * sum(dilations) + 1
        receptive_field += initial_filter_width - 1
        return receptive_field

    def _create_variables(self):
        '''This function creates all variables used by the network.
        This allows us to share them between multiple calls to the loss
        function and generation function.'''

        var = dict()

        with tf.variable_scope('wavenet'):
            with tf.variable_scope('causal_layer'):
                layer = dict()
                initial_channels = 1
                initial_filter_width = self.initial_filter_width
                layer['filter'] = create_variable(
                    'filter',
                    [initial_filter_width,
                     initial_channels,
                     self.residual_channels])
                var['causal_layer'] = layer

            var['dilated_stack'] = list()
            with tf.variable_scope('dilated_stack'):
                for i, dilation in enumerate(self.dilations):
                    with tf.variable_scope('layer{}'.format(i)):
                        current = dict()
                        current['filter'] = create_variable(
                            'filter',
                            [self.filter_width,
                             self.residual_channels,
                             self.dilation_channels])
                        current['gate'] = create_variable(
                            'gate',
                            [self.filter_width,
                             self.residual_channels,
                             self.dilation_channels])
                        current['dense'] = create_variable(
                            'dense',
                            [1,
                             self.dilation_channels,
                             self.residual_channels])
                        current['skip'] = create_variable(
                            'skip',
                            [1,
                             self.dilation_channels,
                             self.skip_channels])

                        if self.use_biases:
                            current['filter_bias'] = create_bias_variable(
                                'filter_bias',
                                [self.dilation_channels])
                            current['gate_bias'] = create_bias_variable(
                                'gate_bias',
                                [self.dilation_channels])
                            current['dense_bias'] = create_bias_variable(
                                'dense_bias',
                                [self.residual_channels])
                            current['skip_bias'] = create_bias_variable(
                                'slip_bias',
                                [self.skip_channels])

                        var['dilated_stack'].append(current)

            with tf.variable_scope('postprocessing'):
                current = dict()
                current['postprocess1'] = create_variable(
                    'postprocess1',
                    [1, self.skip_channels, self.skip_channels])
                current['postprocess2'] = create_variable(
                    'postprocess2',
                    [1, self.skip_channels, self.output_channels])
                if self.use_biases:
                    current['postprocess1_bias'] = create_bias_variable(
                        'postprocess1_bias',
                        [self.skip_channels])
                    current['postprocess2_bias'] = create_bias_variable(
                        'postprocess2_bias',
                        [self.output_channels])
                var['postprocessing'] = current

        return var

    def _create_causal_layer(self, input_batch):
        '''Creates a single causal convolution layer.

        The layer can change the number of channels.
        '''
        with tf.name_scope('causal_layer'):
            weights_filter = self.variables['causal_layer']['filter']
            return causal_conv(input_batch, weights_filter, 1)

    def _create_dilation_layer(self, input_batch, layer_index, dilation,
                               output_width):
        '''Creates a single causal dilated convolution layer.

        Args:
             input_batch: Input to the dilation layer.
             layer_index: Integer indicating which layer this is.
             dilation: Integer specifying the dilation size.
             global_conditioning_batch: Tensor containing the global data upon
                 which the output is to be conditioned upon. Shape:
                 [batch size, 1, channels]. The 1 is for the axis
                 corresponding to time so that the result is broadcast to
                 all time steps.

        The layer contains a gated filter that connects to dense output
        and to a skip connection:

               |-> [gate]   -|        |-> 1x1 conv -> skip output
               |             |-> (*) -|
        input -|-> [filter] -|        |-> 1x1 conv -|
               |                                    |-> (+) -> dense output
               |------------------------------------|

        Where `[gate]` and `[filter]` are causal convolutions with a
        non-linear activation at the output. Biases and global conditioning
        are omitted due to the limits of ASCII art.

        '''
        variables = self.variables['dilated_stack'][layer_index]

        weights_filter = variables['filter']
        weights_gate = variables['gate']

        conv_filter = causal_conv(input_batch, weights_filter, dilation)
        conv_gate = causal_conv(input_batch, weights_gate, dilation)

        if self.use_biases:
            filter_bias = variables['filter_bias']
            gate_bias = variables['gate_bias']
            conv_filter = tf.add(conv_filter, filter_bias)
            conv_gate = tf.add(conv_gate, gate_bias)

        out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)

        # The 1x1 conv to produce the residual output
        weights_dense = variables['dense']
        transformed = tf.nn.conv1d(
            out, weights_dense, stride=1, padding='SAME', name='dense')

        # The 1x1 conv to produce the skip output
        skip_cut = tf.shape(out)[1] - output_width
        out_skip = tf.slice(out, [0, skip_cut, 0], [-1, -1, -1])
        weights_skip = variables['skip']
        skip_contribution = tf.nn.conv1d(
            out_skip, weights_skip, stride=1, padding='SAME', name='skip')

        if self.use_biases:
            dense_bias = variables['dense_bias']
            skip_bias = variables['skip_bias']
            transformed = transformed + dense_bias
            skip_contribution = skip_contribution + skip_bias

        layer = 'layer{}'.format(layer_index)
        tf.summary.histogram(layer + '_filter', weights_filter, [HKEY])
        tf.summary.histogram(layer + '_gate', weights_gate, [HKEY])
        tf.summary.histogram(layer + '_dense', weights_dense, [HKEY])
        tf.summary.histogram(layer + '_skip', weights_skip, [HKEY])
        if self.use_biases:
            tf.summary.histogram(layer + '_biases_filter', filter_bias, [HKEY])
            tf.summary.histogram(layer + '_biases_gate', gate_bias, [HKEY])
            tf.summary.histogram(layer + '_biases_dense', dense_bias, [HKEY])
            tf.summary.histogram(layer + '_biases_skip', skip_bias, [HKEY])

        input_cut = tf.shape(input_batch)[1] - tf.shape(transformed)[1]
        input_batch = tf.slice(input_batch, [0, input_cut, 0], [-1, -1, -1])

        return skip_contribution, input_batch + transformed

    def _create_network(self, input_batch):
        '''Construct the WaveNet network.'''
        outputs = []
        current_layer = input_batch

        # Pre-process the input with a regular convolution
        initial_channels = 1

        current_layer = self._create_causal_layer(current_layer)

        output_width = tf.shape(input_batch)[1] - self.receptive_field + 1

        # Add all defined dilation layers.
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):
                    output, current_layer = self._create_dilation_layer(
                        current_layer, layer_index, dilation,
                        output_width)
                    outputs.append(output)

        with tf.name_scope('postprocessing'):
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
            w1 = self.variables['postprocessing']['postprocess1']
            w2 = self.variables['postprocessing']['postprocess2']
            if self.use_biases:
                b1 = self.variables['postprocessing']['postprocess1_bias']
                b2 = self.variables['postprocessing']['postprocess2_bias']

            tf.summary.histogram('postprocess1_weights', w1, [HKEY])
            tf.summary.histogram('postprocess2_weights', w2, [HKEY])
            if self.use_biases:
                tf.summary.histogram('postprocess1_biases', b1, [HKEY])
                tf.summary.histogram('postprocess2_biases', b2, [HKEY])

            # We skip connections from the outputs of each layer, adding them
            # all up here.
            total = sum(outputs)
            transformed1 = tf.nn.relu(total)
            conv1 = tf.nn.conv1d(transformed1, w1, stride=1, padding='SAME')
            if self.use_biases:
                conv1 = tf.add(conv1, b1)
            transformed2 = tf.nn.relu(conv1)
            conv2 = tf.nn.conv1d(transformed2, w2, stride=1, padding='SAME')
            if self.use_biases:
                conv2 = tf.add(conv2, b2)

        return conv2

    def loss(self,
             input_data,
             input_labels,
             pos_weight,
             l2_reg_str=None,
             name='wavenet'):
        '''Creates the AMT WaveNet network and returns the transcription loss.

        The variables are all scoped to the given name.
        '''
        with tf.name_scope(name):

            network_input = tf.reshape(
                tf.to_float(input_data),
                [self.batch_size, -1, 1])

            raw_output = self._create_network(network_input)

            with tf.name_scope('loss'):
                target_output = tf.reshape(input_labels,
                                           [-1, self.output_channels])
                prediction = tf.reshape(raw_output,
                                        [-1, self.output_channels])
                probabilities = tf.nn.sigmoid(prediction)

                loss = tf.nn.weighted_cross_entropy_with_logits(
                    targets=target_output,
                    logits=prediction,
                    pos_weight=pos_weight)
                reduced_loss = tf.reduce_mean(loss)

                tf.summary.scalar('loss', reduced_loss)

                if l2_reg_str is None or l2_reg_str == 0:
                    return reduced_loss, probabilities
                else:
                    # L2 regularization for all trainable parameters
                    l2_loss = tf.add_n([tf.nn.l2_loss(v)
                                        for v in tf.trainable_variables()
                                        if not('bias' in v.name)])

                    # Add the regularization term to the loss
                    total_loss = (reduced_loss +
                                  l2_reg_str * l2_loss)

                    tf.summary.scalar('l2_loss', l2_loss)
                    tf.summary.scalar('total_loss', total_loss)

                    return total_loss, probabilities
