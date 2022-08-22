from keras_applications import get_submodules_from_kwargs
import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow_addons as tfa
from .utils import get_layer_number, to_tuple
from .._utils import freeze_model, filter_keras_submodules
from ..base_model import BaseModel
from ...backbones.backbones_factory import Backbones

backend = None
layers = None
models = None
keras_utils = None


def Xnet(backbone_name='efficientnetb0',
         input_shape=(None, None, 3),
         input_tensor=None,
         encoder_weights='imagenet',
         freeze_encoder=False,
         skip_connections='default',
         decoder_block_type='upsampling',
         decoder_filters=(256, 128, 64, 32, 16),
         decoder_use_batchnorm=True,
         n_upsample_blocks=5,
         upsample_rates=(2, 2, 2, 2, 2),
         classes=1,
         activation='sigmoid',
         tta=False,
         **kwargs):
    global backend, layers, models, keras_utils
    submodule_args = filter_keras_submodules(kwargs)
    backend, layers, models, keras_utils = get_submodules_from_kwargs(submodule_args)

    backbone = Backbones.get_backbone(
        backbone_name,
        input_shape=input_shape,
        weights=encoder_weights,
        include_top=False,
        **kwargs,
    )

    if skip_connections == 'default':
        skip_connections = Backbones.get_feature_layers(backbone_name, n=4)

    model = build_xnet(backbone,
                       classes,
                       skip_connections,
                       decoder_filters=decoder_filters,
                       block_type=decoder_block_type,
                       activation=activation,
                       n_upsample_blocks=n_upsample_blocks,
                       upsample_rates=upsample_rates,
                       use_batchnorm=decoder_use_batchnorm,
                       tta=tta)

    # lock encoder weights for fine-tuning
    if freeze_encoder:
        freeze_model(backbone)

    return model


def build_xnet(backbone, classes, skip_connection_layers,
               decoder_filters=(256, 128, 64, 32, 16),
               upsample_rates=(2, 2, 2, 2, 2),
               n_upsample_blocks=5,
               block_type='upsampling',
               activation='sigmoid',
               use_batchnorm=True,
               tta=False):
    tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
    input = backbone.input
    if block_type == 'transpose':
        up_block = Transpose2D_block
    else:
        up_block = Upsample2D_block

    if len(skip_connection_layers) > n_upsample_blocks:
        downsampling_layers = skip_connection_layers[int(len(skip_connection_layers) / 2):]
        skip_connection_layers = skip_connection_layers[:int(len(skip_connection_layers) / 2)]
    else:
        downsampling_layers = skip_connection_layers

    # convert layer names to indices
    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                            for l in skip_connection_layers])
    skip_layers_list = [backbone.layers[skip_connection_idx[i]].output for i in range(len(skip_connection_idx))]

    downsampling_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                         for l in downsampling_layers])
    downsampling_list = [backbone.layers[downsampling_idx[i]].output for i in range(len(downsampling_idx))]
    downterm = [None] * (n_upsample_blocks + 1)

    for i in range(len(downsampling_idx)):
        downterm[n_upsample_blocks - i - 1] = downsampling_list[i]

    downterm[-1] = backbone.output
    interm = [None] * (n_upsample_blocks + 1) * (n_upsample_blocks + 1)
    for i in range(len(skip_connection_idx)):
        interm[-i * (n_upsample_blocks + 1) + (n_upsample_blocks + 1) * (n_upsample_blocks - 1)] = skip_layers_list[i]

    interm[(n_upsample_blocks + 1) * n_upsample_blocks] = backbone.output
    for j in range(n_upsample_blocks):
        for i in range(n_upsample_blocks - j):
            upsample_rate = to_tuple(upsample_rates[i])

            if i == 0 and j < n_upsample_blocks - 1 and len(skip_connection_layers) < n_upsample_blocks:
                interm[(n_upsample_blocks + 1) * i + j + 1] = None
            elif j == 0:
                if downterm[i + 1] is not None:
                    interm[(n_upsample_blocks + 1) * i + j + 1] = up_block(decoder_filters[n_upsample_blocks - i - 2],
                                                                           i + 1, j + 1, upsample_rate=upsample_rate,
                                                                           skip=interm[(n_upsample_blocks + 1) * i + j],
                                                                           use_batchnorm=use_batchnorm)(downterm[i + 1])
                else:
                    interm[(n_upsample_blocks + 1) * i + j + 1] = None
            else:
                interm[(n_upsample_blocks + 1) * i + j + 1] = up_block(decoder_filters[n_upsample_blocks - i - 2],
                                                                       i + 1, j + 1, upsample_rate=upsample_rate,
                                                                       skip=interm[(n_upsample_blocks + 1) * i: (
                                                                                                                        n_upsample_blocks + 1) * i + j + 1],
                                                                       use_batchnorm=use_batchnorm)(
                    interm[(n_upsample_blocks + 1) * (i + 1) + j])

    x = layers.Conv2D(classes, (3, 3), padding='same', name='final_conv')(interm[n_upsample_blocks])
    x = layers.Activation(activation, name=activation, dtype='float32')(x)
    # x = layers.Activation(activation, name=activation)(x)

    aaf = False
    if aaf:
        model = BaseModel(input, x)
    else:
        model = Model(input, x)

    return model


def handle_block_names(stage, cols):
    conv_name = 'decoder_stage{}-{}_conv'.format(stage, cols)
    bn_name = 'decoder_stage{}-{}_bn'.format(stage, cols)
    relu_name = 'decoder_stage{}-{}_relu'.format(stage, cols)
    up_name = 'decoder_stage{}-{}_upsample'.format(stage, cols)
    merge_name = 'merge_{}-{}'.format(stage, cols)
    return conv_name, bn_name, relu_name, up_name, merge_name


def ConvRelu(filters, kernel_size, use_batchnorm=False, conv_name='conv', bn_name='bn', relu_name='relu'):
    def layer(x):
        x = layers.Conv2D(filters, kernel_size, padding="same", name=conv_name, use_bias=not (use_batchnorm))(x)
        if use_batchnorm:
            # x = layers.BatchNormalization(name=bn_name)(x)
            x = tfa.layers.GroupNormalization(groups=16, name=bn_name)(x)
        x = layers.Activation('relu', name=relu_name)(x)
        return x

    return layer


def Upsample2D_block(filters, stage, cols, kernel_size=(3, 3), upsample_rate=(2, 2),
                     use_batchnorm=False, skip=None):
    def layer(input_tensor):
        conv_name, bn_name, relu_name, up_name, merge_name = handle_block_names(stage, cols)
        x = layers.UpSampling2D(size=upsample_rate, name=up_name)(input_tensor)
        if (type(skip) != list and skip is not None) or (type(skip) == list and None not in skip):
            if type(skip) is list:
                x = layers.Concatenate(name=merge_name)([x] + skip)
            else:
                x = layers.Concatenate(name=merge_name)([x, skip])
        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '1', bn_name=bn_name + '1', relu_name=relu_name + '1')(x)

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)
        return x

    return layer


def Transpose2D_block(filters, stage, cols, kernel_size=(3, 3), upsample_rate=(2, 2),
                      transpose_kernel_size=(4, 4), use_batchnorm=False, skip=None):
    def layer(input_tensor):
        conv_name, bn_name, relu_name, up_name, merge_name = handle_block_names(stage, cols)
        x = layers.Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate,
                                   padding='same', name=up_name, use_bias=not (use_batchnorm))(input_tensor)
        if use_batchnorm:
            # x = layers.BatchNormalization(name=bn_name + '1')(x)
            x = tfa.layers.GroupNormalization(groups=16, name=bn_name + '1')(x)
        x = layers.Activation('relu', name=relu_name + '1')(x)

        if (type(skip) != list and skip is not None) or (type(skip) == list and None not in skip):
            if type(skip) is list:
                merge_list = []
                merge_list.append(x)
                for l in skip:
                    merge_list.append(l)
                x = layers.Concatenate(name=merge_name)(merge_list)
            else:
                x = layers.Concatenate(name=merge_name)([x, skip])

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)
        return x

    return layer
