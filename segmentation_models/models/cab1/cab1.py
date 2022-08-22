from __future__ import absolute_import

import warnings

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import Activation, concatenate
from tensorflow.python.layers.convolutional import Conv2D
from tensorflow.python.layers.pooling import AveragePooling2D, MaxPooling2D
from tensorflow_addons.layers import AdaptiveAveragePooling2D, AdaptiveMaxPooling2D, TLU

from segmentation_models.models.base_model import BaseModel
from segmentation_models.models.cab1.backbone_zoo import backbone_zoo, bach_norm_checker
from segmentation_models.models.cab1.layer_utils import CONV_stack, decode_layer, encode_layer, CONV_output
from segmentation_models.models.cab1.model_unet_2d import UNET_left, UNET_right


# 2 = s, -c
# 3 = s, c
# 4 = -s, c
# 5 = -s, -c

def AM(X, channel, activation, fused_layers, encoder_threshold, ratio=8, use_hhdc=False, use_cam=False, name=''):
    if use_hhdc == 6:
        X = CONV_stack(X, channel, kernel_size=3, stack_num=1,
                       activation=activation, batch_norm=True, name='{}_am_conv'.format(name))
        return X

    if use_hhdc == 1:
        # First CAM then SAM
        X = CAM(channel, ratio)(X)
        X = SAM(channel, use_hhdc)(X)
        return X

    X_s = X
    X_c = X
    channel_s = channel // 2
    channel_c = channel // 2
    if use_hhdc == 2 or use_hhdc == 3:
        X_s = fused_layers[:-encoder_threshold]
        if len(X_s) > 1:
            X_s = concatenate(X_s, axis=-1)
        else:
            X_s = X_s[0]
        if use_hhdc == 3:
            channel_s = channel / 5 * (5 - encoder_threshold)

    if use_hhdc == 3 or use_hhdc == 4:
        X_c = fused_layers[-encoder_threshold:]
        if len(X_c) > 1:
            X_c = concatenate(X_c, axis=-1)
        else:
            X_c = X_c[0]
        channel_c = channel / 5 * encoder_threshold

    assert channel_s + channel_c == channel

    if use_cam == 2:
        spatial_feature = SAM2(channel_s, len(fused_layers))(fused_layers)
    elif use_cam == 3:
        spatial_feature = SAM3(channel_s, True)(X_s)
    elif use_cam == 4:
        spatial_feature = SAM4(channel_s)(fused_layers)
    elif use_cam == 5:
        spatial_feature = SAM5(channel_s, len(fused_layers))(fused_layers)
    else:
        spatial_feature = SAM(channel_s, use_cam)(X_s)
    channel_feature = CAM(channel_c, ratio)(X_c)
    # X = concatenate([X, channel_feature, spatial_feature], axis=-1) # TODO maybe add X here
    X = concatenate([channel_feature, spatial_feature], axis=-1)
    X = CONV_stack(X, channel, kernel_size=3, stack_num=1,
                   activation=activation, batch_norm=True, name='{}_am_conv'.format(name))
    return X


class CAM(tf.keras.layers.Layer):
    def __init__(self, out_planes, ratio=2):
        super(CAM, self).__init__()
        self.conv = Conv2D(out_planes, 1, use_bias=False, padding='same')
        self.avg_pool = AdaptiveAveragePooling2D(1)
        self.max_pool = AdaptiveMaxPooling2D(1)

        self.fc11 = Conv2D(out_planes // ratio, 1, use_bias=False, padding='same')
        self.fc12 = Conv2D(out_planes, 1, use_bias=False, padding='same')

        self.fc21 = Conv2D(out_planes // ratio, 1, use_bias=False, padding='same')
        self.fc22 = Conv2D(out_planes, 1, use_bias=False, padding='same')
        self.relu1 = tf.keras.layers.Activation('relu')

        self.sigmoid = tf.keras.layers.Activation('sigmoid')

    def call(self, x):
        x = self.conv(x)
        avg_out = self.fc12(self.relu1(self.fc11(self.avg_pool(x))))
        max_out = self.fc22(self.relu1(self.fc21(self.max_pool(x))))
        out = avg_out + max_out
        del avg_out, max_out

        return x * self.sigmoid(out)


class SAM2(tf.keras.layers.Layer):
    def __init__(self, out_planes, n_blocks):
        super(SAM2, self).__init__()
        self.conv = Conv2D(out_planes, 1, use_bias=False, padding='same')
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

        self.hhdc = []
        for i in range(n_blocks):
            self.hhdc.append(HHDC(1))

    def call(self, x):
        out = 0
        for i, block in enumerate(x):
            avg_pool = tf.reduce_mean(block, axis=-1, keepdims=True)
            max_pool = tf.reduce_max(block, axis=-1, keepdims=True)
            out += self.hhdc[i](concatenate([avg_pool, max_pool], axis=-1))

        x = concatenate(x, axis=-1, name='sam_concat_x')
        return self.conv(x) * self.sigmoid(out)


class SAM3(tf.keras.layers.Layer):
    def __init__(self, out_planes, use_hhdc=False):
        super(SAM3, self).__init__()
        self.conv = Conv2D(out_planes, 1, use_bias=False, padding='same')
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

        self.hhdc = HHDC(1)

    def call(self, x):
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)

        out = self.hhdc(concatenate([avg_pool, max_pool], axis=-1))
        return self.conv(x) * self.sigmoid(out)


class SAM4(tf.keras.layers.Layer):
    def __init__(self, out_planes):
        super(SAM4, self).__init__()
        self.conv = Conv2D(out_planes, 1, use_bias=False, padding='same')
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

        self.hhdc = HHDC(1)

    def call(self, x):
        spatial_features = []
        for i, block in enumerate(x):
            avg_pool = tf.reduce_mean(block, axis=-1, keepdims=True)
            max_pool = tf.reduce_max(block, axis=-1, keepdims=True)
            spatial_features.append(avg_pool)
            spatial_features.append(max_pool)

        out = self.hhdc(concatenate(spatial_features, axis=-1))

        x = concatenate(x, axis=-1, name='sam_concat_x')
        return self.conv(x) * self.sigmoid(out)


class SAM5(tf.keras.layers.Layer):
    def __init__(self, out_planes, n_blocks):
        super(SAM5, self).__init__()
        self.conv = []
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

        self.hhdc = []
        for i in range(n_blocks):
            self.conv.append(Conv2D(out_planes / n_blocks, 1, use_bias=False, padding='same'))
            self.hhdc.append(HHDC(1))

    def call(self, x):
        out = []
        for i, block in enumerate(x):
            avg_pool = tf.reduce_mean(block, axis=-1, keepdims=True)
            max_pool = tf.reduce_max(block, axis=-1, keepdims=True)
            block_out = self.hhdc[i](concatenate([avg_pool, max_pool], axis=-1))
            out.append(self.conv[i](block) * self.sigmoid(block_out))

        return concatenate(out, axis=-1, name='sam_concat_out')


class SAM(tf.keras.layers.Layer):
    def __init__(self, out_planes, use_hhdc=False):
        super(SAM, self).__init__()
        self.conv = Conv2D(out_planes, 1, use_bias=False, padding='same')
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

        self.hhdc = HHDC(out_planes, False, use_hhdc)

    def call(self, x):
        x = self.conv(x)
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        assert avg_pool.get_shape()[-1] == 1
        assert max_pool.get_shape()[-1] == 1
        out = concatenate([avg_pool, max_pool], axis=-1)
        assert out.get_shape()[-1] == 2
        out = self.hhdc(out)
        # return self.conv(x) * self.sigmoid(out)
        return x * self.sigmoid(out)


class HHDC(tf.keras.layers.Layer):
    def __init__(self, out_planes, concat=False, dilations=None):
        super(HHDC, self).__init__()
        self.concat = concat
        if concat:
            out_planes -= 2
        self.conv1 = Conv2D(out_planes, 1, use_bias=False, padding='same')
        self.conv2 = Conv2D(out_planes, 1, use_bias=False, padding='same')
        self.conv3 = Conv2D(out_planes, 1, use_bias=False, padding='same')
        self.convd1 = Conv2D(out_planes, 3, dilation_rate=1, use_bias=False, padding='same')
        self.convd2 = Conv2D(out_planes, 3, dilation_rate=2, use_bias=False, padding='same')
        self.convd3 = Conv2D(out_planes, 3, dilation_rate=3, use_bias=False, padding='same')

    def call(self, x):
        out = self.conv1(x) + self.convd1(x) + self.conv2(x) + self.convd2(x) + self.conv3(x) + self.convd3(x)
        if self.concat:
            out = concatenate([out, x], axis=-1)
        return out


def unet_3plus_2d_base(input_tensor, filter_num_down, filter_num_skip, filter_num_aggregate,
                       stack_num_down=2, stack_num_up=1, activation='ReLU', batch_norm=False, pool=True, unpool=True,
                       backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True,
                       name='unet3plus', use_hhdc=False, use_cam=True):
    '''
    The base of UNET 3+ with an optional ImagNet-trained backbone.

    unet_3plus_2d_base(input_tensor, filter_num_down, filter_num_skip, filter_num_aggregate,
                       stack_num_down=2, stack_num_up=1, activation='ReLU', batch_norm=False, pool=True, unpool=True,
                       backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='unet3plus')

    ----------
    Huang, H., Lin, L., Tong, R., Hu, H., Zhang, Q., Iwamoto, Y., Han, X., Chen, Y.W. and Wu, J., 2020.
    UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation.
    In ICASSP 2020-2020 IEEE International Conference on Acoustics,
    Speech and Signal Processing (ICASSP) (pp. 1055-1059). IEEE.

    Input
    ----------
        input_tensor: the input tensor of the base, e.g., `keras.layers.Inpyt((None, None, 3))`.
        filter_num_down: a list that defines the number of filters for each
                         downsampling level. e.g., `[64, 128, 256, 512, 1024]`.
                         the network depth is expected as `len(filter_num_down)`
        filter_num_skip: a list that defines the number of filters after each
                         full-scale skip connection. Number of elements is expected to be `depth-1`.
                         i.e., the bottom level is not included.
                         * Huang et al. (2020) applied the same numbers for all levels.
                           e.g., `[64, 64, 64, 64]`.
        filter_num_aggregate: an int that defines the number of channels of full-scale aggregations.
        stack_num_down: number of convolutional layers per downsampling level/block.
        stack_num_up: number of convolutional layers (after full-scale concat) per upsampling level/block.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., ReLU
        batch_norm: True for batch normalization.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.
        name: prefix of the created keras model and its layers.

        ---------- (keywords of backbone options) ----------
        backbone_name: the bakcbone model name. Should be one of the `tensorflow.keras.applications` class.
                       None (default) means no backbone.
                       Currently supported backbones are:
                       (1) VGG16, VGG19
                       (2) ResNet50, ResNet101, ResNet152
                       (3) ResNet50V2, ResNet101V2, ResNet152V2
                       (4) DenseNet121, DenseNet169, DenseNet201
                       (5) EfficientNetB[0-7]
        weights: one of None (random initialization), 'imagenet' (pre-training on ImageNet),
                 or the path to the weights file to be loaded.
        freeze_backbone: True for a frozen backbone.
        freeze_batch_norm: False for not freezing batch normalization layers.

    * Downsampling is achieved through maxpooling and can be replaced by strided convolutional layers here.
    * Upsampling is achieved through bilinear interpolation and can be replaced by transpose convolutional layers here.

    Output
    ----------
        A list of tensors with the first/second/third tensor obtained from
        the deepest/second deepest/third deepest upsampling block, etc.
        * The feature map sizes of these tensors are different,
          with the first tensor has the smallest size.

    '''

    depth_ = len(filter_num_down)

    X_encoder = []
    X_decoder = []

    # no backbone cases
    if backbone is None:

        X = input_tensor

        # stacked conv2d before downsampling
        X = CONV_stack(X, filter_num_down[0], kernel_size=3, stack_num=stack_num_down,
                       activation=activation, batch_norm=batch_norm, name='{}_down0'.format(name))
        X_encoder.append(X)

        # downsampling levels
        for i, f in enumerate(filter_num_down[1:]):
            # UNET-like downsampling
            X = UNET_left(X, f, kernel_size=3, stack_num=stack_num_down, activation=activation,
                          pool=pool, batch_norm=batch_norm, name='{}_down{}'.format(name, i + 1))
            X_encoder.append(X)

    else:
        # handling VGG16 and VGG19 separately
        if 'VGG' in backbone:
            backbone_ = backbone_zoo(backbone, weights, input_tensor, depth_, freeze_backbone, freeze_batch_norm)
            # collecting backbone feature maps
            X_encoder = backbone_([input_tensor, ])
            depth_encode = len(X_encoder)

        # for other backbones
        else:
            backbone_ = backbone_zoo(backbone, weights, input_tensor, depth_ - 1, freeze_backbone, freeze_batch_norm)
            # collecting backbone feature maps
            X_encoder = backbone_([input_tensor, ])
            depth_encode = len(X_encoder) + 1

        # extra conv2d blocks are applied
        # if downsampling levels of a backbone < user-specified downsampling levels
        if depth_encode < depth_:

            # begins at the deepest available tensor
            X = X_encoder[-1]

            # extra downsamplings
            for i in range(depth_ - depth_encode):
                i_real = i + depth_encode

                X = UNET_left(X, filter_num_down[i_real], stack_num=stack_num_down, activation=activation, pool=pool,
                              batch_norm=batch_norm, name='{}_down{}'.format(name, i_real + 1))
                X_encoder.append(X)

    # treat the last encoded tensor as the first decoded tensor
    X_decoder.append(X_encoder[-1])

    # upsampling levels
    X_encoder = X_encoder[::-1]

    depth_decode = len(X_encoder) - 1

    # loop over upsampling levels
    for i in range(depth_decode):

        f = filter_num_skip[i]

        # collecting tensors for layer fusion
        X_fscale = []

        # for each upsampling level, loop over all available downsampling levels (similar to the unet++)
        for lev in range(depth_decode + 1):  # Edit Gijs: fix bug
            # for lev in range(depth_decode):

            # counting scale difference between the current down- and upsampling levels
            pool_scale = lev - i - 1  # -1 for python indexing

            # deeper tensors are obtained from **decoder** outputs
            if pool_scale < 0:
                if pool_scale < -1:
                    continue
                pool_size = 2 ** (-1 * pool_scale)

                X = decode_layer(X_decoder[lev], f, pool_size, unpool,
                                 activation=activation, batch_norm=batch_norm,
                                 name='{}_up_{}_en{}'.format(name, i, lev))

            # unet skip connection (identity mapping)
            elif pool_scale == 0:

                X = X_encoder[lev]

            # shallower tensors are obtained from **encoder** outputs
            else:
                if pool_scale > 1:
                    continue
                pool_size = 2 ** (pool_scale)

                X = encode_layer(X_encoder[lev], f, pool_size, pool, activation=activation,
                                 batch_norm=batch_norm, name='{}_down_{}_en{}'.format(name, i, lev))

            # a conv layer after feature map scale change
            X = CONV_stack(X, f, kernel_size=3, stack_num=1,
                           activation=activation, batch_norm=batch_norm,
                           name='{}_down_from{}_to{}'.format(name, i, lev))

            X_fscale.append(X)

            # layer fusion at the end of each level
        # stacked conv layers after concat. BatchNormalization is fixed to True

        X = concatenate(X_fscale, axis=-1, name='{}_concat_{}'.format(name, i))
        # X = CONV_stack(X, filter_num_aggregate, kernel_size=3, stack_num=stack_num_up,
        #                activation=activation, batch_norm=True, name='{}_fusion_conv_{}'.format(name, i))
        # Gijs: instead of only conv stack we perform AM: SAM + CAM)
        X = AM(X, channel=filter_num_aggregate, activation=activation, fused_layers=X_fscale,
               encoder_threshold=i + 1, ratio=2,
               use_hhdc=use_hhdc, use_cam=use_cam, name='{}_am_{}'.format(name, i))
        X_decoder.append(X)

    # if tensors for concatenation is not enough
    # then use upsampling without concatenation
    if depth_decode < depth_ - 1:
        for i in range(depth_ - depth_decode - 1):
            i_real = i + depth_decode
            X = UNET_right(X, None, filter_num_aggregate, stack_num=stack_num_up, activation=activation,
                           unpool=unpool, batch_norm=batch_norm, concat=False,
                           name='{}_plain_up{}'.format(name, i_real))
            X_decoder.append(X)

    # return decoder outputs
    return X_decoder


def cab1(input_size, n_labels, filter_num_down, filter_num_skip='auto', filter_num_aggregate='auto',
         stack_num_down=2, stack_num_up=1, activation='ReLU', output_activation='Sigmoid',
         batch_norm=False, pool=True, unpool=True, deep_supervision=False,
         backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='unet3plus',
         aaf=False, use_hhdc=False, use_cam=True):
    '''
    UNET 3+ with an optional ImageNet-trained backbone.

    unet_3plus_2d(input_size, n_labels, filter_num_down, filter_num_skip='auto', filter_num_aggregate='auto',
                  stack_num_down=2, stack_num_up=1, activation='ReLU', output_activation='Sigmoid',
                  batch_norm=False, pool=True, unpool=True, deep_supervision=False,
                  backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='unet3plus')

    ----------
    Huang, H., Lin, L., Tong, R., Hu, H., Zhang, Q., Iwamoto, Y., Han, X., Chen, Y.W. and Wu, J., 2020.
    UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation.
    In ICASSP 2020-2020 IEEE International Conference on Acoustics,
    Speech and Signal Processing (ICASSP) (pp. 1055-1059). IEEE.

    Input
    ----------
        input_size: the size/shape of network input, e.g., `(128, 128, 3)`.
        filter_num_down: a list that defines the number of filters for each
                         downsampling level. e.g., `[64, 128, 256, 512, 1024]`.
                         the network depth is expected as `len(filter_num_down)`
        filter_num_skip: a list that defines the number of filters after each
                         full-scale skip connection. Number of elements is expected to be `depth-1`.
                         i.e., the bottom level is not included.
                         * Huang et al. (2020) applied the same numbers for all levels.
                           e.g., `[64, 64, 64, 64]`.
        filter_num_aggregate: an int that defines the number of channels of full-scale aggregations.
        stack_num_down: number of convolutional layers per downsampling level/block.
        stack_num_up: number of convolutional layers (after full-scale concat) per upsampling level/block.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'
        output_activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interface or 'Sigmoid'.
                           Default option is 'Softmax'.
                           if None is received, then linear activation is applied.
        batch_norm: True for batch normalization.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.
        deep_supervision: True for a model that supports deep supervision. Details see Huang et al. (2020).
        name: prefix of the created keras model and its layers.

        ---------- (keywords of backbone options) ----------
        backbone_name: the bakcbone model name. Should be one of the `tensorflow.keras.applications` class.
                       None (default) means no backbone.
                       Currently supported backbones are:
                       (1) VGG16, VGG19
                       (2) ResNet50, ResNet101, ResNet152
                       (3) ResNet50V2, ResNet101V2, ResNet152V2
                       (4) DenseNet121, DenseNet169, DenseNet201
                       (5) EfficientNetB[0-7]
        weights: one of None (random initialization), 'imagenet' (pre-training on ImageNet),
                 or the path to the weights file to be loaded.
        freeze_backbone: True for a frozen backbone.
        freeze_batch_norm: False for not freezing batch normalization layers.

    * The Classification-guided Module (CGM) is not implemented.
      See https://github.com/yingkaisha/keras-unet-collection/tree/main/examples for a relevant example.
    * Automated mode is applied for determining `filter_num_skip`, `filter_num_aggregate`.
    * The default output activation is sigmoid, consistent with Huang et al. (2020).
    * Downsampling is achieved through maxpooling and can be replaced by strided convolutional layers here.
    * Upsampling is achieved through bilinear interpolation and can be replaced by transpose convolutional layers here.

    Output
    ----------
        model: a keras model.

    '''

    print('AM settings: hhdc={0}, cam={1}'.format(use_hhdc, use_cam))

    depth_ = len(filter_num_down)

    verbose = False

    if filter_num_skip == 'auto':
        verbose = True
        filter_num_skip = [filter_num_down[0] for num in range(depth_ - 1)]

    if filter_num_aggregate == 'auto':
        verbose = True
        filter_num_aggregate = int(depth_ * filter_num_down[0])

    if verbose:
        print('Automated hyper-parameter determination is applied with the following details:\n----------')
        print('\tNumber of convolution filters after each full-scale skip connection: filter_num_skip = {}'.format(
            filter_num_skip))
        print('\tNumber of channels of full-scale aggregated feature maps: filter_num_aggregate = {}'.format(
            filter_num_aggregate))

    if backbone is not None:
        bach_norm_checker(backbone, batch_norm)

    X_encoder = []
    X_decoder = []

    IN = Input(input_size)

    X_decoder = unet_3plus_2d_base(IN, filter_num_down, filter_num_skip, filter_num_aggregate,
                                   stack_num_down=stack_num_down, stack_num_up=stack_num_up, activation=activation,
                                   batch_norm=batch_norm, pool=pool, unpool=unpool,
                                   backbone=backbone, weights=weights, freeze_backbone=freeze_backbone,
                                   freeze_batch_norm=freeze_batch_norm, name=name, use_hhdc=use_hhdc, use_cam=use_cam)
    X_decoder = X_decoder[::-1]

    if deep_supervision:

        # ----- frozen backbone issue checker ----- #
        if ('{}_backbone_'.format(backbone) in X_decoder[0].name) and freeze_backbone:
            backbone_warn = '\n\nThe deepest UNET 3+ deep supervision branch directly connects to a frozen backbone.\nTesting your configurations on `keras_unet_collection.base.unet_plus_2d_base` is recommended.'
            warnings.warn(backbone_warn)
        # ----------------------------------------- #

        OUT_stack = []
        L_out = len(X_decoder)

        print(
            '----------\ndeep_supervision = True\nnames of output tensors are listed as follows ("sup0" is the shallowest supervision layer;\n"final" is the final output layer):\n')

        # conv2d --> upsampling --> output activation.
        # index 0 is final output
        for i in range(1, L_out):

            pool_size = 2 ** (i)

            X = Conv2D(n_labels, 3, padding='same', name='{}_output_conv_{}'.format(name, i - 1))(X_decoder[i])

            X = decode_layer(X, n_labels, pool_size, unpool,
                             activation=None, batch_norm=False, name='{}_output_sup{}'.format(name, i - 1))

            if output_activation:
                print('\t{}_output_sup{}_activation'.format(name, i - 1))

                if output_activation == 'Sigmoid':
                    X = Activation('sigmoid', name='{}_output_sup{}_activation'.format(name, i - 1))(X)
                else:
                    activation_func = eval(output_activation)
                    X = activation_func(name='{}_output_sup{}_activation'.format(name, i - 1))(X)
            else:
                if unpool is False:
                    print('\t{}_output_sup{}_trans_conv'.format(name, i - 1))
                else:
                    print('\t{}_output_sup{}_unpool'.format(name, i - 1))

            OUT_stack.append(X)

        X = CONV_output(X_decoder[0], n_labels, kernel_size=3,
                        activation=output_activation, name='{}_output_final'.format(name))
        OUT_stack.append(X)

        if output_activation:
            print('\t{}_output_final_activation'.format(name))
        else:
            print('\t{}_output_final'.format(name))

        if aaf:
            model = BaseModel([IN, ], OUT_stack)
        else:
            model = Model([IN, ], [OUT_stack, ])

    else:
        OUT = CONV_output(X_decoder[0], n_labels, kernel_size=3,
                          activation=output_activation, name='{}_output_final'.format(name))

        if aaf:
            model = BaseModel([IN, ], [OUT, ])
        else:
            model = Model([IN, ], [OUT, ])

    return model
