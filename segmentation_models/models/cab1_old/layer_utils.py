from __future__ import absolute_import

from tensorflow import expand_dims
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, UpSampling2D, Conv2DTranspose, \
    GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Lambda
from tensorflow.keras.layers import BatchNormalization, Activation, concatenate, multiply, add
from tensorflow.keras.layers import ReLU, LeakyReLU, PReLU, ELU, Softmax
import tensorflow_addons as tfa
import tensorflow as tf


def decode_layer(X, channel, pool_size, unpool, kernel_size=3,
                 activation='ReLU', batch_norm=False, name='decode'):
    '''
    An overall decode layer, based on either upsampling or trans conv.

    decode_layer(X, channel, pool_size, unpool, kernel_size=3,
                 activation='ReLU', batch_norm=False, name='decode')

    Input
    ----------
        X: input tensor.
        pool_size: the decoding factor.
        channel: (for trans conv only) number of convolution filters.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.
        kernel_size: size of convolution kernels.
                     If kernel_size='auto', then it equals to the `pool_size`.
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.

    Output
    ----------
        X: output tensor.

    * The defaut: `kernel_size=3`, is suitable for `pool_size=2`.

    '''
    # parsers
    if unpool is False:
        # trans conv configurations
        bias_flag = not batch_norm

    elif unpool == 'nearest':
        # upsample2d configurations
        unpool = True
        interp = 'nearest'

    elif (unpool is True) or (unpool == 'bilinear'):
        # upsample2d configurations
        unpool = True
        interp = 'bilinear'

    else:
        raise ValueError('Invalid unpool keyword')

    if unpool:
        X = UpSampling2D(size=(pool_size, pool_size), interpolation=interp, name='{}_unpool'.format(name))(X)
    else:
        if kernel_size == 'auto':
            kernel_size = pool_size

        X = Conv2DTranspose(channel, kernel_size, strides=(pool_size, pool_size),
                            padding='same', name='{}_trans_conv'.format(name))(X)

        # batch normalization
        if batch_norm:
            # X = BatchNormalization(axis=3, name='{}_bn'.format(name))(X)
            X = tfa.layers.GroupNormalization(axis=3, groups=16, name='{}_gn'.format(name))(X)
            # X = tfa.layers.FilterResponseNormalization(name='{}_frn'.format(name))(X)

        # activation
        if activation is not None:
            activation_func = eval(activation)
            X = activation_func(name='{}_activation'.format(name))(X)
            # X = tfa.layers.TLU(name='{}_activation'.format(name))(X)

    return X


def encode_layer(X, channel, pool_size, pool, kernel_size='auto',
                 activation='ReLU', batch_norm=False, name='encode'):
    '''
    An overall encode layer, based on one of the:
    (1) max-pooling, (2) average-pooling, (3) strided conv2d.

    encode_layer(X, channel, pool_size, pool, kernel_size='auto',
                 activation='ReLU', batch_norm=False, name='encode')

    Input
    ----------
        X: input tensor.
        pool_size: the encoding factor.
        channel: (for strided conv only) number of convolution filters.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        kernel_size: size of convolution kernels.
                     If kernel_size='auto', then it equals to the `pool_size`.
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.

    Output
    ----------
        X: output tensor.

    '''
    # parsers
    if (pool in [False, True, 'max', 'ave']) is not True:
        raise ValueError('Invalid pool keyword')

    # maxpooling2d as default
    if pool is True:
        pool = 'max'

    elif pool is False:
        # stride conv configurations
        bias_flag = not batch_norm

    if pool == 'max':
        X = MaxPooling2D(pool_size=(pool_size, pool_size), name='{}_maxpool'.format(name))(X)

    elif pool == 'ave':
        X = AveragePooling2D(pool_size=(pool_size, pool_size), name='{}_avepool'.format(name))(X)

    else:
        if kernel_size == 'auto':
            kernel_size = pool_size

        # linear convolution with strides
        X = Conv2D(channel, kernel_size, strides=(pool_size, pool_size),
                   padding='valid', use_bias=bias_flag, name='{}_stride_conv'.format(name))(X)

        # batch normalization
        if batch_norm:
            # X = BatchNormalization(axis=3, name='{}_bn'.format(name))(X)
            X = tfa.layers.GroupNormalization(axis=3, groups=16, name='{}_gn'.format(name))(X)
            # X = tfa.layers.FilterResponseNormalization(name='{}_frn'.format(name))(X)

        # activation
        if activation is not None:
            activation_func = eval(activation)
            X = activation_func(name='{}_activation'.format(name))(X)
            # X = tfa.layers.TLU(name='{}_activation'.format(name))(X)

    return X


def attention_gate(X, g, channel,
                   activation='ReLU',
                   attention='add', name='att'):
    '''
    Self-attention gate modified from Oktay et al. 2018.

    attention_gate(X, g, channel,  activation='ReLU', attention='add', name='att')

    Input
    ----------
        X: input tensor, i.e., key and value.
        g: gated tensor, i.e., query.
        channel: number of intermediate channel.
                 Oktay et al. (2018) did not specify (denoted as F_int).
                 intermediate channel is expected to be smaller than the input channel.
        activation: a nonlinear attnetion activation.
                    The `sigma_1` in Oktay et al. 2018. Default is 'ReLU'.
        attention: 'add' for additive attention; 'multiply' for multiplicative attention.
                   Oktay et al. 2018 applied additive attention.
        name: prefix of the created keras layers.

    Output
    ----------
        X_att: output tensor.

    '''
    activation_func = eval(activation)
    attention_func = eval(attention)

    # mapping the input tensor to the intermediate channel
    theta_att = Conv2D(channel, 1, use_bias=True, name='{}_theta_x'.format(name))(X)

    # mapping the gate tensor
    phi_g = Conv2D(channel, 1, use_bias=True, name='{}_phi_g'.format(name))(g)

    # ----- attention learning ----- #
    query = attention_func([theta_att, phi_g], name='{}_add'.format(name))

    # nonlinear activation
    f = activation_func(name='{}_activation'.format(name))(query)

    # linear transformation
    psi_f = Conv2D(1, 1, use_bias=True, name='{}_psi_f'.format(name))(f)
    # ------------------------------ #

    # sigmoid activation as attention coefficients
    coef_att = Activation('sigmoid', name='{}_sigmoid'.format(name))(psi_f)

    # multiplicative attention masking
    X_att = multiply([X, coef_att], name='{}_masking'.format(name))

    return X_att


class DiagonalWeight(tf.keras.constraints.Constraint):
    """Constrains the weights to be diagonal.
    """

    def __init__(self, shape, flip):
        self.shape = shape
        self.flip = flip

    def __call__(self, w):
        k = tf.eye(self.shape[0], self.shape[1])
        if self.flip:
            k = tf.reverse(k, [-1])
        k = tf.reshape(k, self.shape)
        return w * k


def CONV_stack(X, channel, kernel_size=3, stack_num=2,
               dilation_rate=1, activation='ReLU',
               batch_norm=False, name='conv_stack'):
    '''
    Stacked convolutional layers:
    (Convolutional layer --> batch normalization --> Activation)*stack_num

    CONV_stack(X, channel, kernel_size=3, stack_num=2, dilation_rate=1, activation='ReLU',
               batch_norm=False, name='conv_stack')


    Input
    ----------
        X: input tensor.
        channel: number of convolution filters.
        kernel_size: size of 2-d convolution kernels.
        stack_num: number of stacked Conv2D-BN-Activation layers.
        dilation_rate: optional dilated convolution kernel.
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.

    Output
    ----------
        X: output tensor

    '''

    bias_flag = not batch_norm

    # stacking Convolutional layers
    for i in range(stack_num):

        use_bn = False
        square_conv = Conv2D(channel, (kernel_size, kernel_size), padding='same', use_bias=bias_flag,
                             dilation_rate=dilation_rate, name='{}_{}'.format(name, i))(X)
        # if batch_norm:
        #     if use_bn and channel > 16:
        #         square_conv = BatchNormalization(axis=3, name='{}_{}_bn'.format(name, i))(square_conv)
        #     else:
        #         square_conv = tfa.layers.GroupNormalization(axis=3, groups=16, name='{}_{}_gn'.format(name, i))(
        #             square_conv)
        out = square_conv

        enhance_skeleton = True
        direction_aware = False
        if enhance_skeleton:
            horizontal_conv = Conv2D(channel, (1, kernel_size), padding='same', use_bias=bias_flag,
                                     dilation_rate=dilation_rate, name='h_{}_{}'.format(name, i))(X)
            vertical_conv = Conv2D(channel, (kernel_size, 1), padding='same', use_bias=bias_flag,
                                   dilation_rate=dilation_rate, name='v_{}_{}'.format(name, i))(X)

            # if batch_norm:
            #     if use_bn and channel > 16:
            #         horizontal_conv = BatchNormalization(axis=3, name='h_{}_{}_bn'.format(name, i))(horizontal_conv)
            #         vertical_conv = BatchNormalization(axis=3, name='v_{}_{}_bn'.format(name, i))(vertical_conv)
            #     else:
            #         horizontal_conv = tfa.layers.GroupNormalization(axis=3, groups=16,
            #                                                         name='h_{}_{}_gn'.format(name, i))(
            #             horizontal_conv)
            #         vertical_conv = tfa.layers.GroupNormalization(axis=3, groups=16, name='v_{}_{}_gn'.format(name, i))(
            #             vertical_conv)

            out += horizontal_conv + vertical_conv

        if direction_aware:
            d1_conv = Conv2D(channel, kernel_size, padding='same', use_bias=bias_flag,
                             dilation_rate=dilation_rate, name='d1_{}_{}'.format(name, i),
                             kernel_constraint=DiagonalWeight((kernel_size, kernel_size, 1, 1), False))(X)
            d2_conv = Conv2D(channel, kernel_size, padding='same', use_bias=bias_flag,
                             dilation_rate=dilation_rate, name='d2_{}_{}'.format(name, i),
                             kernel_constraint=DiagonalWeight((kernel_size, kernel_size, 1, 1), True))(X)

            if batch_norm:
                if use_bn and channel > 16:
                    d1_conv = BatchNormalization(axis=3, name='d1_{}_{}_bn'.format(name, i))(d1_conv)
                    d2_conv = BatchNormalization(axis=3, name='d2_{}_{}_bn'.format(name, i))(d2_conv)
                else:
                    d1_conv = tfa.layers.GroupNormalization(axis=3, groups=16, name='d1_{}_{}_gn'.format(name, i))(
                        d1_conv)
                    d2_conv = tfa.layers.GroupNormalization(axis=3, groups=16, name='d2_{}_{}_gn'.format(name, i))(
                        d2_conv)
            out += d1_conv + d2_conv

        X = out

        if batch_norm:
            if use_bn and channel > 16:
                X = BatchNormalization(axis=3, name='{}_{}_bn'.format(name, i))(X)
            else:
                X = tfa.layers.GroupNormalization(axis=3, groups=16, name='{}_{}_gn'.format(name, i))(X)


        # activation
        activation_func = eval(activation)
        X = activation_func(name='{}_{}_activation'.format(name, i))(X)
        # X = tfa.layers.TLU(name='{}_{}_activation'.format(name, i))(X)

    return X


def Res_CONV_stack(X, X_skip, channel, res_num, activation='ReLU', batch_norm=False, name='res_conv'):
    '''
    Stacked convolutional layers with residual path.

    Res_CONV_stack(X, X_skip, channel, res_num, activation='ReLU', batch_norm=False, name='res_conv')

    Input
    ----------
        X: input tensor.
        X_skip: the tensor that does go into the residual path
                can be a copy of X (e.g., the identity block of ResNet).
        channel: number of convolution filters.
        res_num: number of convolutional layers within the residual path.
        activation: one of the `tensorflow.keras.layers` interface, e.g., 'ReLU'.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.

    Output
    ----------
        X: output tensor.

    '''
    X = CONV_stack(X, channel, kernel_size=3, stack_num=res_num, dilation_rate=1,
                   activation=activation, batch_norm=batch_norm, name=name)

    X = add([X_skip, X], name='{}_add'.format(name))

    activation_func = eval(activation)
    X = activation_func(name='{}_add_activation'.format(name))(X)

    return X


def CONV_output(X, n_labels, kernel_size=1, activation='Softmax', name='conv_output'):
    '''
    Convolutional layer with output activation.

    CONV_output(X, n_labels, kernel_size=1, activation='Softmax', name='conv_output')

    Input
    ----------
        X: input tensor.
        n_labels: number of classification label(s).
        kernel_size: size of 2-d convolution kernels. Default is 1-by-1.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interface or 'Sigmoid'.
                    Default option is 'Softmax'.
                    if None is received, then linear activation is applied.
        name: prefix of the created keras layers.

    Output
    ----------
        X: output tensor.

    '''

    X = Conv2D(n_labels, kernel_size, padding='same', use_bias=True, name=name)(X)

    if activation:

        if activation == 'Sigmoid':
            X = Activation('sigmoid', name='{}_activation'.format(name), dtype='float32')(X)

        else:
            activation_func = eval(activation)
            X = activation_func(name='{}_activation'.format(name), dtype='float32')(X)

    return X
