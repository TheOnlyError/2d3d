import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras import layers

class Residual(Model):
    def __init__(self, numIn, numOut):
        super(Residual, self).__init__()
        self.numIn = numIn
        self.numOut = numOut
        self.bn = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.conv1 = layers.Conv2D(self.numOut, kernel_size=1)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(int(self.numOut/2), kernel_size=3, strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(self.numOut, kernel_size=1)

        if self.numIn != self.numOut:
            self.conv4 = layers.Conv2D(self.numOut, kernel_size=1)

        # for m in self.modules():
        #     if isinstance(m, layers.Conv2D) or isinstance(m, nn.ConvTranspose2d):
        #         nn.init.kaiming_normal_(
        #             m.weight, mode='fan_out', nonlinearity='relu')
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, layers.BatchNormalization):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def call(self, x):
        residual = x
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.numIn != self.numOut:
            residual = self.conv4(x)

        return out + residual


class hg_furukawa_original(Model):
    def __init__(self, n_classes):
        # tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
        super(hg_furukawa_original, self).__init__()
        self.conv1_ = layers.Conv2D(64, kernel_size=7, strides=2, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation('relu')
        self.r01 = Residual(64, 128)
        self.maxpool = layers.MaxPool2D(pool_size=2, strides=2)
        self.r02 = Residual(128, 128)
        self.r03 = Residual(128, 128)
        self.r04 = Residual(128, 256)

        self.maxpool1 = layers.MaxPool2D(pool_size=2, strides=2)
        self.r11_a = Residual(256, 256)
        self.r12_a = Residual(256, 256)
        self.r13_a = Residual(256, 256)

        self.maxpool2 = layers.MaxPool2D(pool_size=2, strides=2)
        self.r21_a = Residual(256, 256)
        self.r22_a = Residual(256, 256)
        self.r23_a = Residual(256, 256)

        self.maxpool3 = layers.MaxPool2D(pool_size=2, strides=2)
        self.r31_a = Residual(256, 256)
        self.r32_a = Residual(256, 256)
        self.r33_a = Residual(256, 256)

        self.maxpool4 = layers.MaxPool2D(pool_size=2, strides=2)
        self.r41_a = Residual(256, 256)
        self.r42_a = Residual(256, 256)
        self.r43_a = Residual(256, 256)
        self.r44_a = Residual(256, 512)
        self.r45_a = Residual(512, 512)
        self.upsample4 = layers.Conv2DTranspose(512, kernel_size=2, strides=2)

        self.r41_b = Residual(256, 256)
        self.r42_b = Residual(256, 256)
        self.r43_b = Residual(256, 512)

        self.r4_ = Residual(512, 512)
        self.upsample3 = layers.Conv2DTranspose(512, kernel_size=2, strides=2)

        self.r31_b = Residual(256, 256)
        self.r32_b = Residual(256, 256)
        self.r33_b = Residual(256, 512)

        self.r3_ = Residual(512, 512)
        self.upsample2 = layers.Conv2DTranspose(512, kernel_size=2, strides=2)

        self.r21_b = Residual(256, 256)
        self.r22_b = Residual(256, 256)
        self.r23_b = Residual(256, 512)

        self.r2_ = Residual(512, 512)
        self.upsample1 = layers.Conv2DTranspose(512, kernel_size=2, strides=2)

        self.r11_b = Residual(256, 256)
        self.r12_b = Residual(256, 256)
        self.r13_b = Residual(256, 512)

        self.conv2_ = layers.Conv2D(512, use_bias=True, kernel_size=1)
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.Activation('relu')
        self.conv3_ = layers.Conv2D(256, kernel_size=1)
        self.bn3 = layers.BatchNormalization()
        self.relu3 = layers.Activation('relu')
        self.conv4_ = layers.Conv2D(n_classes, kernel_size=1)
        self.upsample = layers.Conv2DTranspose(n_classes, kernel_size=4, strides=4, dtype='float32')
        self.sigmoid = layers.Activation('sigmoid', dtype='float32')

        # for m in self.modules():
        #     if isinstance(m, layers.Conv2D) or isinstance(m, nn.ConvTranspose2d):
        #         nn.init.kaiming_normal_(
        #             m.weight, mode='fan_out', nonlinearity='relu')
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def save_weights(self,
                     filepath,
                     overwrite=True,
                     save_format=None,
                     options=None):
        self.save(filepath, overwrite=overwrite, save_format=save_format, options=options)

    # def cast_inputs(self, inputs):
    #     # Casts to float16, the policy's lowest-precision dtype
    #     return self._mixed_precision_policy.cast_to_lowest(inputs)

    def call(self, x):
        out = self.conv1_(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.maxpool(out)
        out = self.r01(out)
        out = self.r02(out)
        out = self.r03(out)
        out = self.r04(out)

        out1a = self.maxpool1(out)
        out1a = self.r11_a(out1a)
        out1a = self.r12_a(out1a)
        out1a = self.r13_a(out1a)

        out1b = self.r11_b(out)
        out1b = self.r12_b(out1b)
        out1b = self.r13_b(out1b)

        out2a = self.maxpool2(out1a)
        out2a = self.r21_a(out2a)
        out2a = self.r22_a(out2a)
        out2a = self.r23_a(out2a)

        out2b = self.r21_b(out1a)
        out2b = self.r22_b(out2b)
        out2b = self.r23_b(out2b)

        out3a = self.maxpool3(out2a)
        out3a = self.r31_a(out3a)
        out3a = self.r32_a(out3a)
        out3a = self.r33_a(out3a)

        out3b = self.r31_b(out2a)
        out3b = self.r32_b(out3b)
        out3b = self.r33_b(out3b)

        out4a = self.maxpool4(out3a)
        out4a = self.r41_a(out4a)
        out4a = self.r42_a(out4a)
        out4a = self.r43_a(out4a)
        out4a = self.r44_a(out4a)
        out4a = self.r45_a(out4a)

        out4b = self.r41_b(out3a)
        out4b = self.r42_b(out4b)
        out4b = self.r43_b(out4b)

        out4_ = self.upsample4(out4a)
        out4 = self._upsample_add(out4_, out4b)
        out4 = self.r4_(out4)

        out3_ = self.upsample3(out4)
        out3 = self._upsample_add(out3_, out3b)
        out3 = self.r3_(out3)

        out2_ = self.upsample2(out3)
        out2 = self._upsample_add(out2_, out2b)
        out2 = self.r2_(out2)

        out1_ = self.upsample1(out2)
        out = self._upsample_add(out1_, out1b)

        out = self.conv2_(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3_(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv4_(out)
        out = self.upsample(out)
        # heatmap channels go trough sigmoid
        # out[:, :21] = self.sigmoid(out[:, :21])
        return out

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        return x + y
        # _, _, H, W = y.size()
        # if y.shape != x.shape:
        #     return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y
        # else:
        #     return x + y
    #
    # def init_weights(self):
    #     # Pre-trained network weights from Human pose estimation via Convolutional Part Heatmap Regression
    #     # https://www.adrianbulat.com/human-pose-estimation MPII
    #     model = model_1427.model_1427
    #     model.load_state_dict(torch.load('floortrans/models/model_1427.pth'))
    #
    #     for (src, dst) in zip(model.parameters(), self.parameters()):
    #         dst[:].data.copy_(src[:].data)