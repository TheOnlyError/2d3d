import gc

import tensorflow as tf
from tensorflow.python.keras.callbacks import Callback
import numpy as np


class AutomaticWeightedLoss:
    def __init__(self, loss_functions, names, inds, dec, epochs, log_dir_path):
        self.loss_functions = loss_functions
        self.names = names
        self.inds = inds
        self.dec = dec
        self.sigmas = []
        self.epoch = 0
        self.epochs = epochs
        self.losses = [tf.Variable(name=name, dtype=tf.float32,
                                   initial_value=0.0, trainable=False) for name in self.names]
        for i in list(dict.fromkeys(inds)):
            self.sigmas.append(tf.Variable(name='Sigma_' + str(i), dtype=tf.float32,
                                           initial_value=0.5, trainable=True))

        file_writer = tf.summary.create_file_writer(log_dir_path)
        file_writer.set_as_default()

    def combined_loss(self):
        def loss_function(y_true, y_pred):
            loss_sum = 0
            for i in range(len(self.loss_functions)):
                loss = self.loss_functions[i](y_true, y_pred)
                if self.dec[i]:
                    loss *= tf.math.pow(20.0, -self.epoch/self.epochs)
                self.losses[i].assign_add(loss)
                loss_sum += 0.5 / (self.sigmas[self.inds[i]] ** 2) * loss
            for i in list(dict.fromkeys(self.inds)):
                loss_sum += tf.math.log1p(self.sigmas[i] ** 2)
            return loss_sum

        return loss_function

    def reset(self):
        for loss in self.losses:
            loss.assign(0.0)


class AutomaticWeightedLossCallback(Callback):
    def __init__(self, heatmap, aaf):
        self.heatmap = heatmap
        self.aaf = aaf

    def on_epoch_end(self, epoch, logs=None):
        self.model.automatic_loss.epoch = epoch

        sigmas = []
        with tf.name_scope("Sigmas"):
            for sigma in self.model.loss_sigmas:
                val = sigma.numpy()
                sigmas.append(val)
                tf.summary.scalar(sigma.name, data=val, step=epoch)
            print('sigmas', sigmas)

        if self.aaf:
            w_edge_norm = tf.nn.softmax(self.model.w_edge, axis=-1).numpy()[0][0][0]
            w_not_edge_norm = tf.nn.softmax(self.model.w_not_edge, axis=-1).numpy()[0][0][0]
            print('w_edge_norm', w_edge_norm)
            print('w_not_edge_norm', w_not_edge_norm)
            print('w_edge_norm_mean', np.mean(w_edge_norm, axis=0)[0])
            print('w_not_edge_norm_mean', np.mean(w_not_edge_norm, axis=0)[0])

        with tf.name_scope("Losses"):
            for i in range(len(self.model.automatic_loss.losses)):
                val = self.model.automatic_loss.losses[i].numpy()
                print(val)
                tf.summary.scalar(self.model.automatic_loss.names[i], data=val, step=epoch)
            self.model.automatic_loss.reset()
