import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras.metrics import Metric

from training.post_process import post_process


class CM(Metric):
    def __init__(self, num_classes: int, post_processing: bool = False, conversion: str = False, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.post_processing = post_processing
        self.conversion = conversion

        # Variable to accumulate the predictions in the confusion matrix.
        self.total_cm = self.add_weight(
            'total_confusion_matrix',
            shape=(num_classes, num_classes),
            initializer=tf.compat.v1.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the confusion matrix statistics.

        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.

        Returns:
          Update op.
        """
        if isinstance(y_pred, tuple):
            y_pred = y_pred[1]

        y_true = tf.math.argmax(y_true[0], axis=-1)

        if not self.post_processing:
            y_pred = tf.math.argmax(y_pred[0], axis=-1)

        # if self.post_processing:
            # if self.conversion is not None:
            #     converted_ind = np.zeros(y_pred.shape, dtype=np.uint8)
            #     if self.conversion == 'r3d':
            #         converted_ind[y_pred == 1] = 1
            #         converted_ind[y_pred == 2] = 4
            #     elif self.conversion == 'cubicasa5k':
            #         converted_ind[y_pred == 1] = 1
            #         converted_ind[y_pred == 2] = 3
            #         converted_ind[y_pred == 3] = 4
            #         converted_ind[y_pred == 4] = 6
            #         converted_ind[y_pred == 5] = 7
            #     y_pred = converted_ind
            #
            # y_pred = post_process(y_pred)
            #
            # if self.conversion is not None:
            #     converted_ind = np.zeros(y_pred.shape, dtype=np.uint8)
            #     if self.conversion == 'r3d':
            #         converted_ind[y_pred == 1] = 1
            #         converted_ind[y_pred == 4] = 2
            #     elif self.conversion == 'cubicasa5k':
            #         converted_ind[y_pred == 1] = 1
            #         converted_ind[y_pred == 3] = 2
            #         converted_ind[y_pred == 4] = 3
            #         converted_ind[y_pred == 6] = 4
            #         converted_ind[y_pred == 7] = 5
            #     y_pred = converted_ind

        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)

        # Flatten the input if its rank > 1.
        if y_pred.shape.ndims > 1:
            y_pred = tf.reshape(y_pred, [-1])

        if y_true.shape.ndims > 1:
            y_true = tf.reshape(y_true, [-1])

        # Accumulate the prediction to current confusion matrix.
        current_cm = tf.math.confusion_matrix(
            y_true,
            y_pred,
            self.num_classes,
            dtype=self._dtype)
        return self.total_cm.assign_add(current_cm)

    def reset_state(self):
        backend.set_value(
            self.total_cm, np.zeros((self.num_classes, self.num_classes)))

    def result(self):
        return self.total_cm

    def get_config(self):
        config = {
            'num_classes': self.num_classes,
        }
        return config
