import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter
import numpy as np
from tensorflow.python.keras.engine.training import _minimize
from tensorflow.python.keras.mixed_precision.experimental.loss_scale_optimizer import LossScaleOptimizer


class BaseModel(Model):
    def __init__(self, *args, tta=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.tta = tta

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        optimizer = self.optimizer

        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)
            # if isinstance(optimizer, LossScaleOptimizer):
            #     loss = optimizer.get_scaled_loss(loss)

        trainable_variables = self.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)
        # if isinstance(optimizer, LossScaleOptimizer):
        #     gradients = optimizer.get_unscaled_gradients(gradients)

        # aaf_trainable = trainable_variables[len(trainable_variables)-len([v for v in trainable_variables if 'edge' in v.name]):]

        aaf_len = len([v for v in trainable_variables if 'edge' in v.name])
        # other_grads = gradients[:len(trainable_variables)-aaf_len:]
        # aaf_grads = -gradients[len(trainable_variables)-aaf_len:]

        gradients[-aaf_len:] = [-grad for grad in gradients[-aaf_len:]]

        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        # self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """The logic for one evaluation step.
        This method can be overridden to support custom evaluation logic.
        This method is called by `Model.make_test_function`.
        This function should contain the mathematical logic for one step of
        evaluation.
        This typically includes the forward pass, loss calculation, and metrics
        updates.
        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_test_function`, which can also be overridden.
        Args:
          data: A nested structure of `Tensor`s.
        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned.
        """
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        if self.tta:
            image = x.numpy()[0]
            depth = image.shape[-1]
            image_batch = np.array([
                image,
                np.fliplr(image),
                np.flipud(image),
                np.rot90(image, k=1),
                np.rot90(image, k=3),
            ])
            prediction = self(image_batch, training=False).numpy()
            results = [
                prediction[0].argmax(axis=-1),
                np.fliplr(prediction[1].argmax(axis=-1)),
                np.flipud(prediction[2].argmax(axis=-1)),
                np.rot90(prediction[3].argmax(axis=-1), k=-1),
                np.rot90(prediction[4].argmax(axis=-1), k=-3),
            ]
            result = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=np.array(results)).astype(
                np.uint8)

            # preds = tf.concat([
            #     tf.math.argmax(self(x, training=False), axis=-1),
            #     tf.math.argmax(tf.image.flip_left_right(self(tf.image.flip_left_right(x), training=False)), axis=-1),
            #     tf.math.argmax(tf.image.flip_up_down(self(tf.image.flip_up_down(x), training=False)), axis=-1),
            #     # tf.math.argmax(tf.image.rot90(self(tf.image.rot90(x, k=1), training=False), k=3), axis=-1),
            #     # tf.math.argmax(tf.image.rot90(self(tf.image.rot90(x, k=3), training=False), k=1), axis=-1),
            # ], axis=0)
            y_pred = tf.one_hot(np.expand_dims(result, axis=0), depth=depth)
            # TODO check ypred shape
        else:
            y_pred = self(x, training=False)

        # Updates stateful loss metrics.
        self.compiled_loss(
            y, y_pred, sample_weight, regularization_losses=self.losses)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}
