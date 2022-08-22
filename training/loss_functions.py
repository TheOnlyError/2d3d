from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf

# Helper function to enable loss function to be flexibly used for
# both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn
from training.aaf_layers import ignores_from_label, edges_from_label, eightcorner_activation


def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5:
        return [1, 2, 3]
    # Two dimensional
    elif len(shape) == 4:
        return [1, 2]
    # Exception - Unknown
    else:
        raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')


################################
#           Dice loss          #
################################
def dice_loss(delta=0.5, smooth=0.000001):
    """Dice loss originates from Sørensen–Dice coefficient, which is a statistic developed in 1940s to gauge the similarity between two samples.

    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.5
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """

    def loss_function(y_true, y_pred):
        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1 - y_pred), axis=axis)
        fp = K.sum((1 - y_true) * y_pred, axis=axis)
        # Calculate Dice score
        dice_class = (tp + smooth) / (tp + delta * fn + (1 - delta) * fp + smooth)
        # Average class scores
        dice_loss = K.mean(1 - dice_class)

        return dice_loss

    return loss_function


################################
#         Tversky loss         #
################################
def tversky_loss(delta=0.7, smooth=0.000001):
    """Tversky loss function for image segmentation using 3D fully convolutional deep networks
	Link: https://arxiv.org/abs/1706.05721
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """

    def loss_function(y_true, y_pred):
        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1 - y_pred), axis=axis)
        fp = K.sum((1 - y_true) * y_pred, axis=axis)
        tversky_class = (tp + smooth) / (tp + delta * fn + (1 - delta) * fp + smooth)
        # Average class scores
        tversky_loss = K.mean(1 - tversky_class)

        return tversky_loss

    return loss_function


################################
#       Dice coefficient       #
################################
def dice_coefficient(delta=0.5, smooth=0.000001):
    """The Dice similarity coefficient, also known as the Sørensen–Dice index or simply Dice coefficient, is a statistical tool which measures the similarity between two sets of data.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.5
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """

    def loss_function(y_true, y_pred):
        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1 - y_pred), axis=axis)
        fp = K.sum((1 - y_true) * y_pred, axis=axis)
        dice_class = (tp + smooth) / (tp + delta * fn + (1 - delta) * fp + smooth)
        # Average class scores
        dice = K.mean(dice_class)

        return dice

    return loss_function


################################
#          Combo loss          #
################################
def combo_loss(alpha=0.5, beta=0.5):
    """Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation
    Link: https://arxiv.org/abs/1805.02798
    Parameters
    ----------
    alpha : float, optional
        controls weighting of dice and cross-entropy loss., by default 0.5
    beta : float, optional
        beta > 0.5 penalises false negatives more than false positives., by default 0.5
    """

    def loss_function(y_true, y_pred):
        dice = dice_coefficient()(y_true, y_pred)
        axis = identify_axis(y_true.get_shape())
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)

        if beta is not None:
            beta_weight = np.array([beta, 1 - beta])
            cross_entropy = beta_weight * cross_entropy
        # sum over classes
        cross_entropy = K.mean(K.sum(cross_entropy, axis=[-1]))
        if alpha is not None:
            combo_loss = (alpha * cross_entropy) - ((1 - alpha) * dice)
        else:
            combo_loss = cross_entropy - dice
        return combo_loss

    return loss_function


################################
#      Focal Tversky loss      #
################################
def focal_tversky_loss(delta=0.7, gamma=0.75, smooth=0.000001):
    """A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation
    Link: https://arxiv.org/abs/1810.07842
    Parameters
    ----------
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """

    def loss_function(y_true, y_pred):
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1 - y_pred), axis=axis)
        fp = K.sum((1 - y_true) * y_pred, axis=axis)
        tversky_class = (tp + smooth) / (tp + delta * fn + (1 - delta) * fp + smooth)
        # Average class scores
        focal_tversky_loss = K.mean(K.pow((1 - tversky_class), gamma))

        return focal_tversky_loss

    return loss_function


################################
#          Focal loss          #
################################
def focal_loss(alpha=None, beta=None, gamma_f=2.):
    """Focal loss is used to address the issue of the class imbalance problem. A modulation term applied to the Cross-Entropy loss function.
    Parameters
    ----------
    alpha : float, optional
        controls relative weight of false positives and false negatives. Beta > 0.5 penalises false negatives more than false positives, by default None
    gamma_f : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 2.
    """

    def loss_function(y_true, y_pred):
        axis = identify_axis(y_true.get_shape())
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)

        if alpha is not None:
            alpha_weight = np.array(alpha, dtype=np.float32)
            focal_loss = alpha_weight * K.pow(1 - y_pred, gamma_f) * cross_entropy
        else:
            focal_loss = K.pow(1 - y_pred, gamma_f) * cross_entropy

        focal_loss = K.mean(K.sum(focal_loss, axis=[-1]))
        return focal_loss

    return loss_function


################################
#       Symmetric Focal loss      #
################################
def symmetric_focal_loss(delta=0.7, gamma=2.):
    """
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    """

    def loss_function(y_true, y_pred):
        axis = identify_axis(y_true.get_shape())

        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        # calculate losses separately for each class
        back_ce = K.pow(1 - y_pred[:, :, :, 0], gamma) * cross_entropy[:, :, :, 0]
        back_ce = (1 - delta) * back_ce

        fore_ce = K.pow(1 - y_pred[:, :, :, 1], gamma) * cross_entropy[:, :, :, 1]
        fore_ce = delta * fore_ce

        loss = K.mean(K.sum(tf.stack([back_ce, fore_ce], axis=-1), axis=-1))

        return loss

    return loss_function


#################################
# Symmetric Focal Tversky loss  #
#################################
def symmetric_focal_tversky_loss(delta=0.7, gamma=0.75):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """

    def loss_function(y_true, y_pred):
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1 - y_pred), axis=axis)
        fp = K.sum((1 - y_true) * y_pred, axis=axis)
        dice_class = (tp + epsilon) / (tp + delta * fn + (1 - delta) * fp + epsilon)

        # calculate losses separately for each class, enhancing both classes
        back_dice = (1 - dice_class[:, 0]) * K.pow(1 - dice_class[:, 0], -gamma)
        fore_dice = (1 - dice_class[:, 1]) * K.pow(1 - dice_class[:, 1], -gamma)

        # Average class scores
        loss = K.mean(tf.stack([back_dice, fore_dice], axis=-1))
        return loss

    return loss_function


################################
#     Asymmetric Focal loss    #
################################
def asymmetric_focal_loss(delta=0.7, gamma=2.):
    def loss_function(y_true, y_pred):
        """For Imbalanced datasets
        Parameters
        ----------
        delta : float, optional
            controls weight given to false positive and false negatives, by default 0.7
        gamma : float, optional
            Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
        """

        # axis = identify_axis(y_true.get_shape())

        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)

        # calculate losses separately for each class, only suppressing background class
        back_ce = K.pow(1 - y_pred[:, :, :, 0], gamma) * cross_entropy[:, :, :, 0]
        back_ce = (1 - delta) * back_ce

        # Edit: Support multi class loss
        back_ce = K.expand_dims(back_ce, axis=-1)
        fore_ce = cross_entropy[:, :, :, 1:]
        fore_ce = delta * fore_ce

        loss = K.mean(K.sum(tf.concat([back_ce, fore_ce], axis=-1), axis=-1))

        return loss

    return loss_function


#################################
# Asymmetric Focal Tversky loss #
#################################
def asymmetric_focal_tversky_loss(delta=0.7, gamma=0.75):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """

    def loss_function(y_true, y_pred):
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1 - y_pred), axis=axis)
        fp = K.sum((1 - y_true) * y_pred, axis=axis)
        dice_class = (tp + epsilon) / (tp + delta * fn + (1 - delta) * fp + epsilon)

        # calculate losses separately for each class, only enhancing foreground class
        back_dice = (1 - dice_class[:, 0])
        # Edit: Support multi class loss
        back_dice = K.expand_dims(back_dice, axis=-1)
        fore_dice = (1 - dice_class[:, 1:]) * K.pow(1 - dice_class[:, 1:], -gamma)

        # Average class scores
        loss = K.mean(tf.concat([back_dice, fore_dice], axis=-1))
        return loss

    return loss_function


###########################################
#      Symmetric Unified Focal loss       #
###########################################
def sym_unified_focal_loss(weight=0.5, delta=0.6, gamma=0.5):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to symmetric Focal Tversky loss and symmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    """

    def loss_function(y_true, y_pred):
        symmetric_ftl = symmetric_focal_tversky_loss(delta=delta, gamma=gamma)(y_true, y_pred)
        symmetric_fl = symmetric_focal_loss(delta=delta, gamma=gamma)(y_true, y_pred)
        if weight is not None:
            return (weight * symmetric_ftl) + ((1 - weight) * symmetric_fl)
        else:
            return symmetric_ftl + symmetric_fl

    return loss_function


###########################################
#      Asymmetric Unified Focal loss      #
###########################################
# TODO tune gamma in [0.1, 0.9]
def asym_unified_focal_loss(num_classes, weight=0.5, delta=0.6, gamma=0.5):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to asymmetric Focal Tversky loss and asymmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    """

    def loss_function(y_true, y_pred):
        y_true = y_true[:, :, :, :num_classes]

        asymmetric_ftl = asymmetric_focal_tversky_loss(delta=delta, gamma=gamma)(y_true, y_pred)
        asymmetric_fl = asymmetric_focal_loss(delta=delta, gamma=gamma)(y_true, y_pred)

        if weight is not None:
            return (weight * asymmetric_ftl) + ((1 - weight) * asymmetric_fl)
        else:
            return asymmetric_ftl + asymmetric_fl

    return loss_function


def balanced_entropy(num_classes):
    def loss_function(y_true, y_pred):
        y_true = y_true[:, :, :, :num_classes]

        eps = tf.keras.backend.epsilon()
        z = tf.keras.activations.softmax(y_pred)
        cliped_z = tf.keras.backend.clip(z, eps, 1 - eps)
        log_z = tf.keras.backend.log(cliped_z)

        # num_classes = y_true.shape.as_list()[-1]
        ind = tf.keras.backend.argmax(y_true, axis=-1)
        total = tf.keras.backend.sum(y_true)

        m_c, n_c, loss = [], [], 0
        for c in range(num_classes):
            m_c.append(tf.keras.backend.cast(
                tf.keras.backend.equal(ind, c), dtype=tf.int32))
            n_c.append(tf.keras.backend.cast(
                tf.keras.backend.sum(m_c[-1]), dtype=tf.float32))

        c = []
        for i in range(num_classes):
            c.append(total - n_c[i])
        tc = tf.math.add_n(c)

        for i in range(num_classes):
            w = c[i] / tc
            m_c_one_hot = tf.one_hot((i * m_c[i]), num_classes, axis=-1)
            y_c = m_c_one_hot * y_true
            loss += w * tf.keras.backend.mean(-tf.keras.backend.sum(y_c * log_z, axis=1))

        return loss / num_classes

    return loss_function


def categorical_crossentropy(num_classes):
    def loss_function(y_true, y_pred):
        y_true = y_true[:, :, :, :num_classes]

        axis = -1
        # scale preds so that the class probas of each sample sum to 1
        output = y_pred / tf.reduce_sum(y_pred, axis, True)
        # Compute cross entropy from probabilities.
        epsilon_ = tf.constant(tf.keras.backend.epsilon(), y_pred.dtype.base_dtype)
        output = tf.clip_by_value(output, epsilon_, 1. - epsilon_)
        return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(output), axis))

    return loss_function


# def adaptive_affinity_loss(labels,
#                            one_hot_lab,
#                            probs,
#                            size,
#                            num_classes,
#                            kld_margin,
#                            w_edge,
#                            w_not_edge):
def adaptive_affinity_loss(size,
                           k,
                           num_classes,
                           w_edge,
                           w_not_edge,
                           kld_margin=3.0):
    """Adaptive affinity field (AAF) loss.
    This function computes AAF loss. There are three components in the function:
    1) extracts edges from the ground-truth labels.
    2) extracts ignored pixels and their paired pixels (usually the eight corner
       pixels).
    3) extracts eight corner pixels/predictions from the center in a
       (2*size+1)x(2*size+1) patch
    4) computes KL-Divergence between center pixels and their paired pixels (the
       eight corner).
    5) imposes adaptive weightings on the loss.
    Args:
      labels: A tensor of size [batch_size, height_in, width_in], indicating
        semantic segmentation ground-truth labels.
      one_hot_lab: A tensor of size [batch_size, height_in, width_in, num_classes]
        which is the ground-truth labels in the form of one-hot vector.
      probs: A tensor of size [batch_size, height_in, width_in, num_classes],
        indicating segmentation predictions.
      size: A number indicating the half size of a patch.
      num_classes: A number indicating the total number of valid classes.
      w_edge: A number indicating the weighting for KL-Divergence at edge.
      w_not_edge: A number indicating the weighting for KL-Divergence at non-edge.
      kld_margin: A number indicating the margin for KL-Divergence at edge.
    Returns:
      Two 1-D tensors value indicating the loss at edge and non-edge.
    """

    def loss_function(y_true, y_pred):
        y_true = y_true[:, :, :, :num_classes]

        # Compute ignore map (e.g, label of 255 and their paired pixels).
        labels = tf.keras.backend.argmax(y_true, axis=-1)  # NxHxW TODO maybe need squeeze still check source code
        # labels = tf.squeeze(tf.keras.backend.argmax(y_true, axis=-1), axis=-1)  # NxHxW TODO maybe need squeeze still check source code
        ignore = ignores_from_label(labels, num_classes, size)  # NxHxWx8
        not_ignore = tf.logical_not(ignore)
        not_ignore = tf.expand_dims(not_ignore, axis=3)  # NxHxWx1x8

        # Compute edge map.
        edge = edges_from_label(y_true, size, 0)  # NxHxWxCx8

        # Remove ignored pixels from the edge/non-edge.
        edge = tf.logical_and(edge, not_ignore)
        not_edge = tf.logical_and(tf.logical_not(edge), not_ignore)

        edge_indices = tf.where(tf.reshape(edge, [-1]))
        not_edge_indices = tf.where(tf.reshape(not_edge, [-1]))

        # Extract eight corner from the center in a patch as paired pixels.
        probs_paired = eightcorner_activation(y_pred, size)  # NxHxWxCx8
        probs = tf.expand_dims(y_pred, axis=-1)  # NxHxWxCx1
        bot_epsilon = tf.constant(1e-4, name='bot_epsilon')
        top_epsilon = tf.constant(1.0, name='top_epsilon')

        neg_probs = tf.clip_by_value(
            1 - probs, bot_epsilon, top_epsilon)
        neg_probs_paired = tf.clip_by_value(
            1 - probs_paired, bot_epsilon, top_epsilon)
        probs = tf.clip_by_value(
            probs, bot_epsilon, top_epsilon)
        probs_paired = tf.clip_by_value(
            probs_paired, bot_epsilon, top_epsilon)

        # Compute KL-Divergence.
        kldiv = probs_paired * tf.math.log(probs_paired / probs)
        kldiv += neg_probs_paired * tf.math.log(neg_probs_paired / neg_probs)
        edge_loss = tf.maximum(0.0, kld_margin - kldiv)
        not_edge_loss = kldiv

        w_edge_norm = tf.nn.softmax(w_edge, axis=-1)
        w_not_edge_norm = tf.nn.softmax(w_not_edge, axis=-1)

        # Impose weights on edge/non-edge losses.
        one_hot_lab = tf.expand_dims(y_true, axis=-1)
        w_edge_sum = tf.reduce_sum(w_edge_norm[..., k] * one_hot_lab, axis=3, keepdims=True)  # NxHxWx1x1
        w_not_edge_sum = tf.reduce_sum(w_not_edge_norm[..., k] * one_hot_lab, axis=3, keepdims=True)  # NxHxWx1x1

        edge_loss *= w_edge_sum
        not_edge_loss *= w_not_edge_sum

        not_edge_loss = tf.reshape(not_edge_loss, [-1])
        not_edge_loss = tf.gather(not_edge_loss, not_edge_indices)
        edge_loss = tf.reshape(edge_loss, [-1])
        edge_loss = tf.gather(edge_loss, edge_indices)

        loss = 0.0
        scale = 0.75
        if tf.greater(tf.size(edge_loss), 0):
            loss += 0.5 * 1 / scale * tf.reduce_mean(edge_loss)
        if tf.greater(tf.size(not_edge_loss), 0):
            loss += 20 * scale * tf.reduce_mean(not_edge_loss)
        return loss / num_classes

    return loss_function


def mean_squared_error():
    def loss_function(y_true, y_pred):
        v = tf.reduce_mean(tf.reduce_mean(tf.math.squared_difference(y_pred, y_true), axis=-1))
        if tf.math.is_nan(v):
            tf.print(y_true)
            tf.print(y_pred)
            tf.print(tf.math.squared_difference(y_pred, y_true))
        return v

    return loss_function


def heatmap_regression_loss(num_classes, opening_inds):
    def loss_function(y_true, y_pred):
        loss = 0
        for i, ind in enumerate(opening_inds):
            opening_pred = y_pred[:, :, :, ind]  # Scale this since predictions are in [0,1]
            heatmap_true = y_true[:, :, :, num_classes + i]  # Use heatmap offset
            loss += tf.reduce_mean(tf.square(tf.abs(heatmap_true - opening_pred) + 1) - 1)
        return loss / len(opening_inds)

    return loss_function


def heatmap_regression_loss_nomean(num_classes, opening_inds):
    def loss_function(y_true, y_pred):
        loss = 0
        for i, ind in enumerate(opening_inds):
            opening_pred = y_pred[:, :, :, ind]  # Scale this since predictions are in [0,1]
            heatmap_true = y_true[:, :, :, num_classes + i]  # Use heatmap offset
            loss += tf.reduce_mean(tf.square(tf.abs(heatmap_true - opening_pred) + 1) - 1)
        return loss

    return loss_function


def combined_loss(loss_funcs):
    def loss_function(y_true, y_pred):
        loss_sum = 0
        for func in loss_funcs:
            loss_sum += func(y_true, y_pred)
        return loss_sum

    return loss_function
