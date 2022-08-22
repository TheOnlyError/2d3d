from keras_applications import get_submodules_from_kwargs
import tensorflow_addons as tfa


def freeze_model(model, **kwargs):
    """Set all layers non trainable, excluding BatchNormalization layers"""
    _, layers, _, _ = get_submodules_from_kwargs(kwargs)
    for layer in model.layers:
        if not isinstance(layer, layers.BatchNormalization) and not isinstance(layer, tfa.layers.GroupNormalization):
            layer.trainable = False
    return


def freeze_member(model, n_ensemble, **kwargs):
    """Set all layers non trainable, excluding BatchNormalization layers"""
    _, layers, _, _ = get_submodules_from_kwargs(kwargs)
    for layer in model.layers:
        if not isinstance(layer, layers.BatchNormalization) and not isinstance(layer, tfa.layers.GroupNormalization):
            layer.trainable = False
            layer._name = 'ensemble_' + str(n_ensemble + 1) + '_' + layer.name
    return


def filter_keras_submodules(kwargs):
    """Selects only arguments that define keras_application submodules. """
    submodule_keys = kwargs.keys() & {'backend', 'layers', 'models', 'utils'}
    return {key: kwargs[key] for key in submodule_keys}
