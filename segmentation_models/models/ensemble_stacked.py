from ._utils import freeze_member
import tensorflow as tf


def EnsembleStacked(model_props):
    return build_ensemble(model_props)


def build_ensemble(model_props):
    # Models array according to
    # [model_dir, custom_objects]
    print("========= Loading members =========")
    members = []
    for i, model_prop in enumerate(model_props):
        model = tf.keras.models.load_model(model_prop[0], custom_objects=model_prop[1])
        freeze_member(model, i)  # Set member non trainable
        members.append(model)
    print("Done")

    inputs = members[0].input
    outputs = tf.keras.layers.average(members)  # Output is simple average
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
