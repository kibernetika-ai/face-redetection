import tensorflow as tf

keras = tf.keras
IMG_SIZE = 64
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)


def get_model():
    # Create the base model from the pre-trained model MobileNet V2
    base_model = keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights=None,
    )
    base_model.trainable = False
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = keras.layers.Dense(1)
    model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        prediction_layer
    ])
    return model
