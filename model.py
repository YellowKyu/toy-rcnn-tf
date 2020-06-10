from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

class RCNNModel(object):

    def __init__(self):
        inputs = keras.Input(shape=(108, 192, 3))

        x = layers.Conv2D(filters=8, kernel_size=(3, 3), activation="relu", padding="same")(inputs)

        # objectness score
        objectness = layers.Conv2D(filters=1, kernel_size=(3, 3), activation="sigmoid", padding="same", name="objectness")(x)

        # # bounding boxes (tlbr, ratio, 0-1)
        x_1 = layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")(x)
        x_2 = layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")(x_1)
        x_3 = layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")(x_2)
        # x_4 = layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")(x_3)
        # x_5 = layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")(x_4)
        # bboxes = layers.Conv2D(filters=4, kernel_size=(3, 3), activation="relu", padding="same", name="bboxes")(x_5)

        # bounding boxes
        x_3_flatten = layers.Flatten()(x_3)
        b_1 = layers.Dense(4, activation='relu', name="b_1")(x_3_flatten)
        b_2 = layers.Dense(4, activation='relu', name="b_2")(x_3_flatten)
        b_3 = layers.Dense(4, activation='relu', name="b_3")(x_3_flatten)
        b_4 = layers.Dense(4, activation='relu', name="b_4")(x_3_flatten)
        b_5 = layers.Dense(4, activation='relu', name="b_5")(x_3_flatten)
        cat_bboxes = layers.concatenate([b_1, b_2, b_3, b_4, b_5], axis=0)
        bboxes = K.expand_dims(cat_bboxes, axis=0)

        self.model = keras.Model(inputs=inputs, outputs=[objectness, bboxes])
        self.model.summary()

