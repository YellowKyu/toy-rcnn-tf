from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from tensorflow.keras import Model

class RCNNModel(Model):

    def __init__(self):
        super(RCNNModel, self).__init__()

        # inputs = keras.Input(shape=(108, 192, 3))

        self.conv_1 = layers.Conv2D(filters=8, kernel_size=(3, 3), activation="relu", padding="same")

        # objectness score
        self.objectness = layers.Conv2D(filters=1, kernel_size=(3, 3), activation="sigmoid", padding="same", name="objectness")

        # # bounding boxes (tlbr, ratio, 0-1)
        self.conv_bb_1 = layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")
        self.conv_bb_2 = layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")
        self.conv_bb_3 = layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")
        self.conv_bb_4 = layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")
        self.conv_bb_5 = layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")
        self.bboxes = layers.Conv2D(filters=4, kernel_size=(3, 3), activation="relu", padding="same", name="bboxes")

    def call(self, x):
        x = self.conv_1(x)
        objectness = self.objectness(x)
        x = self.conv_bb_1(x)
        x = self.conv_bb_2(x)
        x = self.conv_bb_3(x)
        x = self.conv_bb_4(x)
        x = self.conv_bb_5(x)
        bboxes = self.bboxes(x)
        return objectness, bboxes

