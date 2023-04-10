import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50V2 as Architecture
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam

from DataSet import DataSet

from time import time


class TimesLogger(Callback):
    def __init__(self):
        super().__init__()
        self.times = []
        self.epoch_time_start = 0

    def on_epoch_begin(self, batch, logs=None):
        self.epoch_time_start = time()

    def on_epoch_end(self, batch, logs=None):
        self.times.append(round(time() - self.epoch_time_start))

    def get_times_log(self):
        return self.times


class NeuralNet:
    def __init__(self, resolution, learing_rate=0.001):
        """
        :param resolution: tuple of int (height, width)
        """
        self.resolution = resolution
        self.data_set_train = DataSet("dataset/train", "jpeg", target_size=self.resolution)
        self.data_set_train.normalize_X()
        self.data_set_train.categorize_Y()
        
        self.data_set_test = DataSet("dataset/val", "jpeg", target_size=self.resolution)
        self.data_set_test.normalize_X()
        self.data_set_test.categorize_Y()

        try:
            self.model = load_model(str(resolution) + "Network")
            print("Ancien modèle chargé")
        except OSError:
            self._create_model(learing_rate)
            print("Nouveau modèle créé")

        self.times_logger = TimesLogger()

    def _create_model(self, learning_rate):
        self.model = Architecture(weights=None, input_shape=self.data_set_train.get_X().shape[1:],
                                  classes=self.data_set_train.number_classes())
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

    def train(self, batch_size, epochs, shuffle_seed):
        """
        :param shuffle_seed: entier détermiant le manière dont sera mélangé les images du dataset
        """
        tf.random.set_seed(shuffle_seed)
        print("Début de l'entrainnement ...")
        self.model.fit(self.data_set_train.get_X(), self.data_set_train.get_Y(),
                       batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[self.times_logger])

    def test(self):
        print("Début du test ...")
        return round(self.model.evaluate(self.data_set_test.get_X(), self.data_set_test.get_Y(), verbose=1)[1], 4)

    def save(self):
        self.model.save(str(self.resolution) + "Network")

    def get_time_history(self):
        return self.times_logger.get_times_log()
