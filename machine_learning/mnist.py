from __future__ import absolute_import, division, print_function, unicode_literals

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
tf.compat.v1.Session(config=config)


class Mnist:

    def run(self, path, model_path):
        (x_train, y_train), (x_test, y_test) = self.load_data(path)
        x_train, x_test = x_train / 255.0, x_test / 255.0

        if os.path.isfile(model_path):
            model = self.load_model(model_path)
        else:
            model = self.gen_model(x_train, y_train)

        self.evaluate_model(model, x_test, y_test)

        self.save_model(model, model_path)

    @staticmethod
    def load_data(path):
        f = np.load(path)
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

        plt.imshow(x_test[10], cmap=plt.cm.binary)
        f.close()
        return (x_train, y_train), (x_test, y_test)

    def gen_model(self, x_train, y_train):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=5)
        return model

    @staticmethod
    def evaluate_model(model, x_test, y_test):
        model.evaluate(x_test, y_test, verbose=2)

    @staticmethod
    def save_model(model, model_path):
        model.save(model_path)

    @staticmethod
    def load_model(model_path):
        return load_model(model_path)


if __name__ == '__main__':
    Mnist().run("E:\code\github\learnAi\machine_learning\data\mnist\mnist.npz",
                "E:\code\github\learnAi\machine_learning\data\mnist\keras_mnist.h5")
