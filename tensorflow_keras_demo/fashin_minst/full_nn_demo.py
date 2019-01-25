import tensorflow as tf
from keras.models import load_model
import keras
import os
import numpy as  np


class FullNnTrain(object):

    def __init__(self):
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        self.model_path = 'full_nn.h5'

    def run(self):
        (train_images, train_labels), (test_images, test_labels) = self.load_data()
        if not os.path.isfile(self.model_path):
            self.train_model(train_images, train_labels)
        self.predict(test_images=test_images, test_labels=test_labels)

    def train(self, train_images, train_labels):
        self.train_model(train_images=train_images, train_labels=train_labels)

    @staticmethod
    def load_data():
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
        train_images, test_images = train_images / 255.0, test_images / 255.0
        return (train_images, train_labels), (test_images, test_labels)

    def train_model(self, train_images, train_labels):
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(train_images, train_labels, epochs=5)
        model.save(self.model_path)

    def predict(self, test_images, test_labels):
        model = load_model(self.model_path)
        predictions = model.predict(test_images)

        for i in range(len(test_images)):
            predict_label_index = int(np.argmax(predictions[i]))
            real_label_index = test_labels[i]

            print('图片标签是{} ,预测结果为{}'.format(self.class_names[real_label_index],
                                            self.class_names[predict_label_index]))


if __name__ == '__main__':
    ft = FullNnTrain()
    ft.run()
