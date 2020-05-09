import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
tf.compat.v1.Session(config=config)


class Cifar(object):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    def run(self, file_path):
        (train_images, train_labels), (test_images, test_labels) = self.load_data(file_path)
        model = self.gen_model()
        history = model.fit(train_images, train_labels, epochs=10,
                            validation_data=(test_images, test_labels))
        test_loss, test_acc = self.evaluate_model(history, model, test_images, test_labels)
        print(test_acc)

    @staticmethod
    def gen_model():
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10))
        model.summary()
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        return model

    @staticmethod
    def evaluate_model(history, model, test_images, test_lables):
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.show()
        return model.evaluate(test_images, test_lables, verbose=2)

    # 展示数据
    def show_data(self, train_images, test_images, train_labels):
        # Normalize pixel values to be between 0 and 1
        train_images, test_images = train_images / 255.0, test_images / 255.0

        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i], cmap=plt.cm.binary)
            # The CIFAR labels happen to be arrays,
            # which is why you need the extra index
            plt.xlabel(self.class_names[train_labels[i][0]])
        plt.show()

    # 加载数据
    @staticmethod
    def load_data(path):
        num_train_samples = 50000

        x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
        y_train = np.empty((num_train_samples,), dtype='uint8')

        for i in range(1, 6):
            fpath = os.path.join(path, 'data_batch_' + str(i))
            (x_train[(i - 1) * 10000:i * 10000, :, :, :],
             y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

        fpath = os.path.join(path, 'test_batch')
        x_test, y_test = load_batch(fpath)

        y_train = np.reshape(y_train, (len(y_train), 1))
        y_test = np.reshape(y_test, (len(y_test), 1))

        if K.image_data_format() == 'channels_last':
            x_train = x_train.transpose(0, 2, 3, 1)
            x_test = x_test.transpose(0, 2, 3, 1)

        return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    Cifar().run("E:\code\github\learnAi\machine_learning\data\cifar-10-batches-py")
