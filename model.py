import tensorflow as tf
from tensorflow.keras import layers, models

class MiniAlexNet:
    @staticmethod
    def build(input_shape=(150, 150, 3), num_classes=3):
        model = models.Sequential()
        # Bloque 1
        model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2,2)))
        # Bloque 2
        model.add(layers.Conv2D(64, (3,3), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2,2)))
        # Bloque 3
        model.add(layers.Conv2D(128, (3,3), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2,2)))
        # Clasificación
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(num_classes, activation='softmax'))
        return model
