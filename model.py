import tensorflow as tf

from tensorflow.keras import Sequential, layers

def build_classifier(input_shape):
    model = Sequential([
        layers.Permute((2, 1), input_shape=input_shape),      # (3, 12000) -> (12000, 3)
        layers.Conv1D(64, 9, strides=2, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(4),
        layers.Conv1D(128, 7, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(256, 5, activation='relu', padding='same'),
        layers.Dropout(0.5),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

