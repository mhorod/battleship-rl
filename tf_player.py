
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt
import numpy as np

from dataset import *
from player import *
from random_player import *
from hybrid_player import *
from stats import *

class NeuralNetworkPredictor(ShipPredictor):
    '''
    Predicts ship placement using a neural network.
    '''
    def __init__(self, model):
        self.model = model

    def predict_ships(self, board):
        x, _ = board_to_sample(board)
        return np.array(self.model(np.array([x]), training=False)[0])

    def predict_ships_many(self, boards):
        xs = np.array([board_to_sample(board)[0] for board in boards])
        ys = np.array(self.model(xs, training=False))
        return ys

class NeuralNetworkShooter(PredictionShooter):
    def __init__(self, model):
        super().__init__(NeuralNetworkPredictor(model))

def fit_model(model, shooter, games, epochs):
    xs, ys = make_dataset(RandomPlacer(), shooter, games)
    dataset = tf.data.Dataset.from_tensor_slices((xs, ys))
    size = len(list(dataset))
    train, val = dataset.take(int(0.8 * size)), dataset.skip(int(0.8 * size))
    history = model.fit(train.batch(64), epochs=epochs, validation_data=val.batch(64))
    return history, PredictionShooter(NeuralNetworkPredictor(model))

def make_perceptron_model():
    perceptron_model = models.Sequential([
        layers.Flatten(input_shape=(BOARD_SIZE, BOARD_SIZE, len(Tile))),
        layers.Dense(BOARD_SIZE * BOARD_SIZE, activation='sigmoid'),
        layers.Reshape((BOARD_SIZE, BOARD_SIZE))
    ])

    perceptron_model.compile(optimizer='adam',loss='binary_crossentropy')
    return perceptron_model

def make_dense_model():
    perceptron_model = models.Sequential([
        layers.Flatten(input_shape=(BOARD_SIZE, BOARD_SIZE, len(Tile))),
        layers.Dense(128, activation='relu'),
        layers.Dense(BOARD_SIZE * BOARD_SIZE, activation='sigmoid'),
        layers.Reshape((BOARD_SIZE, BOARD_SIZE))
    ])

    perceptron_model.compile(optimizer='adam',loss='binary_crossentropy')

    return perceptron_model

def make_cnn_model():
    cnn_model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(BOARD_SIZE, BOARD_SIZE, len(Tile))),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(BOARD_SIZE * BOARD_SIZE, activation='sigmoid'),
        layers.Reshape((BOARD_SIZE, BOARD_SIZE))
    ])

    cnn_model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    return cnn_model
