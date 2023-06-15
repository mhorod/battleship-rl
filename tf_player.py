
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt
import numpy as np

from dataset import *
from player import *
from random_player import *
from prediction_shooter import *

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

def fit_model(model, dataset, epochs, filename=None):
    size = len(list(dataset))
    train, val = dataset.take(int(0.8 * size)), dataset.skip(int(0.8 * size))
    history = model.fit(train.batch(64), epochs=epochs, validation_data=val.batch(64), )
    if filename is not None:
        model.save(filename)
    return history, NeuralNetworkShooter(model)

def load_model(filename):
    return NeuralNetworkShooter(tf.keras.models.load_model(filename))

def make_perceptron_model(board_config):
    BOARD_SIZE = board_config.size
    perceptron_model = models.Sequential([
        layers.Flatten(input_shape=(BOARD_SIZE, BOARD_SIZE, len(Tile))),
        layers.Dense(BOARD_SIZE * BOARD_SIZE, activation='sigmoid'),
        layers.Reshape((BOARD_SIZE, BOARD_SIZE))
    ])

    perceptron_model.compile(optimizer='adam',loss='binary_crossentropy')
    return perceptron_model

def make_dense_model(board_config):
    BOARD_SIZE = board_config.size
    perceptron_model = models.Sequential([
        layers.Flatten(input_shape=(BOARD_SIZE, BOARD_SIZE, len(Tile))),
        layers.Dense(128, activation='relu'),
        layers.Dense(BOARD_SIZE * BOARD_SIZE, activation='sigmoid'),
        layers.Reshape((BOARD_SIZE, BOARD_SIZE))
    ])

    perceptron_model.compile(optimizer='adam',loss='binary_crossentropy')

    return perceptron_model

def make_cnn_model(board_config):
    BOARD_SIZE = board_config.size
    cnn_model = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(BOARD_SIZE, BOARD_SIZE, len(Tile))),
        layers.Flatten(),
        layers.Dense(BOARD_SIZE * BOARD_SIZE, activation='sigmoid'),
        layers.Reshape((BOARD_SIZE, BOARD_SIZE))
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    cnn_model.compile(optimizer, loss='binary_crossentropy')

    return cnn_model
