
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt
import numpy as np

from dataset import *
from player import *
from random_player import *
from stats import *

class TFShooter(Shooter):
    def __init__(self, model):
        self.model = model
    
    def shoot_many(self, boards):
        xs = np.array([board_to_sample(board)[0] for board in boards])
        ys = self.model.predict(xs, verbose=0)
        positions = []
        for board, y in zip(boards, ys):
            pos = self.select_best_pos(board, y)
            positions.append(pos)
        return positions
    
    def shoot(self, board):
        return self.shoot_many([board])[0]
    
    def select_best_pos(self, board, predictions):
        tried = 0
        while tried < BOARD_SIZE * BOARD_SIZE:
            pos = np.unravel_index(np.argmax(predictions), predictions.shape)
            if board[pos] == Tile.EMPTY:
                return pos
            predictions[pos] = -1
            tried += 1



def plot_predictions(shooter):
    board = Board(RandomPlacer().place_ships())
    while board.count_ship_tiles() > 0:
        fig, ax = plt.subplots(1, 3)
        x, y = board_to_sample(board)

        xx = np.zeros((BOARD_SIZE, BOARD_SIZE))
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                xx[row, col] = np.argmax(x[row, col])


        ax[0].matshow(xx, vmin=0, vmax=3)
        ax[1].matshow(y, vmin=0, vmax=1)

        ax[2].matshow(shooter.model.predict(np.array([x]), verbose=0)[0])
        plt.show()
        board.shoot(shooter.shoot(board))



def iterate_fitting(model_func, start_shooter, games, epochs, iterations):
    shooter = start_shooter
    model = model_func()
    for i in range(iterations):
        print("Iteration ", i)
        xs, ys = make_dataset(RandomPlacer(), shooter, games)
        dataset = tf.data.Dataset.from_tensor_slices((xs, ys))
        size = len(list(dataset))
        train, val = dataset.take(int(0.8 * size)), dataset.skip(int(0.8 * size))
        model.fit(train.batch(32), epochs=epochs, validation_data=val.batch(32))
        shooter = TFShooter(model)
    return shooter


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
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(BOARD_SIZE, BOARD_SIZE, len(Tile))),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(BOARD_SIZE * BOARD_SIZE, activation='sigmoid'),
        layers.Reshape((BOARD_SIZE, BOARD_SIZE))
    ])

    cnn_model.compile(optimizer='adam',
                    loss='binary_crossentropy')

    return cnn_model


#perceptron_shooter = iterate_fitting(make_perceptron_model, RandomPlayer(), games=10000, epochs=200, iterations=1)
dense_shooter = iterate_fitting(make_dense_model, RandomPlayer(), games=10000, epochs=100, iterations=1)

players = [
    (RandomPlayer(), "Random"),
    #(perceptron_shooter, "Perceptron"),
    (dense_shooter, "Dense"),
]

game_lengths =[]
GAMES = 200
for player, name in players:
    player_lengths = compare_placer_with_shooter(RandomPlacer(), player, matches=GAMES)
    game_lengths.append(player_lengths)
    print(name, ":", np.mean(player_lengths), "±", np.std(player_lengths))

fig, ax = plt.subplots(1, len(players))
for i, (player, name) in enumerate(players):
    ax[i].hist(game_lengths[i], bins=range(1, 100))
    ax[i].set_title(name)
plt.show()