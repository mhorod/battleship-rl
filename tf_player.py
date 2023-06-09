from player import *
from random_player import *

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import numpy as np
from stats import *

import matplotlib.pyplot as plt

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
    ships = RandomPlacer().place_ships()
    shoot_board = Board()

    while ships.count(Tile.SHIP) > 0:
        fig, ax = plt.subplots(1, 3)

        x, y = board_to_sample(shoot_board, ships)
        ax[0].matshow(x, vmin=-1, vmax=1)
        ax[1].matshow(y, vmin=-1, vmax=1)
        ax[2].matshow(shooter.model.predict(np.array([x]), verbose=0)[0])
        plt.show()

        pos = shooter.shoot(shoot_board)
        shoot_board[pos] = Tile.HIT if ships[pos] == Tile.SHIP else Tile.MISS


def board_to_sample(board):
    return board.get_repr(), board.get_ship_repr()

def make_dataset(placer, shooter, games):
    xs = []
    ys = []
    boards = [Board(placer.place_ships()) for _ in range(games)]
    ship_counts = [board.count_ship_tiles() for board in boards]
    moments_to_extract = [random.randint(10, BOARD_SIZE * BOARD_SIZE - 10) for _ in range(games)]
    game_length = 0

    while len(boards) > 0:
        game_length += 1
        indices_to_remove = []
        for i, (board, moment) in enumerate(zip(boards, moments_to_extract)):
            if game_length == moment:
                x, y = board_to_sample(boards[i])
                xs.append(x)
                ys.append(y)
                indices_to_remove.append(i)

        for i in indices_to_remove[::-1]:
            boards.pop(i)
            ship_counts.pop(i)
            moments_to_extract.pop(i)

        if len(boards) == 0:
            break

        shots = shooter.shoot_many(boards)
        indices_to_remove = []
        for i, (board, shot) in enumerate(zip(boards, shots)):
            result = board.shoot(shot)
            if result == ShotResult.HIT or result == ShotResult.SUNK:
                ship_counts[i] -= 1
                if ship_counts[i] == 0:
                    indices_to_remove.append(i)
        
        for i in indices_to_remove[::-1]:
            x, y = board_to_sample(board)
            xs.append(x)
            ys.append(y)

            boards.pop(i)
            ship_counts.pop(i)
            moments_to_extract.pop(i)

    dataset = tf.data.Dataset.from_tensor_slices((xs, ys))
    return dataset

def iterate_fitting(model_func, start_shooter, games, epochs, iterations):
    shooter = start_shooter
    for i in range(iterations):
        print("Iteration ", i)
        dataset = make_dataset(RandomPlacer(), shooter, games)
        size = len(list(dataset))
        train, val = dataset.take(int(0.8 * size)), dataset.skip(int(0.8 * size))
        model = model_func()
        model.fit(train.batch(32), epochs=epochs, validation_data=val.batch(32))
        shooter = TFShooter(model)
    return shooter


def make_dense_model():
    dense_model = models.Sequential([
        layers.Flatten(input_shape=(BOARD_SIZE, BOARD_SIZE, len(Tile))),
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(BOARD_SIZE * BOARD_SIZE, activation='sigmoid'),
        layers.Reshape((BOARD_SIZE, BOARD_SIZE))
    ])

    dense_model.compile(optimizer='adam',loss='binary_crossentropy')

    return dense_model

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


dense_shooter = iterate_fitting(make_dense_model, RandomPlayer(), games=5000, epochs=100, iterations=1)
plot_predictions(dense_shooter)

#cnn_shooter = iterate_fitting(make_cnn_model, RandomPlayer(), games=2000, epochs=200, iterations=1)
#plot_predictions(cnn_shooter)

GAMES = 200
print("random playing...")
random_results = compare_placer_with_shooter(RandomPlayer(), RandomPlayer(), GAMES)

print("dense playing...")
dense_results = compare_placer_with_shooter(RandomPlayer(), dense_shooter, GAMES)

#print("cnn playing...")
#cnn_results = compare_placer_with_shooter(RandomPlayer(), cnn_shooter, GAMES)

print("Random player: ", np.mean(random_results))
print("Dense player:  ", np.mean(dense_results))
#print("CNN player:    ", np.mean(cnn_results))