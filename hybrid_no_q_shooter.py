from game import *
from game import np
from player import *
from prediction_shooter import *
from hunter_player import *

import tensorflow as tf
import numpy as np
from tensorflow import keras

from collections import deque
import time
import random


def agent(board_config):
    BOARD_SIZE = board_config.size
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(BOARD_SIZE, BOARD_SIZE, len(Tile)))),
    model.add(keras.layers.Dense(BOARD_SIZE * BOARD_SIZE, activation='relu'))
    model.add(keras.layers.Dense(BOARD_SIZE * BOARD_SIZE, activation='relu'))
    model.add(keras.layers.Dense(BOARD_SIZE * BOARD_SIZE, activation='sigmoid'))
    model.add(keras.layers.Reshape((BOARD_SIZE, BOARD_SIZE)))
    return model

class HybridNoQPredictor(ShipPredictor):

    def __init__(self, board_config: BoardConfig):
        self.board_config = board_config
        self.hunter_predictor = HunterPredictor()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
        self.model = agent(board_config)

    def save(self, filename):
        self.model.save_weights(filename)

    def load(self, filename):
        self.model.load_weights(filename)

    def predict_ships(self, board: Board):
        if board.count(Tile.HIT) > 0:
            return self.hunter_predictor.predict_ships(board)
        else:
            return self.model(np.array([board.get_repr()]))[0].numpy()

    def predict_ships_many(self, boards) -> list:
        hunter_pred = self.hunter_predictor.predict_ships_many(boards)
        my_pred = self.model(np.array([board.get_repr() for board in boards])).numpy()
        return [hunter_pred[i] if boards[i].count(Tile.HIT) > 0 else my_pred[i] for i in range(len(boards))]



    def update_weights(self, replay_memory):
        MIN_REPLAY_SIZE = 1000
        if len(replay_memory) < MIN_REPLAY_SIZE:
            return

        batch_size = 64
        mini_batch = random.sample(replay_memory, batch_size)

        X = []
        Y = []
        for (board, ships) in mini_batch:
            X.append(board)
            Y.append(ships)

        with tf.GradientTape() as tape:
            predictions = self.model(np.array(X), training=True)
            loss = tf.reduce_mean((np.array(Y) - predictions)**2)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    
    def train(self, placer: Placer):
        BOARD_SIZE = self.board_config.size
        train_episodes = 5000
        epsilon = 1
        max_epsilon = 1
        min_epsilon = 0.1
        decay = 0.001
        replay_memory = deque(maxlen=10_000)
        steps_to_update_target_model = 0

        for episode in range(train_episodes):
            shots = 0
            board = Board(placer.place_ships())
            done = False
            while not done:
                state = board.get_repr()
                ships = board.get_ship_repr()
                replay_memory.append((state, ships))

                steps_to_update_target_model += 1
                random_number = np.random.rand()
                if random_number <= epsilon:
                    while True:
                        action = (np.random.randint(0, BOARD_SIZE), np.random.randint(0, BOARD_SIZE))
                        if board[action] == Tile.EMPTY:
                            break
                else:
                    action = PredictionShooter(self).shoot(board)
                
                board.shoot(action)
                shots += 1
                while board.count(Tile.HIT) != 0:
                    board.shoot(PredictionShooter(self.hunter_predictor).shoot(board))
                    shots += 1

                done = (board.count_ship_tiles() == 0)

                if steps_to_update_target_model % 4 == 0 or done:
                    self.update_weights(replay_memory)


                if done:
                    print('Total shots : {} after n steps = {}, epsilon={}.'.format(shots, episode, np.round(epsilon*100)/100))

            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

class HybridNoQShooter(PredictionShooter):
    def __init__(self, board_config: BoardConfig):
        self.predictor = HybridNoQPredictor(board_config)
        super().__init__(self.predictor)

    def save(self, filename):
        self.predictor.save(filename)

    def load(self, filename):
        self.predictor.load(filename)

    def train(self, placer):
        self.predictor.train(placer)

def load_hybrid_no_q_shooter(board_config, filename):
    shooter = HybridNoQShooter(board_config)
    shooter.load(filename)
    return shooter
