from game import *
from game import np
from player import *
from prediction_shooter import *
from random_player import *
from configs import *
from pathlib import Path

import tensorflow as tf
import numpy as np
from tensorflow import keras

from collections import deque
import time
import random


def agent(board_config):
    BOARD_SIZE = board_config.size
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(BOARD_SIZE, BOARD_SIZE, len(Tile)))),
    model.add(keras.layers.Dense(BOARD_SIZE * BOARD_SIZE, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(BOARD_SIZE * BOARD_SIZE, activation='linear', kernel_initializer=init))
    model.add(keras.layers.Reshape((BOARD_SIZE, BOARD_SIZE)))
    return model

class QPredictor(ShipPredictor):

    def __init__(self, board_config: BoardConfig):
        self.board_config = board_config
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
        self.model = agent(board_config)
        self.target_model = agent(board_config)
        self.target_model.set_weights(self.model.get_weights())

    def save(self, filename):
        self.model.save_weights(filename)

    def load(self, filename):
        self.model.load_weights(filename)
        self.target_model.set_weights(self.model.get_weights())

    def predict_ships(self, board):
        return self.model(np.array([board.get_repr()]))[0].numpy()

    def predict_ships_many(self, boards) -> list:
        return self.model(np.array([board.get_repr() for board in boards])).numpy()



    def update_weights(self, replay_memory):
        BOARD_SIZE = self.board_config.size
        discount_factor = 0.3

        MIN_REPLAY_SIZE = 1000
        if len(replay_memory) < MIN_REPLAY_SIZE:
            return

        batch_size = 64
        mini_batch = random.sample(replay_memory, batch_size)
        current_states = np.array([transition[0] for transition in mini_batch])
        new_current_states = np.array([transition[3] for transition in mini_batch])
        future_qs_list = self.target_model(new_current_states, training=False).numpy()

        Y = []
        for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    if np.argmax(new_observation[i, j, :]) != Tile.EMPTY:
                        future_qs_list[index, i, j] = -1000
            if not done:
                max_future_q = reward + discount_factor * np.max(future_qs_list[index])
            else:
                max_future_q = reward

            Y.append(max_future_q)

        actions = np.array([transition[1] for transition in mini_batch])
        masks = np.zeros((batch_size, BOARD_SIZE, BOARD_SIZE))
        for i in range(batch_size):
            masks[i, actions[i][0], actions[i][1]] = 1
        with tf.GradientTape() as tape:
            q_values = self.model(current_states, training=True)
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=(1, 2))
            loss = tf.keras.losses.huber(Y, q_action)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    
    def train(self, placer: Placer):
        BOARD_SIZE = self.board_config.size
        train_episodes = 2000
        epsilon = 1
        max_epsilon = 1
        min_epsilon = 0.03
        decay = 0.003
        replay_memory = deque(maxlen=10_000)
        steps_to_update_target_model = 0

        game_lengths = []
        for episode in range(train_episodes):
            shots = 0
            board = Board(placer.place_ships())
            observation = board.get_repr()
            done = False
            while not done:
                steps_to_update_target_model += 1
                random_number = np.random.rand()
                if random_number <= epsilon:
                    while True:
                        action = (np.random.randint(0, BOARD_SIZE), np.random.randint(0, BOARD_SIZE))
                        if board[action] == Tile.EMPTY:
                            break
                else:
                    predicted = self.model(np.array([observation]), training=False)[0].numpy()
                    for i in range(BOARD_SIZE):
                        for j in range(BOARD_SIZE):
                            if board[i, j] != Tile.EMPTY:
                                predicted[i, j] = -10000
                    action = np.unravel_index(np.argmax(predicted), predicted.shape)
                            
                reward = get_reward(board.shoot(action))
                new_observation = board.get_repr()
                done = (board.count_ship_tiles() == 0)
                replay_memory.append([observation, action, reward, new_observation, done])

                if steps_to_update_target_model % 4 == 0 or done:
                    self.update_weights(replay_memory)

                observation = new_observation
                shots += 1

                if done:
                    print('Total shots : {} after n steps = {}, epsilon={}.'.format(shots, episode, np.round(epsilon*100)/100))
                    game_lengths.append(shots)

                    if steps_to_update_target_model >= 100:
                        print('Copying main network weights to the target network weights')
                        self.target_model.set_weights(self.model.get_weights())
                        steps_to_update_target_model = 0
                    break

            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
        return game_lengths

class QShooter(PredictionShooter):
    def __init__(self, board_config: BoardConfig):
        self.predictor = QPredictor(board_config)
        super().__init__(self.predictor)

    def train(self, placer):
        return self.predictor.train(placer)

    def save(self, filename):
        self.predictor.save(filename)

    def load(self, filename):
        self.predictor.load(filename)
    
def get_reward(shot_result):
    if shot_result == ShotResult.HIT:
        return 1
    elif shot_result == ShotResult.SUNK:
        return 1
    elif shot_result == ShotResult.MISS:
        return -1
    elif shot_result == ShotResult.ILLEGAL:
        return 0

if __name__ == '__main__':
    path = 'plots/loss/standard/q_shooter'
    Path(path).mkdir(parents=True, exist_ok=True)
    shooter = QShooter(STANDARD_CONFIG)
    placer = RandomPlacer(STANDARD_CONFIG)
    lengths = shooter.train(placer)
    plt.plot(lengths)
    plt.title("Q Shooter game lengths during training on standard board")
    plt.xlabel("games played")
    plt.ylabel("game length")
    plt.savefig("plots/loss/standard/q_shooter/game_lengths2.png")
    shooter.save("models/standard/q_shooter")