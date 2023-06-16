from player import *
from random_player import *

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import clone_model

import numpy as np
from stats import *

import matplotlib.pyplot as plt

GAMMA = 0.0


class PolicyGradientShooter(Shooter):
    def __init__(self, board_config: BoardConfig):
        self.board_config = board_config
        self.optimizer = tf.optimizers.SGD(0.01)
        self.policy_model = make_dense_model(board_config)
    
    def shoot_many(self, boards):
        xs = np.array([board.get_repr() for board in boards])
        ys = self.q_model(xs)
        positions = []
        for board, y in zip(boards, ys):
            pos = self.select_best_pos(board, y)
            positions.append(pos)
        return positions
    
    def shoot(self, board):
        return self.shoot_many([board])[0]

    def shoot_many_epsilon_greedy(self, boards):
        xs = np.array([board.get_repr() for board in boards])
        ys = self.q_model(xs)
        positions = []
        for board, y in zip(boards, ys.numpy()):
            pos = self.select_epsilon_greedy(board, y)
            positions.append(pos)
        return positions
    
    def select_epsilon_greedy(self, board, predictions): 
        # if np.random.rand() < EPSILON:
        #     predictions = np.abs(np.random.normal(0, 1, predictions.shape))
        return self.select_best_pos(board, predictions)
    
    def select_best_pos(self, board, predictions):
        BOARD_SIZE = self.board_config.size
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[(i, j)] != Tile.EMPTY:
                    predictions[i, j] = 0
        probs = predictions.reshape(BOARD_SIZE * BOARD_SIZE)
        probs /= sum(probs)
        pos = np.unravel_index(np.random.choice(BOARD_SIZE * BOARD_SIZE, p = probs), predictions.shape)
        return pos

    def play_game(self, placer):
        board = Board(placer.place_ships())
        shots = []
        results = []
        board_states = []
        while(board.count_ship_tiles() > 0):
            board_states.append(board.get_repr())
            predictions = self.policy_model(np.array([board.get_repr()])).numpy().reshape(self.board_config.dimensions)
            shot = self.select_epsilon_greedy(board, predictions)
            shots.append(shot)
            results.append(board.shoot(shot))
        return board_states, shots, results

    def train(self, placer):
        game_lengths = []
        for i in range(5000):
            board_states, shots, results = self.play_game(placer)
            rewards = [get_reward(result) for result in results]
            sum_reward = 0
            discnt_rewards = []
            rewards.reverse()
            for r in rewards:
                sum_reward = r + GAMMA*sum_reward
                discnt_rewards.append(sum_reward)
            discnt_rewards.reverse()  

            game_lengths.append(len(shots))
            loss_sum = 0

            for reward, board_state, shot in zip(discnt_rewards, board_states, shots):
                with tf.GradientTape() as tape:
                    predicted = self.policy_model(np.array([board_state]), training=True)[0]
                    shot = np.ravel_multi_index(shot, self.board_config.dimensions)
                    loss = a_loss(predicted, shot, reward)
                    loss_sum += loss.numpy()
                grads = tape.gradient(loss, self.policy_model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.policy_model.trainable_variables))
            print(len(shots), loss_sum)
        return game_lengths
    
    def save(self, filename):
        self.policy_model.save_weights(filename)
    def load(self, filename):
        self.policy_model.load_weights(filename)


def a_loss(prob, action, reward): 
    log_prob = tf.math.log(prob[action])
    loss = -log_prob*reward
    return loss 

def make_dense_model(board_config):
    BOARD_SIZE = board_config.size
    dense_model = models.Sequential([
        layers.Flatten(input_shape=(BOARD_SIZE, BOARD_SIZE, len(Tile))),
        # layers.Dense(BOARD_SIZE * BOARD_SIZE, activation='relu'),
        # layers.Dense(BOARD_SIZE * BOARD_SIZE, activation='relu'),
        layers.Dense(BOARD_SIZE * BOARD_SIZE, activation='softmax'),
    ])

    return dense_model

def get_reward(shot_result):
    if shot_result == ShotResult.HIT:
        return 1
    elif shot_result == ShotResult.SUNK:
        return 1
    elif shot_result == ShotResult.MISS:
        return 0
    elif shot_result == ShotResult.ILLEGAL:
        return 0

# shooter = PolicyGradientShooter()
# shooter.load("simple_model")
# placer = RandomPlacer()
# game_lengths = shooter.train(placer)
# shooter.save("simple_model")
# plt.plot(range(len(game_lengths)), game_lengths)
# plt.show()