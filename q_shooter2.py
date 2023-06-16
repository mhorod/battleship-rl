from player import *
from random_player import *

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import clone_model

import numpy as np
from stats import *

import matplotlib.pyplot as plt

EPSILON = 1
START_EPSILON = 0.2
FINAL_EPSILON = 0.05
GAMMA = 0

class QShooter(Shooter):
    def __init__(self, board_config):
        self.board_config = board_config
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.observations = []
        self.q_model = make_dense_model()
        self.target_model = make_dense_model()
    
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
        ys = self.q_model(xs, training=False)
        positions = []
        for board, y in zip(boards, ys.numpy()):
            pos = self.select_epsilon_greedy(board, y)
            positions.append(pos)
        return positions
    
    def select_epsilon_greedy(self, board, predictions): 
        if np.random.rand() < EPSILON:
            predictions = np.random.normal(0, 1, predictions.shape) 
        return self.select_best_pos(board, predictions)
    
    def select_best_pos(self, board, predictions):
        BOARD_SIZE = self.board_config.size
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i, j] != Tile.EMPTY:
                    predictions[i, j] = -1000
        tried = 0
        while tried < BOARD_SIZE * BOARD_SIZE:
            pos = np.unravel_index(np.argmax(predictions), predictions.shape)
            if board[pos] == Tile.EMPTY:
                return pos
            predictions[pos] = -1000
            tried += 1

    def make_observations(self, placer, games):
        boards = [Board(placer.place_ships()) for _ in range(games)]
        observations = []
        game_lengths = []
        i = 0
        while len(boards) > 0:
            i += 1
            initial_states = [board.get_repr() for board in boards]
            actions = self.shoot_many_epsilon_greedy(boards)
            shots = [board.shoot(shot) for board, shot in zip(boards, actions)]
            rewards = [get_reward(shot) for shot in shots]
            final_states = [board.get_repr() if shot != ShotResult.ILLEGAL else None for board, shot in zip(boards, shots)]
            observations.extend(zip(initial_states, actions, rewards, final_states))
            diff = len(boards)
            boards = [board for board, shot in zip(boards, shots) if board.get_alive_ship_tiles() > 0]
            diff -= len(boards)
            game_lengths.extend([i] * diff)
        print(game_lengths)

        return observations

    def update_target(self):
        self.target_model = clone_model(self.q_model)
        self.target_model.build((None, BOARD_SIZE, BOARD_SIZE, 4))
        self.target_model.set_weights(self.q_model.get_weights())
    
    def training_step(self, placer):
        while(len(self.observations) < 10000):
            self.observations.extend(self.make_observations(placer, (10000 - len(self.observations) + 100) // 100))
        np.random.shuffle(self.observations)
        self.observations = self.observations[:9900]

        observations = self.observations[:64]

        losses = []
        for i in range(4):
            losses.append(self.gradient_step(observations[16*i : 16*(i+1)]))
        print(tf.reduce_mean(losses))

    def gradient_step(self, observations):
        initial_states = np.array([i for i, _, _, _ in observations])
        rewards = [r for _, _, r, _ in observations]
        final_states = [f for _, _, _, f in observations]
        actions = [a for (_, a, _, _) in observations]
        remaining_rewards = GAMMA * np.max(self.target_model(np.array([np.zeros((BOARD_SIZE, BOARD_SIZE, 4)) if f is None else f for f in final_states]), training=False), axis=(1,2)) \
            + np.array(rewards)
        remaining_rewards = [r if f is None else rr for r, rr, f in zip(rewards, remaining_rewards, final_states)]
        masks = np.zeros((len(observations), BOARD_SIZE, BOARD_SIZE))
        for i in range(len(observations)):
            masks[i, actions[i][0], actions[i][1]] = 1
        with tf.GradientTape() as tape:
            q_values = self.q_model(initial_states, training=True)
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=(1, 2))
            loss = tf.keras.losses.MSE(remaining_rewards, q_action)
        grads = tape.gradient(loss, self.q_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_model.trainable_variables))
        return loss






def board_to_sample(board):
    return board.get_repr(), board.get_ship_repr()

def get_reward(shot_result):
    if shot_result == ShotResult.HIT:
        return 1
    elif shot_result == ShotResult.SUNK:
        return 1
    elif shot_result == ShotResult.MISS:
        return 0
    elif shot_result == ShotResult.ILLEGAL:
        return 0

def make_dense_model():
    # def loss(true_y, pred_y):
    #     result = .0
    #     for i in range(len(true_y)):
    #         result += tf.square(pred_y[i, int(true_y[i, 0, 0]), int(true_y[i,0,1])] - true_y[i, 1, 0])
    #     return result / float(len(true_y))

    dense_model = models.Sequential([
        layers.Flatten(input_shape=(BOARD_SIZE, BOARD_SIZE, len(Tile))),
        layers.Dense(BOARD_SIZE * BOARD_SIZE, activation='sigmoid'),
        layers.Reshape((BOARD_SIZE, BOARD_SIZE))
    ])

    # dense_model.compile(optimizer='adam',loss=loss)

    return dense_model

shooter = QShooter()
placer = RandomPlacer()
for i in range(1000):
    EPSILON = START_EPSILON - (START_EPSILON - FINAL_EPSILON)*i/1000
    # print(EPSILON)
    shooter.training_step(placer)
    if i % 10 == 0:
        shooter.update_target()