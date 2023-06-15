import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt

from dataset import *
from player import *
from random_player import *
from tf_player import *


class ActorCriticModel:
    def __init__(self, actor, critic, discount_factor=0.9):
        self.actor = actor
        self.actor_optimizer = tf.optimizers.Adam(5e-4)

        self.critic = critic
        self.critic_optimizer = tf.optimizers.Adam(5e-4)

        self.discount_factor = discount_factor

    def act(self, board):
        board_repr = board.get_repr()
        probabilities = self.actor(np.array([board_repr]))[0]

        choices = []
        weights = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r, c] == Tile.EMPTY and probabilities[r, c] > 0:
                    choices.append((r, c))
                    weights.append(probabilities[r, c])
        weights = np.array(weights)
        weights /= np.sum(weights)
        return choices[np.random.choice(len(choices), p=weights)]

    def learn(self, board, shot, reward, result_board, done):
        board_repr = np.array([board.get_repr()])
        result_board_repr = np.array([result_board.get_repr()])

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            probabilities = self.actor(board_repr, training=True)[0]
            value = self.critic(board_repr, training=True)[0]
            result_value = self.critic(result_board_repr, training=True)[0]
            temporal_difference = reward + self.discount_factor * \
                result_value * (1 - int(done)) - value

            actor_loss = self.actor_loss(
                probabilities, shot, temporal_difference)
            critic_loss = temporal_difference ** 2

        grads1 = tape1.gradient(actor_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(critic_loss, self.critic.trainable_variables)

        self.actor_optimizer.apply_gradients(
            zip(grads1, self.actor.trainable_variables))

        self.critic_optimizer.apply_gradients(
            zip(grads2, self.critic.trainable_variables))

    def actor_loss(self, probabilities, action, td):
        log_prob = tf.math.log(probabilities[action[0], action[1]])
        return -log_prob * td


class ActorCriticShooter(NeuralNetworkShooter):
    def __init__(self, model):
        super().__init__(model.actor)


def make_actor_critic_model():
    actor = models.Sequential([
        layers.Flatten(input_shape=(BOARD_SIZE, BOARD_SIZE, len(Tile))),
        #layers.Dense(128, activation='relu'),
        layers.Dense(BOARD_SIZE * BOARD_SIZE, activation='softmax'),
        layers.Reshape((BOARD_SIZE, BOARD_SIZE))
    ])

    critic = models.Sequential([
        layers.Flatten(input_shape=(BOARD_SIZE, BOARD_SIZE, len(Tile))),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='linear'),
    ])

    return ActorCriticModel(actor, critic)


def play_one_game(model: ActorCriticModel, board):
    moves = 0
    initial_ship_tiles = board.count_ship_tiles()
    game_history = []
    while board.count_ship_tiles() > 0:
        moves += 1
        preshot_board = board.clone()
        shot = model.act(board)
        result = board.shoot(shot)
        reward = 0
        done = False
        if board.count_ship_tiles() == 0:
            reward = 1 - (moves - initial_ship_tiles) / (BOARD_SIZE * BOARD_SIZE - initial_ship_tiles)
            done = True
        game_history.append((preshot_board, shot, board.clone(), reward, done))
        
    for _ in range(3):
        for preshot_board, shot, board, reward, done in game_history:
            model.learn(preshot_board, shot, reward, board, done)

        

    print("moves: ", moves)
    print("reward: ", reward)
    return moves


model = make_actor_critic_model()
placer = RandomPlacer()
boards = [Board(placer.place_ships()) for _ in range(100)]
lengths = []
for i in range(10000):
    print(i)
    board = random.choice(boards)
    try:
        lengths.append(play_one_game(model, board.clone()))
    except:
        pass

print(np.mean(lengths))
plt.plot(lengths)
plt.show()
plt.hist(lengths)
plt.show()
