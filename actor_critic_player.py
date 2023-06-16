from pathlib import Path

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

from dataset import *
from player import *
from neural_network_player import NeuralNetworkShooter

class ActorCriticModel:
    def __init__(self, actor, critic):
        self.actor = actor
        self.actor_optimizer = tf.optimizers.Adam(5e-3)

        self.critic = critic
        self.critic_optimizer = tf.optimizers.Adam(5e-3)

    def act(self, board):
        board_repr = board.get_repr()
        probabilities = self.actor(np.array([board_repr]))[0]

        choices = []
        weights = []
        for pos in board_positions(board.config):
            if board[pos] == Tile.EMPTY and probabilities[pos] > 0:
                choices.append(pos)
                weights.append(probabilities[pos])
        weights = np.array(weights)
        weights /= np.sum(weights)
        return choices[np.random.choice(len(choices), p=weights)]

    def learn(self, states, actions, discounted_rewards):
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            probabilities = self.actor(states, training=True)
            values = self.critic(states, training=True)
            temporal_difference = tf.math.subtract(discounted_rewards, values)

            actor_loss = self.actor_loss(probabilities, actions, temporal_difference)
            critic_loss = self.critic_loss(values, discounted_rewards)

        actor_gradients = tape1.gradient(actor_loss, self.actor.trainable_variables)
        critic_gradients = tape2.gradient(critic_loss, self.critic.trainable_variables)

        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        return actor_loss, critic_loss


    def actor_loss(self, probabilities, actions, td):
        action_probabilities = []
        for prob, action in zip(probabilities, actions):
            r, c = action
            action_probabilities.append(prob[r][c])


        action_probabilities = tf.convert_to_tensor(action_probabilities)
        log_probabilities = tf.math.log(action_probabilities)

        policy_losses = []
        entropy_losses = []

        policy_losses = tf.math.multiply(log_probabilities, td)
        entropy_losses = tf.math.negative(tf.math.multiply(action_probabilities, log_probabilities))

        p_loss = tf.reduce_mean(policy_losses)
        e_loss = tf.reduce_mean(entropy_losses)

        loss = -p_loss - 0.001 * e_loss

        return loss


    def critic_loss(self, values, discounted_rewards):
        return tf.reduce_mean(tf.square(discounted_rewards - values))


class ActorCriticShooter(NeuralNetworkShooter):
    def __init__(self, model):
        super().__init__(model.actor)


class HybridActorCriticModel(ActorCriticModel):
    def __init__(self, actor, critic, good_player, cheat_probability):
        super().__init__(actor, critic)
        self.good_player = good_player
        self.cheat_probability = cheat_probability
        
    def act(self, board):
        if random.random() < self.cheat_probability:
            return self.good_player.shoot(board)
        else:
            return super().act(board)



def make_actor_critic_model(board_config):
    BOARD_SIZE = board_config.size
    actor = models.Sequential([
        layers.Flatten(input_shape=(BOARD_SIZE, BOARD_SIZE, len(Tile))),
        layers.Dense(32),
        layers.Dense(BOARD_SIZE * BOARD_SIZE, activation='softmax'),
        layers.Reshape((BOARD_SIZE, BOARD_SIZE))
    ])

    critic = models.Sequential([
        layers.Flatten(input_shape=(BOARD_SIZE, BOARD_SIZE, len(Tile))),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='relu'),
    ])

    return ActorCriticModel(actor, critic)

def make_hybrid_actor_critic_model(board_config, good_player, cheat_probability):
    BOARD_SIZE = board_config.size
    hidden = int(BOARD_SIZE * BOARD_SIZE * 1.25)
    actor = models.Sequential([
        layers.Flatten(input_shape=(BOARD_SIZE, BOARD_SIZE, len(Tile))),
        layers.Dense(hidden),
        layers.Dense(BOARD_SIZE * BOARD_SIZE, activation='softmax'),
        layers.Reshape((BOARD_SIZE, BOARD_SIZE))
    ])

    critic = models.Sequential([
        layers.Flatten(input_shape=(BOARD_SIZE, BOARD_SIZE, len(Tile))),
        layers.Dense(hidden, activation='relu'),
        layers.Dense(1, activation='relu'),
    ])

    return HybridActorCriticModel(actor, critic, good_player, cheat_probability)


def preprocess_history(states, actions, rewards, gamma):
    discounted_rewards = []
    cumulative_reward = 0
    for reward in rewards[::-1]:
        cumulative_reward = reward + gamma * cumulative_reward
        discounted_rewards.append(cumulative_reward)
    discounted_rewards.reverse()
    states = np.array(states)
    actions = np.array(actions)
    discounted_rewards = np.array(discounted_rewards)
    return states, actions, discounted_rewards

def play_one_game(model: ActorCriticModel, board):
    moves = 0
    initial_ship_tiles = board.count_ship_tiles()
    spare_tiles = board.config.size * board.config.size - initial_ship_tiles

    states = []
    actions = []
    rewards = []

    while board.count_ship_tiles() > 0:
        moves += 1
        preshot_board = board.clone()
        shot = model.act(board)
        board.shoot(shot)
        reward = 0
        if board.count_ship_tiles() == 0:
            reward = 1 - (moves - initial_ship_tiles) / spare_tiles

        states.append(preshot_board.get_repr())
        actions.append(shot)
        rewards.append(reward)
        
    states, actions, discounted_rewards = preprocess_history(states, actions, rewards, 0.9)
    model.learn(states, actions, discounted_rewards)
    return moves, rewards[-1]

class ActorCriticShooter(NeuralNetworkShooter):
    def __init__(self, model):
        super().__init__(model.actor)

def save_actor_critic_model(model, path):
    Path(path).mkdir(parents=True, exist_ok=True)
    model.actor.save(f'{path}/actor.model')
    model.critic.save(f'{path}/critic.model')

def load_actor_critic_shooter(path):
    actor = tf.keras.models.load_model(f'{path}/actor.model')
    critic = tf.keras.models.load_model(f'{path}/critic.model')
    return ActorCriticShooter(ActorCriticModel(actor, critic))