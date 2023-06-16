from pathlib import Path

from configs import *
from neural_network_player import *
from random_player import *
from naive_bayes_player import *
from actor_critic_player import *
from hunter_player import *

import matplotlib.pyplot as plt


BOARD_CONFIGS = [
    (TINY_CONFIG, 'tiny'),
    (SMALL_CONFIG, 'small'),
    (STANDARD_CONFIG, 'standard'),
]

def make_tensorflow_dataset(placer, shooter, size):
    xs, ys = make_dataset(placer, shooter, size)
    dataset = tf.data.Dataset.from_tensor_slices((xs, ys))
    return dataset

def fit_neural_network_models_on_config(board_config, config_name):
    model_path = f'models/{config_name}/neural_network'
    plot_path = f'plots/loss/{config_name}/neural_network'

    Path(model_path).mkdir(parents=True, exist_ok=True)
    Path(plot_path).mkdir(parents=True, exist_ok=True)

    placer = RandomPlacer(board_config)
    shooter = RandomShooter()
    dataset = make_tensorflow_dataset(placer, shooter, 10000)
    models = [
        (make_perceptron_model(board_config), 'perceptron'),
        (make_dense_model(board_config), 'dense'),
        (make_cnn_model(board_config), 'cnn'),
    ]
    
    for model, name in models:
        history, _ = fit_model(model, dataset, 50, f'{model_path}/{name}.model')
        plot_history(history, name, config_name, plot_path)

def plot_history(history, model_name, config_name, plot_path):
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='loss')
    ax.plot(history.history['val_loss'], label='val_loss')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Loss for {model_name} model on {config_name} board')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(f'{plot_path}/{model_name}.png')


def fit_neural_network_models():
    print('Fitting neural network models')
    for board_config, config_name in BOARD_CONFIGS:
        print(f'Fitting neural network models for {config_name} board')
        fit_neural_network_models_on_config(board_config, config_name)

def fit_naive_bayes_models():
    print('Fitting naive bayes models')
    for board_config, config_name in BOARD_CONFIGS:
        print(f'Fitting naive bayes model for {config_name} board')
        nb = make_naive_bayes(board_config, RandomPlacer(board_config), RandomShooter(), 10000)
        path = f'models/{config_name}/naive_bayes.model'
        save_naive_bayes_shooter(nb, path)


def fit_actor_critic_model(board_config, config_name):
    model_path = f'models/{config_name}/actor_critic'
    plot_path = f'plots/loss/{config_name}/actor_critic'

    Path(model_path).mkdir(parents=True, exist_ok=True)
    Path(plot_path).mkdir(parents=True, exist_ok=True)

    placer = RandomPlacer(board_config)
    model = make_actor_critic_model(board_config)

    lengths = []
    rewards = []

    for i in range(1000):
        print(f'Game {i}')
        board = Board(placer.place_ships())
        length, reward = play_one_game(model, board)

        lengths.append(length)
        rewards.append(reward)

    save_actor_critic_model(model, model_path)

    f1, ax1 = plt.subplots()
    ax1.plot(lengths)
    ax1.set_xlabel('Game')
    ax1.set_ylabel('Length')
    ax1.set_title(f'Lengths for actor-critic model on {config_name} board')
    f1.tight_layout()
    f1.savefig(f'{plot_path}/lengths.png')

    f2, ax2 = plt.subplots()
    ax2.plot(rewards)
    ax2.set_xlabel('Game')
    ax2.set_ylabel('Reward')
    ax2.set_title(f'Rewards for actor-critic model on {config_name} board')
    f2.tight_layout()
    f2.savefig(f'{plot_path}/rewards.png')

def fit_actor_critic_models():
    print('Fitting actor-critic models')
    for board_config, config_name in BOARD_CONFIGS:
        print(f'Fitting actor-critic model for {config_name} board')
        fit_actor_critic_model(board_config, config_name)

def fit_actor_critic_model_on_one_board(board_config, config_name):
    model_path = f'models/{config_name}/actor_critic_one_board'
    plot_path = f'plots/loss/{config_name}/actor_critic_one_board'

    Path(model_path).mkdir(parents=True, exist_ok=True)
    Path(plot_path).mkdir(parents=True, exist_ok=True)

    placer = RandomPlacer(board_config)
    model = make_actor_critic_model(board_config)

    lengths = []
    rewards = []

    board = Board(placer.place_ships())
    for i in range(100):
        print(f'Game {i}')
        length, reward = play_one_game(model, board.clone())

        lengths.append(length)
        rewards.append(reward)

    save_actor_critic_model(model, model_path)

    f1, ax1 = plt.subplots()
    ax1.plot(lengths)
    ax1.set_xlabel('Game')
    ax1.set_ylabel('Length')
    ax1.set_title(f'Lengths for actor-critic model on one {config_name} board')
    f1.tight_layout()
    f1.savefig(f'{plot_path}/lengths.png')

    f2, ax2 = plt.subplots()
    ax2.plot(rewards)
    ax2.set_xlabel('Game')
    ax2.set_ylabel('Reward')
    ax2.set_title(f'Rewards for actor-critic model on one {config_name} board')
    f2.tight_layout()
    f2.savefig(f'{plot_path}/rewards.png')

def fit_actor_critic_models_on_one_board():
    print('Fitting actor-critic models')
    for board_config, config_name in BOARD_CONFIGS:
        print(f'Fitting actor-critic model for {config_name} board')
        fit_actor_critic_model_on_one_board(board_config, config_name)

def fit_hybrid_actor_critic_model(board_config, config_name):
    model_path = f'models/{config_name}/hybrid_actor_critic'
    plot_path = f'plots/loss/{config_name}/hybrid_actor_critic'

    Path(model_path).mkdir(parents=True, exist_ok=True)
    Path(plot_path).mkdir(parents=True, exist_ok=True)

    model = make_hybrid_actor_critic_model(board_config, EvenHunterShooter(), 0.25)

    lengths = []
    rewards = []
    placer = RandomPlacer(board_config)
    for i in range(1000):
        print(f'Game {i}')
        board = Board(placer.place_ships())
        length, reward = play_one_game(model, board)

        lengths.append(length)
        rewards.append(reward)

    save_actor_critic_model(model, model_path)

    f1, ax1 = plt.subplots()
    ax1.plot(lengths)
    ax1.set_xlabel('Game')
    ax1.set_ylabel('Length')
    ax1.set_title(f'Lengths for hybrid actor-critic model on {config_name} board')
    f1.tight_layout()
    f1.savefig(f'{plot_path}/lengths.png')

    f2, ax2 = plt.subplots()
    ax2.plot(rewards)
    ax2.set_xlabel('Game')
    ax2.set_ylabel('Reward')
    ax2.set_title(f'Rewards for hybrid actor-critic model on {config_name} board')
    f2.tight_layout()
    f2.savefig(f'{plot_path}/rewards.png')

def fit_hybrid_actor_critic_models():
    print('Fitting hybrid actor-critic models')
    for board_config, config_name in BOARD_CONFIGS:
        print(f'Fitting hybrid actor-critic model for {config_name} board')
        fit_hybrid_actor_critic_model(board_config, config_name)






#fit_neural_network_models()
#fit_naive_bayes_models()
#fit_actor_critic_models()
#fit_hybrid_actor_critic_models()
fit_actor_critic_models_on_one_board()