from tf_player import *

def make_tensorflow_dataset(placer, shooter, size):
    xs, ys = make_dataset(placer, shooter, size)
    dataset = tf.data.Dataset.from_tensor_slices((xs, ys))
    return dataset

def fit_default_config_models():
    board_config = DEFAULT_BOARD_CONFIG
    placer = RandomPlacer(board_config)
    shooter = RandomShooter()
    dataset = make_tensorflow_dataset(placer, shooter, 10000)
    #fit_model(make_perceptron_model(board_config), dataset, 100, 'models/default/perceptron.model')
    #fit_model(make_dense_model(board_config), dataset, 100, 'models/default/dense.model')
    fit_model(make_cnn_model(board_config), dataset, 100, 'models/default/cnn.model')


def fit_tiny_config_models():
    board_config = TINY_BOARD_CONFIG
    placer = RandomPlacer(board_config)
    shooter = RandomShooter()
    dataset = make_tensorflow_dataset(placer, shooter, 10000)
    fit_model(make_perceptron_model(board_config), dataset, 100, 'models/tiny/perceptron.model')
    fit_model(make_dense_model(board_config), dataset, 100, 'models/tiny/dense.model')
    fit_model(make_cnn_model(board_config), dataset, 100, 'models/tiny/cnn.model')


fit_default_config_models()
#fit_tiny_config_models()