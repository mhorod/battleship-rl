from tf_player import *

def make_tensorflow_dataset(size):
    xs, ys = make_dataset(RandomPlacer(), RandomShooter(), size)
    dataset = tf.data.Dataset.from_tensor_slices((xs, ys))
    return dataset


dataset = make_tensorflow_dataset(10000)
fit_model(make_perceptron_model(), dataset, 100, 'models/perceptron.model')
fit_model(make_dense_model(), dataset, 100, 'models/dense.model')
fit_model(make_cnn_model(), dataset, 100, 'models/cnn.model')