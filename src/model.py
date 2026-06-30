# Neural network architecture for FPINN
import tensorflow as tf


class FPINN(tf.keras.Model):
    """
    Feedforward neural network used for the fractional inverse problem.

    Architecture:
    Input layer: 1 neuron
    Hidden layer: n_hidden neurons with tanh activation
    Output layer: 1 neuron

    Trainable parameters:
    a      -> growth parameter
    alpha  -> fractional order parameter
    """

    def __init__(self, n_hidden=10):
        super().__init__()

        self.dense1 = tf.keras.layers.Dense(
            n_hidden,
            activation='tanh',
            kernel_initializer=tf.keras.initializers.GlorotNormal()
        )

        self.out = tf.keras.layers.Dense(
            1,
            kernel_initializer=tf.keras.initializers.GlorotNormal()
        )

        # Unknown parameters of the inverse problem
        self.a = tf.Variable(
            tf.random.normal(shape=[], mean=0.0, stddev=1.0),
            name="a",
            dtype=tf.float32
        )

        self.alpha = tf.Variable(
            tf.random.normal(shape=[], mean=0.0, stddev=1.0),
            name="alpha",
            dtype=tf.float32
        )


    def call(self, x):

        x = tf.expand_dims(x, axis=-1)

        layer_1  = self.dense1(x)

        output = self.out(layer_1)

        return output
