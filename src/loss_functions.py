# Loss functions for inverse problem
import tensorflow as tf



def equation_loss(solution, caputo_derivative, a):
    """
    Loss function associated with the fractional differential equation.

    Measures the residual:

        D^alpha Y(t) = a Y(t)

    Parameters
    ----------
    solution : tf.Tensor
        Neural network approximation of the solution.

    caputo_derivative : tf.Tensor
        Approximation of the Caputo fractional derivative.

    a : tf.Tensor
        Growth parameter.

    Returns
    -------
    tf.Tensor
        Mean squared residual of the differential equation.
    """

    residual = caputo_derivative - (a * solution)

    return tf.reduce_mean(
        tf.square(residual)
    )




def data_loss(solution, observations):
    """
    Loss function associated with experimental data.

    Parameters
    ----------
    solution : tf.Tensor
        Neural network prediction.

    observations : tf.Tensor
        Experimental data.

    Returns
    -------
    tf.Tensor
        Mean squared error between prediction and data.
    """

    error = solution - observations

    return tf.reduce_mean(
        tf.square(error)
    )
