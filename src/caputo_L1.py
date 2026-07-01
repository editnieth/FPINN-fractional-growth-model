# Caputo fractional derivative implementation
import tensorflow as tf


def caputo_L1(Y, h, alpha, idx_eval):
    """
    L1 approximation of the Caputo fractional derivative.

    Parameters
    ----------
    Y : tf.Tensor
        Approximation of the solution.
        
    h : tf.Tensor
        Time step size.
        
    alpha : tf.Tensor
        Fractional order.

    idx_eval : tf.Tensor
        Points where the fractional derivative is evaluated.

    Returns
    -------
    tf.Tensor
        Approximation of the Caputo fractional derivative.
    """

    # Normalization constant
    C_gamma = 1.0 / (
        h**alpha * tf.exp(tf.math.lgamma(2.0 - alpha))
    )


    def compute_for_n(n):

        # Previous points
        j_vals = tf.range(
            0, 
            n, 
            dtype=tf.int32
        )


        j_vals_f = tf.cast(
            j_vals, 
            tf.float32
        )


        n_f = tf.cast(
            n, 
            tf.float32
        )


        # Finite differences
        diff = (
            tf.gather(Y, j_vals + 1)
            -
            tf.gather(Y, j_vals)
        )


        # L1 coefficients
        delta = (
            tf.pow(n_f - j_vals_f, 1.0 - alpha)
            -
            tf.pow(n_f - j_vals_f - 1.0, 1.0 - alpha)
        )


        return C_gamma * tf.reduce_sum(diff * delta)



    derivative = tf.map_fn(
        lambda n: compute_for_n(n),
        idx_eval,
        fn_output_signature=tf.float32
    )


    return derivative
