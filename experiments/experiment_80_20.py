# 80-20 validation experiment
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from src.model import FPINN
from src.caputo_L1 import caputo_L1
from src.loss_functions import equation_loss, data_loss



# ============================
# Reproducibility
# ============================

np.random.seed(123)
tf.random.set_seed(123)



# ============================
# Load experimental data
# ============================

data = pd.read_csv(
    "../data/mobile_web_growth.csv"
)


cuotas = data["Cuota"].values



# 80% of the data

N_train = int(0.8 * len(cuotas))


cuotas_80 = cuotas[:N_train]



# Initial condition

y0 = cuotas_80[0]



# ============================
# Time domain
# ============================

N = len(cuotas_80)


t = np.linspace(
    0,
    1,
    N+1
)


t_tensor = tf.cast(
    t,
    tf.float32
)


h = t[1]-t[0]


h = tf.constant(
    h,
    dtype=tf.float32
)



# Evaluation points

idx_eval = np.arange(
    N+1,
    dtype=np.int32
)


idx_eval = tf.constant(
    idx_eval,
    dtype=tf.int32
)



# ============================
# FPINN model
# ============================

model = FPINN(
    n_hidden=10
)



# ============================
# Optimizer
# ============================

learning_rate = 0.01


optimizer = tf.keras.optimizers.Adam(
    learning_rate
)



# ============================
# Neural network solution
# ============================


def solution(t):

    nn_output = tf.reshape(
        model(t),
        [-1]
    )

    return t * nn_output + y0



# ============================
# Training step
# ============================


lambda_loss = 600



@tf.function
def train_step():


    with tf.GradientTape() as tape:


        Y = solution(
            t_tensor
        )


        alpha = tf.sigmoid(
            model.alpha
        )


        derivative = caputo_L1(
            Y,
            h,
            alpha,
            idx_eval
        )


        loss_edf = equation_loss(
            Y,
            derivative,
            model.a
        )


        loss_data = data_loss(
            Y[:N_train],
            tf.constant(
                cuotas_80,
                dtype=tf.float32
            )
        )


        loss = (
            lambda_loss * loss_edf
            +
            loss_data
        )



    variables = (
        model.trainable_variables
        +
        [
            model.a,
            model.alpha
        ]
    )


    gradients = tape.gradient(
        loss,
        variables
    )


    optimizer.apply_gradients(
        zip(
            gradients,
            variables
        )
    )


    return loss



# ============================
# Training
# ============================


epochs = 100


loss_history = []


for epoch in range(epochs):


    loss = train_step()


    loss_history.append(
        loss.numpy()
    )


    if (epoch+1) % 1000 == 0:

        print(
            f"Epoch {epoch+1}: loss = {loss.numpy():.6f}"
        )



# ============================
# Estimated parameters
# ============================


alpha_est = tf.sigmoid(
    model.alpha
).numpy()


a_est = model.a.numpy()



print("-----------------------")
print("Estimated parameters")
print("a =", a_est)
print("alpha =", alpha_est)



# ============================
# Prediction plot
# ============================


prediction = solution(
    t_tensor
).numpy()



plt.figure(figsize=(10,6))


plt.scatter(
    t[:-1],
    cuotas_80,
    label="Mobile web data"
)


plt.plot(
    t,
    prediction,
    label="FPINN approximation"
)


plt.legend()

plt.grid()

plt.xlabel("t")

plt.ylabel("Percent")


plt.show()
