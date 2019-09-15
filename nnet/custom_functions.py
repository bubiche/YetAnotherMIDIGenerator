import tensorflow as tf


# https://arxiv.org/pdf/1710.05941.pdf
def swish(x):
   beta = 1
   return beta * x * tf.keras.backend.hard_sigmoid(x)
