#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from tensorflow import keras
tf.random.set_seed(0)




def main():
    n_input, n_hidden_1, n_hidden_2, n_output=1,5,2,1

    #---- Defined to match the results from previous exercise
    weights=[ 
        tf.Variable(tf.random.normal([n_input, n_hidden_1])),
        tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2])),
        tf.Variable(tf.random.normal([n_hidden_2, n_output])),
    ]

    biases=[
        tf.Variable(tf.random.normal([n_hidden_1])),
        tf.Variable(tf.random.normal([n_hidden_2])),
        tf.Variable(tf.random.normal([n_output])),
    ]
    #--------

    model = keras.Sequential(
        [
            keras.layers.Dense(n_hidden_1, activation="sigmoid",input_dim=n_input),
            keras.layers.Dense(n_hidden_2, activation="sigmoid"),
            keras.layers.Dense(n_output, activation="linear", ),
        ]
        )
    start = tf.constant([-1 for x in range( n_input )], dtype = tf.float32)
    end = tf.constant([1 for x in range( n_input )], dtype = tf.float32)
    x=tf.linspace(start, end, 10)
    model.set_weights([  weights[0],biases[0],
                        weights[1],biases[1],
                        weights[2],biases[2],
                        ])



    
    print("Print the Prevision:")
    print(model(x)) 

    print("Print the Summary:")
    model.summary()

    print("Print the Weights:")
    print(model.weights)


if __name__=='__main__':
    main()
