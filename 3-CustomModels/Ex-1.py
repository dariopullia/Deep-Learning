#!/usr/bin/env python
import numpy as np
import tensorflow as tf
import numpy as np
tf.random.set_seed(0)


def multilayer_perceptron(x, weights, biases):
    if (len(weights)!=len(biases)): return

    res=x

    for i in range(len(weights)-1):
        res=tf.math.sigmoid(tf.add(tf.matmul(res,weights[i]), biases[i]))

    res=tf.matmul(res,weights[len(weights)-1])+biases[len(weights)-1]

    return res







def main():

    n_input, n_hidden_1, n_hidden_2, n_output=1,5,2,1
      
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

    start = tf.constant([-1 for x in range( n_input )], dtype = tf.float32)
    end = tf.constant([1 for x in range( n_input )], dtype = tf.float32)
    x=tf.linspace(start, end, 10)
    print(multilayer_perceptron(x, weights, biases))


    


if __name__=='__main__':
    main()



