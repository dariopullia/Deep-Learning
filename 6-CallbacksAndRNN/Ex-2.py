#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import pandas as pd
from tensorflow import keras
tf.random.set_seed(0)


def main():
    test_data=np.load('test_data.npy')
    test_label=np.load('test_label.npy')
    training_data=np.load('training_data.npy')
    training_label=np.load('training_label.npy')


    model=keras.Sequential()
    model.add(keras.layers.LSTM(30, input_shape=(training_data.shape[1], training_data.shape[2])))
    model.add(keras.layers.Dense(1,))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',        
        metrics=['accuracy']
        )

    model.fit(training_data, training_label, epochs=25, batch_size=32, verbose=1)

    plt.figure(figsize=(10,8))

    plt.subplot(3,1,1)
    plt.plot(range(training_label.shape[0]),training_label, label='Training data')
    plt.plot(range(training_label.shape[0], training_label.shape[0]+test_label.shape[0]) ,test_label, label='Test data')
    plt.title('Training data')
    plt.xlabel('Time')
    plt.ylabel('Data')
    plt.legend()

    plt.subplot(3,2,3)
    plt.plot(range(test_label.shape[0]) ,test_label, label='true')
    plt.plot(range(test_label.shape[0]) ,model.predict(test_data), label='pred')
    plt.title('Test prediction')
    plt.xlabel('Time')
    plt.ylabel('Data')
    plt.legend()

    plt.subplot(3,2,4)
    plt.plot(range(100) ,test_label[0:100], label='true')
    plt.plot(range(100) ,model.predict(test_data)[0:100], label='pred')
    plt.title('First 100 days')
    plt.xlabel('Time')
    plt.ylabel('Data')
    plt.legend()

    plt.subplot(3,2,5)
    plt.plot(range(test_label.shape[0]) ,test_label-model.predict(test_data), label='true')
    plt.title('True - prediction')
    plt.xlabel('Time')
    plt.ylabel('Difference')
    plt.legend()

    plt.subplot(3,2,6)
    plt.scatter(model.predict(test_data), test_label, s=2, color='black')
    plt.title('Scatter plot')
    plt.xlabel('Prediction')
    plt.ylabel('True')
    plt.legend()

    plt.subplots_adjust(hspace = 0.5, wspace=0.3)
    plt.show()


    return 0    





if __name__ == '__main__':  
    main()
