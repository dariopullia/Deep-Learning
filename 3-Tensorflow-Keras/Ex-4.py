#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
tf.random.set_seed(0)


def main():

    # Load dataset
    x_tr, y_tr, x_val, y_val=np.loadtxt('data.dat', unpack=True)

    # Create baseline linear model
    model = keras.Sequential(
        [
            keras.layers.Dense(1, activation="linear", input_dim=1),
        ]
        )

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                loss='mean_squared_error'
    )
    #Fit model
    history=model.fit(x_tr,y_tr, batch_size=len(x_tr), epochs=500,validation_data=(x_val, y_val))

    #Show improvement
    plt.figure()
    plt.title('Loss evolution baseline model')
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    #Predict

    plt.figure()
    plt.title('Dataset')
    plt.scatter(x_tr, y_tr, label='Training data')
    plt.scatter(x_val, y_val, label='Validation data')
    plt.plot(x_tr, model(x_tr), label='Model', color='red', linewidth=3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    # NN approach

    NN_model=keras.Sequential(
        [
            keras.layers.Dense(10, activation="relu", input_dim=1),
            keras.layers.Dense(10, activation="relu", ),
            keras.layers.Dense(10, activation="relu", ),
            keras.layers.Dense(1, activation="linear",),
        ]
    )



    NN_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                loss='mean_squared_error'
    )
    #Fit model
    history=NN_model.fit(x_tr,y_tr, batch_size=len(x_tr), epochs=500,validation_data=(x_val, y_val))

    #Show improvement
    plt.figure()
    plt.title('Loss evolution NN model')
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    #Predict

    plt.figure()
    plt.title('Dataset')
    plt.scatter(x_tr, y_tr, label='Training data')
    plt.scatter(x_val, y_val, label='Validation data')
    plt.plot(x_tr, NN_model(x_tr), label='Model', color='red', linewidth=3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.show()




if __name__=='__main__':
    main()



