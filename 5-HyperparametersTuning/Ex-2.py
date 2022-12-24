#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import hyperopt as hp
from tensorflow import keras
tf.random.set_seed(0)


def create_model(x_train, y_train, parameters):
    model=keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(parameters['layer_size'], activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=parameters['learning_rate']),
        loss='sparse_categorical_crossentropy',        
        metrics=['accuracy']
        )
    model.fit(x_train, y_train, epochs=5)
    return model


def hypertest_model(parameters, x_train, y_train, x_test, y_test):
    model=create_model(x_train, y_train, parameters)
    loss, accuracy=model.evaluate(x_test, y_test)
    return {'loss': -accuracy, 'status': hp.STATUS_OK}



def main():

    (x_train, y_train), (x_test, y_test) = (tf.keras.datasets.mnist.load_data())
    x_train, x_test = x_train / 255.0, x_test / 255.0


    print(x_train.shape, x_test.shape)

    model=keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(10, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])

    model.summary()
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',        
        metrics=['accuracy']
        )
    
    #model.fit(x_train, y_train, epochs=5)

    parameters={
        'layer_size': 10,
        'learning_rate':1.0,
    }

    space={
        'layer_size': hp.hp.choice('layer_size', [10, 20, 30, 40, 50]),
        'learning_rate': hp.hp.uniform('learning_rate', 0.001, 0.1),
    }

    trials=hp.Trials()

    best=hp.fmin(
        fn=lambda x: hypertest_model(x, x_train, y_train, x_test, y_test),
        space=space,
        algo=hp.tpe.suggest,
        max_evals=5,
        trials=trials,
    )
    print(hp.space_eval(space, best))

    plt.figure(figsize=(10, 10))
    plt.suptitle('Hyperparameters tuning')
    plt.subplot(1, 3, 1)
    plt.title('Accuracy vs Trial ID')
    plt.scatter([t['tid'] for t in trials.trials], [-t['result']['loss'] for t in trials.trials], label='Loss')
    plt.xlabel('Trial ID')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.title('Accuracy vs Layer Size')
    plt.scatter([t['misc']['vals']['layer_size'] for t in trials.trials], [-t['result']['loss'] for t in trials.trials], label='Layer Size')
    plt.xlabel('Layer Size')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.title('Accuracy vs Learning Rate')
    plt.scatter([t['misc']['vals']['learning_rate'] for t in trials.trials], [-t['result']['loss'] for t in trials.trials], label='Learning Rate')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()





    return 0    





if __name__ == '__main__':  
    main()
