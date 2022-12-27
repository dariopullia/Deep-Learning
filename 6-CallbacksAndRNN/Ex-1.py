#!/usr/bin/env python
import seaborn as sns
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import pandas as pd
import hyperopt as hp
from tensorflow import keras
tf.random.set_seed(0)

def create_model(x_train, y_train, parameters):
    model=keras.Sequential()
    model.add(keras.layers.Dense(parameters['layer_size'], activation='relu', input_shape=(4,)))
    for i in range(parameters['layer_number']-1):
        model.add(keras.layers.Dense(parameters['layer_size'], activation='relu'))
    model.add(keras.layers.Dense(3, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=parameters['learning_rate']),
        loss='categorical_crossentropy',        
        metrics=['accuracy']
        )

    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, monitor='val_loss'),
        keras.callbacks.TensorBoard(log_dir='./EX1_logs')
    ]
    model.fit(x_train, y_train, epochs=200, batch_size=32, validation_split=0.4, callbacks=callbacks)

    return model


def hypertest_model(parameters, x_train, y_train, x_test, y_test):
    model=create_model(x_train, y_train, parameters)
    loss, accuracy=model.evaluate(x_test, y_test)
    return {'loss': -accuracy, 'status': hp.STATUS_OK}



def main():

    data=pd.DataFrame(sklearn.datasets.load_iris().data, columns=sklearn.datasets.load_iris().feature_names)
    data['label']=sklearn.datasets.load_iris().target_names[sklearn.datasets.load_iris().target]
    
    sns.pairplot(data, hue='label')

    print(data)
    data = pd.get_dummies(data, columns=["label"], prefix="label")
    print(data)
    train = data.sample(frac=0.8, random_state=1)

    test = data.drop(train.index)

    x_train = train[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
    y_train = train[['label_setosa', 'label_versicolor', 'label_virginica']]
    x_test = test[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
    y_test = test[['label_setosa', 'label_versicolor', 'label_virginica']]
    

    model=keras.Sequential()
    model.add(keras.layers.Dense(64, activation='relu', input_shape=(4,)))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(10, activation='relu'))
    model.add(keras.layers.Dense(3, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.1),
        loss='categorical_crossentropy',        
        metrics=['accuracy']
        )
    # Dangerous, too many parameters
    # history = model.fit(x_train, y_train, epochs=200, batch_size=32, validation_split=0.4)

    # Add callbacks to avoid overfitting and to monitor the training process
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, monitor='val_loss'),
        keras.callbacks.TensorBoard(log_dir='./EX1_logs')
    ]
    history = model.fit(x_train, y_train, epochs=200, batch_size=32, validation_split=0.4, callbacks=callbacks)

    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Now we search for the best parameters

    parameters={
        'layer_size': 10,
        'learning_rate':1.0,
        'layer_number': 3,

    }

    space={
        'layer_size': hp.hp.choice('layer_size', [10, 20, 30, 40, 50]),
        'learning_rate': hp.hp.uniform('learning_rate', 0.001, 0.1),
        'layer_number': hp.hp.choice('layer_number', [2, 3, 4, 5, 6])
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
