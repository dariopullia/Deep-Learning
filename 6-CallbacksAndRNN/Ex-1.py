#!/usr/bin/env python
import seaborn as sns

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import pandas as pd
from tensorflow import keras
tf.random.set_seed(0)


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

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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
    plt.legend()
    plt.show()




    return 0    





if __name__ == '__main__':  
    main()
