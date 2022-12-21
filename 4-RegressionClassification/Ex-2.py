#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

tf.random.set_seed(0)



def main():
    (x_train, y_train), (x_test, y_test) = (tf.keras.datasets.fashion_mnist.load_data())
    x_train, x_test = x_train / 255.0, x_test / 255.0
    #plot example image
    # plt.figure()
    # plt.imshow(x_train[0])
    # plt.colorbar()
    # plt.grid(False)
    # plt.figure()
    # plt.imshow(x_train[1])
    # plt.colorbar()
    # plt.grid(False)
    #plt.show() #uncomment to show images


    #flatten images
    x_train = x_train.reshape(-1, 28*28)
    x_test = x_test.reshape(-1, 28*28)
 

    classes=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


    model=keras.Sequential([
            keras.layers.Dense(128, activation="relu", input_dim=28*28),
            keras.layers.Dense(10, activation="softmax"),
    ])
    

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',        
        metrics=['accuracy']
        )
    model.summary()

    history=model.fit(
        x_train,y_train,
        epochs=5,
        batch_size=128,
        validation_data=(x_test, y_test),
    )

    print(history.history.keys())

    #Show improvement
    plt.figure()
    plt.title('Loss evolution NN model')
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    #Show improvement
    plt.figure()
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    #Predict
    predictions=model.predict(x_test)

    #Detect wrong predictions
    wrong_predictions=[]
    right_predictions=[]
    for i in range(len(predictions)):
        if np.argmax(predictions[i])!=y_test[i]:
            wrong_predictions.append(i)
        else:
            right_predictions.append(i)



    plt.figure(figsize=(20,5))
    plt.suptitle('Right and wrong predictions')
    for i in range(10):
        plt.subplot(2,10,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_test[right_predictions[i]].reshape(28,28), cmap=plt.cm.binary)
        plt.xlabel(classes[np.argmax(predictions[right_predictions[i]])]+' \n '+classes[(y_test[right_predictions[i]])] )
    for i in range(10,20):
        plt.subplot(2,10,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_test[wrong_predictions[i]].reshape(28,28), cmap=plt.cm.binary)
        plt.xlabel(classes[np.argmax(predictions[wrong_predictions[i]])]+' \n '+classes[(y_test[wrong_predictions[i]])] )

    plt.show()





    return 0    





if __name__ == '__main__':  
    main()
