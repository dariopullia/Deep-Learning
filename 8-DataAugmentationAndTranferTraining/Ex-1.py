#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import pathlib
tf.random.set_seed(0)

def create_CNN_classifier(input_shape):
    model=tf.keras.models.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=input_shape),
        tf.keras.layers.Conv2D(15, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(15, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(15, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(5)
    ])
    return model
    

def data_augmentation_model(input_shape):
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=input_shape),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )
    return data_augmentation


def data_augmentation_and_classifier(input_shape):
    model = tf.keras.Sequential([
        data_augmentation_model(input_shape),
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(15, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(15, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(15, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(5)
    ])
    return model




def main():

    #import data
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

    train_data = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        image_size=(180, 180),
        batch_size=32,
        seed=0,
    )
    valid_data = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        image_size=(180, 180),
        batch_size=32,
        seed=0
    )

    class_names = train_data.class_names
    print(class_names)

 
    plt.figure(figsize=(10, 10))
    for images, labels in train_data.take(1):

        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")

    train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    valid_data = valid_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    #create model
    model = create_CNN_classifier((180, 180, 3))
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    # history = model.fit(train_data, epochs=8, validation_data=valid_data)
    # model.save('Classifier.h5')
    # np.save('Classifier_history.npy', history.history)

    #load model
    model = tf.keras.models.load_model('Classifier.h5')
    history = np.load('Classifier_history.npy', allow_pickle='TRUE').item()
    
    #plot accuracy and loss
    plt.figure(figsize=(10, 10))
    plt.suptitle("Classifier")
    plt.subplot(2, 1, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')

    #show data augmentation
    data_augmentation = data_augmentation_model((180, 180, 3))
    plt.figure(figsize=(10, 10))
    plt.suptitle("Data Augmentation")
    for images, _ in train_data.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")
    


    #create model with data augmentation
    model = data_augmentation_and_classifier((180, 180, 3))
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    # history = model.fit(train_data, epochs=8, validation_data=valid_data)

    # model.save('Data_Aug_and_Classifier.h5')
    # np.save('Data_Aug_and_Classifier_history.npy', history.history)

    #load model
    model = tf.keras.models.load_model('Data_Aug_and_Classifier.h5')
    history = np.load('Data_Aug_and_Classifier_history.npy', allow_pickle='TRUE').item()
    

    #plot accuracy and loss
    plt.figure(figsize=(10, 10))
    plt.suptitle("Data Augmentation and Classifier")
    plt.subplot(2, 1, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(2, 1, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    
    
    
    plt.show()



    return 0

if __name__ == "__main__":
    main()


