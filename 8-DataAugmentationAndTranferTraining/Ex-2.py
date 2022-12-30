#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import pathlib
import os
tf.random.set_seed(0)



def data_augmentation_model():
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
        ]
    )
    return data_augmentation


def main():

    dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
    path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=dataset_url, extract=True)
    path = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
    train_dir = pathlib.Path(path) / 'train'
    validation_dir = pathlib.Path(path) / 'validation'

    train_data = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(160, 160),
        batch_size=32,
        seed=0,
    )
    valid_data = tf.keras.utils.image_dataset_from_directory(
        validation_dir,
        image_size=(160, 160),
        batch_size=32,
        seed=0
    )

    class_names = train_data.class_names
    print(class_names)

    train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    valid_data = valid_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


    data_augmentation_input =tf.keras.Input(shape=(160, 160, 3))
    data_augmentation = data_augmentation_model()(data_augmentation_input)

    data_augmentation_mod= tf.keras.Model(inputs=data_augmentation_input, outputs=data_augmentation)

    plt.figure(figsize=(10, 10))
    plt.suptitle("Data Augmentation")
    for images, _ in train_data.take(1):
        for i in range(9):
            augmented_images = data_augmentation_mod(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")
    

    base_model= tf.keras.applications.MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
    base_model.trainable = False


    input = tf.keras.Input(shape=(160, 160, 3))
    model = data_augmentation_model()(input)
    model=tf.keras.applications.mobilenet_v2.preprocess_input(model)
    model=base_model(model, training=False)
    model=tf.keras.layers.GlobalAveragePooling2D()(model)
    model=tf.keras.layers.Dropout(0.2)(model)
    model=tf.keras.layers.Dense(1)(model)
    model=tf.keras.Model(inputs=input, outputs=model)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

    # history = model.fit(train_data, epochs=8, validation_data=valid_data)

    # model.save('Classifier_tranfered.h5')
    # np.save('Classifier_tranfered_history.npy', history.history)

    # Load the model
    model = tf.keras.models.load_model('Classifier_tranfered.h5')
    history = np.load('Classifier_tranfered_history.npy', allow_pickle='TRUE').item()




    plt.figure(figsize=(10, 10))
    plt.suptitle("Training and Validation Accuracy")
    plt.subplot(2, 1, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    

    plt.show()



    return 0

if __name__ == "__main__":
    main()


