#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import time
tf.random.set_seed(0)

BATCH_SIZE = 50
seed = tf.random.normal([25, 100])
cross_entropy=tf.keras.losses.BinaryCrossentropy(from_logits=True)


# Define the generator
def Create_Generator():
    model = keras.Sequential()
    model.add(keras.layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Reshape((7, 7, 256)))
    model.add(keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model


# Define the discriminator
def Create_Discriminator():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1, activation='linear'))

    return model


def discriminator_loss(real_value, fake_value):
    real_loss= cross_entropy(tf.ones_like(real_value), real_value)
    fake_loss=cross_entropy(tf.zeros_like(fake_value),fake_value)
    return fake_loss+real_loss


def generator_loss(fake_value):
    return cross_entropy(tf.ones_like(fake_value),fake_value)


def generate_and_save_image(generator, discriminator, epoch):

    plt.figure(figsize=(5, 5))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(generator(seed)[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.savefig(f'images/{epoch}.png')


def continue_train_GAN(dataset,starting_epoch, epochs, generator, discriminator, generator_optimizer, discriminator_optimizer):
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)
    checkpoint.restore('./training_checkpoints/ckpt-4')                                

    for epoch in range(starting_epoch, epochs):
        start=time.time()
        print(f"Starting Epoch: {epoch}")
        for image_batch in dataset:
            train_step(generator, discriminator, image_batch,generator_optimizer, discriminator_optimizer)
        generate_and_save_image(generator, discriminator, epoch+1)

        if epoch%5==0:
            checkpoint.save(file_prefix = checkpoint_prefix)
        print(f"Epoch {epoch} done in {time.time()-start} s")



def train_GAN(dataset, epochs, generator, discriminator, generator_optimizer, discriminator_optimizer):
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)

    for epoch in range(epochs):
        start=time.time()
        print(f"Starting Epoch: {epoch}")
        for image_batch in dataset:
            train_step(generator, discriminator, image_batch,generator_optimizer, discriminator_optimizer)
        generate_and_save_image(generator, discriminator, epoch+1)

        if epoch%5==0:
            checkpoint.save(file_prefix = checkpoint_prefix)
        print(f"Epoch {epoch} done in {time.time()-start} s")





@tf.function
def train_step(generator, discriminator,image_batch,generator_optimizer, discriminator_optimizer):
    noise = tf.random.normal([BATCH_SIZE, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:


        creations=generator(noise, training=True )
        predictions_fakes=discriminator(creations,training=True)
        predictions_reals=discriminator(image_batch,training=True)


        gen_loss=generator_loss(predictions_fakes)
        disc_loss=discriminator_loss(predictions_reals, predictions_fakes)
    grad_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    grad_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(grad_disc, discriminator.trainable_variables))



def main():

    # Load the MNIST dataset
    (x_train, y_train), (_,_) = keras.datasets.mnist.load_data()
    print(x_train.shape, y_train.shape)

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 175.5 - 1
    
    dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(BATCH_SIZE)

    generator = Create_Generator()
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    discriminator = Create_Discriminator()
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    # train_GAN(dataset=dataset, epochs=20, generator=generator, discriminator=discriminator, generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer)
    continue_train_GAN(dataset=dataset, starting_epoch=20, epochs=50, generator=generator, discriminator=discriminator, generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer)

    return 0




if __name__ == '__main__':
    main()
