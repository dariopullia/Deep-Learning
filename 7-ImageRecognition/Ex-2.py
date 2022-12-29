#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tensorflow import keras
tf.random.set_seed(0)


class Class_and_Boxes_model(tf.keras.Model()):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)



def plot_image(train_img, train_box, train_label, names):
    plt.figure()
    n_fig = len(train_img)
    for i in range(n_fig):
        plt.subplot(3,int(n_fig/3),i+1)
        plt.imshow(train_img[i])
        plt.gca().add_patch(Rectangle((train_box[i][0],train_box[i][1]),train_box[i][2]-train_box[i][0],train_box[i][3]-train_box[i][1],fill=False,color='red',lw=3))
        plt.title(names[np.argmax(train_label[i])])
        plt.xticks([])
        plt.yticks([])




def create_feature_extractor(input):
    x = tf.keras.layers.Conv2D(32, 3, activation='relu',)(input)
    x = tf.keras.layers.AveragePooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
    x = tf.keras.layers.AveragePooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(16, 3, activation='relu')(x)
    x = tf.keras.layers.AveragePooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)

    return x


def create_classifier(feature_extractor):
    classifier=keras.layers.Dense(10, activation='softmax', name="classifier")(feature_extractor)
    return classifier

def create_regressor(feature_extractor):
    regressor=keras.layers.Dense(4, name='box')(feature_extractor)
    return regressor


def create_model(input_shape):
    input=tf.keras.layers.Input(shape=input_shape)
    feature_extractor = create_feature_extractor(input)
    classifier = create_classifier(feature_extractor)
    regressor = create_regressor(feature_extractor)
    model = keras.models.Model(inputs=input, outputs=[classifier, regressor])
    return model


def InteresectionOverUnion(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection
    return intersection / union



def main():

    train_box = np.load('data/training_boxes.npy')
    train_label = np.load('data/training_labels.npy')
    train_img = np.load('data/training_images.npy')
    valid_box = np.load('data/validation_boxes.npy')
    valid_label = np.load('data/validation_labels.npy')
    valid_img = np.load('data/validation_images.npy')
    names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


    #multiply elements of train_box by the size of the image
    train_box=train_box*train_img.shape[1]
    valid_box=valid_box*valid_img.shape[1]
    # plot_image(train_img, train_box)


    model = create_model((75, 75, 1,))
    model.summary()

    model.compile(optimizer='adam',
                loss={
                    'classifier': 'categorical_crossentropy',
                    'box': 'mse'
                },
                metrics={
                    'classifier': 'acc',
                    'box': 'mse'
                })
    model.summary()
    # history = model.fit(train_img, (train_label, train_box),
    #                     validation_data=(valid_img, (valid_label, valid_box)),
    #                     epochs=10)


    # #save model and history
    # model.save('BB_model.h5')
    # np.save('BB_history.npy', history.history)

    #load model and history
    model = keras.models.load_model('BB_model.h5')
    history = np.load('BB_history.npy', allow_pickle='TRUE').item()


    #plot history
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(history['classifier_loss'], label='train classifier')
    plt.plot(history['val_classifier_loss'], label='validation classifier')

    plt.plot(history['box_loss'], label='train box', c='green')
    #plt.plot(history['val_box_loss'], label='validation box', c='purple')


    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc="upper left")

    plt.subplot(2,1,2)
    plt.plot(history['classifier_acc'], label='train classifier acc')
    plt.plot(history['val_classifier_acc'], label='validation classifier acc')


    plt.title('Accuracy')

    plt.ylabel('Accuracy')
    plt.legend(loc="upper left")

    plt.twinx()
    plt.plot(history['box_mse'], label='train box mse',c='green')
    #plt.plot(history['val_box_mse'], label='validation box mse',c='purple')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(loc="lower left")


    sample_valid_img = valid_img[0:12]
    sample_valid_label = model.predict(sample_valid_img)[0]
    sample_valid_box = model.predict(sample_valid_img)[1]

    #plot sample images
    plot_image(sample_valid_img, sample_valid_box, sample_valid_label, names)

    #Evaluate Intersection over Union

    goodIOU = 0
    allIOU = 0
    pred_box=model.predict(valid_img)
    for i in range(len(valid_img)):
        allIOU += 1
        if InteresectionOverUnion(valid_box[i], pred_box[1][i] ) > 0.6:
            goodIOU += 1

    print('Good over All Intersection over Union: ', goodIOU/allIOU)



    plt.show()


    return 0







if __name__ == '__main__':
    main()
