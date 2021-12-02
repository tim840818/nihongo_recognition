import os, re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

import tensorflow
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from tensorflow.keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.layers import Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model


def get_romanji(filename): # extract romanji from a filename
    keyword = re.search(r'kana(\w+?)\d+', filename).group(1)
    return keyword.lower()

def romanji_to_dict(S): # construct dict from Romanji series
    romanji_list = list(set(S))
    romanji_dict = {}
    for i, romanji in enumerate(romanji_list):
        romanji_dict[romanji] = i
    return romanji_dict

def std_X(X):  # standardize X_train and X_test from 0-255 to 0-1
    X = X.astype('float32') / 255
    return X

if __name__ == "__main__":
    ## import data
    img_test = cv.imread("hiragana-dataset-master/hiragana_images/kanaBA0.jpg", cv.IMREAD_GRAYSCALE)
    (img_rows, img_cols) = img_test.shape

    ## construct datatable
    handwriting_list = []
    dir_name = "hiragana-dataset-master/hiragana_images"
    for filename in os.listdir(dir_name):
        picture_label_list = [cv.imread(
            f"{dir_name}/{filename}", cv.IMREAD_GRAYSCALE), filename, get_romanji(filename)]
        handwriting_list.append(picture_label_list)
    handwriting_table = pd.DataFrame(handwriting_list, columns=["Handwriting", "Filename", "Romanji"])

    ## add labels to the datatable w.r.t. the romanji
    romanji_dict = romanji_to_dict(handwriting_table["Romanji"])
    romanji_categories = len(romanji_dict)
    handwriting_table["Label"] = handwriting_table["Romanji"].map(lambda x: romanji_dict[x])

    ## construct X and y data from data-table
    X = handwriting_table["Handwriting"].to_numpy()
    y = handwriting_table["Label"].to_numpy()
    X = np.array([X[i].reshape(img_rows, img_cols) for i in range(X.shape[0])])
    X = X.reshape(X.shape[0], img_rows, img_cols, 1)

    ## split the data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    Y_train = np_utils.to_categorical(y_train, romanji_categories)
    Y_test = np_utils.to_categorical(y_test, romanji_categories)

    ## standardize X_train and X_test
    X_train = std_X(X_train)
    X_test = std_X(X_test)

    ## construct a CNN model
    input_shape = (img_rows, img_cols, 1)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu', padding='same',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(romanji_categories, activation='softmax'))

    print('model.summary:')
    model.summary()

    model.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])

    ## uae augmentation to generate more data from the training data
    datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        # shear_range=0.1,
        zoom_range=0.1,
        fill_mode='nearest',
        validation_split=0.2)

    ## train the model with data (including augmentaion datagen)
    batch_size = 128
    nb_epoch = 32

    datagen.fit(X_train)
    train_history = model.fit(datagen.flow(X_train, Y_train, batch_size=32, subset='training'),
                            validation_data=datagen.flow(X_train, Y_train, batch_size=8, subset='validation'),
                            batch_size=batch_size, epochs=nb_epoch, verbose=1)

    ## show the training details
    plt.figure(figsize=(18,4))

    plt.subplot(1, 2, 1)
    plt.suptitle('Optimizer : RMSprop', fontsize=12)
    plt.ylabel('Accuracy', fontsize=20)
    plt.plot(train_history.history['accuracy'], color='b', label='Training Accuracy')
    plt.plot(train_history.history['val_accuracy'], color='r', label='Validation Accuracy')
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.ylabel('Loss', fontsize=20)
    plt.plot(train_history.history['loss'], color='b', label='Training Loss')
    plt.plot(train_history.history['val_loss'], color='r', label='Validation Loss')
    plt.legend(loc='upper right')

    plt.show()
    
    ## evaluate the model with test data
    print("------evaluation------------")
    score = model.evaluate(X_test, Y_test, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    print("----------------------------")

    ## construct inverse romanji dict to translate Label back to Romanji
    inv_romanji_dict = dict((v, k) for k, v in romanji_dict.items())

    ## get the predicted results from the model
    nVal = X_test.shape[0]
    print(f"number of testing data: {nVal}")
    prob = model.predict(X_test)
    predictions = np.argmax(prob, axis=1)
    ground_truth = y_test

    ## show twenty results (of total 200 testing data) in plots
    num_test = 20
    num_eachrow = 5
    for i in range(num_test):
        ax = plt.subplot(num_test/num_eachrow, num_eachrow, i+1)
        plt.title(inv_romanji_dict[predictions[i]])
        plt.imshow(X_test[i])
    plt.show()

    ## check all results
    print(predictions)
    print(ground_truth)
    errors = np.where(predictions != ground_truth)[0]
    print(errors)
    print("Number of errors = {}/{}".format(len(errors), nVal))
    for error in errors:
        plt.title(inv_romanji_dict[predictions[error]])
        plt.imshow(X_test[error])
        plt.show()

    ## creates a HDF5 file
    # model.save('model_hiragana_recognition_cnn.h5') 