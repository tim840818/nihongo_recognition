from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

#for CNN model
from keras.layers import Flatten
from keras.layers import Conv2D, MaxPooling2D

def build_hiragana_cnn(input_shape, romanji_categories):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape,
                 padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.25))
    model.add(Dense(romanji_categories, activation='softmax'))
    return model