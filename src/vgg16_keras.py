import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import pickle
import math
import os
import boto

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential, Model, load_model
from keras import optimizers
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.layers import Input, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from helpers import *

sys.setrecursionlimit(1000000)


os.environ["THEANO_FLAGS"] = 'allow_gc=False, optimizer_including="local_ultra_fast_sigmoid", nvcc.fastmath=True, use_fast_math=True, optimizer=fast_compile, borrow=True'

def split_data(X, y, num_classes):

    # le = LabelEncoder()
    # y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1903)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    X_test, X_val = X_test[:-4000], X_test[-4000:]
    y_test, y_val = y_test[:-4000], y_test[-4000:]

    # convert class vectors to binary class matrices
    Y_train = to_categorical(y_train, num_classes)
    Y_val = to_categorical(y_val, num_classes)
    Y_test = to_categorical(y_test, num_classes)

    print('\nX_train shape: {}\n y_train shape: {}\n X_val shape: {}\n y_val shape: {}\n X_test shape: {}\n y_test shape: {}'.format(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape))

    return X_train, X_test, Y_train, Y_test, X_val, Y_val, y_test, y_val

def build_vgg16(num_classes=200):
    #Get back the convolutional part of a VGG network trained on ImageNet
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
    #model_vgg16_conv.summary()

    #Create your own input format (here 3x200x200)
    input_ = Input(shape=(224,224,3), name = 'image_input')

    #Use the generated model
    output_vgg16_conv = model_vgg16_conv(input_)

    #Add the fully-connected layers
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    #Create your own model
    my_model = Model(inputs=input_, outputs=x)

    # Specify SGD optimizer parameters
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    # Compile model
    my_model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    #In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
    return my_model#, my_model.summary()

def _image_generator(X_train, Y_train):
    train_datagen = image.ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True)
    train_datagen.fit(X_train, seed=1919)
    return train_datagen

#Then training with your data !
def fit_model_vgg16(X_train, X_test, Y_train, Y_test, batch_size, epochs):
    generator = _image_generator(X_train, Y_train)

    #filepath="weights/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    #checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')

    # Change learning rate when learning plateaus
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
              patience=4, min_lr=0.00001)

    # Stop model once it stops improving to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

    # put all callback functions in a list
    callbacks_list = [early_stop, reduce_lr]

    history = my_model.fit_generator(
        generator.flow(X_train, Y_train, batch_size=batch_size),
        steps_per_epoch=(X_train.shape[0] // batch_size),
        epochs=epochs,
        validation_data=(X_val, Y_val),
        callbacks=callbacks_list
        )

    score = my_model.evaluate(X_val, Y_val, verbose=1)
    probs = my_model.predict(X_val, batch_size=batch_size)
    return my_model, score, probs, history


def predictions_test_data(model, X_test, Y_test):
    score = model.evaluate(X_test, Y_test, verbose=1)
    probs = model.predict(X_test, batch_size=32)
    return score, probs


if __name__ == '__main__':
    # Load npz files
    x_ = np.load('X_200.npz')
    y_ = np.load('y_200.npz')

    # Extract numpy arrays from npz
    X, y = load_npz(x_, y_)

    # Split data into train, validation, and test sets
    X_train, X_test, Y_train, Y_test, X_val, Y_val, y_test, y_val = split_data(X, y, 200)

    # Instantiate model
    my_model = build_vgg16(200)

    # Fit model and evaluate
    fit_model, score, probs, history = fit_model_vgg16(X_train, X_val, Y_train, Y_val, batch_size=32, epochs=20)

    # Print results and save performance chart
    print('Val score:', score[0])
    print('Val accuracy:', score[1])
    score_top5(y_val, probs)
    score_top3(y_val, probs)
    model_summary_plots(history, 'top200_vgg16-final')

    # Predict on test set
    t_score, t_probs = predictions_test_data(fit_model, X_test, Y_test)

    # Print results from test set
    print('Test score:', t_score[0])
    print('Test accuracy:', t_score[1])
    score_top5(y_test, t_probs)
    score_top3(y_test, t_probs)

    # Save model architecture to json
    model_json = fit_model.to_json()
    with open("vgg16-top200-final.json", "w") as json_file:
        json_file.write(model_json)

    # Serialize weights to HDF5
    fit_model.save('vgg16-top200-final.h5')

    # Save model history to pickle
    with open('vgg16-top200-history-final.pkl', 'wb') as f:
        for obj in [history]:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save y_test to npz
    np.savez('y_test_vgg16top200-final.npz', y_test)

    # Save predictions matrix to pickle
    with open('test_probs_vgg16top200-final.pkl', 'wb') as ff:
        pickle.dump(t_probs, ff, protocol=pickle.HIGHEST_PROTOCOL)
