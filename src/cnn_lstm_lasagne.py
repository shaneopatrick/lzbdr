# VGG_CNN_S, model from the paper:
# "Return of the Devil in the Details: Delving Deep into Convolutional Nets"
# 13.1% top-5 error on ILSVRC-2012-val
# Original source: https://gist.github.com/ksimonyan/fd8800eeb36e276cd6f9
# License: non-commercial use only

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
#import cv2
import math
import os
import boto
import time
import io
import pdb

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

import theano
import theano.tensor as T

#from nolearn.lasagne import NeuralNet
from lasagne.layers import InputLayer, get_output
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import SpatialPyramidPoolingLayer
from lasagne.layers import DimshuffleLayer, LSTMLayer, ReshapeLayer, CustomRecurrentLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import softmax
from lasagne.layers import set_all_param_values, get_all_params
from lasagne.objectives import categorical_crossentropy
from lasagne.updates import nesterov_momentum
import lasagne

from helpers import score_top5, score_top3, model_summary_plots

os.environ["THEANO_FLAGS"] = 'allow_gc=False, optimizer_including="local_ultra_fast_sigmoid", nvcc.fastmath=True, use_fast_math=True, optimizer=fast_compile'
access_key = os.environ['AWS_ACCESS_KEY_ID']
sec_access_key = os.environ['AWS_SECRET_ACCESS_KEY']


def split_data(X_train, X_test, y_train, y_test):
    img_rows, img_cols = 224, 224

    X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('int32')
    y_test = y_test.astype('int32')

    X_train /= 255
    X_test /= 255

    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    X_test, X_val = X_test[:-64], X_test[-64:]
    y_test, y_val = y_test[:-64], y_test[-64:]

    print('\nX_train shape: {}\n y_train shape: {}\n X_val shape: {}\n y_val shape: {}\n X_test shape: {}\n y_test shape: {}'.format(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape))

    return X_train, y_train, X_val, y_val, X_test, y_test

def build_model(input_var):
    net = {}

    net['input'] = InputLayer((None, 3, 224, 224), input_var=input_var)
    net['conv1'] = ConvLayer(net['input'],
                             num_filters=96,
                             filter_size=7,
                             stride=2,
                             flip_filters=False)
    # caffe has alpha = alpha * pool_size
    net['norm1'] = NormLayer(net['conv1'], alpha=0.0001)
    net['pool1'] = PoolLayer(net['norm1'],
                             pool_size=3,
                             stride=3,
                             ignore_border=False)
    net['conv2'] = ConvLayer(net['pool1'],
                             num_filters=256,
                             filter_size=5,
                             flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2'],
                             pool_size=2,
                             stride=2,
                             ignore_border=False)
    net['conv3'] = ConvLayer(net['pool2'],
                             num_filters=512,
                             filter_size=3,
                             pad=1,
                             flip_filters=False)
    net['conv4'] = ConvLayer(net['conv3'],
                             num_filters=512,
                             filter_size=3,
                             pad=1,
                             flip_filters=False)
    net['conv5'] = ConvLayer(net['conv4'],
                             num_filters=512,
                             filter_size=3,
                             pad=1,
                             flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5'],
                             pool_size=3,
                             stride=3,
                             ignore_border=False)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['drop6'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['drop6'], num_units=4096)
    net['drop7'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(net['drop7'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net

def build_cnn(input_var, num_classes):
    #Get back the convolutional part of a VGG network trained on ImageNet
    network = build_model(input_var)

    with open('vgg_cnn_s.pkl', 'rb') as f:
        params = pickle.load(f, encoding='latin-1')
    set_all_param_values(network.values(), params['values'])
    # pool5 shape:  (None, 512, 6, 6)

    del network['fc6']
    del network['drop6']
    del network['fc7']
    del network['drop7']
    del network['fc8']

    network['shuff6'] = DimshuffleLayer(network['pool5'], (0, 2, 1, 3))
    network['lstm7'] = LSTMLayer((None, 512, 6, 6), network['shuff6'])
    network['rshp8'] = ReshapeLayer(network['lstm7'], (-1, 64))
    network['fc9'] = DenseLayer(network['rshp8'], num_units=4096)
    network['drop9'] = DropoutLayer(network['fc9'], p=0.5)
    network['fc10'] = DenseLayer(network['drop9'], num_units=num_classes, nonlinearity=None)
    network['prob'] = NonlinearityLayer(network['fc10'], softmax)
    return network



def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    #pdb.set_trace()
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    else:
        indices = np.arange(len(inputs))
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs[excerpt], targets[excerpt]


if __name__ == '__main__':

    num_epochs = 50
    b_size = 32

    X_train = np.load('X_train10_canv.npy')
    X_test = np.load('X_test10_nocanv.npy')
    y_train = np.load('y_train10_canv.npy')
    y_test = np.load('y_test10_nocanv.npy')

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X_train, X_test, y_train, y_test)

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network = build_cnn(input_var, 10)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network['prob'])
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network.values(), trainable=True)
    updates = lasagne.updates.sgd(
            loss, params, learning_rate=0.001)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network['prob'], deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], outputs=loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], outputs=[test_loss, test_acc])

    # Compilation of theano functions
    # Obtaining the probability distribution over classes
    test_pred = lasagne.layers.get_output(network['prob'], deterministic=True)
    # Returning the predicted output for the given minibatch
    test_fn2 =  theano.function([network['input'].input_var], [T.argmax(test_pred, axis=1)])



    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, b_size, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        #pdb.set_trace()
        for batch in iterate_minibatches(X_val, y_val, b_size, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_pred = np.empty((0))
    test_true = np.empty((0))
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, b_size, shuffle=False):
        inputs, targets = batch
        y_pred, = test_fn2(inputs)
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
        test_pred = np.append(test_pred, y_pred, axis=0)
        test_true = np.append(test_true, targets, axis=0)
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))


    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model_lasagne_full.npz', *lasagne.layers.get_all_param_values(network.values()))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)
