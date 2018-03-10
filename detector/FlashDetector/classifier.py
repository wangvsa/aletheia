import os
import math
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import preprocess

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation, Dropout, Conv2D, Flatten
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

INPUT_ROWS = 60
INPUT_COLS = 60
OVERLAP = 20

def create_dnn():
    model = Sequential([
            Conv2D(46, (3,3), input_shape=(INPUT_ROWS, INPUT_COLS, 3), activation='relu'),
            #Conv2D(46, (3,3), activation='relu'),
            Dropout(0.2),
            Flatten(),
            Dense(1, activation='sigmoid')    # last layer output 0 or 1
    ])
    model.compile('adadelta', 'binary_crossentropy')
    #model.compile('rmsprop', 'binary_crossentropy')
    #model.compile('sgd', 'binary_crossentropy', metrics=['accuracy'])
    return model

def print_results(pred, real):
    # recall, false positive, true positive
    tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0
    no_error_count = 0

    for i in range(len(pred)):
        #if i < 20: print pred[i], "---", real[i]
        if pred[i] == 1 and real[i] == 1 : tp = tp + 1      # true positive
        if pred[i] == 0 and real[i] == 0 : tn = tn + 1      # true negative
        if pred[i] == 1 and real[i] == 0 : fp = fp + 1      # false positive
        if pred[i] == 0 and real[i] == 1 : fn = fn + 1      # false negative

    recall, precision, fpr  = 'NA', 'NA', 'NA'
    if tp+fn != 0: recall = tp/(tp+fn)
    if tp+fp != 0: precision = tp/(tp+fp)
    if fp+tn != 0: fpr = fp/len(pred) #fpr = fp/(fp+tn)
    print 'tp:', tp, 'tn:', tn, 'fp:', fp, 'fn:', fn
    print 'recall:', recall, ', precision:', precision, ', false positive rate:', fpr


def train(model_file, data_dir, N, epochs):
    # save the best model after each epoch
    checkpoint = ModelCheckpoint(model_file)
    train_X, train_y = preprocess.get_classifier_training_data(data_dir, INPUT_ROWS, INPUT_COLS)
    train_X = train_X.reshape(train_X.shape+(1,))
    print "X shape:", train_X.shape
    print "y shape:", train_y.shape

    # Load existing model if exits
    if os.path.isfile(model_file):
        model = load_model(model_file)
    else :
        model = create_dnn()
    model.fit(train_X, train_y, epochs=2, validation_split=0.25, verbose=2, callbacks=[checkpoint])

def train_multi(model_file, train_X, train_y, epochs=10):
    print train_X.shape, train_y.shape
    # save the best model after each epoch
    checkpoint = ModelCheckpoint(model_file)
    # Load existing model if exits
    if os.path.isfile(model_file):
        model = load_model(model_file)
    else :
        model = create_dnn()
    model.fit(train_X, train_y, epochs=epochs, validation_split=0.25, verbose=2, callbacks=[checkpoint])

def evaluation(model_file, data_dir):
    print "Evaluating..."
    dataset, has_error = preprocess.get_classifier_test_data(data_dir, INPUT_ROWS, INPUT_COLS, 0)
    model = load_model(model_file)
    pred = []
    for frame in dataset:
        X = frame.reshape(frame.shape+(1,))
        pred.append(np.round(np.max(model.predict(X))))
    print_results(pred, has_error)

def predict(dataset_dir):
    model = load_model('Sod_multi.h5')
    for i in range(50):
        total = 0; acc = 0
        suffix = ("0000"+str(i))[-4:]
        for filename in glob.iglob(dataset_dir+"*_"+suffix):
            total = total + 1.0
            frame = preprocess.hdf5_to_numpy(filename)
            X = frame.reshape(frame.shape+(1,))

            prediction = np.round(np.max(model.predict(X)))
            if math.isnan(prediction): prediction = 1.0
            acc = acc + prediction
        print suffix, ", acc: ", acc / total, ", total:", total
