import sys
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

INPUT_ROWS = 50
INPUT_COLS = 50

def create_dnn():
    model = Sequential([
            Conv2D(46, (3,3), input_shape=(INPUT_ROWS, INPUT_COLS, 1), activation='relu'),
            #Conv2D(46, (3,3), activation='relu'),
            Dropout(0.2),
            Flatten(),
            Dense(1, activation='sigmoid')    # last layer output 0 or 1
    ])
    model.compile('adadelta', 'binary_crossentropy')
    #model.compile('rmsprop', 'binary_crossentropy')
    #model.compile('sgd', 'binary_crossentropy', metrics=['accuracy'])
    return model

def get_training_data(data_dir, N):
    dataset = preprocess.read_hdf5_dataset(data_dir, INPUT_ROWS, INPUT_COLS)
    dataset, has_error = preprocess.preprocess_for_classifier(dataset, N)
    dataset = dataset.reshape(dataset.shape+(1,))
    print "dataset shape(X):", dataset.shape
    print "hash error shape(Y):", has_error.shape
    return dataset, has_error

def print_results(pred, real):
    # recall, false positive, true positive
    tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0
    no_error_count = 0

    for i in range(len(pred)):
        if i < 20: print pred[i], "---", real[i]
        if pred[i] == 1 and real[i] == 1 : tp = tp + 1      # true positive
        if pred[i] == 0 and real[i] == 0 : tn = tn + 1      # true negative
        if pred[i] == 1 and real[i] == 0 : fp = fp + 1      # false positive
        if pred[i] == 0 and real[i] == 1 : fn = fn + 1      # false negative

    recall, precision, fpr  = 'NA', 'NA', 'NA'
    if tp+fn != 0: recall = tp/(tp+fn)
    if tp+fp != 0: precision = tp/(tp+fp)
    if fp+tn != 0: fpr = fp/(fp+tn)
    print 'tp:', tp, 'tn:', tn, 'fp:', fp, 'fn:', fn
    print 'recall:', recall, ', precision:', precision, ', false positive rate:', fpr


def train(data_dir, N=20):
    # save the best model after each epoch
    checkpoint = ModelCheckpoint("classifier.h5")
    train_X, train_y = get_training_data(data_dir, N)
    #model = create_dnn()
    #model.fit(train_X, train_y, epochs=3, validation_split=0.25, verbose=2, callbacks=[checkpoint])

    # Load existing model and continue to train
    model = load_model('classifier.h5')
    model.fit(train_X, train_y, epochs=3, validation_split=0.25, verbose=2, callbacks=[checkpoint])

def evaluation(data_dir, N=1):
    print "Evaluating..."
    eva_X, eva_y = get_training_data(data_dir, N)
    model = load_model('classifier.h5')
    pred = model.predict(eva_X)
    print_results(np.round(pred), eva_y)

def predict(dataset_dir):
    model = load_model('classifier.h5')
    paths = ["/*_51.h5", "/*_52.h5", "/*_53.h5", "/*_54.h5", "/*_55.h5",
                "/*_56.h5", "/*_57.h5", "/*_58.h5", "/*_59.h5", "/*_60.h5",
                "/*_61.h5", "/*_62.h5", "/*_63.h5", "/*_64.h5", "/*_65.h5",
                "/*_66.h5", "/*_67.h5", "/*_68.h5", "/*_69.h5", "/*_70.h5",
                "/*_71.h5", "/*_72.h5", "/*_73.h5", "/*_74.h5", "/*_75.h5"]
    for path in paths:
        total = 0; acc = 0
        for filename in glob.iglob(dataset_dir+path):
            total = total + 1.0
            frame = preprocess.hdf5_to_numpy(filename)
            blocks = preprocess.split_to_blocks(frame, INPUT_ROWS, INPUT_COLS)
            X = np.array(blocks)
            X = X.reshape(X.shape+(1,))
            prediction = np.round(np.max(model.predict(X)))
            acc = acc + prediction
            #print filename, prediction
        print path, ", acc: ", acc / total, ", total:", total

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "classifier.py data_dir N"
    else :
        #train(sys.argv[1], int(sys.argv[2]))
        #evaluation(sys.argv[1])
        predict(sys.argv[1])
