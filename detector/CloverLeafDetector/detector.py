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
            Conv2D(46, (3,3), activation='relu'),
            Dropout(0.2),
            Flatten(),
            Dense(2)    # last layer output the position of the error
    ])
    model.compile('adadelta', 'mse')
    return model

def get_training_data(data_dir, N):
    dataset = preprocess.read_hdf5_dataset(data_dir, INPUT_ROWS, INPUT_COLS)
    dataset, error_positions = preprocess.preprocess_for_detector(dataset, N)
    error_positions[:,0] = error_positions[:,0] / INPUT_ROWS
    error_positions[:,1] = error_positions[:,1] / INPUT_COLS
    dataset = dataset.reshape(dataset.shape+(1,))
    print "dataset shape(X):", dataset.shape
    print "error position shape(Y):", error_positions.shape
    return dataset, error_positions

def print_results(pred, truth):
    pred[:,0] = pred[:,0] * INPUT_ROWS
    pred[:,1] = pred[:,1] * INPUT_COLS
    truth[:,0] = truth[:,0] * INPUT_ROWS
    truth[:,1] = truth[:,1] * INPUT_COLS
    pred = np.round(pred)
    truth = np.round(truth)

    acc = 0.0
    for i in range(len(pred)):
        if i < 20: print pred[i], "---", truth[i]
        x1, y1, x2, y2 = int(pred[i,0]), int(pred[i,1]), int(truth[i,0]), int(truth[i,1])
        if x1==x2 and y1==y2: acc = acc+1
    acc = acc / len(pred)
    print acc


def train(data_dir, N = 20):
    # save the best model after each epoch
    checkpoint = ModelCheckpoint("classifier.h5")
    train_X, train_y = get_training_data(data_dir, N)
    model = create_dnn()
    model.fit(train_X, train_y, epochs=2, validation_split=0.25, verbose=2, callbacks=[checkpoint])

    # Load existing model and continue to train
    #model = load_model('detector.h5')
    #model.fit(train_X, train_y, epochs=5, validation_split=0.25, verbose=2, callbacks=[checkpoint])


def evaluation(data_dir, N = 1):
    print "Evaluating..."
    eva_X, eva_y = get_training_data(data_dir, N)
    model = load_model('detector.h5')
    pred = model.predict(eva_X)
    print_results(pred, eva_y)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "classifier.py data_dir N"
    else :
        train(sys.argv[1], N=1)
        evaluation(sys.argv[1])
