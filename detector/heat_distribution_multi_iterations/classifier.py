import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import heat_distribution as hd

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation, Dropout, Conv2D, Flatten
from keras.optimizers import SGD

# Get heat distribution matrices and its error positions
def get_imgs_and_errors():
    MM, error_positions = hd.get_random_error_data()
    has_error = np.zeros(len(error_positions))
    for i in range(len(error_positions)) :
        if error_positions[i, 0] != -1 or error_positions[i, 1] != -1:
            has_error[i] = 1
    imgs = MM.reshape(MM.shape+(1,))  # add a channel dimension, make it to the shape of (MM.shape, 1)
    print "imgs:", imgs.shape, ", has_error", has_error.shape
    return imgs, has_error

def create_dnn():
    model = Sequential([
            Conv2D(46, (3,3), input_shape=(hd.ROWS, hd.COLUMNS, 1), activation='relu'),
            Conv2D(46, (3,3), activation='relu'),
            Dropout(0.2),
            Flatten(),
            Dense(1, activation='sigmoid')    # last layer output 0 or 1
    ])
    model.compile('adadelta', 'binary_crossentropy')
    return model

def get_train_and_test_set(imgs, has_errors):
    # Split training and test
    i = int(0.8 * len(imgs))
    train_X = imgs[:i]
    test_X = imgs[i:]
    train_y = has_errors[:i]
    test_y = has_errors[i:]
    eva_X = imgs[i:]
    eva_y = has_errors[i:]
    return train_X, test_X, train_y, test_y, eva_X, eva_y

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

    print len(pred)
    print 'tp:', tp, 'tn:', tn, 'fp:', fp, 'fn:', fn
    print 'recall:', tp/(tp+fn), ', precision:', tp/(tp+fp)
    print 'false positive rate:', fp/(fp+tn)

imgs, has_errors = get_imgs_and_errors()
train_X, test_X, train_y, test_y, eva_X, eva_y = get_train_and_test_set(imgs, has_errors)

model = create_dnn()
model.fit(train_X, train_y, epochs=20, validation_data=(test_X, test_y), verbose=2)
model.save_weights('heat_classifier_weight.h5')
#model.load_weights('heat_classifier_weight.h5')
#model = load_model('heat_classifier_model.h5')

print "Test..."
pred = model.predict(eva_X)
print_results(np.round(pred), eva_y)

