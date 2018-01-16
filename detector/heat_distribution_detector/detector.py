import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import heat_distribution as hd

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation, Dropout, Conv2D, Flatten
from keras.optimizers import SGD

# Get heat distribution matrices and its error positions
def get_imgs_and_bboxes():
    MM = hd.heat_distribution()
    MM, bboxes = hd.insert_error(MM)
    bboxes[:,0] = bboxes[:,0] / hd.ROWS
    bboxes[:,1] = bboxes[:,1] / hd.COLUMNS
    imgs = MM.reshape(MM.shape+(1,))  # add a channel dimension, make it to the shape of (MM.shape, 1)

    print "imgs:", imgs.shape, ", bboxes", bboxes.shape
    return imgs, bboxes

def create_dnn():
    model = Sequential([
            Conv2D(46, (3,3), input_shape=(hd.ROWS, hd.COLUMNS, 1), activation='relu'),
            Conv2D(46, (3,3), activation='relu'),
            Dropout(0.2),
            Flatten(),
            Dense(2)    # last layer output (x,y)
    ])
    model.compile('adadelta', 'mse')
    return model

def get_train_and_test_set(imgs, bboxes):
    # Split training and test
    i = int(0.8 * len(imgs))
    train_X = imgs[:i]
    test_X = imgs[i:]
    train_y = bboxes[:i]
    test_y = bboxes[i:]
    test_imgs = imgs[i:]
    test_bboxes = bboxes[i:]
    return train_X, test_X, train_y, test_y, test_imgs, test_bboxes

imgs, bboxes = get_imgs_and_bboxes()
train_X, test_X, train_y, test_y, test_imgs, test_bboxes = get_train_and_test_set(imgs, bboxes)

#model = create_dnn()
#model.fit(train_X, train_y, epochs=50, validation_data=(test_X, test_y), verbose=2)
#model.save_weights('heat_detector_weights.h5')
#model.load_weights('heat_detector_weights.h5')
#model.save('heat_detector_model.h5')
model = load_model('heat_detector_model.h5')

print "Test..."
pred_bboxes = model.predict(test_imgs)
print pred_bboxes.shape

acc = 0.0
for i in range(len(pred_bboxes)):
    pred, truth = np.round(pred_bboxes[i]*hd.ROWS), np.round(test_bboxes[i]*hd.ROWS)
    if i < 20: print pred, "---", truth
    x1, y1, x2, y2 = int(pred[0]), int(pred[1]), int(truth[0]), int(truth[1])
    if x1==x2 and y1==y2: acc = acc+1
acc = acc / len(pred_bboxes)
print acc
