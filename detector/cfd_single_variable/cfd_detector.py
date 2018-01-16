import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cfd_cavity_flow as caf

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, Flatten
from keras.optimizers import SGD
from keras.models import load_model

# Get heat distribution matrices and its error positions
def get_imgs_and_bboxes():
    MM, error_positions = caf.get_with_error_data()
    error_positions[:,0] = error_positions[:,0] / caf.nx
    error_positions[:,1] = error_positions[:,1] / caf.ny
    imgs = MM.reshape(MM.shape+(1,))  # add a channel dimension, make it to the shape of (MM.shape, 1), i.e. grayscale image

    print "imgs:", imgs.shape, ", error_positions:", error_positions.shape
    return imgs, error_positions

def create_dnn():
    model = Sequential([
            Conv2D(46, (3,3), input_shape=(caf.nx, caf.ny, 1), activation='relu'),
            Conv2D(46, (3,3), activation='relu'),
            Dropout(0.2),
            Flatten(),
            Dense(2)    # last layer output (x,y)
    ])
    model.compile('adadelta', 'mse')
    return model

def split_train_and_test_set(imgs, error_positions):
    # Split training and test
    i = int(0.8 * len(imgs))
    train_X = imgs[:i]
    test_X = imgs[i:]
    train_y = error_positions[:i]
    test_y = error_positions[i:]
    eva_imgs = imgs[i:]
    eva_positions = error_positions[i:]
    return train_X, test_X, train_y, test_y, eva_imgs, eva_positions

imgs, bboxes = get_imgs_and_bboxes()
train_X, test_X, train_y, test_y, eva_imgs, eva_positions = split_train_and_test_set(imgs, bboxes)

#model = create_dnn()
#model.fit(train_X, train_y, epochs=50, validation_data=(test_X, test_y), verbose=2)
# save both model and weights
#model.save('my_model.h5')
model = load_model('my_model.h5')

print "Test..."
pred_bboxes = model.predict(eva_imgs)
print pred_bboxes.shape

acc = 0.0
for i in range(len(pred_bboxes)):
    pred, truth = np.round(pred_bboxes[i]*caf.nx), np.round(eva_positions[i]*caf.nx)
    if i < 20: print pred, "---", truth
    x1, y1, x2, y2 = int(pred[0]), int(pred[1]), int(truth[0]), int(truth[1])
    if x1==x2 and y1==y2: acc = acc+1
acc = acc / len(pred_bboxes)
print acc
