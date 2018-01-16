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
    imgs, has_error = caf.get_with_error_data()
    print "imgs:", imgs.shape, ", has_error: ", has_error.shape
    return imgs, has_error

def create_dnn():
    model = Sequential([
            Conv2D(46, (3,3), input_shape=(caf.nx, caf.ny, 3), activation='relu'),
            Conv2D(46, (3,3), activation='relu'),
            Dropout(0.2),
            Flatten(),
            Dense(1, activation='sigmoid')    # last layer output whether there's an error
    ])
    model.compile('adadelta', 'binary_crossentropy')
    return model

def split_train_and_test_set(imgs, bboxes):
    # Split training and test
    i = int(0.8 * len(imgs))
    train_X = imgs[:i]
    test_X = imgs[i:]
    train_y = bboxes[:i]
    test_y = bboxes[i:]
    eva_imgs = imgs[i:]
    eva_bboxes = bboxes[i:]
    return train_X, test_X, train_y, test_y, eva_imgs, eva_bboxes

imgs, bboxes = get_imgs_and_bboxes()
train_X, test_X, train_y, test_y, eva_imgs, eva_bboxes = split_train_and_test_set(imgs, bboxes)

#model = create_dnn()
#model.fit(train_X, train_y, epochs=3, validation_data=(test_X, test_y), verbose=2)
# save both model and weights
#model.save('my_model.h5')
model = load_model('my_model.h5')

print "Test..."
pred_bboxes = model.predict(eva_imgs)
print pred_bboxes.shape

acc = 0.0
for i in range(len(pred_bboxes)):
    if i < 20: print np.round(pred_bboxes[i]), "---", eva_bboxes[i]
    if np.round(pred_bboxes[i]) == eva_bboxes[i]:
        acc = acc+1
acc = acc / len(pred_bboxes)
print acc
