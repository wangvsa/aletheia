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
    print "frames: ", MM.shape, ", error_positions: ", error_positions.shape
    return MM, error_positions

eva_imgs, eva_positions = get_imgs_and_bboxes()

classifier = load_model('classifier_model.h5')
detector = load_model('detector_model.h5')

print "Evaluation..."
classifier_pred = classifier.predict(eva_imgs)
detector_pred = detector.predict(eva_imgs)
classifier_pred = np.round(classifier_pred)
detector_pred = np.round(detector_pred[:]*caf.nx)


fp = 0.0
recall = 0.0
errors_count = 0
for i in range(len(eva_positions)):
    has_error = 1
    if(eva_positions[i,0] == -caf.nx and eva_positions[i,1] == -caf.ny):
        has_error = 0
    # calculate false positive rate
    if has_error == 0 and classifier_pred[i] == 1:
        print 'error', i
        fp = fp + 1

    # calculate recall
    if has_error == 1 :
        errors_count = errors_count + 1
    if classifier_pred[i] == 1 and eva_positions[i, 0] == detector_pred[i, 0] and eva_positions[i, 1] == detector_pred[i, 1]:
        recall = recall + 1

print fp, fp/(len(eva_positions)-errors_count)
print recall,  errors_count, recall/errors_count

