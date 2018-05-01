import h5py
import sys
import random, math
import numpy as np
from sklearn.svm import SVR
from bits import bit_flip

IMPACT_ERROR_BOUND = 0.00078125

def get_flip_error(val):
    tmp = val
    # flip bits 30 times
    for i in range(30):
        while True:
            pos = random.randint(0,63)
            tmp = bit_flip(tmp, pos)
            if not math.isnan(tmp) and not math.isinf(tmp):
                break
    error = tmp
    return error


# Read dens data from hdf5 file
# and flatten it
def read_data(filename):
    f = h5py.File(filename)
    data = f['dens'][0,0]
    return data.flatten()


def detect(svr, data, prev_max_error, fp):

    has_error = False   # wether this iteration has errors
    max_error = 0       # keep record the max prediction error made in this iteration

    threshold = (np.max(data) - np.min(data)) * IMPACT_ERROR_BOUND
    threshold = (threshold + prev_max_error) * max(fp, 1)
    #print("threashold:", threshold)

    for i in range(1, len(data)-1):

        X, y = np.array([[i-1], [i+1]]), np.array([data[i-1], data[i+1]])
        svr.fit(X, y)
        pred, real = svr.predict([[i]])[0], data[i]

        err = abs(pred - real)
        max_error = max(max_error, err)
        if err > threshold:
            has_error = True
            #print(X, y, pred, real)

    return  has_error, max_error


def test_ssd(start_iter, error_iter):
    #error_iter = int(sys.argv[1])    # at which iteration we inject the error

    prev_max_error, fp = 0, 0

    svr = SVR(kernel="rbf", C=1e3, gamma=0.1)

    for i in range(start_iter, error_iter+1):
        #if i < error_iter:
        #    filename = "./clean/sod_hdf5_plt_cnt_" + ("0000"+str(i))[-4:]
        #else:
        #    filename = "./error_30_38_147_237hdf5_plt_cnt_" + ("0000"+str(i))[-4:]
        #print(filename)

        filename = "./clean/sod_hdf5_plt_cnt_" + ("0000"+str(i))[-4:]
        data = read_data(filename)

        # Inject an error here
        if i == error_iter:
            pos = random.randint(2, len(data)-2)
            old = data[pos]
            data[pos] = get_flip_error(data[pos])
            print("insert error at %s, old: %s, new: %s" %(pos, old, data[pos]))

        has_error, prev_max_error = detect(svr, data, prev_max_error, fp)
        if has_error and i != error_iter:
            fp += 1
        print(i, fp, has_error, prev_max_error)
    return has_error


if __name__ == "__main__":
    results = []
    for start_iter in range(50, 51):
        res = test_ssd(start_iter, start_iter+10)
        results.append(res)

    print(np.sum(results))

