from data_reader import read_data


class AdaptiveDetector:
    def __init__(self):
        self.fp = 0
        self.filterBound = 0.000001

    def lcf(self, v1, v2):
        return 2*v1-v2
    def qcf(self, v1, v2, v3):
        return 3*v1-3*v2+v3
    def ccf(self, v1, v2, v3, v4):
        return 4*v1-6*v2+4*v3-v4

    '''
    Perform detection for the given data point
    args:
        err_v: the data value after one bit flip
        v, v1, v2, v3, v4, v5: 6 consecutive data points, v is the current one
    return:
        True if its an error, Flase otherwise
    '''
    def detect_one_point(self, v, v1, v2, v3, v4, v5):
        # lcf:
        pred = self.lcf(v1, v2)
        lastPredErr = abs(self.lcf(v2, v3) - v1)
        # qcf
        #pred = qcf(v1, v2, v3)
        #lastPredErr = abs(qcf(v2, v3, v4) - v1)
        # ccf
        #pred = self.ccf(v1, v2, v3, v4)
        #lastPredErr = abs(self.ccf(v2, v3, v4, v5) - v1)

        valueRange = 1
        radius = valueRange * self.filterBound + lastPredErr

        upperBound = pred + radius + radius * self.fp
        lowerBound = pred - radius - radius * self.fp

        result = False
        if v > upperBound or v < lowerBound:
            print(v, pred, upperBound, lowerBound)
            result = True

        return result

    '''
    Perform detection on a data frame
    args:
        d, d1, d2, d3, d4, d5: 5 consecutive data frames, d is the current one
    '''
    def detect(self, d, d1, d2, d3, d4, d5):
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                hasError = self.detect_one_point(d[i,j], d1[i,j], d2[i,j], d3[i,j], d4[i,j], d5[i,j])
                if hasError:
                    return True
        return False



'''
Test the recall of the AID
We insert an error in each frame and see if AID can detect it
'''
def test_recall(prefix):
    aid = AdaptiveDetector()

    d5 = read_data(prefix, 0)
    d4 = read_data(prefix, 1)
    d3 = read_data(prefix, 2)
    d2 = read_data(prefix, 3)
    d1 = read_data(prefix, 4)

    # start from the 6th frame
    recall = 0
    for it in range(5, 200):

        d = read_data(prefix, it)

        # insert an error
        org = d[100, 100]
        d[100, 100] = org * 3

        recall += aid.detect(d, d1, d2, d3, d4)

        d[100, 100] = org   # restore the correct value before next detection

        d5 = d4
        d4 = d3
        d3 = d2
        d2 = d1
        d1 = d

        print("it:", it, " recall:", recall)


# Perform the detection on clean data to get false positive rate
def test_fp(prefix):
    aid = AdaptiveDetector()

    d5 = read_data(prefix, 0)
    d4 = read_data(prefix, 1)
    d3 = read_data(prefix, 2)
    d2 = read_data(prefix, 3)
    d1 = read_data(prefix, 4)

    # start from the 6th frame
    for it in range(5, 200):
        d = read_data(prefix, it)
        aid.fp += aid.detect(d, d1, d2, d3, d4, d5)
        d5 = d4
        d4 = d3
        d3 = d2
        d2 = d1
        d1 = d
        print("it:", it, " fp:", aid.fp)

#test_recall("/home/wangchen/Flash/Sedov/clean/sedov_hdf5_chk_")
test_fp("/home/wangchen/Flash/Sedov/clean/sedov_hdf5_chk_")
