import sys
import argparse
import classifier
import numpy as np

DETECTOR_DATA_DIR = "/home/wangchen/Sources/aletheia/detector/FlashDetector/data/"
FLASH_APP_DIR = "/home/wangchen/Sources/FLASH4.4/objects/"
FLASH_APPS = ["Blast2", "BlastBS", "BrioWu", "Cellular",
        "DMReflection", "RHD_Sod", "Sedov", "Sod"]

def fun_train(app, N):
    data_dir = DETECTOR_DATA_DIR + "train/"+app
    model_file = app + ".h5"        # existing model file
    classifier.train(model_file, data_dir, N, epochs=10)
    classifier.evaluation(model_file, data_dir)
    return 0

def fun_train_multi():
    error_data_dir = FLASH_APP_DIR + "Sod_multi/data/"
    clean_data_dir = FLASH_APP_DIR + "Sod/"

    model_file = "Sod_multi.h5"        # existing model file

    # Get train_X and train_y
    import preprocess
    import glob
    import random

    clean_X = preprocess.read_hdf5_dataset(clean_data_dir)
    clean_X, clean_y = preprocess.preprocess_for_classifier(clean_X, 1)

    error_X = []
    error_y = []
    for filename in glob.iglob(error_data_dir+"*0001"):
        step = int(filename.split(',')[0].split('[')[1])
        step = step + int(filename.split('_')[-1])

        blockId = int(filename.split(',')[1].replace(' ', ''))
        print filename, blockId

        frame = preprocess.hdf5_to_numpy(filename)
        if frame.shape[0] > blockId:
            block = frame[blockId]
            if np.isnan(block).any() or np.isinf(block).any() or np.max(block) > 100:
                continue
            else:
                # Append error data
                std = np.std(block)
                if std == 0 : std = np.max(block)
                block = block / std
                error_X.append(block)
                error_y.append(1)

                # Append clean data
                '''
                step_str = ("0000"+str(step))[-4:]
                clean_frame = preprocess.hdf5_to_numpy(clean_data_dir + "sod_hdf5_plt_cnt_" + step_str)
                if clean_frame.shape[0] > blockId:
                    clean_block = clean_frame[blockId]
                    std = np.std(clean_block)
                    if std == 0 : std = np.max(block)
                    clean_block = clean_block / std
                    train_X.append(clean_block)
                    train_y.append(0)
                '''

    print clean_X.shape
    train_X = np.vstack((clean_X, np.array(error_X)))
    train_y = np.concatenate((clean_y, np.array(error_y)))
    train_X = train_X.reshape(train_X.shape+(1,))
    classifier.train_multi(model_file, train_X, train_y)


def fun_test(app):
    data_dir = DETECTOR_DATA_DIR + "test/"+app
    model_file = app + ".h5"        # existing model file
    classifier.evaluation(model_file, data_dir)
    return 0

def fun_testall():
    for app in FLASH_APPS:
        print "Testing", app
        fun_test(app)
    return 0



if __name__ == "__main__":

    # allow three actions: {train, test, testall}
    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers(dest="action")
    sp_train = sp.add_parser("train", help="train a model")
    sp_train_multi = sp.add_parser("trainmulti", help="train a model")
    sp_test = sp.add_parser("test", help="test a model")
    sp_testall = sp.add_parser("testall", help="test all available models")

    # arguments for action "train"
    sp_train.add_argument("app", help="the app's name", choices=FLASH_APPS)
    sp_train.add_argument("--N", help="duplicate N times of dataset for training", type=int, default=1)

    # arguments for action "test"
    sp_test.add_argument("app", help="the app's name", choices=FLASH_APPS)

    args = parser.parse_args()
    if args.action == "train":
        fun_train(args.app, args.N)
    elif args.action == "trainmulti":
        fun_train_multi()
    elif args.action == "test":
        fun_test(args.app)
    elif args.action == "testall":
        fun_testall()
